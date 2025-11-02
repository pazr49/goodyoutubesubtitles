import os
import uuid
import shutil
import logging
from pathlib import Path # Import Path
import time # Add for timestamps
# subprocess might be removable now from here if not used by other parts
# import subprocess # Add subprocess for yt-dlp

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, BackgroundTasks, Form
from fastapi.responses import FileResponse # Import FileResponse
from sse_starlette.sse import EventSourceResponse
import asyncio, json
from typing import Optional, Dict, Any, List
import uuid as _uuid  # for thread helper
import re # Import re

# Import config, models, dependencies, processing, utils
from ..config import Config, YouTube, PytubeFixError # Go up one level for imports
from ..models import TranscriptionResponse, YouTubeRequest, VideoUploadRequest
from ..dependencies import get_elevenlabs_client, get_gemini_client
from ..processing import extract_audio_from_video, process_audio_to_transcript
from ..utils import clean_temp_file
from ..translation import translate_sbv_file
# NEW: Import for refactored YouTube download functions
from ..youtube_download import download_youtube_pytubefix_sync, download_youtube_yt_dlp_sync
# NEW: Import for refactored task management
from ..task_management import (
    update_task_progress,
    get_or_create_task_event_queue,
    get_task_current_progress,
    remove_task_event_queue
)

logger = logging.getLogger(__name__)
router = APIRouter() # Create a router instance

# --- Task and Event Queue stores are now in src/task_management.py ---
# --- _update_progress helper is now update_task_progress in src/task_management.py ---

# --- API Endpoints using the router ---
@router.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Service is running"}

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "api": Config.API_TITLE,
        "version": Config.API_VERSION,
        "status": "online"
    }

@router.post("/transcribe-video")
async def transcribe_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    target_languages: Optional[str] = Form(default=None, description="Comma-separated list of target languages for translation"),
    client = Depends(get_elevenlabs_client)
):
    """Enqueue video or audio file transcription and return a task_id for progress streaming."""
    # Validate file type - now accepts both video and audio files
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    if file_ext not in Config.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(Config.ALLOWED_FILE_EXTENSIONS)}")
    
    # Parse target languages
    parsed_target_languages = None
    if target_languages:
        parsed_target_languages = [lang.strip() for lang in target_languages.split(',') if lang.strip()]
    
    # Get Gemini client if translation is requested
    gemini_client = None
    if parsed_target_languages:
        try:
            gemini_client = await get_gemini_client()
        except HTTPException:
            raise HTTPException(status_code=503, detail="Translation requested but Gemini service not available. Check GEMINI_API_KEY.")
    
    # Save upload
    temp_video_path = os.path.join(Config.TEMP_DIR, f"{uuid.uuid4()}{file_ext}")
    with open(temp_video_path, "wb") as buf:
        shutil.copyfileobj(video_file.file, buf)
    await video_file.close()
    # Prepare task
    task_id = str(uuid.uuid4())
    get_or_create_task_event_queue(task_id) # Initialize queue
    update_task_progress(task_id, {"status": "queued", "stage": "uploaded", "message": "Video uploaded, ready to start transcription."})
    # Schedule background task
    background_tasks.add_task(
        run_transcription_task,
        task_id,
        video_file_path=temp_video_path,
        original_video_filename=video_file.filename,
        youtube_url=None,
        target_languages=parsed_target_languages,
        client=client,
        gemini_client=gemini_client
    )
    return {"task_id": task_id}

@router.post("/transcribe-youtube")
async def transcribe_youtube(
    request: YouTubeRequest,
    background_tasks: BackgroundTasks,
    client = Depends(get_elevenlabs_client)
):
    url = str(request.url)
    target_languages = request.target_languages
    logger.info(f"YOUTUBE REQUEST RECEIVED: {url[:30]}...")

    # If pytubefix is the *only* strategy or a possible one, this check is fine.
    # If we *only* allow yt-dlp, this check could be removed or conditional.
    # For now, keeping it, as the download function for pytubefix also has an internal check.
    if Config.YOUTUBE_DOWNLOAD_STRATEGY == "pytubefix" and (not YouTube or not PytubeFixError):
        logger.error(f"Pytubefix strategy selected but Pytubefix/YouTube library not available.")
        raise HTTPException(status_code=501, detail="Pytubefix (YouTube) processing dependency not available for selected strategy.")
    
    # Get Gemini client if translation is requested
    gemini_client = None
    if target_languages:
        try:
            gemini_client = await get_gemini_client()
        except HTTPException:
            raise HTTPException(status_code=503, detail="Translation requested but Gemini service not available. Check GEMINI_API_KEY.")
    
    task_id = str(uuid.uuid4())
    logger.info(f"[{task_id}] TASK ID GENERATED")
    
    get_or_create_task_event_queue(task_id) # Initialize queue
    logger.info(f"[{task_id}] CREATED QUEUE (via task_management)")
    
    update_task_progress(task_id, {"status": "queued", "stage": "initialized", "message": "YouTube transcription queued."})
    
    logger.info(f"[{task_id}] SCHEDULING BACKGROUND TASK (with 0.5s delay)")
    
    async def delayed_task():
        logger.info(f"[{task_id}] BACKGROUND TASK DELAY ACTIVE - waiting 0.5s before starting")
        await asyncio.sleep(0.5)
        logger.info(f"[{task_id}] BACKGROUND TASK STARTING PROCESSING")
        await run_transcription_task(
            task_id,
            video_file_path=None,
            original_video_filename=None,
            youtube_url=url,
            target_languages=target_languages,
            client=client,
            gemini_client=gemini_client
        )
    
    background_tasks.add_task(delayed_task)
    
    logger.info(f"[{task_id}] RETURNING TASK ID TO CLIENT")
    return {"task_id": task_id}

@router.get("/progress/{task_id}")
async def stream_progress(task_id: str, request: Request):
    """Server-Sent Events endpoint to stream task progress updates."""
    logger.info(f"[{task_id}] SSE CONNECTION ESTABLISHED")
    
    queue = get_or_create_task_event_queue(task_id)
    logger.info(f"[{task_id}] QUEUE RETRIEVED/CREATED (via task_management)")
    
    current_state = get_task_current_progress(task_id)
    if current_state:
        logger.info(f"[{task_id}] FOUND EXISTING STATE: {current_state.get('status')} - {current_state.get('stage')}")
        await queue.put(current_state.copy()) # Use .copy() here too
        logger.info(f"[{task_id}] QUEUED EXISTING STATE")
    else:
        logger.warning(f"[{task_id}] NO EXISTING STATE FOUND for SSE connection")
    
    async def event_generator():
        count = 0
        logger.info(f"[{task_id}] EVENT GENERATOR STARTED")
        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"[{task_id}] CLIENT DISCONNECTED after {count} events")
                    break
                    
                logger.info(f"[{task_id}] WAITING FOR UPDATE #{count+1}")
                data = await queue.get()
                count += 1
                logger.info(f"[{task_id}] RECEIVED UPDATE #{count}: {data.get('status')} - {data.get('stage')}")
                
                yield {"data": json.dumps(data)}
                logger.info(f"[{task_id}] SENT UPDATE #{count} TO CLIENT")
                
                if data.get("status") in ("complete", "error", "cancelled"):
                    logger.info(f"[{task_id}] FINAL STATUS REACHED: {data.get('status')} after {count} events")
                    break
        finally:
            logger.info(f"[{task_id}] CLEANING UP QUEUE (via task_management)")
            remove_task_event_queue(task_id)
            logger.info(f"[{task_id}] EVENT GENERATOR ENDING")

    logger.info(f"[{task_id}] RETURNING EVENT SOURCE RESPONSE")
    return EventSourceResponse(event_generator())

# --- ENHANCED DOWNLOAD ENDPOINT --- 
@router.get("/download/{filename}")
async def download_transcript(filename: str, file_type: Optional[str] = "processed"):
    """Downloads transcript files with support for different formats.
    
    Args:
        filename: Base filename (without extension)
        file_type: Type of file to download:
            - "processed": Download .sbv subtitle file (default)
            - "raw": Download .json file with word-level timestamps
            - "zip": Download zip file containing all formats (processed, raw, translations)
    """
    from fastapi.responses import StreamingResponse
    import zipfile
    import io
    
    logger.info(f"Received download request for filename: {filename}, file_type: {file_type}")
    
    # Basic sanitization: remove potential path traversal characters
    safe_filename = filename.replace("..", "").replace("/", "").replace("\\", "")
    if safe_filename != filename:
        logger.warning(f"Attempted download with potentially unsafe filename: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    try:
        # Construct the base path safely
        base_path = Path(Config.TEMP_DIR).resolve()
        
        # Regex to find the main SBV file (ends in _<8-char-uuid>.sbv)
        # This helps distinguish it from translated files like _<uuid>_spanish.sbv
        main_sbv_pattern = re.compile(rf"^{re.escape(safe_filename)}_[a-f0-9]{{8}}\.sbv$")
        
        if file_type == "processed":
            import glob
            # Find the main processed SBV file
            all_sbv_files = glob.glob(os.path.join(Config.TEMP_DIR, f"{safe_filename}_*.sbv"))
            main_processed_file = next((f for f in all_sbv_files if main_sbv_pattern.match(os.path.basename(f))), None)
            
            if main_processed_file:
                file_path = Path(main_processed_file).resolve()
                if not str(file_path).startswith(str(base_path)):
                    logger.error(f"Path traversal attempt detected for filename: {filename} -> {file_path}")
                    raise HTTPException(status_code=403, detail="Access forbidden.")
                
                logger.info(f"Serving processed file: {file_path}")
                return FileResponse(
                    path=file_path,
                    filename=f"{safe_filename}.sbv",
                    media_type='text/plain'
                )
            else:
                logger.warning(f"Processed transcript file not found for base name: {safe_filename}")
                raise HTTPException(status_code=404, detail="Processed transcript file not found.")
                
        elif file_type == "raw":
            import glob
            # Find raw file. Assume it's the most recent one matching the pattern.
            pattern = os.path.join(Config.TEMP_DIR, f"{safe_filename}_*_raw.json")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                file_path = Path(max(matching_files, key=os.path.getmtime)).resolve()
                if not str(file_path).startswith(str(base_path)):
                    logger.error(f"Path traversal attempt detected for filename: {filename} -> {file_path}")
                    raise HTTPException(status_code=403, detail="Access forbidden.")
                    
                logger.info(f"Serving raw file: {file_path}")
                return FileResponse(
                    path=file_path,
                    filename=f"{safe_filename}_raw.json",
                    media_type='application/json'
                )
            else:
                logger.warning(f"Raw transcript file not found for pattern: {pattern}")
                raise HTTPException(status_code=404, detail="Raw transcript file not found.")
                
        elif file_type == "zip":
            import glob
            files_to_zip = []

            # 1. Find the main processed SBV file
            all_sbv_files = glob.glob(os.path.join(Config.TEMP_DIR, f"{safe_filename}_*.sbv"))
            main_processed_file = next((f for f in all_sbv_files if main_sbv_pattern.match(os.path.basename(f))), None)

            if not main_processed_file:
                logger.warning(f"No main transcript file found for '{safe_filename}' to create a zip.")
                raise HTTPException(status_code=404, detail="No main transcript file found to create a zip archive.")
            
            main_processed_path = Path(main_processed_file).resolve()
            if not str(main_processed_path).startswith(str(base_path)):
                raise HTTPException(status_code=403, detail="Access forbidden.")
            files_to_zip.append((main_processed_path, f"{safe_filename}.sbv"))

            # From the main file, get the full stem with UUID
            stem_with_uuid = main_processed_path.stem

            # 2. Find the raw JSON file using the full stem
            raw_path = base_path / f"{stem_with_uuid}_raw.json"
            if raw_path.exists():
                files_to_zip.append((raw_path, f"{safe_filename}_raw.json"))

            # 3. Find all translated files
            translation_pattern = os.path.join(Config.TEMP_DIR, f"{stem_with_uuid}_*.sbv")
            for f_path in glob.glob(translation_pattern):
                # Add to zip, but skip the main file which is already included
                if os.path.basename(f_path) != main_processed_path.name:
                    translated_path = Path(f_path).resolve()
                    if str(translated_path).startswith(str(base_path)):
                        # E.g., MyVideo_uuid_spanish.sbv -> MyVideo_spanish.sbv
                        lang_suffix = Path(f_path).stem.split(stem_with_uuid + '_')[-1]
                        arc_name = f"{safe_filename}_{lang_suffix}.sbv"
                        files_to_zip.append((translated_path, arc_name))
            
            if not files_to_zip:
                logger.warning(f"No transcript files found for: {filename}")
                raise HTTPException(status_code=404, detail="No transcript files found.")
            
            # Create zip file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path, arc_name in files_to_zip:
                    zip_file.write(file_path, arc_name)
            
            zip_buffer.seek(0)
            logger.info(f"Serving zip file with {len(files_to_zip)} files for: {filename}")
            
            return StreamingResponse(
                io.BytesIO(zip_buffer.read()),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={safe_filename}_transcripts.zip"}
            )
            
        else:
            logger.warning(f"Invalid file_type requested: {file_type}")
            raise HTTPException(status_code=400, detail="Invalid file_type. Use 'processed', 'raw', or 'zip'.")
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error during file download for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while retrieving file.") 

@router.get("/list-files")
async def list_files():
    """Lists all files in the temp directory, grouped by transcript sets."""
    try:
        files = []
        file_groups = {}
        
        for filename in os.listdir(Config.TEMP_DIR):
            file_path = os.path.join(Config.TEMP_DIR, filename)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                modified = os.path.getmtime(file_path)
                
                file_info = {
                    "name": filename,
                    "size_bytes": size,
                    "modified": modified,
                    "type": "sbv" if filename.endswith(".sbv") else "json" if filename.endswith(".json") else "other"
                }
                
                # Group related files together
                if filename.endswith(".sbv") or filename.endswith("_raw.json"):
                    # Extract base name (remove UUID and extension)
                    base_name = filename
                    if filename.endswith("_raw.json"):
                        base_name = filename[:-9]  # Remove "_raw.json"
                    elif filename.endswith(".sbv"):
                        base_name = filename[:-4]   # Remove ".sbv"
                    
                    # Remove UUID suffix if present
                    if '_' in base_name:
                        parts = base_name.split('_')
                        if len(parts) > 1 and len(parts[-1]) == 8:  # UUID is 8 chars
                            base_name = '_'.join(parts[:-1])
                    
                    if base_name not in file_groups:
                        file_groups[base_name] = {
                            "base_name": base_name,
                            "files": [],
                            "latest_modified": modified
                        }
                    
                    file_groups[base_name]["files"].append(file_info)
                    file_groups[base_name]["latest_modified"] = max(file_groups[base_name]["latest_modified"], modified)
                else:
                    files.append(file_info)
        
        # Convert grouped files to list format
        grouped_files = []
        for group_name, group_data in file_groups.items():
            grouped_files.append({
                "group_name": group_name,
                "files": sorted(group_data["files"], key=lambda x: x["name"]),
                "latest_modified": group_data["latest_modified"],
                "file_count": len(group_data["files"])
            })
        
        # Sort groups by latest modification time
        grouped_files.sort(key=lambda x: x["latest_modified"], reverse=True)
        
        return {
            "grouped_files": grouped_files,
            "individual_files": sorted(files, key=lambda x: x["modified"], reverse=True),
            "total_groups": len(grouped_files),
            "total_individual_files": len(files)
        }
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Background task handler ---
async def run_transcription_task(
    task_id: str,
    video_file_path: Optional[str],
    original_video_filename: Optional[str],
    youtube_url: Optional[str],
    target_languages: Optional[List[str]],
    client: Any, # Assuming client is the ElevenLabs client instance
    gemini_client: Any # Assuming gemini_client is the Gemini client instance
):
    """Background transcription task, updates progress via _update_progress."""
    current_task = asyncio.current_task()
    if current_task:
        current_task.set_name(f"task-{task_id}-{original_video_filename or youtube_url or 'task'}") # More descriptive task name
        
    logger.info(f"[{task_id}] *** BACKGROUND TASK EXECUTION STARTED for '{original_video_filename or youtube_url}'. ***")

    # Use the imported progress callback function from task_management
    # The `run_transcription_task` itself will call `update_task_progress` directly.
    # The `process_audio_to_transcript` function needs a callback that matches the expected signature.
    # So we pass `update_task_progress` directly if its signature matches, or wrap it if needed.
    # For now, assuming `process_audio_to_transcript` expects `def callback(update: Dict[str, Any])`
    # and `update_task_progress` has `task_id` as its first arg.
    # We need a small wrapper here for the callback passed to `process_audio_to_transcript`.
    def callback_for_processing(progress_update_dict: Dict[str, Any]):
        update_task_progress(task_id, progress_update_dict)

    audio_path_to_clean: Optional[str] = None # Keep track of audio file for cleanup

    try:
        if youtube_url:
            logger.info(f"[{task_id}] Preparing to download audio from YouTube: {youtube_url} using strategy: {Config.YOUTUBE_DOWNLOAD_STRATEGY}")
            update_task_progress(task_id, {"status": "processing", "stage": "download", "message": f"Downloading audio from YouTube ({Config.YOUTUBE_DOWNLOAD_STRATEGY})..."})
            
            if Config.YOUTUBE_DOWNLOAD_STRATEGY == "yt-dlp":
                audio_path, original_name = await asyncio.to_thread(download_youtube_yt_dlp_sync, task_id, youtube_url)
            elif Config.YOUTUBE_DOWNLOAD_STRATEGY == "pytubefix":
                # Ensure Pytubefix is available if this strategy is chosen
                if not YouTube or not PytubeFixError: # This check is now also inside download_youtube_pytubefix_sync
                    logger.error(f"[{task_id}] Pytubefix strategy selected but Pytubefix/YouTube library not available.")
                    raise RuntimeError("Pytubefix (YouTube) library not available for selected strategy.")
                audio_path, original_name = await asyncio.to_thread(download_youtube_pytubefix_sync, task_id, youtube_url)
            else:
                logger.error(f"[{task_id}] Invalid YOUTUBE_DOWNLOAD_STRATEGY: {Config.YOUTUBE_DOWNLOAD_STRATEGY}")
                raise ValueError(f"Invalid YOUTUBE_DOWNLOAD_STRATEGY: {Config.YOUTUBE_DOWNLOAD_STRATEGY}. Choose 'pytubefix' or 'yt-dlp'.")

            audio_path_to_clean = audio_path # Mark for cleanup
            logger.info(f"[{task_id}] YouTube audio download complete. Path: {audio_path}, Original Name: {original_name}")
            update_task_progress(task_id, {"status": "processing", "stage": "downloaded", "message": "YouTube audio downloaded."})
        
        elif video_file_path:
            file_ext = os.path.splitext(video_file_path)[1].lower()
            
            # Check if it's an audio file that doesn't need extraction
            if file_ext in Config.ALLOWED_AUDIO_EXTENSIONS:
                logger.info(f"[{task_id}] Audio file detected, using directly: {video_file_path}")
                update_task_progress(task_id, {"status": "processing", "stage": "extract", "message": f"Processing audio file '{original_video_filename}'"})
                
                audio_path = video_file_path  # Use the audio file directly
                audio_path_to_clean = None  # Don't clean the original file, we'll clean it in finally
                original_name = original_video_filename
                logger.info(f"[{task_id}] Audio file ready for transcription. Path: {audio_path}, Original Name: {original_name}")
                update_task_progress(task_id, {"status": "processing", "stage": "extracted", "message": "Audio file ready for transcription."})
            else:
                # It's a video file, extract audio
                logger.info(f"[{task_id}] Preparing to extract audio from video file: {video_file_path}")
                update_task_progress(task_id, {"status": "processing", "stage": "extract", "message": f"Extracting audio from '{original_video_filename}'"})
                
                # Note: extract_audio_from_video now takes task_id and logs with it
                audio_path = await asyncio.to_thread(extract_audio_from_video, task_id, video_file_path)
                audio_path_to_clean = audio_path # Mark for cleanup
                original_name = original_video_filename # Ensure original_name is set
                logger.info(f"[{task_id}] Audio extraction complete. Path: {audio_path}, Original Name: {original_name}")
                update_task_progress(task_id, {"status": "processing", "stage": "extracted", "message": "Audio extraction complete."})
        
        else:
            logger.error(f"[{task_id}] No video_file_path or youtube_url provided for transcription.")
            raise ValueError("No input source (video file or YouTube URL) provided for transcription.")

        logger.info(f"[{task_id}] Queuing transcription process for '{original_name}' (audio at {audio_path}).")
        update_task_progress(task_id, {"status": "processing", "stage": "transcribing_queued", "message": f"Transcription process for '{original_name}' starting..."})
        
        # process_audio_to_transcript logs extensively with task_id and original_name
        processed_path = await process_audio_to_transcript( 
            task_id,              
            audio_path, 
            client, 
            original_name,
            target_languages,
            callback_for_processing, # Pass the wrapped callback
            gemini_client
        )
        
        logger.info(f"[{task_id}] Transcription process for '{original_name}' completed. Processed transcript at: {processed_path}")
        
        # Check for translated files in the temp directory for this task
        translated_files = []
        if target_languages:
            original_filename = Path(processed_path).stem
            for language in target_languages:
                translated_filename = f"{original_filename}_{language}.sbv"
                translated_path = os.path.join(Config.TEMP_DIR, translated_filename)
                if os.path.exists(translated_path):
                    translated_files.append(os.path.basename(translated_path))
        
        # Extract base filename for dual-file downloads (remove .sbv extension and UUID)
        base_filename = Path(processed_path).stem
        # Remove the UUID suffix to get clean base name for downloads
        if '_' in base_filename:
            # Split by underscore and remove the last part (UUID)
            parts = base_filename.split('_')
            if len(parts) > 1 and len(parts[-1]) == 8:  # UUID is 8 chars
                base_filename = '_'.join(parts[:-1])
        
        # Prepare completion message
        completion_data = {
            "status": "complete", 
            "stage": "finished", 
            "message": f"Transcription for '{original_name}' complete.", 
            "filename": base_filename,  # Use base filename for dual-file downloads
            "processed_file": os.path.basename(processed_path),
            "raw_file": os.path.basename(processed_path).replace('.sbv', '_raw.json')
        }
        
        if translated_files:
            completion_data["translated_files"] = translated_files
            completion_data["message"] += f" {len(translated_files)} translations created."
        
        update_task_progress(task_id, completion_data)
        logger.info(f"[{task_id}] *** BACKGROUND TASK COMPLETED SUCCESSFULLY for '{original_name}'. ***")

    except Exception as e:
        # Log the error with task_id and original_name if available
        name_for_log = original_video_filename or youtube_url or "unknown_source"
        logger.error(f"[{task_id}] TASK FAILED for '{name_for_log}': {e}", exc_info=True)
        update_task_progress(task_id, {"status": "error", "stage": "failed", "message": f"Error during transcription for '{name_for_log}': {str(e)}"})
        logger.info(f"[{task_id}] *** BACKGROUND TASK FAILED for '{name_for_log}'. ***")
    
    finally:
        name_for_log_cleanup = original_video_filename or youtube_url or "task"
        logger.info(f"[{task_id}] Starting cleanup for '{name_for_log_cleanup}'.")
        
        # Clean the downloaded/extracted audio file
        if audio_path_to_clean:
            logger.info(f"[{task_id}] Removing temporary audio file: {audio_path_to_clean}")
            clean_temp_file(audio_path_to_clean, task_id_for_log=task_id)
        
        # Clean the uploaded video file if it exists (it's passed as video_file_path)
        if video_file_path: # This was the originally uploaded temp file
            logger.info(f"[{task_id}] Removing temporary uploaded video file: {video_file_path}")
            clean_temp_file(video_file_path, task_id_for_log=task_id)
            
        logger.info(f"[{task_id}] Cleanup complete for '{name_for_log_cleanup}'. TASK FINISHED.") 