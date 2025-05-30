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
    """Enqueue video transcription and return a task_id for progress streaming."""
    # Validate file type
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    if file_ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}")
    
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

# --- NEW DOWNLOAD ENDPOINT --- 
@router.get("/download/{filename}")
async def download_transcript(filename: str):
    """Downloads the specified transcript file (.sbv).
    
    Ensures the file requested is within the temporary directory.
    """
    logger.info(f"Received download request for filename: {filename}")
    
    # Basic sanitization: remove potential path traversal characters
    # Although Path() helps, double sanitization is often good practice.
    safe_filename = filename.replace("..", "").replace("/", "").replace("\\", "")
    if safe_filename != filename:
        logger.warning(f"Attempted download with potentially unsafe filename: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    try:
        # Construct the full path safely
        base_path = Path(Config.TEMP_DIR).resolve() # Get absolute path of TEMP_DIR
        file_path = (base_path / safe_filename).resolve()

        # SECURITY CHECK: Ensure the resolved file path is still within the base_path directory
        if not str(file_path).startswith(str(base_path)):
            logger.error(f"Path traversal attempt detected for filename: {filename} -> {file_path}")
            raise HTTPException(status_code=403, detail="Access forbidden.")
            
        # Check if file exists and is a file
        if file_path.is_file():
            logger.info(f"Serving file: {file_path}")
            # Return the file as a response
            return FileResponse(
                path=file_path,
                filename=safe_filename, # Suggest filename to browser
                media_type='text/plain' # Explicitly set for .sbv if needed, else FileResponse guesses
            )
        else:
            logger.warning(f"Requested transcript file not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found.")
            
    except Exception as e:
        logger.error(f"Error during file download for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while retrieving file.") 

@router.get("/list-files")
async def list_files():
    """Lists all files in the temp directory."""
    try:
        files = []
        for filename in os.listdir(Config.TEMP_DIR):
            file_path = os.path.join(Config.TEMP_DIR, filename)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                modified = os.path.getmtime(file_path)
                files.append({
                    "name": filename,
                    "size_bytes": size,
                    "modified": modified,
                    "type": "sbv" if filename.endswith(".sbv") else "other"
                })
        return {"files": files}
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
        transcript_path = await process_audio_to_transcript( 
            task_id,              
            audio_path, 
            client, 
            original_name,
            target_languages,
            callback_for_processing, # Pass the wrapped callback
            gemini_client
        )
        
        logger.info(f"[{task_id}] Transcription process for '{original_name}' completed. Transcript at: {transcript_path}")
        
        # Check for translated files in the temp directory for this task
        translated_files = []
        if target_languages:
            original_filename = Path(transcript_path).stem
            for language in target_languages:
                translated_filename = f"{original_filename}_{language}.sbv"
                translated_path = os.path.join(Config.TEMP_DIR, translated_filename)
                if os.path.exists(translated_path):
                    translated_files.append(os.path.basename(translated_path))
        
        # Prepare completion message
        completion_data = {
            "status": "complete", 
            "stage": "finished", 
            "message": f"Transcription for '{original_name}' complete.", 
            "filename": os.path.basename(transcript_path)
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