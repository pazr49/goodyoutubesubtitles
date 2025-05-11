import os
import uuid
import shutil
import logging
from pathlib import Path # Import Path
import time # Add for timestamps

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse # Import FileResponse
from sse_starlette.sse import EventSourceResponse
import asyncio, json
from typing import Optional, Dict, Any
import uuid as _uuid  # for thread helper

# Import config, models, dependencies, processing, utils
from ..config import Config, YouTube, PytubeFixError # Go up one level for imports
from ..models import TranscriptionResponse, YouTubeRequest
from ..dependencies import get_elevenlabs_client
from ..processing import extract_audio_from_video, process_audio_to_transcript
from ..utils import clean_temp_file

logger = logging.getLogger(__name__)
router = APIRouter() # Create a router instance

# Add in-memory task progress store
# Keyed by task_id, each value is a dict with status, stage, message, and optional filename or error
task_progress_store: Dict[str, Dict[str, Any]] = {}

# Queue store for SSE subscribers, keyed by task_id
event_queue_store: Dict[str, asyncio.Queue] = {}

# Helper to update progress_store and notify SSE clients
def _update_progress(task_id: str, progress: Dict[str, Any]):
    logger.info(f"[{task_id}] UPDATING PROGRESS: {progress['status']} - {progress['stage']}")
    task_progress_store[task_id] = progress
    queue = event_queue_store.get(task_id)
    if queue:
        queue.put_nowait(progress.copy())
        logger.info(f"[{task_id}] QUEUED UPDATE: {progress['status']} - {progress['stage']}")
    else:
        logger.warning(f"[{task_id}] NO QUEUE FOUND when trying to update progress: {progress['status']} - {progress['stage']}")

# Helper to download YouTube audio in a sync thread
def _download_youtube_sync(task_id_for_log: str, youtube_url: str) -> tuple[str, str]:
    """Synchronously download YouTube audio and return file path and title."""
    logger.info(f"[{task_id_for_log}] _DOWNLOAD_YOUTUBE_SYNC: Initializing YouTube object for {youtube_url[:30]}...")
    yt = YouTube(youtube_url)
    title = yt.title or "youtube_video"
    logger.info(f"[{task_id_for_log}] _DOWNLOAD_YOUTUBE_SYNC: Video Title - {title}")
    
    logger.info(f"[{task_id_for_log}] _DOWNLOAD_YOUTUBE_SYNC: Selecting audio stream...")
    stream = yt.streams.filter(
        only_audio=True,
        file_extension=Config.PREFERRED_AUDIO_FORMAT
    ).order_by('abr').desc().first() or yt.streams.get_audio_only()
    
    if not stream:
        logger.error(f"[{task_id_for_log}] _DOWNLOAD_YOUTUBE_SYNC: No suitable audio stream found.")
        raise RuntimeError("No suitable audio stream found.")
        
    audio_ext = f".{stream.subtype}"
    # Use a more descriptive temp_audio_name that includes a UUID for uniqueness
    temp_audio_name = f"{_uuid.uuid4()}_youtube{audio_ext}" 
    temp_audio_path = os.path.join(Config.TEMP_DIR, temp_audio_name)
    logger.info(f"[{task_id_for_log}] _DOWNLOAD_YOUTUBE_SYNC: Downloading stream to {temp_audio_path}")
    
    download_start_time = time.time()
    stream.download(output_path=Config.TEMP_DIR, filename=temp_audio_name)
    download_duration = time.time() - download_start_time
    logger.info(f"[{task_id_for_log}] _DOWNLOAD_YOUTUBE_SYNC: Download complete in {download_duration:.2f} seconds. Path: {temp_audio_path}")
    
    return temp_audio_path, title

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
    client = Depends(get_elevenlabs_client)
):
    """Enqueue video transcription and return a task_id for progress streaming."""
    # Validate file type
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    if file_ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}")
    # Save upload
    temp_video_path = os.path.join(Config.TEMP_DIR, f"{uuid.uuid4()}{file_ext}")
    with open(temp_video_path, "wb") as buf:
        shutil.copyfileobj(video_file.file, buf)
    await video_file.close()
    # Prepare task
    task_id = str(uuid.uuid4())
    # Initialize queue and publish initial state
    event_queue_store[task_id] = asyncio.Queue()
    _update_progress(task_id, {"status": "queued", "stage": "uploaded", "message": "Video uploaded, ready to start transcription."})
    # Schedule background task
    background_tasks.add_task(
        run_transcription_task,
        task_id,
        video_file_path=temp_video_path,
        original_video_filename=video_file.filename,
        youtube_url=None,
        client=client
    )
    return {"task_id": task_id}

@router.post("/transcribe-youtube")
async def transcribe_youtube(
    request: YouTubeRequest,
    background_tasks: BackgroundTasks,
    client = Depends(get_elevenlabs_client)
):
    # Validate youtube dependency
    if not YouTube or not PytubeFixError:
        raise HTTPException(status_code=501, detail="YouTube processing dependency not available.")
    url = str(request.url)
    
    # Log receipt of request
    logger.info(f"YOUTUBE REQUEST RECEIVED: {url[:30]}...")
    
    # Prepare task
    task_id = str(uuid.uuid4())
    logger.info(f"[{task_id}] TASK ID GENERATED")
    
    # Initialize queue and publish initial state
    event_queue_store[task_id] = asyncio.Queue()
    logger.info(f"[{task_id}] CREATED QUEUE")
    
    _update_progress(task_id, {"status": "queued", "stage": "initialized", "message": "YouTube transcription queued."})
    
    # Schedule background task but add a small delay to ensure endpoints return first
    logger.info(f"[{task_id}] SCHEDULING BACKGROUND TASK (with 0.5s delay)")
    
    async def delayed_task():
        logger.info(f"[{task_id}] BACKGROUND TASK DELAY ACTIVE - waiting 0.5s before starting")
        await asyncio.sleep(0.5)  # Small delay to ensure endpoint returns and SSE can connect
        logger.info(f"[{task_id}] BACKGROUND TASK STARTING PROCESSING")
        await run_transcription_task(
            task_id,
            video_file_path=None,
            original_video_filename=None,
            youtube_url=url,
            client=client
        )
    
    background_tasks.add_task(delayed_task)
    
    logger.info(f"[{task_id}] RETURNING TASK ID TO CLIENT")
    return {"task_id": task_id}

@router.get("/progress/{task_id}")
async def stream_progress(task_id: str, request: Request):
    """Server-Sent Events endpoint to stream task progress updates."""
    logger.info(f"[{task_id}] SSE CONNECTION ESTABLISHED")
    
    # Ensure a queue exists for this task
    queue = event_queue_store.setdefault(task_id, asyncio.Queue())
    logger.info(f"[{task_id}] QUEUE RETRIEVED/CREATED")
    
    # Push current state if available
    if task_progress_store.get(task_id):
        current_state = task_progress_store[task_id].copy()
        logger.info(f"[{task_id}] FOUND EXISTING STATE: {current_state['status']} - {current_state['stage']}")
        await queue.put(current_state)
        logger.info(f"[{task_id}] QUEUED EXISTING STATE")
    else:
        logger.warning(f"[{task_id}] NO EXISTING STATE FOUND for SSE connection")
    
    async def event_generator():
        count = 0
        logger.info(f"[{task_id}] EVENT GENERATOR STARTED")
        while True:
            # Disconnect check
            if await request.is_disconnected():
                logger.info(f"[{task_id}] CLIENT DISCONNECTED after {count} events")
                break
                
            # Log waiting for data
            logger.info(f"[{task_id}] WAITING FOR UPDATE #{count+1}")
            
            # Get data from queue
            data = await queue.get()
            count += 1
            logger.info(f"[{task_id}] RECEIVED UPDATE #{count}: {data['status']} - {data['stage']}")
            
            # Send to client
            logger.info(f"[{task_id}] SENDING UPDATE #{count} TO CLIENT")
            yield {"data": json.dumps(data)}
            logger.info(f"[{task_id}] SENT UPDATE #{count} TO CLIENT")
            
            # Check for completion
            if data.get("status") in ("complete", "error", "cancelled"):
                logger.info(f"[{task_id}] FINAL STATUS REACHED: {data['status']} after {count} events")
                break
                
        # Cleanup queue
        logger.info(f"[{task_id}] CLEANING UP QUEUE")
        event_queue_store.pop(task_id, None)
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
    client: Any # Assuming client is the ElevenLabs client instance
):
    """Background transcription task, updates progress via _update_progress."""
    current_task = asyncio.current_task()
    if current_task:
        current_task.set_name(f"task-{task_id}-{original_video_filename or youtube_url or 'task'}") # More descriptive task name
        
    logger.info(f"[{task_id}] *** BACKGROUND TASK EXECUTION STARTED for '{original_video_filename or youtube_url}'. ***")

    def progress_callback_for_processing(progress_update: Dict[str, Any]):
        _update_progress(task_id, progress_update)

    audio_path_to_clean: Optional[str] = None # Keep track of audio file for cleanup

    try:
        if youtube_url:
            logger.info(f"[{task_id}] Preparing to download audio from YouTube: {youtube_url}")
            _update_progress(task_id, {"status": "processing", "stage": "download", "message": "Downloading audio from YouTube..."})
            
            # Note: _download_youtube_sync already logs extensively with task_id
            audio_path, original_name = await asyncio.to_thread(_download_youtube_sync, task_id, youtube_url)
            audio_path_to_clean = audio_path # Mark for cleanup
            logger.info(f"[{task_id}] YouTube audio download complete. Path: {audio_path}, Original Name: {original_name}")
            _update_progress(task_id, {"status": "processing", "stage": "downloaded", "message": "YouTube audio downloaded."})
        
        elif video_file_path:
            logger.info(f"[{task_id}] Preparing to extract audio from video file: {video_file_path}")
            _update_progress(task_id, {"status": "processing", "stage": "extract", "message": f"Extracting audio from '{original_video_filename}'"})
            
            # Note: extract_audio_from_video now takes task_id and logs with it
            audio_path = await asyncio.to_thread(extract_audio_from_video, task_id, video_file_path)
            audio_path_to_clean = audio_path # Mark for cleanup
            original_name = original_video_filename # Ensure original_name is set
            logger.info(f"[{task_id}] Audio extraction complete. Path: {audio_path}, Original Name: {original_name}")
            _update_progress(task_id, {"status": "processing", "stage": "extracted", "message": "Audio extraction complete."})
        
        else:
            logger.error(f"[{task_id}] No video_file_path or youtube_url provided for transcription.")
            raise ValueError("No input source (video file or YouTube URL) provided for transcription.")

        logger.info(f"[{task_id}] Queuing transcription process for '{original_name}' (audio at {audio_path}).")
        _update_progress(task_id, {"status": "processing", "stage": "transcribing_queued", "message": f"Transcription process for '{original_name}' starting..."})
        
        # process_audio_to_transcript logs extensively with task_id and original_name
        transcript_path = await asyncio.to_thread(
            process_audio_to_transcript, 
            task_id,              
            audio_path, 
            client, 
            original_name,
            progress_callback_for_processing 
        )
        
        logger.info(f"[{task_id}] Transcription process for '{original_name}' completed. Transcript at: {transcript_path}")
        _update_progress(task_id, {"status": "complete", "stage": "finished", "message": f"Transcription for '{original_name}' complete.", "filename": os.path.basename(transcript_path)})
        logger.info(f"[{task_id}] *** BACKGROUND TASK COMPLETED SUCCESSFULLY for '{original_name}'. ***")

    except Exception as e:
        # Log the error with task_id and original_name if available
        name_for_log = original_video_filename or youtube_url or "unknown_source"
        logger.error(f"[{task_id}] TASK FAILED for '{name_for_log}': {e}", exc_info=True)
        _update_progress(task_id, {"status": "error", "stage": "failed", "message": f"Error during transcription for '{name_for_log}': {str(e)}"})
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