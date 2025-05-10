import os
import uuid
import shutil
import logging
from pathlib import Path # Import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse # Import FileResponse
from sse_starlette.sse import EventSourceResponse
import asyncio, json
from typing import Optional, Dict, Any

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
    task_progress_store[task_id] = {"status": "queued", "stage": "uploaded", "message": "Video uploaded, ready to start transcription."}
    # Schedule background task
    background_tasks.add_task(
        run_transcription_task,
        task_id,
        task_progress_store,
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
    # Prepare task
    task_id = str(uuid.uuid4())
    task_progress_store[task_id] = {"status": "queued", "stage": "initialized", "message": "YouTube transcription queued."}
    # Schedule background task
    background_tasks.add_task(
        run_transcription_task,
        task_id,
        task_progress_store,
        video_file_path=None,
        original_video_filename=None,
        youtube_url=url,
        client=client
    )
    return {"task_id": task_id}

@router.get("/progress/{task_id}")
async def stream_progress(task_id: str, request: Request):
    """Server-Sent Events endpoint to stream task progress updates."""
    async def event_generator():
        last_data = None
        while True:
            # Check client disconnect
            if await request.is_disconnected():
                break
            progress = task_progress_store.get(task_id)
            if progress is None:
                # If no such task, notify once and end
                payload = {"status": "not_found", "message": "Task ID not found."}
                yield {"data": json.dumps(payload)}
                break
            data = json.dumps(progress)
            if data != last_data:
                yield {"data": data}
                last_data = data
            # Stop streaming on completion or error
            if progress.get("status") in ("complete", "error", "cancelled"):
                break
            await asyncio.sleep(1)
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

# --- Background task handler ---
async def run_transcription_task(
    task_id: str,
    progress_store: Dict[str, Dict[str, Any]],
    video_file_path: Optional[str],
    original_video_filename: Optional[str],
    youtube_url: Optional[str],
    client
):
    """Background transcription task, updates progress_store as it goes."""
    try:
        # Step 1: Audio preparation
        if youtube_url:
            progress_store[task_id] = {"status": "processing", "stage": "download", "message": "Downloading audio from YouTube..."}
            yt = YouTube(youtube_url)
            title = yt.title or "youtube_video"
            stream = yt.streams.filter(only_audio=True, file_extension=Config.PREFERRED_AUDIO_FORMAT).order_by('abr').desc().first() or yt.streams.get_audio_only()
            if not stream:
                raise RuntimeError("No suitable audio stream found.")
            audio_ext = f".{stream.subtype}"
            temp_audio_name = f"{uuid.uuid4()}_youtube{audio_ext}"
            temp_audio_path = os.path.join(Config.TEMP_DIR, temp_audio_name)
            stream.download(output_path=Config.TEMP_DIR, filename=temp_audio_name)
            progress_store[task_id] = {"status": "processing", "stage": "downloaded", "message": "YouTube audio downloaded."}
            audio_path = temp_audio_path
            original_name = title
        else:
            progress_store[task_id] = {"status": "processing", "stage": "extract", "message": "Extracting audio from uploaded video..."}
            audio_path = extract_audio_from_video(video_file_path)
            progress_store[task_id] = {"status": "processing", "stage": "extracted", "message": "Audio extraction complete."}
            original_name = original_video_filename
        # Step 2: Transcription
        progress_store[task_id] = {"status": "processing", "stage": "transcribing", "message": "Transcribing audio..."}
        transcript_path = process_audio_to_transcript(audio_path, client, original_name)
        # Step 3: Complete
        progress_store[task_id] = {"status": "complete", "stage": "finished", "message": "Transcription complete.", "filename": os.path.basename(transcript_path)}
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        progress_store[task_id] = {"status": "error", "stage": "failed", "message": str(e)}
    finally:
        # Cleanup temp files
        if youtube_url and 'temp_audio_path' in locals():
            clean_temp_file(temp_audio_path)
        if video_file_path:
            clean_temp_file(video_file_path) 