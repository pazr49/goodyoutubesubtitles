import os
import uuid
import shutil
import logging
from pathlib import Path # Import Path
import asyncio # Added
import json # Added

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse # Import FileResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional, Dict, Any, List # Added List

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

# New: Stores a list of asyncio.Queue for each task_id to notify listeners
# Each queue in the list corresponds to one active SSE connection for that task_id.
task_event_queues: Dict[str, List[asyncio.Queue]] = {}
task_event_queues_lock = asyncio.Lock() # To protect concurrent access to task_event_queues

async def _update_task_progress_and_notify(task_id: str, progress_data: Dict[str, Any]):
    """
    Updates the task_progress_store and notifies all listening SSE generators.
    """
    logger.debug(f"Updating progress for task {task_id}: {progress_data}")
    task_progress_store[task_id] = progress_data

    async with task_event_queues_lock:
        queues_for_task = task_event_queues.get(task_id, [])
        if not queues_for_task:
            logger.debug(f"No active SSE listeners for task {task_id} to notify.")
        else:
            logger.debug(f"Notifying {len(queues_for_task)} SSE listener(s) for task {task_id}.")
        for queue in queues_for_task:
            try:
                await queue.put(progress_data)
            except Exception as e:
                logger.error(f"Error putting progress into queue for task {task_id}: {e}", exc_info=True)

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
    try:
        with open(temp_video_path, "wb") as buf:
            shutil.copyfileobj(video_file.file, buf)
    finally:
        await video_file.close()
    # Prepare task
    task_id = str(uuid.uuid4())
    await _update_task_progress_and_notify(
        task_id,
        {"status": "queued", "stage": "uploaded", "message": "Video uploaded, ready to start transcription."}
    )
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
    # Prepare task
    task_id = str(uuid.uuid4())
    # Initial progress update
    await _update_task_progress_and_notify(
        task_id,
        {"status": "queued", "stage": "initialized", "message": "YouTube transcription queued."}
    )
    # Schedule background task
    background_tasks.add_task(
        run_transcription_task,
        task_id,
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
        my_queue = asyncio.Queue()
        last_sent_progress_content: Optional[Dict[str, Any]] = None

        try:
            # Register the queue for this client
            async with task_event_queues_lock:
                if task_id not in task_event_queues:
                    task_event_queues[task_id] = []
                task_event_queues[task_id].append(my_queue)
            logger.info(f"SSE client connected for task {task_id}. Registered queue.")

            current_snapshot_progress = task_progress_store.get(task_id)

            if current_snapshot_progress:
                yield {"data": json.dumps(current_snapshot_progress)}
                last_sent_progress_content = current_snapshot_progress
                logger.debug(f"Sent initial snapshot progress for task {task_id}: {current_snapshot_progress}")
                # Do not return early if terminal, queue might have earlier states.
            elif not current_snapshot_progress:
                 # If task isn't even in the store, it's likely a bogus task_id from client
                 not_found_payload = {"status": "not_found", "message": "Task ID not found in progress store."}
                 yield {"data": json.dumps(not_found_payload)}
                 logger.warning(f"Task ID {task_id} not found in task_progress_store. Sent not_found and closing SSE for this client.")
                 return

            while True:
                if await request.is_disconnected():
                    logger.info(f"SSE client for task {task_id} disconnected.")
                    break

                try:
                    # Wait for a new progress update from the queue
                    # Add a timeout to periodically check for client disconnect
                    progress_from_queue = await asyncio.wait_for(my_queue.get(), timeout=30.0) 
                except asyncio.TimeoutError:
                    # No new progress within timeout, just loop to check disconnect status
                    logger.debug(f"SSE for task {task_id} timed out waiting for queue, checking disconnect.")
                    continue 
                
                my_queue.task_done() # Acknowledge message processed from queue

                if progress_from_queue != last_sent_progress_content:
                    yield {"data": json.dumps(progress_from_queue)}
                    last_sent_progress_content = progress_from_queue
                    logger.debug(f"Sent progress update from queue for task {task_id}: {progress_from_queue}")
                else:
                    logger.debug(f"Skipping send from queue for task {task_id} as data is same as last sent: {progress_from_queue}")

                if progress_from_queue.get("status") in ("complete", "error", "cancelled", "not_found"):
                    logger.info(f"Task {task_id} reached terminal state ({progress_from_queue.get('status')}) via queue. Closing SSE stream.")
                    break
        
        except Exception as e:
            logger.error(f"Error in SSE event_generator for task {task_id}: {e}", exc_info=True)
            # Attempt to send an error to the client if possible before closing
            try:
                error_payload = {"status": "error", "stage": "sse_stream_failure", "message": "SSE stream failed on server."}
                yield {"data": json.dumps(error_payload)}
            except Exception: # Ignore if we can't send (e.g., client already disconnected)
                pass 
        finally:
            # Unregister the queue
            async with task_event_queues_lock:
                if task_id in task_event_queues and my_queue in task_event_queues[task_id]:
                    task_event_queues[task_id].remove(my_queue)
                    if not task_event_queues[task_id]: # If list is empty, remove task_id entry itself
                        del task_event_queues[task_id]
                    logger.info(f"SSE client for task {task_id} cleaned up its queue.")
                else:
                    # This might happen if cleanup was already attempted or if the queue wasn't properly registered
                    logger.warning(f"SSE queue for task {task_id} not found during cleanup or already removed.")
            logger.info(f"SSE event_generator for task {task_id} finished.")

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
    video_file_path: Optional[str],
    original_video_filename: Optional[str],
    youtube_url: Optional[str],
    client
):
    """Background transcription task, updates progress_store as it goes."""
    try:
        # Step 1: Audio preparation
        if youtube_url:
            await _update_task_progress_and_notify(
                task_id,
                {"status": "processing", "stage": "download", "message": "Downloading audio from YouTube..."}
            )
            yt = YouTube(youtube_url)
            title = yt.title or "youtube_video"
            stream = yt.streams.filter(only_audio=True, file_extension=Config.PREFERRED_AUDIO_FORMAT).order_by('abr').desc().first() or yt.streams.get_audio_only()
            if not stream:
                raise RuntimeError("No suitable audio stream found.")
            audio_ext = f".{stream.subtype}"
            temp_audio_name = f"{uuid.uuid4()}_youtube{audio_ext}"
            temp_audio_path = os.path.join(Config.TEMP_DIR, temp_audio_name)
            stream.download(output_path=Config.TEMP_DIR, filename=temp_audio_name)
            await _update_task_progress_and_notify(
                task_id,
                {"status": "processing", "stage": "downloaded", "message": "YouTube audio downloaded."}
            )
            audio_path = temp_audio_path
            original_name = title
        else:
            await _update_task_progress_and_notify(
                task_id,
                {"status": "processing", "stage": "extract", "message": "Extracting audio from uploaded video..."}
            )
            audio_path = extract_audio_from_video(video_file_path)
            await _update_task_progress_and_notify(
                task_id,
                {"status": "processing", "stage": "extracted", "message": "Audio extraction complete."}
            )
            original_name = original_video_filename
        
        # Step 2: Transcription
        await _update_task_progress_and_notify(
            task_id,
            {"status": "processing", "stage": "transcribing", "message": "Transcribing audio (this may take a while)..."}
        )
        transcript_path = process_audio_to_transcript(audio_path, client, original_name)

        # Step 3: Complete
        await _update_task_progress_and_notify(
            task_id,
            {"status": "complete", "stage": "finished", "message": "Transcription complete.", "filename": os.path.basename(transcript_path)}
        )
    except PytubeFixError as e: # Specific error handling for pytube issues
        logger.error(f"PytubeFixError in Task {task_id}: {e}", exc_info=True)
        await _update_task_progress_and_notify(
            task_id,
            {"status": "error", "stage": "youtube_download_failed", "message": f"YouTube video processing error: {str(e)}"}
        )
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        await _update_task_progress_and_notify(
            task_id,
            {"status": "error", "stage": "failed", "message": str(e)}
        )
    finally:
        # Cleanup temp files
        if youtube_url and 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            clean_temp_file(temp_audio_path)
        if video_file_path and os.path.exists(video_file_path): # Ensure it exists before trying to clean
            clean_temp_file(video_file_path)
        # No need to clean audio_path if it's the same as temp_audio_path or derived from video_file_path's audio extraction
        # (as those are handled above). If extract_audio_from_video creates a new distinct temp file, it should be cleaned there
        # or its path returned for cleaning here. The current `extract_audio_from_video` seems to do that.
        # The `process_audio_to_transcript` also cleans its own chunks.

        # Final check for task_progress_store to ensure a terminal state is set if not already
        final_state = task_progress_store.get(task_id)
        if final_state and final_state.get("status") not in ("complete", "error", "cancelled"):
            # This case should ideally not be hit if all exit paths in try/except update the status.
            # However, as a safeguard:
            logger.warning(f"Task {task_id} ended without a clear terminal status. Marking as error.")
            await _update_task_progress_and_notify(
                task_id,
                {"status": "error", "stage": "unknown_failure", "message": "Task ended with an undetermined error."}
            ) 