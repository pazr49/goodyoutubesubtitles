import os
import uuid
import shutil
import logging
from pathlib import Path # Import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse # Import FileResponse

# Import config, models, dependencies, processing, utils
from ..config import Config, YouTube, PytubeFixError # Go up one level for imports
from ..models import TranscriptionResponse, YouTubeRequest
from ..dependencies import get_elevenlabs_client
from ..processing import extract_audio_from_video, process_audio_to_transcript
from ..utils import clean_temp_file

logger = logging.getLogger(__name__)
router = APIRouter() # Create a router instance

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

@router.post("/transcribe-video", response_model=TranscriptionResponse)
async def transcribe_video(
    video_file: UploadFile = File(...),
    client = Depends(get_elevenlabs_client) # Inject dependency
):
    """
    Generate subtitles from uploaded video file.
    
    Steps:
    1. Validate input file type.
    2. Save uploaded video temporarily.
    3. Extract audio from video.
    4. Process audio using transcription service.
    5. Return path to the generated transcript file.
    6. Clean up temporary files.
    """
    # Validate file type
    file_extension = os.path.splitext(video_file.filename)[1].lower()
    if file_extension not in Config.ALLOWED_VIDEO_EXTENSIONS:
        logger.warning(f"Invalid video file type uploaded: {file_extension}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type '{file_extension}'. Allowed: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Prepare temporary file paths
    temp_video_path = os.path.join(Config.TEMP_DIR, f"{uuid.uuid4()}{file_extension}")
    temp_audio_path = None
    
    try:
        # 1. Save Uploaded Video
        logger.info(f"Saving uploaded video {video_file.filename} to {temp_video_path}")
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        logger.info("Video saved successfully.")
        
        # 2. Extract audio
        temp_audio_path = extract_audio_from_video(temp_video_path)
        
        # 3. Process audio to transcript
        transcript_file_path = process_audio_to_transcript(
            temp_audio_path,
            client,
            video_file.filename
        )
        
        logger.info(f"Successfully processed video: {video_file.filename}")
        return {
            "message": "Transcription successful",
            "transcript_file": os.path.basename(transcript_file_path)
        }
        
    except Exception as e:
        # Log error with file info
        error_msg = str(e)
        logger.error(f"Error processing video '{video_file.filename}': {error_msg}", exc_info=True)
        
        # Re-raise HTTP exceptions, wrap others in 500
        if isinstance(e, HTTPException):
            raise
        elif isinstance(e, (ValueError, IOError, RuntimeError)): # More specific handling
             raise HTTPException(status_code=500, detail=f"Processing error: {error_msg}")
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred during video processing.")
        
    finally:
        # 4. Cleanup Temporary Files
        logger.debug(f"Cleaning up temporary files for video: {video_file.filename}")
        await video_file.close()
        clean_temp_file(temp_video_path)
        clean_temp_file(temp_audio_path)

@router.post("/transcribe-youtube", response_model=TranscriptionResponse)
async def transcribe_youtube(
    request: YouTubeRequest,
    client = Depends(get_elevenlabs_client) # Inject dependency
):
    """
    Generate subtitles from YouTube URL.
    
    Steps:
    1. Validate YouTube library availability.
    2. Download audio from YouTube URL.
    3. Process downloaded audio using transcription service.
    4. Return path to the generated transcript file.
    5. Clean up temporary audio file.
    """
    # 1. Validate YouTube library
    if not YouTube or not PytubeFixError:
        logger.error("YouTube processing dependency (pytubefix) not available.")
        raise HTTPException(
            status_code=501, # Not Implemented
            detail="YouTube processing dependency (pytubefix) not installed or available."
        )
        
    url = str(request.url)
    temp_audio_path = None
    video_title = "youtube_video"  # Default fallback title
    
    logger.info(f"--- Entering transcribe_youtube for URL: {url} ---")
    try:
        # 2. Download Audio from YouTube
        logger.info("Attempting to initialize YouTube object...")
        yt = YouTube(url)
        logger.info("YouTube object initialized.")
        
        if hasattr(yt, 'title') and yt.title:
            video_title = yt.title
            logger.info(f"Processing YouTube video: '{video_title}'")
        else:
            logger.info("Processing YouTube video (title not found). Using default.")
        
        logger.info("Filtering for audio streams...")
        stream = yt.streams.filter(
            only_audio=True, 
            file_extension=Config.PREFERRED_AUDIO_FORMAT
        ).order_by('abr').desc().first()
        
        if not stream:
            logger.info(f"Preferred format '{Config.PREFERRED_AUDIO_FORMAT}' not found, trying get_audio_only()...")
            stream = yt.streams.get_audio_only()
            
        if not stream:
            logger.warning(f"No suitable audio stream found for URL: {url}")
            raise HTTPException(
                status_code=404, 
                detail="No suitable audio stream found for this YouTube video"
            )
        else:
            logger.info(f"Selected audio stream: {stream}")
            
        logger.info("Preparing to download audio...")
        audio_extension = f".{stream.subtype}"
        temp_audio_filename = f"{uuid.uuid4()}_youtube_audio{audio_extension}"
        temp_audio_path = os.path.join(Config.TEMP_DIR, temp_audio_filename)
        
        logger.info(f"Attempting download to: {temp_audio_path}")
        stream.download(output_path=Config.TEMP_DIR, filename=temp_audio_filename)
        logger.info(f"YouTube audio download complete: {temp_audio_path}")
        
        # 3. Process Audio
        logger.info(f"Calling process_audio_to_transcript for {temp_audio_path}...")
        transcript_file_path = process_audio_to_transcript(
            temp_audio_path,
            client,
            video_title # Use fetched title
        )
        logger.info(f"process_audio_to_transcript completed. Result path: {transcript_file_path}")
        
        # 4. Return Result
        logger.info(f"Successfully processed YouTube URL: {url}")
        logger.info("--- Exiting transcribe_youtube successfully ---")
        return {
            "message": "Transcription successful",
            "transcript_file": os.path.basename(transcript_file_path)
        }
        
    except PytubeFixError as e:
        logger.error(f"PytubeFixError processing {url}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing YouTube URL (pytube): {str(e)}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Generic error processing YouTube URL {url}: {error_msg}", exc_info=True)
        
        # Re-raise HTTP exceptions, wrap others
        if isinstance(e, HTTPException):
            raise
        elif isinstance(e, (ValueError, IOError, RuntimeError)): # Handle known processing errors
             raise HTTPException(status_code=500, detail=f"Processing error: {error_msg}")
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred during YouTube processing.")
        
    finally:
        # 5. Cleanup Temporary File
        logger.info(f"--- Cleaning up transcribe_youtube for URL: {url} ---")
        clean_temp_file(temp_audio_path)

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