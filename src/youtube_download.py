import os
import uuid
import logging
import subprocess
import time
import uuid as _uuid # Keep original alias if used internally by functions

# Attempt to import Config, handling potential relative import issues if this file is run directly
try:
    from .config import Config, YouTube # Use relative import for sibling module
except ImportError:
    # Fallback for potential direct execution or different project structures (less likely for this refactor)
    from config import Config, YouTube 

logger = logging.getLogger(__name__)

# Helper to download YouTube audio in a sync thread (using pytubefix)
def download_youtube_pytubefix_sync(task_id_for_log: str, youtube_url: str) -> tuple[str, str]:
    """Synchronously download YouTube audio using pytubefix and return file path and title."""
    logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_PYTUBEFIX_SYNC: Initializing YouTube object for {youtube_url[:30]}... with use_po_token=True")
    
    # Ensure YouTube (pytubefix) is available
    if not YouTube:
        logger.error(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_PYTUBEFIX_SYNC: Pytubefix (YouTube) library not available/loaded.")
        raise RuntimeError("Pytubefix (YouTube) library not available. Cannot download with this method.")

    yt = YouTube(youtube_url, use_po_token=True)
    title = yt.title or "youtube_video"
    logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_PYTUBEFIX_SYNC: Video Title - {title}")
    
    logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_PYTUBEFIX_SYNC: Selecting audio stream...")
    stream = yt.streams.filter(
        only_audio=True,
        file_extension=Config.PREFERRED_AUDIO_FORMAT # Note: This might be for pytubefix's preferred *container*
    ).order_by('abr').desc().first() or yt.streams.get_audio_only()
    
    if not stream:
        logger.error(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_PYTUBEFIX_SYNC: No suitable audio stream found.")
        raise RuntimeError("No suitable audio stream found by pytubefix.")
        
    audio_ext = f".{stream.subtype}" 
    temp_audio_name = f"{_uuid.uuid4()}_youtube_pytubefix{audio_ext}" 
    temp_audio_path = os.path.join(Config.TEMP_DIR, temp_audio_name)
    logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_PYTUBEFIX_SYNC: Downloading stream to {temp_audio_path}")
    
    download_start_time = time.time()
    try:
        stream.download(output_path=Config.TEMP_DIR, filename=temp_audio_name)
    except Exception as e: # Catch potential errors from pytubefix download
        logger.error(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_PYTUBEFIX_SYNC: pytubefix download failed: {e}", exc_info=True)
        raise RuntimeError(f"pytubefix download failed: {e}")

    download_duration = time.time() - download_start_time
    logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_PYTUBEFIX_SYNC: Download complete in {download_duration:.2f} seconds. Path: {temp_audio_path}")
    
    return temp_audio_path, title

# Helper to download YouTube audio using yt-dlp
def download_youtube_yt_dlp_sync(task_id_for_log: str, youtube_url: str) -> tuple[str, str]:
    """Synchronously download YouTube audio using yt-dlp and return file path and title."""
    logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: Starting download for {youtube_url[:50]}...")

    temp_audio_basename = f"{_uuid.uuid4()}_youtube_yt_dlp"
    output_template = os.path.join(Config.TEMP_DIR, f"{temp_audio_basename}.%(ext)s")
    
    title = "youtube_video" # Default title
    try:
        title_command = [
            "yt-dlp",
            "--get-title",
            "--no-warnings",
            youtube_url
        ]
        title_process = subprocess.run(title_command, capture_output=True, text=True, check=True, encoding='utf-8')
        title = title_process.stdout.strip() or title
        logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: Video Title - {title}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: Failed to get title. stderr: {e.stderr}")
    except FileNotFoundError:
        logger.error(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: yt-dlp command not found. Ensure it's installed and in PATH.")
        raise RuntimeError("yt-dlp not found. Please ensure it's installed and accessible.")

    command = [
        "yt-dlp",
        "-x", 
        "--audio-format", Config.PREFERRED_AUDIO_FORMAT_FOR_EXTRACTION, 
        "-o", output_template,
        "--ffmpeg-location", Config.FFMPEG_PATH,
        "--no-warnings",
        "--no-playlist",
        youtube_url
    ]

    logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: Executing command: {' '.join(command)}")
    download_start_time = time.time()

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        if process.stderr:
            logger.debug(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: yt-dlp stderr: {process.stderr}")
        if process.stdout:
            logger.debug(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: yt-dlp stdout: {process.stdout}")
        
        download_duration = time.time() - download_start_time

        expected_extension = Config.PREFERRED_AUDIO_FORMAT_FOR_EXTRACTION
        temp_audio_path = os.path.join(Config.TEMP_DIR, f"{temp_audio_basename}.{expected_extension}")

        if not os.path.exists(temp_audio_path):
            logger.warning(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: Expected file {temp_audio_path} not found. Searching in TEMP_DIR...")
            found_files = [f for f in os.listdir(Config.TEMP_DIR) if f.startswith(temp_audio_basename)]
            if found_files:
                temp_audio_path = os.path.join(Config.TEMP_DIR, found_files[0])
                logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: Found file {temp_audio_path}")
            else:
                logger.error(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: Output audio file not found for basename {temp_audio_basename}")
                raise RuntimeError("yt-dlp downloaded, but output file could not be found.")

        logger.info(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: Download complete in {download_duration:.2f} seconds. Path: {temp_audio_path}")
        return temp_audio_path, title

    except subprocess.CalledProcessError as e:
        logger.error(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: yt-dlp failed. RC: {e.returncode}. stderr: {e.stderr}. stdout: {e.stdout}")
        raise RuntimeError(f"yt-dlp failed: {e.stderr or 'Unknown error'}")
    except FileNotFoundError:
        logger.error(f"[{task_id_for_log}] DOWNLOAD_YOUTUBE_YT_DLP_SYNC: yt-dlp command not found (should have been caught earlier).")
        raise RuntimeError("yt-dlp not found. Please ensure it's installed and accessible.") 