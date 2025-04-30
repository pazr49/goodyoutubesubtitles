import os
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# --- Add Pytubefix Import --- 
logger = logging.getLogger(__name__)

try:
    from pytubefix import YouTube
    from pytubefix.exceptions import PytubeFixError
    logger.info("Pytubefix library loaded successfully.")
except ImportError:
    logger.warning("Pytubefix library not found. YouTube functionality will be disabled.")
    YouTube = None
    PytubeFixError = None
# --- End Pytubefix Import ---

logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Application configuration"""
    # Load environment variables
    load_dotenv()
    
    # API settings
    API_TITLE = "YouTube Subtitles Generator"
    API_VERSION = "1.0.0"
    
    # ElevenLabs settings
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_MODEL_ID = "scribe_v1"
    
    # Storage settings
    TEMP_DIR = "temp_files"
    
    # Transcript settings
    PAUSE_THRESHOLD = 1.0  # Seconds
    MAX_CHARS_PER_SEGMENT = 85  # Approx 2 lines on YouTube
    
    # Video settings
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    
    # Audio settings
    PREFERRED_AUDIO_FORMAT = "m4a"
    AUDIO_BITRATE = "320k"

# --- Setup ---
# Create directory for temporary files
try:
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    logger.info(f"Temporary directory '{Config.TEMP_DIR}' ensured.")
except OSError as e:
    logger.error(f"Failed to create temporary directory '{Config.TEMP_DIR}': {e}")
    # Decide if this is fatal. Maybe raise an exception?
    # raise RuntimeError(f"Failed to create required temporary directory: {e}") from e

# Initialize ElevenLabs client
elevenlabs_client = None
if Config.ELEVENLABS_API_KEY:
    try:
        elevenlabs_client = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)
        logger.info("ElevenLabs client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ElevenLabs client: {e}")
        # Non-fatal, but endpoints relying on it will fail.
else:
    logger.warning("ELEVENLABS_API_KEY not found. ElevenLabs functionality will be disabled.") 