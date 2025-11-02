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
    
    # Gemini settings for translation
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # Storage settings
    TEMP_DIR = "temp_files"
    
    # Transcript settings
    PAUSE_THRESHOLD = 1.0  # Seconds
    MAX_CHARS_PER_SEGMENT = 85  # Approx 2 lines on YouTube
    
    # Video and Audio settings
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
    ALLOWED_FILE_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
    
    # Audio settings
    PREFERRED_AUDIO_FORMAT = "m4a"
    AUDIO_BITRATE = "320k"
    
    # FFmpeg settings (new)
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")  # Use environment variable or default to 'ffmpeg' command
    PREFERRED_AUDIO_FORMAT_FOR_EXTRACTION = "mp3"  # Format for extracted audio

    # YouTube Download Strategy
    # Options: "pytubefix", "yt-dlp"
    YOUTUBE_DOWNLOAD_STRATEGY = os.getenv("YOUTUBE_DOWNLOAD_STRATEGY", "yt-dlp") 
    YT_DLP_COOKIES_PATH = os.getenv("YT_DLP_COOKIES_PATH", None) # Path to cookies.txt file for yt-dlp

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

# Initialize Gemini client for translation
gemini_client = None
if Config.GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully.")
    except ImportError:
        logger.error("Google GenAI library not found. Install with: pip install google-genai")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        # Non-fatal, but translation functionality will fail.
else:
    logger.warning("GEMINI_API_KEY not found. Translation functionality will be disabled.") 