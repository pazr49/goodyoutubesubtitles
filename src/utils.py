import os
import uuid
import logging
from typing import Optional

from .config import Config # Relative import

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def format_timestamp(seconds: float) -> str:
    """Converts seconds to YouTube SBV timestamp format (HH:MM:SS.ms)"""
    if seconds < 0:
        logger.warning(f"Received negative timestamp: {seconds}. Clamping to 0.")
        seconds = 0
        # Or raise ValueError("Timestamp must be non-negative") if preferred
        
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds_part = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"

def ends_with_sentence_end(text: str) -> bool:
    """Checks if text ends with sentence ending punctuation"""
    return text.strip().endswith(('.', '?', '!'))

def save_transcript_to_file(transcript_text: str, original_filename: str) -> str:
    """Saves transcript text to an .sbv file and returns the file path"""
    try:
        # Sanitize the filename - remove invalid characters
        base_filename = os.path.splitext(os.path.basename(original_filename))[0]
        # Replace invalid filename characters more strictly
        base_filename = "".join(c for c in base_filename if c.isalnum() or c in " _-").strip()
        if not base_filename: # Handle cases where sanitization removes everything
            base_filename = "transcript"
        
        # Generate a unique filename
        transcript_filename = f"{base_filename}_{uuid.uuid4().hex[:8]}.sbv"
        transcript_path = os.path.join(Config.TEMP_DIR, transcript_filename)

        # Write the transcript to the file
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        
        logger.info(f"Transcript saved successfully to: {transcript_path}")
        return transcript_path
    except IOError as e:
        logger.error(f"Failed to save transcript file for '{original_filename}': {e}", exc_info=True)
        raise IOError(f"Failed to save transcript file: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred saving transcript for '{original_filename}': {e}", exc_info=True)
        raise RuntimeError(f"An unexpected error occurred saving the transcript: {e}") from e

def clean_temp_file(file_path: Optional[str]) -> None:
    """Safely remove a temporary file if it exists"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")
        except OSError as e:
            # Log as warning, usually not critical if cleanup fails
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
    elif file_path:
        logger.debug(f"Temporary file not found for removal (already cleaned or never created): {file_path}") 