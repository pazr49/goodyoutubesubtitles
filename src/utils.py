import os
import uuid
import logging
from typing import Optional, List, Any
import time

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

def save_transcript_to_file(
    transcript_text: str, 
    original_filename: str, 
    task_id_for_log: Optional[str] = None
) -> str:
    """Saves the transcript text to a .sbv file in the temp directory."""
    log_prefix = f"[{task_id_for_log}] " if task_id_for_log else ""
    
    base_name = os.path.splitext(original_filename)[0]
    # Sanitize base_name: replace non-alphanumeric (excluding space, underscore, hyphen) with underscore, limit length
    safe_base_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in base_name).strip()
    safe_base_name = safe_base_name.replace(' ', '_') # Replace spaces with underscores
    safe_base_name = safe_base_name[:50] # Limit length to avoid overly long filenames

    # Use a generic but unique name if safe_base_name becomes empty after sanitization
    if not safe_base_name:
        safe_base_name = "transcript"

    transcript_filename = f"{safe_base_name}_{uuid.uuid4().hex[:8]}.sbv"
    # Assuming Config.TEMP_DIR is correctly configured and accessible, e.g., 'temp_files'
    # For direct use here, ensure 'temp_files' is appropriate or use Config.TEMP_DIR
    temp_dir = getattr(__import__('src.config', fromlist=['Config']).Config, 'TEMP_DIR', 'temp_files')
    transcript_path = os.path.join(temp_dir, transcript_filename)

    try:
        os.makedirs(temp_dir, exist_ok=True) 
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        logger.info(f"{log_prefix}Transcript saved successfully to: {transcript_path}")
        return transcript_path
    except IOError as e:
        logger.error(f"{log_prefix}Failed to save transcript to {transcript_path}: {e}", exc_info=True)
        raise
    except Exception as e: # Catch any other potential errors, like related to Config access
        logger.error(f"{log_prefix}An unexpected error occurred while saving transcript to {transcript_path}: {e}", exc_info=True)
        raise

def save_raw_transcript_to_file(
    words_data: List[Any],
    original_filename: str,
    task_id_for_log: Optional[str] = None
) -> str:
    """Saves the raw word-level transcript data to a JSON file."""
    import json
    import time
    
    log_prefix = f"[{task_id_for_log}] " if task_id_for_log else ""
    
    base_name = os.path.splitext(original_filename)[0]
    # Sanitize base_name: replace non-alphanumeric (excluding space, underscore, hyphen) with underscore, limit length
    safe_base_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in base_name).strip()
    safe_base_name = safe_base_name.replace(' ', '_') # Replace spaces with underscores
    safe_base_name = safe_base_name[:50] # Limit length to avoid overly long filenames

    # Use a generic but unique name if safe_base_name becomes empty after sanitization
    if not safe_base_name:
        safe_base_name = "transcript"

    raw_filename = f"{safe_base_name}_{uuid.uuid4().hex[:8]}_raw.json"
    temp_dir = getattr(__import__('src.config', fromlist=['Config']).Config, 'TEMP_DIR', 'temp_files')
    raw_path = os.path.join(temp_dir, raw_filename)

    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convert words data to JSON-serializable format
        words_json = []
        total_duration = 0.0
        
        for word_info in words_data:
            if hasattr(word_info, 'type') and word_info.type == 'word':
                word_dict = {
                    'text': word_info.text,
                    'start': word_info.start,
                    'end': word_info.end,
                    'type': 'word'
                }
                words_json.append(word_dict)
                total_duration = max(total_duration, word_info.end)
        
        # Create metadata
        metadata = {
            'original_filename': original_filename,
            'total_words': len(words_json),
            'duration': total_duration,
            'created_at': time.time(),
            'format': 'word_level_timestamps'
        }
        
        # Create final JSON structure
        raw_data = {
            'metadata': metadata,
            'words': words_json
        }
        
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"{log_prefix}Raw transcript saved successfully to: {raw_path} ({len(words_json)} words, {total_duration:.1f}s)")
        return raw_path
        
    except IOError as e:
        logger.error(f"{log_prefix}Failed to save raw transcript to {raw_path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"{log_prefix}An unexpected error occurred while saving raw transcript to {raw_path}: {e}", exc_info=True)
        raise

def save_both_transcripts(
    transcript_text: str,
    words_data: List[Any],
    original_filename: str,
    task_id_for_log: Optional[str] = None
) -> tuple[str, str]:
    """Saves both processed (.sbv) and raw (.json) transcript files.
    
    Returns:
        tuple: (processed_file_path, raw_file_path)
    """
    log_prefix = f"[{task_id_for_log}] " if task_id_for_log else ""
    
    base_name = os.path.splitext(original_filename)[0]
    # Sanitize base_name: replace non-alphanumeric (excluding space, underscore, hyphen) with underscore, limit length
    safe_base_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in base_name).strip()
    safe_base_name = safe_base_name.replace(' ', '_') # Replace spaces with underscores
    safe_base_name = safe_base_name[:50] # Limit length to avoid overly long filenames

    # Use a generic but unique name if safe_base_name becomes empty after sanitization
    if not safe_base_name:
        safe_base_name = "transcript"

    # Generate unique identifier for this pair of files
    unique_id = uuid.uuid4().hex[:8]
    
    # Create filenames using the same base name
    processed_filename = f"{safe_base_name}_{unique_id}.sbv"
    raw_filename = f"{safe_base_name}_{unique_id}_raw.json"
    
    temp_dir = getattr(__import__('src.config', fromlist=['Config']).Config, 'TEMP_DIR', 'temp_files')
    processed_path = os.path.join(temp_dir, processed_filename)
    raw_path = os.path.join(temp_dir, raw_filename)

    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save processed transcript
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        
        # Save raw transcript
        import json
        import time
        
        # Convert words data to JSON-serializable format
        words_json = []
        total_duration = 0.0
        
        for word_info in words_data:
            if hasattr(word_info, 'type') and word_info.type == 'word':
                word_dict = {
                    'text': word_info.text,
                    'start': word_info.start,
                    'end': word_info.end,
                    'type': 'word'
                }
                words_json.append(word_dict)
                total_duration = max(total_duration, word_info.end)
        
        # Create metadata
        metadata = {
            'original_filename': original_filename,
            'total_words': len(words_json),
            'duration': total_duration,
            'created_at': time.time(),
            'format': 'word_level_timestamps'
        }
        
        # Create final JSON structure
        raw_data = {
            'metadata': metadata,
            'words': words_json
        }
        
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"{log_prefix}Both transcripts saved successfully:")
        logger.info(f"{log_prefix}  Processed: {processed_path}")
        logger.info(f"{log_prefix}  Raw: {raw_path} ({len(words_json)} words, {total_duration:.1f}s)")
        
        return processed_path, raw_path
        
    except IOError as e:
        logger.error(f"{log_prefix}Failed to save transcripts: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"{log_prefix}An unexpected error occurred while saving transcripts: {e}", exc_info=True)
        raise

def clean_temp_file(file_path: str, task_id_for_log: Optional[str] = None):
    """Safely removes a temporary file if it exists."""
    log_prefix = f"[{task_id_for_log}] " if task_id_for_log else ""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"{log_prefix}Successfully removed temporary file: {file_path}")
        elif file_path:
            logger.warning(f"{log_prefix}Attempted to clean temporary file, but it does not exist: {file_path}")
        # If file_path is None or empty, do nothing silently
    except OSError as e:
        logger.error(f"{log_prefix}Error removing temporary file {file_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"{log_prefix}An unexpected error occurred while cleaning temp file {file_path}: {e}", exc_info=True)

def cleanup_old_sbv_files(max_age_seconds: int = 300): # Default 5 minutes
    """
    Scans the TEMP_DIR for .sbv files and deletes those older than max_age_seconds.
    """
    logger.info(f"Running cleanup task for .sbv files older than {max_age_seconds} seconds.")
    now = time.time()
    files_deleted_count = 0
    files_scanned_count = 0

    try:
        for filename in os.listdir(Config.TEMP_DIR):
            if filename.endswith(".sbv"):
                files_scanned_count += 1
                file_path = os.path.join(Config.TEMP_DIR, filename)
                try:
                    file_mod_time = os.path.getmtime(file_path)
                    if (now - file_mod_time) > max_age_seconds:
                        os.remove(file_path)
                        logger.info(f"Deleted old .sbv file: {file_path}")
                        files_deleted_count += 1
                except FileNotFoundError:
                    logger.warning(f"File not found during cleanup (possibly deleted by another process): {file_path}")
                except OSError as e:
                    logger.error(f"Error deleting file {file_path} during cleanup: {e}")
        
        if files_scanned_count > 0:
            logger.info(f"Cleanup task finished. Scanned: {files_scanned_count} .sbv files. Deleted: {files_deleted_count}.")
        else:
            logger.info("Cleanup task finished. No .sbv files found to scan.")
            
    except FileNotFoundError:
        logger.warning(f"Temporary directory {Config.TEMP_DIR} not found during cleanup scan. It might be created later.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during .sbv cleanup: {e}", exc_info=True) 