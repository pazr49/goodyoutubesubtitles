import os
import logging
import uuid
import subprocess
import json
from typing import List, Any, Callable, Optional, Dict
import shutil # Added for cleaning up chunk files
import asyncio
from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip
from elevenlabs import ElevenLabs
# Commenting out pydub which has compatibility issues with Python 3.13
# from pydub import AudioSegment # Added for audio manipulation
# from pydub.utils import make_chunks # We might use this, or implement manually for overlap

from .config import Config # Relative imports
from .utils import format_timestamp, ends_with_sentence_end, save_transcript_to_file, clean_temp_file # Added clean_temp_file

logger = logging.getLogger(__name__)

# --- Constants for audio chunking ---
CHUNK_LENGTH_S = 5 * 60  # 5 minutes in seconds
OVERLAP_S = 5          # 5 seconds overlap

def _get_audio_duration(task_id_for_log: str, audio_path: str) -> float:
    """
    Get the duration of an audio file using ffprobe.
    Returns the duration in seconds.
    """
    logger.info(f"[{task_id_for_log}] Getting duration for audio file: {audio_path}")
    
    try:
        # Use ffprobe to get audio duration
        command = [
            os.path.join(os.path.dirname(Config.FFMPEG_PATH), "ffprobe"),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            audio_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        
        logger.info(f"[{task_id_for_log}] Audio duration: {duration:.2f}s")
        return duration
    except Exception as e:
        logger.error(f"[{task_id_for_log}] Error getting audio duration: {e}")
        # Return a reasonable default (30 minutes) if we can't determine the duration
        # This will cause chunking to proceed with potentially incorrect chunks
        return 30 * 60

def _split_audio_into_chunks(task_id_for_log: str, audio_path: str, original_name: str) -> List[tuple[str, float]]:
    """
    Splits an audio file into chunks with overlap using FFmpeg.
    Returns a list of tuples: (chunk_file_path, chunk_start_time_seconds).
    """
    logger.info(f"[{task_id_for_log}] Splitting audio file: {audio_path} into {CHUNK_LENGTH_S}s chunks with {OVERLAP_S}s overlap.")
    chunk_paths_with_starts = []
    
    try:
        # Get the audio duration
        audio_duration = _get_audio_duration(task_id_for_log, audio_path)
        
        # If audio is shorter than a chunk, just return it as is
        if audio_duration <= CHUNK_LENGTH_S:
            logger.info(f"[{task_id_for_log}] Audio shorter than chunk length, using as single chunk.")
            ext = os.path.splitext(audio_path)[1]
            single_chunk_filename = f"{os.path.splitext(original_name)[0]}_single_chunk_{uuid.uuid4().hex[:8]}{ext}"
            single_chunk_path = os.path.join(Config.TEMP_DIR, single_chunk_filename)
            
            # Copy the file instead of transcoding
            shutil.copy2(audio_path, single_chunk_path)
            return [(single_chunk_path, 0.0)]
        
        # Calculate step size for chunks (accounting for overlap)
        step_size = CHUNK_LENGTH_S - OVERLAP_S
        
        # Generate chunks
        idx = 0
        current_pos = 0.0
        
        while current_pos < audio_duration:
            chunk_start = current_pos
            chunk_end = min(current_pos + CHUNK_LENGTH_S, audio_duration)
            
            # Skip if we'd create an empty chunk
            if chunk_end <= chunk_start:
                break
                
            chunk_duration = chunk_end - chunk_start
            
            # Skip very short chunks at the end
            if chunk_duration < 1.0 and idx > 0:  # 1 second minimum, and not the first chunk
                break
                
            # Create chunk filename
            chunk_base_name = os.path.splitext(original_name)[0]
            ext = os.path.splitext(audio_path)[1]
            chunk_filename = f"{chunk_base_name}_chunk_{idx}_{uuid.uuid4().hex[:8]}{ext}"
            chunk_path = os.path.join(Config.TEMP_DIR, chunk_filename)
            
            # FFmpeg command to extract the chunk
            command = [
                Config.FFMPEG_PATH,
                "-ss", str(chunk_start),
                "-i", audio_path,
                "-t", str(chunk_duration),
                "-c", "copy",  # Use copy codec for speed 
                "-y",
                chunk_path
            ]
            
            logger.info(f"[{task_id_for_log}] Extracting chunk {idx}: covers {chunk_start:.2f}s to {chunk_end:.2f}s of original")
            
            try:
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                if process.stderr:
                    logger.debug(f"[{task_id_for_log}] FFmpeg chunk {idx} stderr: {process.stderr}")
                
                chunk_paths_with_starts.append((chunk_path, chunk_start))
                logger.info(f"[{task_id_for_log}] Chunk {idx} extracted to: {chunk_path}")
            except Exception as e:
                logger.error(f"[{task_id_for_log}] Failed to extract chunk {idx}: {e}")
                # Continue to next chunk if this one fails
            
            idx += 1
            current_pos += step_size
            
            # Break if we've reached the end of the audio
            if chunk_end >= audio_duration:
                break
        
        if not chunk_paths_with_starts:
            logger.warning(f"[{task_id_for_log}] No chunks created for {audio_path}. Treating as a single file.")
            ext = os.path.splitext(audio_path)[1]
            single_chunk_filename = f"{os.path.splitext(original_name)[0]}_single_chunk_{uuid.uuid4().hex[:8]}{ext}"
            single_chunk_path = os.path.join(Config.TEMP_DIR, single_chunk_filename)
            
            # Copy the file instead of transcoding
            shutil.copy2(audio_path, single_chunk_path)
            return [(single_chunk_path, 0.0)]
            
        logger.info(f"[{task_id_for_log}] Successfully split '{original_name}' into {len(chunk_paths_with_starts)} chunks.")
        return chunk_paths_with_starts
        
    except Exception as e:
        logger.error(f"[{task_id_for_log}] Error splitting audio: {e}")
        # If chunking fails, return the original audio file as a single chunk
        ext = os.path.splitext(audio_path)[1]
        single_chunk_filename = f"{os.path.splitext(original_name)[0]}_error_chunk_{uuid.uuid4().hex[:8]}{ext}"
        single_chunk_path = os.path.join(Config.TEMP_DIR, single_chunk_filename)
        
        # Copy the file instead of transcoding
        shutil.copy2(audio_path, single_chunk_path)
        return [(single_chunk_path, 0.0)]

def _stitch_transcriptions(task_id_for_log: str, chunk_word_data_list: List[tuple[List[Any], float]], overlap_s: float) -> List[Any]:
    """
    Stitches word data from multiple chunks, handling overlaps.
    chunk_word_data_list: List of (list of word objects, chunk_start_time_in_original_audio_seconds)
    overlap_s: The duration of the overlap in seconds.
    """
    logger.info(f"[{task_id_for_log}] Stitching {len(chunk_word_data_list)} transcript chunks with {overlap_s}s overlap.")
    stitched_words = []
    last_global_end_time = 0.0  # Tracks the end time of the last word added from the *previous* chunk

    for i, (words_from_chunk, chunk_original_start_s) in enumerate(chunk_word_data_list):
        logger.debug(f"[{task_id_for_log}] Processing chunk {i} for stitching (original start: {chunk_original_start_s:.2f}s).")
        
        current_chunk_processed_words = []
        for word_info in words_from_chunk:
            if not all(hasattr(word_info, attr) for attr in ['text', 'start', 'end', 'type']) or word_info.type != 'word':
                continue

            global_word_start = word_info.start + chunk_original_start_s
            global_word_end = word_info.end + chunk_original_start_s

            adjusted_word_info = {
                'text': word_info.text,
                'start': global_word_start,
                'end': global_word_end,
                'type': 'word' 
            }
            current_chunk_processed_words.append(adjusted_word_info)

        if i == 0:
            stitched_words.extend(current_chunk_processed_words)
            if stitched_words:
                last_global_end_time = stitched_words[-1]['end']
            logger.debug(f"[{task_id_for_log}] Added {len(current_chunk_processed_words)} words from first chunk. Last global end time: {last_global_end_time:.2f}s")
        else:
            added_count = 0
            for word_info in current_chunk_processed_words:
                if word_info['start'] >= last_global_end_time:
                    stitched_words.append(word_info)
                    added_count +=1
            
            if stitched_words and added_count > 0: # ensure last_global_end_time is updated only if new words were added
                last_global_end_time = stitched_words[-1]['end']
            logger.debug(f"[{task_id_for_log}] Added {added_count} words from chunk {i}. Last global end time now: {last_global_end_time:.2f}s")

    logger.info(f"[{task_id_for_log}] Stitching complete. Total words: {len(stitched_words)}.")
    
    class PseudoWord:
        def __init__(self, text, start, end, type='word'):
            self.text = text
            self.start = start
            self.end = end
            self.type = type

    final_pseudo_words = [PseudoWord(text=w['text'], start=w['start'], end=w['end']) for w in stitched_words]
    return final_pseudo_words


def create_transcript_format(
    words_data: List[Any],
    pause_threshold: float = Config.PAUSE_THRESHOLD,
    max_chars_per_segment: int = Config.MAX_CHARS_PER_SEGMENT
) -> str:
    """Processes ElevenLabs word data into YouTube subtitle format (.sbv)"""
    if not words_data:
        logger.warning("Received empty words_data for transcript formatting.")
        return ""

    transcript_segments = []
    current_segment_words = []
    segment_start_time = None
    previous_word_end_time = None
    current_char_count = 0
    PUNCTUATION_TO_ATTACH = {'.', '?', '!'} # Define punctuation that should attach

    # Helper to build segment text, attaching punctuation tokens to preceding words
    def _assemble_text(words: List[str]) -> str:
        assembled = []
        for token in words:
            if token in PUNCTUATION_TO_ATTACH and assembled:
                assembled[-1] += token
            else:
                assembled.append(token)
        return " ".join(assembled)

    for word_info in words_data:
        if not hasattr(word_info, 'type') or word_info.type != 'word':
            continue
        
        if not all(hasattr(word_info, attr) for attr in ['text', 'start', 'end']):
            logger.warning(f"Skipping malformed word_info object: {vars(word_info) if hasattr(word_info, '__dict__') else word_info}")
            continue

        word_text = word_info.text
        word_start_time = word_info.start
        word_end_time = word_info.end
        current_word_char_count = len(word_text) # Use a distinct name
        is_current_word_standalone_punctuation = word_text in PUNCTUATION_TO_ATTACH

        # --- Decision Point 1: Should we split *before* processing the current word_info? ---
        perform_split_before_this_word = False
        calculated_gap_before_this_word = None

        if current_segment_words and previous_word_end_time is not None:
            calculated_gap_before_this_word = word_start_time - previous_word_end_time

            # Reason 1: Previous segment ended with sentence punctuation
            if ends_with_sentence_end(current_segment_words[-1]):
                perform_split_before_this_word = True
            # Reason 2: Adding current word would exceed character limit
            elif (current_char_count + (1 if current_segment_words else 0) + current_word_char_count) > max_chars_per_segment:
                # Allow sentence-ending words to overflow and split after
                if not ends_with_sentence_end(word_text):
                    perform_split_before_this_word = True
            # Reason 3: Pause occurred
            elif calculated_gap_before_this_word > pause_threshold:
                if not is_current_word_standalone_punctuation:
                    perform_split_before_this_word = True
        
        if perform_split_before_this_word and \
           len(current_segment_words) == 1 and \
           not (calculated_gap_before_this_word is not None and calculated_gap_before_this_word > pause_threshold):
            perform_split_before_this_word = False

        if perform_split_before_this_word:
            try:
                if segment_start_time is not None and previous_word_end_time is not None: 
                    start_str = format_timestamp(segment_start_time)
                    end_str = format_timestamp(previous_word_end_time)
                    text = _assemble_text(current_segment_words)
                    transcript_segments.append(f"{start_str},{end_str}\n{text}")
            except ValueError as e:
                logger.error(f"Error formatting timestamp during segment split (before word): {e} (start: {segment_start_time}, end: {previous_word_end_time})")
            
            current_segment_words = []
            current_char_count = 0
            segment_start_time = None
            previous_word_end_time = None

        # --- Process the current word_info: Add it to the (potentially new) segment ---
        if not current_segment_words:
            segment_start_time = word_start_time

        current_segment_words.append(word_text)
        current_char_count += (current_word_char_count + (1 if len(current_segment_words) > 1 else 0))
        previous_word_end_time = word_end_time # End time is now of the current word

        # --- Decision Point 2: Should we split *after* having added the current word_info? ---
        perform_split_after_this_word = False
        
        if current_segment_words: # Check if there are words in the current segment
            if ends_with_sentence_end(current_segment_words[-1]):
                perform_split_after_this_word = True
            elif is_current_word_standalone_punctuation and \
                 calculated_gap_before_this_word is not None and \
                 calculated_gap_before_this_word > pause_threshold:
                perform_split_after_this_word = True
        
        # No anti-single-word logic here, as splitting after specific punctuation or sentence end is usually intended.

        if perform_split_after_this_word:
            try:
                if segment_start_time is not None and previous_word_end_time is not None: 
                    start_str = format_timestamp(segment_start_time)
                    end_str = format_timestamp(previous_word_end_time)
                    text = _assemble_text(current_segment_words)
                    transcript_segments.append(f"{start_str},{end_str}\n{text}")
            except ValueError as e:
                logger.error(f"Error formatting timestamp during segment split (after word): {e} (start: {segment_start_time}, end: {previous_word_end_time})")

            current_segment_words = []
            current_char_count = 0
            segment_start_time = None
            previous_word_end_time = None

    if current_segment_words and segment_start_time is not None and previous_word_end_time is not None:
        try:
            start_str = format_timestamp(segment_start_time)
            end_str = format_timestamp(previous_word_end_time)
            text = _assemble_text(current_segment_words)
            transcript_segments.append(f"{start_str},{end_str}\n{text}")
        except ValueError as e:
            logger.error(f"Error formatting timestamp for final segment: {e} (start: {segment_start_time}, end: {previous_word_end_time})")

    result = "\n\n".join(transcript_segments)
    logger.info(f"Formatted transcript with {len(transcript_segments)} segments.")
    return result

async def process_audio_to_transcript(
    task_id_for_log: str,
    audio_path: str, 
    client: ElevenLabs, 
    original_name: str,
    target_languages: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    gemini_client: Optional[Any] = None
) -> str:
    """Process an audio file to generate transcript using chunking"""
    logger.info(f"[{task_id_for_log}] Starting transcription process for '{original_name}' (path: {audio_path})")
    
    if not client:
        logger.error(f"[{task_id_for_log}] ElevenLabs client not available for '{original_name}'.")
        raise ValueError("ElevenLabs client is not available")

    chunk_files_with_starts = []
    all_chunk_word_data = [] # Stores (word_list_from_chunk, chunk_start_time_s)

    try:
        # 1. Split audio into chunks
        logger.info(f"[{task_id_for_log}] Step 1: Splitting audio into chunks for '{original_name}'.")
        chunk_files_with_starts = _split_audio_into_chunks(task_id_for_log, audio_path, original_name)
        
        if not chunk_files_with_starts:
            logger.warning(f"[{task_id_for_log}] Audio splitting yielded no chunks for '{original_name}'. Aborting transcription.")
            # No need to raise here if we want to return an empty transcript; empty list will be handled below.
            # However, if this is a critical failure, raising is appropriate.
            # For now, let's assume it might be valid (e.g. very short silent audio)
            # Fall through to stitch and format, which will produce an empty transcript.
            # If an error is preferred: raise ValueError("Audio splitting failed to produce any chunks.")
            pass # Let it proceed to return an empty transcript path if no chunks

        # 2. Process each chunk
        logger.info(f"[{task_id_for_log}] Step 2: Processing {len(chunk_files_with_starts)} audio chunks for '{original_name}' through ElevenLabs.")
        total_chunks = len(chunk_files_with_starts)
        
        for i, (chunk_path, chunk_start_s) in enumerate(chunk_files_with_starts):
            current_chunk_number = i + 1
            logger.info(f"[{task_id_for_log}] Processing chunk {current_chunk_number}/{total_chunks} for '{original_name}': {chunk_path}")
            
            if progress_callback:
                progress_callback({
                    "status": "processing", 
                    "stage": f"transcribing_chunk_{current_chunk_number}_of_{total_chunks}", 
                    "message": f"Transcribing audio chunk {current_chunk_number} of {total_chunks} for '{original_name}'...",
                    "current_chunk": current_chunk_number,
                    "total_chunks": total_chunks
                })

            try:
                with open(chunk_path, 'rb') as audio_chunk_file:
                    response = client.speech_to_text.convert(
                        file=audio_chunk_file,
                        model_id=Config.ELEVENLABS_MODEL_ID
                    )
                
                if hasattr(response, 'words') and response.words:
                    logger.info(f"[{task_id_for_log}] Received {len(response.words)} words from chunk {current_chunk_number} for '{original_name}'.")
                    all_chunk_word_data.append((response.words, chunk_start_s))
                    if progress_callback:
                        progress_callback({
                            "status": "processing", 
                            "stage": f"chunk_{current_chunk_number}_of_{total_chunks}_complete", 
                            "message": f"Completed transcription for chunk {current_chunk_number} of {total_chunks} for '{original_name}'.",
                            "current_chunk": current_chunk_number,
                            "total_chunks": total_chunks
                        })
                else:
                    logger.warning(f"[{task_id_for_log}] Empty or invalid transcription response for chunk {chunk_path} of '{original_name}'. Response: {vars(response) if hasattr(response, '__dict__') else response}")
            except Exception as e:
                logger.error(f"[{task_id_for_log}] Error transcribing audio chunk {chunk_path} for '{original_name}': {e}", exc_info=True)
                # Optionally, re-raise or decide if partial transcription is acceptable
                # For now, log and continue to allow processing of other chunks.
            finally:
                clean_temp_file(chunk_path)

        if not all_chunk_word_data and chunk_files_with_starts: # Only error if there were chunks but no data
            logger.error(f"[{task_id_for_log}] No words returned from any audio chunks for '{original_name}'.")
            raise ValueError("Transcription service returned no words from any chunks.")

        # 3. Stitch transcriptions
        logger.info(f"[{task_id_for_log}] Step 3: Stitching transcriptions for '{original_name}'.")
        if progress_callback:
            progress_callback({"status": "processing", "stage": "stitching", "message": f"Stitching transcriptions for '{original_name}'..."})
        
        stitched_word_objects = _stitch_transcriptions(task_id_for_log, all_chunk_word_data, OVERLAP_S)
        
        if not stitched_word_objects and (all_chunk_word_data or chunk_files_with_starts) : # Log warning if there was data/chunks but stitching failed to produce words
            logger.warning(f"[{task_id_for_log}] Stitching resulted in no words for '{original_name}', though chunks were processed.")
            
        # 4. Format transcript
        logger.info(f"[{task_id_for_log}] Step 4: Formatting final stitched transcript for '{original_name}'.")
        transcript_text = create_transcript_format(stitched_word_objects)
        
        # 5. Save to file
        logger.info(f"[{task_id_for_log}] Step 5: Saving final transcript for '{original_name}'.")
        transcript_path = save_transcript_to_file(transcript_text, original_name, task_id_for_log) # Pass task_id to save_transcript
        logger.info(f"[{task_id_for_log}] Successfully processed '{original_name}'. Transcript at: {transcript_path}")
        
        # 6. Handle translations if requested
        if target_languages and gemini_client:
            # Import translation function here to avoid circular imports
            from .translation import translate_sbv_file
            
            logger.info(f"[{task_id_for_log}] Step 6: Processing translations for '{original_name}' to languages: {target_languages}")
            translated_files = []
            
            for language in target_languages:
                logger.info(f"[{task_id_for_log}] Starting translation to {language} for '{original_name}'")
                
                try:
                    translated_path = await translate_sbv_file(
                        transcript_path,
                        language,
                        gemini_client,
                        task_id_for_log,
                        original_name,
                        progress_callback
                    )
                    translated_files.append(translated_path)
                    logger.info(f"[{task_id_for_log}] Translation to {language} completed: {translated_path}")
                    
                except Exception as e:
                    logger.error(f"[{task_id_for_log}] Error translating to {language}: {e}")
                    # Continue with other languages even if one fails
                    continue
            
            if translated_files:
                logger.info(f"[{task_id_for_log}] Created {len(translated_files)} translated versions for '{original_name}'")
        
        return transcript_path
        
    except Exception as e:
        logger.error(f"[{task_id_for_log}] Critical error during transcription process for '{original_name}': {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to process audio to transcript for '{original_name}': {e}") from e
    finally:
        logger.debug(f"[{task_id_for_log}] Final cleanup of any remaining temporary chunk files for '{original_name}'.")
        for chunk_path, _ in chunk_files_with_starts: # Ensure this list is always defined
             clean_temp_file(chunk_path)

def extract_audio_from_video(task_id_for_log: str, video_path: str) -> str:
    """
    Extracts audio from a video file using ffmpeg directly for efficiency.
    Returns the path to the extracted audio file.
    """
    logger.info(f"[{task_id_for_log}] Starting audio extraction for {video_path} using direct ffmpeg.")

    os.makedirs(Config.TEMP_DIR, exist_ok=True)

    output_audio_filename = f"{uuid.uuid4()}_extracted_audio.{Config.PREFERRED_AUDIO_FORMAT_FOR_EXTRACTION}"
    output_audio_path = os.path.join(Config.TEMP_DIR, output_audio_filename)

    command = [
        Config.FFMPEG_PATH,
        '-i', video_path,
        '-vn',  # No video output
        '-c:a', 'aac',  # Default to AAC codec
        '-ar', '44100', # Audio sample rate
        '-b:a', '192k', # Audio bitrate
        '-y', # Overwrite output file without asking
        output_audio_path
    ]

    if Config.PREFERRED_AUDIO_FORMAT_FOR_EXTRACTION.lower() == "mp3":
        command[5] = 'libmp3lame' # Change codec if mp3 is preferred

    logger.info(f"[{task_id_for_log}] Executing ffmpeg command: {' '.join(command)}")

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        if process.stdout: # Log stdout only if it has content
            logger.info(f"[{task_id_for_log}] FFmpeg stdout: {process.stdout}")
        if process.stderr: # Log stderr only if it has content (ffmpeg often uses stderr for info)
            logger.debug(f"[{task_id_for_log}] FFmpeg stderr: {process.stderr}")
        logger.info(f"[{task_id_for_log}] Audio extracted successfully to {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"[{task_id_for_log}] FFmpeg failed for {video_path}.")
        logger.error(f"[{task_id_for_log}] FFmpeg stderr: {e.stderr}")
        logger.error(f"[{task_id_for_log}] FFmpeg stdout: {e.stdout}")
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}") from e
    except FileNotFoundError:
        logger.error(f"[{task_id_for_log}] ffmpeg command not found at '{Config.FFMPEG_PATH}'. Ensure ffmpeg is installed and in PATH, or Config.FFMPEG_PATH is set correctly.")
        raise RuntimeError(f"ffmpeg not found. Please ensure it's installed and accessible.") 