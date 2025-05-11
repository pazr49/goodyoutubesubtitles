import os
import logging
import uuid
from typing import List, Any, Callable, Optional, Dict
import shutil # Added for cleaning up chunk files

from moviepy.video.io.VideoFileClip import VideoFileClip
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment # Added for audio manipulation
# from pydub.utils import make_chunks # We might use this, or implement manually for overlap

from .config import Config # Relative imports
from .utils import format_timestamp, ends_with_sentence_end, save_transcript_to_file, clean_temp_file # Added clean_temp_file

logger = logging.getLogger(__name__)

# --- Constants for audio chunking ---
CHUNK_LENGTH_S = 5 * 60  # 5 minutes in seconds
OVERLAP_S = 5          # 5 seconds overlap

def _split_audio_into_chunks(task_id_for_log: str, audio_path: str, original_name: str) -> List[tuple[str, float]]:
    """
    Splits an audio file into chunks with overlap.
    Returns a list of tuples: (chunk_file_path, chunk_start_time_seconds).
    """
    logger.info(f"[{task_id_for_log}] Splitting audio file: {audio_path} into {CHUNK_LENGTH_S}s chunks with {OVERLAP_S}s overlap.")
    chunk_paths_with_starts = []
    
    try:
        audio = AudioSegment.from_file(audio_path)
        logger.info(f"[{task_id_for_log}] Audio duration: {len(audio) / 1000.0:.2f}s")
    except Exception as e:
        logger.error(f"[{task_id_for_log}] Failed to load audio file {audio_path} with pydub: {e}", exc_info=True)
        raise ValueError(f"Pydub failed to load audio: {e}")

    chunk_length_ms = CHUNK_LENGTH_S * 1000
    overlap_ms = OVERLAP_S * 1000
    
    step_ms = chunk_length_ms - overlap_ms
    if step_ms <= 0:
        logger.warning(f"[{task_id_for_log}] Overlap is greater than or equal to chunk length for {original_name}. Audio will not be split effectively.")
        if len(audio) <= chunk_length_ms:
            # Determine file extension safely
            audio_format = getattr(audio, 'format', None) or os.path.splitext(audio_path)[1].lstrip('.') or 'mp3'
            chunk_filename = f"{os.path.splitext(original_name)[0]}_chunk_0_{uuid.uuid4().hex[:8]}.{audio_format}"
            chunk_path = os.path.join(Config.TEMP_DIR, chunk_filename)
            audio.export(chunk_path, format=audio_format)
            logger.info(f"[{task_id_for_log}] Audio shorter than a chunk or invalid step, processing as one: {chunk_path}")
            return [(chunk_path, 0.0)]
        else:
            step_ms = chunk_length_ms

    idx = 0
    current_pos_ms = 0
    while current_pos_ms < len(audio):
        chunk_start_ms = current_pos_ms
        chunk_end_ms = current_pos_ms + chunk_length_ms
        actual_chunk_end_ms = min(chunk_end_ms, len(audio))
        
        chunk = audio[chunk_start_ms:actual_chunk_end_ms]
        
        if len(chunk) == 0:
            logger.warning(f"[{task_id_for_log}] Empty chunk generated at index {idx} for {original_name}, start_ms {chunk_start_ms}. Skipping.")
            current_pos_ms += step_ms
            continue

        chunk_base_name = os.path.splitext(original_name)[0]
        file_extension = getattr(audio, 'format', None) or os.path.splitext(audio_path)[1].lstrip('.') or "mp3"
        chunk_filename = f"{chunk_base_name}_chunk_{idx}_{uuid.uuid4().hex[:8]}.{file_extension}"
        chunk_path = os.path.join(Config.TEMP_DIR, chunk_filename)
        
        try:
            logger.info(f"[{task_id_for_log}] Exporting chunk {idx}: {chunk_path} (covers {chunk_start_ms/1000.0:.2f}s to {actual_chunk_end_ms/1000.0:.2f}s of original)")
            chunk.export(chunk_path, format=file_extension)
            chunk_paths_with_starts.append((chunk_path, chunk_start_ms / 1000.0))
        except Exception as e:
            logger.error(f"[{task_id_for_log}] Failed to export audio chunk {chunk_path} for {original_name}: {e}", exc_info=True)
            
        idx += 1
        if actual_chunk_end_ms == len(audio):
            break
        current_pos_ms += step_ms
        if len(audio) - current_pos_ms < overlap_ms and current_pos_ms < len(audio):
             pass # Already handled by loop condition and min()

    if not chunk_paths_with_starts and len(audio) > 0:
        logger.warning(f"[{task_id_for_log}] Splitting resulted in no chunks for {audio_path} (original: {original_name}). Treating as a single chunk.")
        audio_format = getattr(audio, 'format', None) or os.path.splitext(audio_path)[1].lstrip('.') or 'mp3'
        single_chunk_filename = f"{os.path.splitext(original_name)[0]}_single_chunk_{uuid.uuid4().hex[:8]}.{audio_format}"
        single_chunk_path = os.path.join(Config.TEMP_DIR, single_chunk_filename)
        audio.export(single_chunk_path, format=audio_format)
        return [(single_chunk_path, 0.0)]
        
    logger.info(f"[{task_id_for_log}] Successfully split '{original_name}' into {len(chunk_paths_with_starts)} chunks.")
    return chunk_paths_with_starts

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

    for word_info in words_data:
        if not hasattr(word_info, 'type') or word_info.type != 'word':
            continue
        
        if not all(hasattr(word_info, attr) for attr in ['text', 'start', 'end']):
            logger.warning(f"Skipping malformed word_info object: {vars(word_info) if hasattr(word_info, '__dict__') else word_info}")
            continue

        word_text = word_info.text
        word_start_time = word_info.start
        word_end_time = word_info.end
        word_char_count = len(word_text)

        should_split = False
        if current_segment_words and previous_word_end_time is not None:
            gap = word_start_time - previous_word_end_time
            
            if gap > pause_threshold:
                should_split = True
            elif ends_with_sentence_end(current_segment_words[-1]): # Relies on current_segment_words having text
                should_split = True
            elif (current_char_count + 1 + word_char_count) > max_chars_per_segment:
                should_split = True

            if should_split and len(current_segment_words) == 1 and gap <= pause_threshold: # Avoid single-word segments unless long pause
                should_split = False

        if should_split:
            try:
                if segment_start_time is not None and previous_word_end_time is not None: 
                    start_str = format_timestamp(segment_start_time)
                    end_str = format_timestamp(previous_word_end_time)
                    text = " ".join(current_segment_words)
                    transcript_segments.append(f"{start_str},{end_str}\n{text}")
            except ValueError as e:
                logger.error(f"Error formatting timestamp during segment split: {e} (start: {segment_start_time}, end: {previous_word_end_time})")

            current_segment_words = []
            current_char_count = 0
            segment_start_time = None
            previous_word_end_time = None

        if not current_segment_words:
            segment_start_time = word_start_time

        current_segment_words.append(word_text)
        current_char_count += (word_char_count + (1 if len(current_segment_words) > 1 else 0))
        previous_word_end_time = word_end_time

    if current_segment_words and segment_start_time is not None and previous_word_end_time is not None:
        try:
            start_str = format_timestamp(segment_start_time)
            end_str = format_timestamp(previous_word_end_time)
            text = " ".join(current_segment_words)
            transcript_segments.append(f"{start_str},{end_str}\n{text}")
        except ValueError as e:
            logger.error(f"Error formatting timestamp for final segment: {e} (start: {segment_start_time}, end: {previous_word_end_time})")

    result = "\n\n".join(transcript_segments)
    logger.info(f"Formatted transcript with {len(transcript_segments)} segments.")
    return result

def process_audio_to_transcript(
    task_id_for_log: str,
    audio_path: str, 
    client: ElevenLabs, 
    original_name: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
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
        return transcript_path
        
    except Exception as e:
        logger.error(f"[{task_id_for_log}] Critical error during transcription process for '{original_name}': {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to process audio to transcript for '{original_name}': {e}") from e
    finally:
        logger.debug(f"[{task_id_for_log}] Final cleanup of any remaining temporary chunk files for '{original_name}'.")
        for chunk_path, _ in chunk_files_with_starts: # Ensure this list is always defined
             clean_temp_file(chunk_path)

def extract_audio_from_video(task_id_for_log: str, video_path: str) -> str:
    """Extract audio from video file and return audio file path"""
    logger.info(f"[{task_id_for_log}] Starting audio extraction from video: {video_path}")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_filename = f"{base_name}_{uuid.uuid4().hex[:8]}.mp3"
    audio_path = os.path.join(Config.TEMP_DIR, audio_filename)
    
    try:
        logger.info(f"[{task_id_for_log}] Creating VideoFileClip for: {video_path}")
        video_clip = VideoFileClip(video_path)
        logger.info(f"[{task_id_for_log}] Extracting audio to {audio_path} (codec=mp3, bitrate={Config.AUDIO_BITRATE})")
        video_clip.audio.write_audiofile(
            audio_path, 
            codec='mp3', 
            bitrate=Config.AUDIO_BITRATE, 
            logger=None # Suppress moviepy's own verbose logs
        )
        video_clip.close()
        logger.info(f"[{task_id_for_log}] Audio extraction successful: {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"[{task_id_for_log}] Failed to extract audio from video {video_path}: {str(e)}", exc_info=True)
        if os.path.exists(audio_path):
            try:
                logger.info(f"[{task_id_for_log}] Attempting to clean up partially created audio file: {audio_path}")
                os.remove(audio_path)
                logger.info(f"[{task_id_for_log}] Cleaned up partially created audio file: {audio_path}")
            except OSError as cleanup_e:
                logger.warning(f"[{task_id_for_log}] Failed to clean up partial audio file {audio_path}: {cleanup_e}")
        raise ValueError(f"Failed to extract audio from video '{video_path}': {e}") from e 