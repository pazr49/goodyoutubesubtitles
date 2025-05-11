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
        logger.info(f"[{task_id_for_log}] Audio duration: {len(audio) / 1000.0}s")
    except Exception as e:
        logger.error(f"[{task_id_for_log}] Failed to load audio file {audio_path} with pydub: {e}", exc_info=True)
        raise ValueError(f"Pydub failed to load audio: {e}")

    chunk_length_ms = CHUNK_LENGTH_S * 1000
    overlap_ms = OVERLAP_S * 1000
    
    # Calculate effective chunk length (how much to step forward each time)
    step_ms = chunk_length_ms - overlap_ms
    if step_ms <= 0:
        logger.warning("Overlap is greater than or equal to chunk length. Audio will not be split effectively.")
        # Handle this case: perhaps process as a single chunk if too short or step is invalid
        if len(audio) <= chunk_length_ms: # If audio is shorter than one chunk
            chunk_filename = f"{os.path.splitext(original_name)[0]}_chunk_0_{uuid.uuid4().hex[:8]}.{audio.format or 'mp3'}"
            chunk_path = os.path.join(Config.TEMP_DIR, chunk_filename)
            audio.export(chunk_path, format=audio.format or 'mp3')
            logger.info(f"[{task_id_for_log}] Audio shorter than a chunk or invalid step, processing as one: {chunk_path}")
            return [(chunk_path, 0.0)]
        else: # If step is invalid but audio is longer, adjust step to avoid issues.
            step_ms = chunk_length_ms # Effectively no overlap if step_ms was problematic

    idx = 0
    current_pos_ms = 0
    while current_pos_ms < len(audio):
        chunk_start_ms = current_pos_ms
        chunk_end_ms = current_pos_ms + chunk_length_ms
        
        # Ensure the chunk doesn't go beyond the audio length for the main part
        # The actual segment might be shorter if it's the last one.
        actual_chunk_end_ms = min(chunk_end_ms, len(audio))
        
        chunk = audio[chunk_start_ms:actual_chunk_end_ms]
        
        if len(chunk) == 0: # Should not happen if logic is correct
            logger.warning(f"[{task_id_for_log}] Empty chunk generated at index {idx}, start_ms {chunk_start_ms}. Skipping.")
            current_pos_ms += step_ms
            continue

        # Use part of original name + chunk index + UUID for uniqueness
        chunk_base_name = os.path.splitext(original_name)[0]
        # Determine file extension from pydub's audio segment if possible, else default
        file_extension = getattr(audio, 'format', os.path.splitext(audio_path)[1].lstrip('.'))
        if not file_extension: # Fallback if format is not detected or path has no extension
            file_extension = "mp3" # A common default

        chunk_filename = f"{chunk_base_name}_chunk_{idx}_{uuid.uuid4().hex[:8]}.{file_extension}"
        chunk_path = os.path.join(Config.TEMP_DIR, chunk_filename)
        
        try:
            logger.info(f"[{task_id_for_log}] Exporting chunk {idx}: {chunk_path} (covers {chunk_start_ms/1000.0}s to {actual_chunk_end_ms/1000.0}s of original)")
            chunk.export(chunk_path, format=file_extension)
            chunk_paths_with_starts.append((chunk_path, chunk_start_ms / 1000.0))
        except Exception as e:
            logger.error(f"[{task_id_for_log}] Failed to export audio chunk {chunk_path}: {e}", exc_info=True)
            # Decide if we should skip this chunk or raise an error
            # For now, skip and log
            
        idx += 1
        if actual_chunk_end_ms == len(audio): # Reached the end of the audio
            break
        current_pos_ms += step_ms
        # Ensure we don't create a tiny sliver chunk at the very end if the step perfectly aligns
        # or if the remaining audio is smaller than the overlap.
        if len(audio) - current_pos_ms < overlap_ms and current_pos_ms < len(audio) :
             # If remaining part is smaller than overlap, and we haven't finished,
             # it means the previous chunk already covered most of this.
             # We can stop, or adjust the last chunk to cover everything.
             # The current logic should already handle this by min(chunk_end_ms, len(audio))
             pass


    if not chunk_paths_with_starts and len(audio) > 0:
        # If splitting resulted in no chunks (e.g., very short audio not caught by initial check)
        # Treat the whole audio as a single chunk.
        logger.warning(f"[{task_id_for_log}] Splitting resulted in no chunks for {audio_path}. Treating as a single chunk.")
        single_chunk_filename = f"{os.path.splitext(original_name)[0]}_single_chunk_{uuid.uuid4().hex[:8]}.{getattr(audio, 'format', 'mp3')}"
        single_chunk_path = os.path.join(Config.TEMP_DIR, single_chunk_filename)
        audio.export(single_chunk_path, format=getattr(audio, 'format', 'mp3'))
        return [(single_chunk_path, 0.0)]
        
    logger.info(f"[{task_id_for_log}] Successfully split audio into {len(chunk_paths_with_starts)} chunks.")
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
        logger.debug(f"[{task_id_for_log}] Processing chunk {i} which originally started at {chunk_original_start_s}s.")
        
        current_chunk_processed_words = []
        for word_info in words_from_chunk:
            if not all(hasattr(word_info, attr) for attr in ['text', 'start', 'end', 'type']) or word_info.type != 'word':
                continue

            # Adjust word timestamps to be relative to the global timeline
            global_word_start = word_info.start + chunk_original_start_s
            global_word_end = word_info.end + chunk_original_start_s

            # Create a new word object or update existing to avoid modifying original response objects if they are complex
            # For now, let's assume word_info can be augmented with global times if not already dicts
            # A safer way would be to create new dicts/objects:
            adjusted_word_info = {
                'text': word_info.text,
                'start': global_word_start,
                'end': global_word_end,
                'type': 'word' 
                # copy other relevant attributes if ElevenLabs response.words items have more
            }
            # If word_info is an object with attributes:
            # word_info.global_start = global_word_start (add new attribute)
            # word_info.global_end = global_word_end

            current_chunk_processed_words.append(adjusted_word_info)

        if i == 0: # First chunk
            stitched_words.extend(current_chunk_processed_words)
            if stitched_words:
                last_global_end_time = stitched_words[-1]['end']
            logger.debug(f"[{task_id_for_log}] Added {len(current_chunk_processed_words)} words from the first chunk. Last end time: {last_global_end_time}")
        else:
            # For subsequent chunks, only add words that start after the last added word from the previous chunk,
            # considering the overlap.
            # We want to find the point in the current_chunk_processed_words that truly follows last_global_end_time.
            # The overlap means the current chunk started transcribing 'overlap_s' seconds *before* last_global_end_time
            # (approximately, as chunk_original_start_s is the actual start).
            
            # Words from the current chunk should be added if their start time is
            # greater than or equal to the effective end of the previous useful segment.
            # The previous chunk's words went up to last_global_end_time.
            # The current chunk started at chunk_original_start_s.
            # The overlap region is from chunk_original_start_s to chunk_original_start_s + overlap_s.
            # We trust the words from the *previous* chunk for times < chunk_original_start_s + overlap_s.
            # So, we take words from the *current* chunk that start at/after (chunk_original_start_s + overlap_s)
            # No, this is not quite right. We need to compare with last_global_end_time.

            effective_start_for_this_chunk = last_global_end_time - overlap_s
            
            added_count = 0
            for word_info in current_chunk_processed_words:
                # word_info['start'] is already global here
                if word_info['start'] >= last_global_end_time:
                    stitched_words.append(word_info)
                    added_count +=1
                # A more sophisticated approach might try to find a seam in the overlap.
                # For now, a hard cutoff: if current word starts after previous chunk's last word, add it.
                # This might lose a fraction of a word if cut exactly.
                # A slightly better heuristic: if a word starts *within* the overlap but *after* the previous last word's start + tiny_delta, consider it.
                # Let's keep it simple: if it starts after the previous last_global_end_time, it's new.
                # This means we prefer the end of the previous chunk over the start of the current one in the overlap.
            
            if stitched_words: # Update last_global_end_time with the newly added words
                last_global_end_time = stitched_words[-1]['end']
            logger.debug(f"[{task_id_for_log}] Added {added_count} words from chunk {i}. Last end time now: {last_global_end_time}")

    logger.info(f"[{task_id_for_log}] Stitching complete. Total words: {len(stitched_words)}.")
    # The 'words' objects from ElevenLabs might not be simple dicts.
    # We need to ensure create_transcript_format can handle what we pass it.
    # Let's convert our list of dicts back to a list of objects similar to what ElevenLabs returns, if necessary.
    # For now, assuming create_transcript_format can handle list of dicts with 'text', 'start', 'end', 'type'.
    # If not, we'd need to reconstruct objects:
    # from elevenlabs.types.speech_to_text import Word # or similar
    # final_words_objects = [Word(text=w['text'], start=w['start'], end=w['end'], type='word', confidence=1.0) for w in stitched_words]
    # For now, let's assume our dicts are fine. We should check the `Word` object structure from `elevenlabs` sdk.
    # Based on the original create_transcript_format, it expects objects with .text, .start, .end, .type attributes.
    # So we need to convert back.
    
    class PseudoWord: # Helper class to mimic ElevenLabs Word object structure
        def __init__(self, text, start, end, type='word'):
            self.text = text
            self.start = start
            self.end = end
            self.type = type
            # Add other attributes if create_transcript_format uses them (e.g., confidence)
            # For now, these are the ones checked in create_transcript_format.

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
        if not hasattr(word_info, 'type') or word_info.type != 'word': # More robust check
            # Ignore non-word items like spacing
            continue
        
        # Basic validation for word structure (add more if needed)
        if not all(hasattr(word_info, attr) for attr in ['text', 'start', 'end']):
            logger.warning(f"Skipping malformed word_info object: {word_info}")
            continue

        word_text = word_info.text
        word_start_time = word_info.start
        word_end_time = word_info.end
        word_char_count = len(word_text)

        # Determine if a split should happen before adding this word
        should_split = False
        if current_segment_words and previous_word_end_time is not None:
            # Calculate gap between words
            gap = word_start_time - previous_word_end_time
            
            # Split criteria
            if gap > pause_threshold:
                should_split = True
            elif ends_with_sentence_end(current_segment_words[-1]):
                should_split = True
            elif (current_char_count + 1 + word_char_count) > max_chars_per_segment:
                should_split = True

            # Prevent single-word segments unless there's a long pause
            if should_split and len(current_segment_words) == 1 and gap <= pause_threshold:
                should_split = False

        # Finalize previous segment if splitting
        if should_split:
            try:
                start_str = format_timestamp(segment_start_time)
                end_str = format_timestamp(previous_word_end_time)
                text = " ".join(current_segment_words)
                transcript_segments.append(f"{start_str},{end_str}\n{text}")
            except ValueError as e:
                logger.error(f"Error formatting timestamp during segment split: {e}")
                # Decide how to handle: skip segment? raise error?
                # For now, log and continue, potentially creating a gap

            # Reset for the new segment
            current_segment_words = []
            current_char_count = 0
            segment_start_time = None
            previous_word_end_time = None # Reset this too

        # Add the current word to the segment
        if not current_segment_words:
            segment_start_time = word_start_time

        current_segment_words.append(word_text)
        current_char_count += (word_char_count + (1 if len(current_segment_words) > 1 else 0))
        previous_word_end_time = word_end_time

    # Add the final remaining segment
    if current_segment_words and segment_start_time is not None and previous_word_end_time is not None:
        try:
            start_str = format_timestamp(segment_start_time)
            end_str = format_timestamp(previous_word_end_time)
            text = " ".join(current_segment_words)
            transcript_segments.append(f"{start_str},{end_str}\n{text}")
        except ValueError as e:
            logger.error(f"Error formatting timestamp for final segment: {e}")
            # Handle as above

    # Join all segments with double newlines for SBV format
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
    logger.info(f"[{task_id_for_log}] Processing audio file via chunking: {audio_path} (original: {original_name})")
    
    if not client:
        logger.error(f"[{task_id_for_log}] process_audio_to_transcript called with uninitialized ElevenLabs client.")
        raise ValueError("ElevenLabs client is not available")

    chunk_files_with_starts = []
    all_chunk_word_data = [] # Stores (word_list_from_chunk, chunk_start_time_s)

    try:
        # 1. Split audio into chunks
        logger.info(f"[{task_id_for_log}] Step 1: Splitting audio into chunks...")
        chunk_files_with_starts = _split_audio_into_chunks(task_id_for_log, audio_path, original_name)
        
        if not chunk_files_with_starts:
            logger.warning(f"[{task_id_for_log}] Audio splitting yielded no chunks for {original_name}. Aborting transcription.")
            raise ValueError("Audio splitting failed to produce any chunks.")

        # 2. Process each chunk
        logger.info(f"[{task_id_for_log}] Step 2: Processing {len(chunk_files_with_starts)} audio chunks through ElevenLabs...")
        total_chunks = len(chunk_files_with_starts)
        for i, (chunk_path, chunk_start_s) in enumerate(chunk_files_with_starts):
            current_chunk_number = i + 1
            logger.info(f"[{task_id_for_log}] Processing chunk {current_chunk_number}/{total_chunks}: {chunk_path} (starts at {chunk_start_s}s)")
            
            # Optional: Send per-chunk processing start update
            if progress_callback:
                progress_callback({
                    "status": "processing", 
                    "stage": f"transcribing_chunk_{current_chunk_number}_of_{total_chunks}", 
                    "message": f"Transcribing audio chunk {current_chunk_number} of {total_chunks}...",
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
                    logger.info(f"[{task_id_for_log}] Received {len(response.words)} words from chunk {current_chunk_number}.")
                    all_chunk_word_data.append((response.words, chunk_start_s))
                    # Optional: Send per-chunk completion update
                    if progress_callback:
                        progress_callback({
                            "status": "processing", 
                            "stage": f"chunk_{current_chunk_number}_of_{total_chunks}_complete", 
                            "message": f"Completed transcription for chunk {current_chunk_number} of {total_chunks}.",
                            "current_chunk": current_chunk_number,
                            "total_chunks": total_chunks
                        })
                else:
                    logger.warning(f"[{task_id_for_log}] Empty or invalid transcription response for chunk {chunk_path}. Response: {response}")
            except Exception as e:
                logger.error(f"[{task_id_for_log}] Error transcribing audio chunk {chunk_path}: {e}", exc_info=True)
            finally:
                clean_temp_file(chunk_path)

        if not all_chunk_word_data:
            logger.error(f"[{task_id_for_log}] No words returned from any audio chunks for {original_name}.")
            raise ValueError("Transcription service returned no words from any chunks.")

        # 3. Stitch transcriptions
        logger.info(f"[{task_id_for_log}] Step 3: Stitching transcriptions from chunks...")
        if progress_callback: # <<< Send STITCHING stage update
            progress_callback({"status": "processing", "stage": "stitching", "message": "Stitching transcriptions from chunks..."})
        
        stitched_word_objects = _stitch_transcriptions(task_id_for_log, all_chunk_word_data, OVERLAP_S)
        
        if not stitched_word_objects:
            logger.warning(f"[{task_id_for_log}] Stitching resulted in no words for {original_name}.")
            
        # 4. Format transcript
        logger.info(f"[{task_id_for_log}] Step 4: Formatting final stitched transcript...")
        transcript_text = create_transcript_format(stitched_word_objects) # Pass the stitched list of PseudoWord objects
        
        # 5. Save to file
        logger.info(f"[{task_id_for_log}] Step 5: Saving final transcript to file...")
        transcript_path = save_transcript_to_file(transcript_text, original_name)
        logger.info(f"[{task_id_for_log}] Successfully processed and chunked audio: {original_name}. Transcript at: {transcript_path}")
        return transcript_path
        
    except Exception as e:
        logger.error(f"[{task_id_for_log}] Error during chunked audio processing for {original_name} (path: {audio_path}): {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to process audio to transcript using chunking: {e}") from e
    finally:
        logger.debug(f"[{task_id_for_log}] Final cleanup of any remaining chunk files for {original_name}")
        for chunk_path, _ in chunk_files_with_starts:
             clean_temp_file(chunk_path)

def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video file and return audio file path"""
    logger.info(f"Starting audio extraction from video: {video_path}")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # Generate a unique name for the audio file to avoid collisions
    audio_filename = f"{base_name}_{uuid.uuid4().hex[:8]}.mp3"
    audio_path = os.path.join(Config.TEMP_DIR, audio_filename)
    
    try:
        video_clip = VideoFileClip(video_path)
        logger.info(f"Extracting audio to {audio_path} (codec=mp3, bitrate={Config.AUDIO_BITRATE})")
        video_clip.audio.write_audiofile(
            audio_path, 
            codec='mp3', 
            bitrate=Config.AUDIO_BITRATE, 
            logger=None # Suppress moviepy logs
        )
        video_clip.close() # Ensure the clip is closed
        logger.info(f"Audio extraction successful: {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Failed to extract audio from video {video_path}: {str(e)}", exc_info=True)
        # Attempt cleanup if the audio file was partially created
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Cleaned up partially created audio file: {audio_path}")
            except OSError as cleanup_e:
                logger.warning(f"Failed to clean up partial audio file {audio_path}: {cleanup_e}")
        raise ValueError(f"Failed to extract audio from video: {e}") from e 