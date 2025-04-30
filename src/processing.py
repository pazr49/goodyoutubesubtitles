import os
import logging
import uuid
from typing import List, Any

from moviepy.video.io.VideoFileClip import VideoFileClip
from elevenlabs.client import ElevenLabs

from .config import Config # Relative imports
from .utils import format_timestamp, ends_with_sentence_end, save_transcript_to_file

logger = logging.getLogger(__name__)

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
    audio_path: str, 
    client: ElevenLabs, # Type hint added
    original_name: str
) -> str:
    """Process an audio file to generate transcript"""
    logger.info(f"Processing audio file: {audio_path} (original: {original_name})")
    
    if not client:
        logger.error("process_audio_to_transcript called with uninitialized ElevenLabs client.")
        raise ValueError("ElevenLabs client is not available")

    try:
        # Send audio to ElevenLabs for transcription
        logger.info(f"Sending audio to ElevenLabs model '{Config.ELEVENLABS_MODEL_ID}'")
        with open(audio_path, 'rb') as audio_file:
            response = client.speech_to_text.convert(
                file=audio_file,
                model_id=Config.ELEVENLABS_MODEL_ID
            )
        logger.info(f"Received transcription response from ElevenLabs for {original_name}.")
        
        # Check for valid response with words
        if not hasattr(response, 'words') or not response.words:
            logger.warning(f"Empty or invalid transcription response for {original_name}. Response: {response}")
            # Consider if this should be an error or just an empty transcript
            raise ValueError("Transcription service returned no words or invalid format")
            
        # Format transcript
        logger.info("Formatting transcript...")
        transcript_text = create_transcript_format(response.words)
        
        # Save to file
        logger.info("Saving transcript to file...")
        transcript_path = save_transcript_to_file(transcript_text, original_name)
        return transcript_path
        
    except Exception as e:
        logger.error(f"Error during audio processing for {original_name} (path: {audio_path}): {str(e)}", exc_info=True)
        # Re-raise a more specific error or a generic one
        raise RuntimeError(f"Failed to process audio to transcript: {e}") from e

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