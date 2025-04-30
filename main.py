import os
import shutil
import uuid
import math
import logging
# import subprocess # Remove subprocess import
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from moviepy.video.io.VideoFileClip import VideoFileClip
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
# Use pytubefix imports
try:
    from pytubefix import YouTube
    from pytubefix.exceptions import PytubeFixError # Use pytubefix exception
    # No need for request object from pytubefix for basic usage
except ImportError:
    YouTube = None
    PytubeFixError = None

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# --- ElevenLabs Client Setup ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY not found in environment variables.")
    # Optionally, raise an error or handle it gracefully
    # raise ValueError("Missing ELEVENLABS_API_KEY environment variable")

# Initialize client only if API key exists
client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# --- Temporary Directory Setup ---
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Helper Functions ---
def format_timestamp(seconds: float) -> str:
    """Converts seconds (float) to HH:MM:SS.ms string format."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds_part = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"

# Helper Function to check for sentence ending punctuation
def ends_with_sentence_end(text: str) -> bool:
    return text.strip().endswith(('.', '?', '!'))

def create_transcript_format(words_data: list,
                             pause_threshold: float = 1.0, # Increased pause threshold
                             max_chars_per_segment: int = 85) -> str: # Approx 2 lines on YT
    """Processes ElevenLabs word data into YouTube subtitle format (.sbv)."""
    if not words_data:
        return ""

    transcript_segments = []
    current_segment_words = []
    segment_start_time = None
    previous_word_end_time = None
    current_char_count = 0

    for i, word_info in enumerate(words_data):
        if word_info.type != 'word':
            # Ignore non-word items like spacing for splitting logic
            # We will add spaces during the final join
            continue

        word_text = word_info.text
        word_start_time = word_info.start
        word_end_time = word_info.end
        word_char_count = len(word_text)

        # --- Determine if a split should happen *before* adding this word ---
        should_split = False
        if current_segment_words: # Can only split if there's an existing segment
            # 1. Check for long pause
            gap = word_start_time - previous_word_end_time
            if gap > pause_threshold:
                should_split = True

            # 2. Check for sentence end on the *previous* word
            elif ends_with_sentence_end(current_segment_words[-1]):
                 should_split = True

            # 3. Check character limit
            # Add 1 for the space before the new word
            elif (current_char_count + 1 + word_char_count) > max_chars_per_segment:
                should_split = True

            # 4. Prevent single-word segments unless absolutely necessary (long pause)
            if should_split and len(current_segment_words) == 1 and gap <= pause_threshold:
                 # Avoid splitting if it creates a single-word line, unless forced by pause
                 should_split = False


        # --- Finalize previous segment if splitting ---
        if should_split:
            start_str = format_timestamp(segment_start_time)
            # Use the end time of the last word in the segment
            end_str = format_timestamp(previous_word_end_time)
            text = " ".join(current_segment_words) # Use single space
            transcript_segments.append(f"{start_str},{end_str}\n{text}")

            # Reset for the new segment
            current_segment_words = []
            current_char_count = 0
            segment_start_time = None # Will be set when the first word is added


        # --- Add the current word to the (new or existing) segment ---
        if not current_segment_words:
            segment_start_time = word_start_time # Start time of the first word

        current_segment_words.append(word_text)
        # Add 1 for space if not the first word
        current_char_count += (word_char_count + (1 if len(current_segment_words) > 1 else 0))
        previous_word_end_time = word_end_time # Update the end time marker


    # --- Add the final remaining segment ---
    if current_segment_words:
        start_str = format_timestamp(segment_start_time)
        end_str = format_timestamp(previous_word_end_time)
        text = " ".join(current_segment_words)
        transcript_segments.append(f"{start_str},{end_str}\n{text}")

    # Join all segments with double newlines for SBV format
    return "\n\n".join(transcript_segments)

def save_transcript_to_file(transcript_text: str, original_filename: str) -> str:
    """Saves transcript text to an .sbv file and returns the file path."""
    # Create a filename based on the original video filename but with .sbv extension
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    transcript_filename = f"{base_filename}_{uuid.uuid4().hex[:8]}.sbv"
    transcript_path = os.path.join(TEMP_DIR, transcript_filename)

    # Write the transcript to the file
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    return transcript_path

def _process_video_to_sbv(video_path: str, original_filename: str) -> str:
    """Internal helper to process a local video file to an SBV transcript file."""
    logger.info(f"[_process_video_to_sbv] Processing video file: {video_path} from original: {original_filename}") # Added identifier
    if not client:
        logger.error("[_process_video_to_sbv] ElevenLabs client is not initialized.")
        raise HTTPException(status_code=500, detail="ElevenLabs client not initialized. Check API key.")

    # Use a unique ID based on the video path for derived files
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    extracted_audio_filename = f"{base_name}.mp3"
    extracted_audio_path = os.path.join(TEMP_DIR, extracted_audio_filename)

    transcript_text = ""
    transcript_file_path = ""

    try:
        # 1. Extract Audio (moved from endpoint)
        logger.info(f"[_process_video_to_sbv] Starting audio extraction from {video_path} to {extracted_audio_path}")
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(extracted_audio_path, codec='mp3', bitrate="320k", logger=None) # Suppress moviepy logs if desired
        video_clip.close()
        logger.info("[_process_video_to_sbv] Audio extraction complete.")

        # 2. Call ElevenLabs Speech-to-Text (moved from endpoint)
        logger.info(f"[_process_video_to_sbv] Sending audio {extracted_audio_path} to ElevenLabs...")
        with open(extracted_audio_path, 'rb') as audio_f:
            response = client.speech_to_text.convert(
                file=audio_f,
                model_id='scribe_v1'
            )
        logger.info("[_process_video_to_sbv] ElevenLabs transcription received.")

        # 3. Format Transcript (moved from endpoint)
        if hasattr(response, 'words') and response.words:
            logger.info("[_process_video_to_sbv] Formatting transcript...")
            transcript_text = create_transcript_format(response.words)
            # 4. Save transcript to .sbv file (moved from endpoint)
            transcript_file_path = save_transcript_to_file(transcript_text, original_filename)
            logger.info(f"[_process_video_to_sbv] Transcript saved to {transcript_file_path}")
        else:
            transcript_text = "Transcription complete but no words found or result format unexpected."
            logger.warning(f"[_process_video_to_sbv] Unexpected ElevenLabs response or empty words for {original_filename}: {response}")

    except Exception as e:
        logger.error(f"[_process_video_to_sbv] Error during processing video {original_filename} (path: {video_path}): {e}", exc_info=True)
        # Re-raise the exception to be caught by the endpoint handler
        raise e
    finally:
        # 5. Cleanup Temporary Audio File (video cleanup handled by caller)
        if os.path.exists(extracted_audio_path):
            logger.info(f"[_process_video_to_sbv] Cleaning up temporary audio file: {extracted_audio_path}")
            os.remove(extracted_audio_path)
        else:
            logger.warning(f"[_process_video_to_sbv] Temporary audio file not found for cleanup: {extracted_audio_path}")

    logger.info(f"[_process_video_to_sbv] Finished processing. Returning SBV path: {transcript_file_path}")
    return transcript_file_path # Return the path to the generated SBV file

# --- NEW AUDIO PROCESSING HELPER ---
def _process_audio_to_sbv(audio_path: str, original_filename_base: str) -> str:
    """Internal helper to process a local audio file to an SBV transcript file."""
    logger.info(f"[_process_audio_to_sbv] Processing audio file: {audio_path} from original base: {original_filename_base}")
    if not client:
        logger.error("[_process_audio_to_sbv] ElevenLabs client is not initialized.")
        raise HTTPException(status_code=500, detail="ElevenLabs client not initialized. Check API key.")

    transcript_text = ""
    transcript_file_path = ""

    try:
        # 1. Call ElevenLabs Speech-to-Text (directly with audio path)
        logger.info(f"[_process_audio_to_sbv] Sending audio {audio_path} to ElevenLabs...")
        with open(audio_path, 'rb') as audio_f:
            response = client.speech_to_text.convert(
                file=audio_f,
                model_id='scribe_v1'
            )
        logger.info("[_process_audio_to_sbv] ElevenLabs transcription received.")

        # 2. Format Transcript
        if hasattr(response, 'words') and response.words:
            logger.info("[_process_audio_to_sbv] Formatting transcript...")
            transcript_text = create_transcript_format(response.words)
            # 3. Save transcript to .sbv file
            # Use the original base name provided (e.g., video title)
            transcript_file_path = save_transcript_to_file(transcript_text, original_filename_base)
            logger.info(f"[_process_audio_to_sbv] Transcript saved to {transcript_file_path}")
        else:
            transcript_text = "Transcription complete but no words found or result format unexpected."
            logger.warning(f"[_process_audio_to_sbv] Unexpected ElevenLabs response or empty words for {original_filename_base}: {response}")

    except Exception as e:
        logger.error(f"[_process_audio_to_sbv] Error during processing audio file {audio_path}: {e}", exc_info=True)
        # Re-raise the exception to be caught by the endpoint handler
        raise e
    # No finally block needed here, cleanup is handled by the caller endpoint
    # because this function receives the temp file path directly.

    logger.info(f"[_process_audio_to_sbv] Finished processing. Returning SBV path: {transcript_file_path}")
    return transcript_file_path

# --- API Endpoints ---
@app.get("/ping")
async def ping():
    """Simple endpoint to check if the server is running."""
    return {"message": "pong"}

@app.get("/")
async def root():
    return {"message": "Hello from the Python API!"}

@app.post("/transcribe-video")
async def transcribe_video(video_file: UploadFile = File(...)):
    """Accepts video upload, extracts audio, transcribes, returns formatted transcript file path."""

    # Basic validation remains here
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"} # Added webm
    file_extension = os.path.splitext(video_file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file_extension}'. Allowed: {', '.join(allowed_extensions)}")

    unique_id = uuid.uuid4()
    # Use a more robust temp filename based on UUID, keep original extension
    temp_video_filename = f"{unique_id}{file_extension}"
    temp_video_path = os.path.join(TEMP_DIR, temp_video_filename)

    transcript_file_path = None
    try:
        # 1. Save Uploaded Video
        logger.info(f"Saving uploaded video {video_file.filename} to {temp_video_path}")
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        logger.info("Video saved.")

        # 2. Process video using the helper function
        transcript_file_path = _process_video_to_sbv(temp_video_path, video_file.filename)

    except Exception as e:
        # Catch potential errors from saving or processing
        logger.error(f"Error in /transcribe-video endpoint for {video_file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
    finally:
        # 3. Cleanup Temporary Video File
        if os.path.exists(temp_video_path):
            logger.info(f"Cleaning up temporary video file: {temp_video_path}")
            os.remove(temp_video_path)
        await video_file.close()

    # 4. Return result
    if transcript_file_path:
        return {
            "message": "Transcription successful",
            "transcript_file": os.path.basename(transcript_file_path)
        }
    else:
        # This case might occur if _process_video_to_sbv returns None or empty path
        raise HTTPException(status_code=500, detail="Transcription process completed but failed to generate file path.")

@app.post("/transcribe-youtube")
async def transcribe_youtube(url: str = Body(..., embed=True)):
    """Accepts YouTube URL, downloads audio using pytubefix, transcribes, returns transcript file path."""
    logger.info(f"[/transcribe-youtube] Received request for URL: {url}")
    # Check pytubefix dependency
    if not YouTube or not PytubeFixError:
        logger.error("[/transcribe-youtube] Pytubefix dependency not available.")
        raise HTTPException(status_code=501, detail="YouTube processing dependency (pytubefix) not installed or available.")

    logger.info(f"[/transcribe-youtube] Processing URL: {url} using pytubefix (audio download)")

    temp_audio_path = None
    transcript_file_path = None
    original_filename_base = "youtube_video" # Placeholder for base name (e.g., title)

    try:
        # --- Use pytubefix library to download AUDIO ---
        logger.info(f"[/transcribe-youtube] Initializing YouTube object from pytubefix for {url}")
        yt = YouTube(url)
        logger.info(f"[/transcribe-youtube] YouTube object initialized. Video Title: '{yt.title}'")

        # Get the best audio-only stream
        logger.info("[/transcribe-youtube] Filtering for audio streams...")
        # stream = yt.streams.get_audio_only() # Simple way, often picks webm/opus
        # More specific filter for common formats like m4a (often better compatibility)
        stream = yt.streams.filter(only_audio=True, file_extension='m4a').order_by('abr').desc().first()
        if not stream:
            logger.warning("[/transcribe-youtube] No M4A audio stream found, trying get_audio_only()...")
            stream = yt.streams.get_audio_only()

        if not stream:
            logger.warning(f"[/transcribe-youtube] No suitable audio stream found for URL: {url}")
            raise HTTPException(status_code=404, detail="No suitable audio stream found for this YouTube video.")
        else:
            logger.info(f"[/transcribe-youtube] Selected audio stream: {stream}")

        # Use video title as the base for the SBV filename
        original_filename_base = yt.title
        # Download audio to temp dir with a unique name
        unique_id = uuid.uuid4()
        # Get extension from the stream (e.g., .m4a, .webm)
        audio_extension = f".{stream.subtype}"
        # Ensure stream.default_filename is safe (less critical with library, but keep)
        # Use a simpler temp name since we base SBV on title
        temp_audio_filename = f"{unique_id}_youtube_audio{audio_extension}"
        temp_audio_path = os.path.join(TEMP_DIR, temp_audio_filename)

        logger.info(f"[/transcribe-youtube] Attempting to download audio '{yt.title}' to {temp_audio_path}")
        stream.download(output_path=TEMP_DIR, filename=temp_audio_filename)
        logger.info(f"[/transcribe-youtube] YouTube audio download complete: {temp_audio_path}")
        # --- End pytubefix audio download modification ---

        # Process the downloaded audio using the new helper function
        logger.info(f"[/transcribe-youtube] Calling _process_audio_to_sbv for {temp_audio_path}")
        transcript_file_path = _process_audio_to_sbv(temp_audio_path, original_filename_base)
        logger.info(f"[/transcribe-youtube] _process_audio_to_sbv finished for {temp_audio_path}")

    # Catch pytubefix specific exceptions if needed, otherwise general Exception
    except PytubeFixError as e:
        logger.error(f"[/transcribe-youtube] PytubeFixError occurred for URL {url}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing YouTube URL (PytubeFixError): {str(e)}")
    except Exception as e:
        logger.error(f"[/transcribe-youtube] Generic exception occurred for URL {url}: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e # Re-raise it directly
        else:
            raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
    finally:
        # Cleanup downloaded YouTube AUDIO file
        if temp_audio_path and os.path.exists(temp_audio_path):
            logger.info(f"[/transcribe-youtube] Cleaning up temporary YouTube audio file: {temp_audio_path}")
            os.remove(temp_audio_path)
        elif temp_audio_path:
            logger.warning(f"[/transcribe-youtube] Temporary YouTube audio file {temp_audio_path} not found for cleanup (might have failed before creation).")

    # Return result
    if transcript_file_path:
        logger.info(f"[/transcribe-youtube] Request successful. Returning SBV file: {os.path.basename(transcript_file_path)}")
        return {
            "message": "Transcription successful",
            "transcript_file": os.path.basename(transcript_file_path)
        }
    else:
        logger.error("[/transcribe-youtube] Processing finished but no transcript file path was generated.")
        raise HTTPException(status_code=500, detail="Transcription process completed but failed to generate file path.") 