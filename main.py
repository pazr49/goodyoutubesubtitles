import os
import shutil
import uuid
import math
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from moviepy.video.io.VideoFileClip import VideoFileClip
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

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

# --- API Endpoints ---
@app.get("/ping")
async def ping():
    """Simple endpoint to check if the server is running."""
    return {"message": "pong"}

@app.get("/")
async def root():
    return {"message": "Hello from the Python API!"}

@app.post("/transcribe-video") # Renamed endpoint for clarity
async def transcribe_video(video_file: UploadFile = File(...)):
    """Accepts MP4 video, extracts audio, transcribes using ElevenLabs, and returns formatted transcript file path."""

    if not client:
         raise HTTPException(status_code=500, detail="ElevenLabs client not initialized. Check API key.")

    if not video_file.filename.endswith((".mp4", ".mov", ".avi", ".mkv")): # Added more video types
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file (e.g., MP4, MOV).")

    unique_id = uuid.uuid4()
    temp_video_path = os.path.join(TEMP_DIR, f"{unique_id}_{video_file.filename}")
    extracted_audio_filename = f"{unique_id}.mp3"
    extracted_audio_path = os.path.join(TEMP_DIR, extracted_audio_filename)

    transcript_text = ""
    transcript_file_path = ""

    try:
        # 1. Save Uploaded Video
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        # 2. Extract Audio
        video_clip = VideoFileClip(temp_video_path)
        # Use a high bitrate for potentially better quality for STT
        video_clip.audio.write_audiofile(extracted_audio_path, codec='mp3', bitrate="320k")
        video_clip.close()

        # 3. Call ElevenLabs Speech-to-Text
        with open(extracted_audio_path, 'rb') as audio_f:
            # According to docs, convert expects file, model_id
            # Default timestamps_granularity is 'word' which is what we need.
            response = client.speech_to_text.convert(
                file=audio_f,
                model_id='scribe_v1' # Corrected model ID for Speech-to-Text
            )

        # 4. Format Transcript
        # Access 'words' directly from the response object
        if hasattr(response, 'words') and response.words:
             transcript_text = create_transcript_format(response.words)

             # 5. Save transcript to .sbv file
             transcript_file_path = save_transcript_to_file(transcript_text, video_file.filename)
        else:
             # Handle case where transcription might be empty or failed partially
             transcript_text = "Transcription complete but no words found or result format unexpected."
             # Log the actual response for debugging if needed
             logger.warning(f"Unexpected ElevenLabs response format or empty words list for file {video_file.filename}: {response}")


    except Exception as e:
        # Log the full error for better debugging
        logger.error(f"Error during processing video {video_file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
    finally:
        # 6. Cleanup Temporary Files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(extracted_audio_path):
             os.remove(extracted_audio_path)
        await video_file.close()

    # 7. Return Formatted Transcript File Path
    # Return just the filename or a path relative to some defined output dir
    return {
        "message": "Transcription successful",
        "transcript_file": os.path.basename(transcript_file_path) if transcript_file_path else None
        # Removed the full transcript text from the response for brevity
    } 