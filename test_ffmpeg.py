import os
import subprocess
import uuid
import logging
import traceback  # Added for better error reporting

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test settings
TEMP_DIR = "temp_files"
# Path to ffmpeg executable
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"  # Confirmed path from the user
PREFERRED_AUDIO_FORMAT = "mp3"
SAMPLE_VIDEO = r"C:\Users\danci\Downloads\Untitled video - Made with Clipchamp.mp4"

def test_ffmpeg_extraction():
    """Test FFmpeg audio extraction from a sample video file"""
    
    sample_video = SAMPLE_VIDEO
    
    if not os.path.exists(sample_video):
        print(f"Sample video not found at: {sample_video}")
        return False
    
    print(f"Using sample video: {sample_video}")
    
    # Verify FFmpeg exists at the specified path
    if not os.path.exists(FFMPEG_PATH):
        print(f"FFmpeg executable not found at: {FFMPEG_PATH}")
        print("Please update FFMPEG_PATH in the script to point to your ffmpeg.exe location.")
        return False
    
    # Output path
    output_audio_filename = f"{uuid.uuid4()}_test_extracted_audio.{PREFERRED_AUDIO_FORMAT}"
    output_audio_path = os.path.join(TEMP_DIR, output_audio_filename)
    
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # FFmpeg command
    command = [
        FFMPEG_PATH,
        '-i', sample_video,
        '-vn',  # No video output
        '-c:a', 'aac' if PREFERRED_AUDIO_FORMAT == "aac" else 'libmp3lame',
        '-ar', '44100',
        '-b:a', '192k',
        '-y',
        output_audio_path
    ]
    
    print(f"Executing FFmpeg command: {' '.join(command)}")
    
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        
        if process.stdout:
            print(f"FFmpeg stdout: {process.stdout}")
        
        if process.stderr:
            print(f"FFmpeg stderr: {process.stderr}")
            
        print(f"Audio extracted successfully to: {output_audio_path}")
        print(f"File exists: {os.path.exists(output_audio_path)}")
        print(f"File size: {os.path.getsize(output_audio_path)} bytes")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed with error code {e.returncode}")
        print(f"FFmpeg stderr: {e.stderr}")
        print(f"FFmpeg stdout: {e.stdout}")
        return False
    except FileNotFoundError:
        print(f"ffmpeg command not found at '{FFMPEG_PATH}'. Ensure ffmpeg is installed and in PATH.")
        return False
    except Exception as e:  # Added general exception handler
        print(f"Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing FFmpeg audio extraction...")
    success = test_ffmpeg_extraction()
    print(f"Test {'successful' if success else 'failed'}") 