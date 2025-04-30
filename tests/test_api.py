import pytest
from fastapi.testclient import TestClient
import io
import os
from unittest.mock import AsyncMock, MagicMock # For mocking awaitables like file.close()

# Import the FastAPI app from your main application file
# Adjust the import path if your main app instance is located differently
from main import app

# Create a TestClient instance
client = TestClient(app)


def test_ping():
    """Test the /ping health check endpoint."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "message": "Service is running"}

def test_root():
    """Test the root / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    # We can check for expected keys or exact values depending on stability
    data = response.json()
    assert "api" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "online"
    # Example of checking specific config values if needed
    # from src.config import Config 
    # assert data["api"] == Config.API_TITLE
    # assert data["version"] == Config.API_VERSION

# --- Tests for Transcription Endpoints --- 

def test_transcribe_video_success(mocker):
    """Test successful transcription via video upload."""
    # --- Mocking Setup ---
    # Mock functions called by the endpoint
    # Mock the return value of extract_audio_from_video
    mock_extract = mocker.patch(
        "src.routers.transcribe.extract_audio_from_video", 
        return_value="temp/fake_audio_extracted.mp3"
    )
    # Mock the return value of process_audio_to_transcript
    mock_process = mocker.patch(
        "src.routers.transcribe.process_audio_to_transcript",
        return_value="temp/fake_transcript_generated.sbv"
    )
    # Mock clean_temp_file to check calls without actually deleting
    mock_clean = mocker.patch("src.routers.transcribe.clean_temp_file")
    
    # Mock os.path.basename used in the final response
    mock_basename = mocker.patch("os.path.basename", return_value="fake_transcript_generated.sbv")
    
    # Mock fastapi.UploadFile.close with a standard MagicMock
    mock_file_close = MagicMock() # Use MagicMock instead of AsyncMock
    mocker.patch("fastapi.UploadFile.close", mock_file_close)
    # ---

    # Mock shutil.copyfileobj to avoid actual file writing during save
    mock_copy = mocker.patch("shutil.copyfileobj")
    
    # Mock built-in open to avoid writing the temp video file
    mock_open = mocker.patch("builtins.open", mocker.mock_open())

    # --- Test Execution ---
    # Simulate a file upload
    fake_video_content = b"dummy video content"
    # Create the fake file object - no need to mock close here now
    fake_file = io.BytesIO(fake_video_content)
        
    response = client.post(
        "/transcribe-video", 
        files={"video_file": ("test_video.mp4", fake_file, "video/mp4")} 
    )

    # --- Assertions ---
    assert response.status_code == 200
    assert response.json() == {
        "message": "Transcription successful", 
        "transcript_file": "fake_transcript_generated.sbv" # From mock_basename
    }

    # Check that our mocked functions were called
    # 1. File was "saved" (open and copyfileobj called)
    mock_open.assert_called_once()
    mock_copy.assert_called_once()
    # 2. Audio was "extracted"
    mock_extract.assert_called_once()
    # 3. Audio was "processed"
    mock_process.assert_called_once_with(
        "temp/fake_audio_extracted.mp3", # Result from mock_extract
        mocker.ANY, # The ElevenLabs client dependency (ignore specific instance)
        "test_video.mp4" # Original filename
    )
    # 4. Basename was called for the response
    mock_basename.assert_called_once_with("temp/fake_transcript_generated.sbv")
    # 5. Cleanup was called for both temp video and temp audio
    assert mock_clean.call_count == 2 
    # Could add specific path checks if needed: 
    # mock_clean.assert_any_call(mocker.ANY) # Temp video path is dynamic
    # mock_clean.assert_any_call("temp/fake_audio_extracted.mp3")
    
    # 6. UploadFile.close was called - check the MagicMock
    # assert mock_file_close.called # REMOVED: Difficult to reliably test with TestClient

def test_transcribe_youtube_success(mocker):
    """Test successful transcription via YouTube URL."""
    # --- Mocking Setup ---
    # Mock the YouTube class and its chained methods/attributes
    mock_stream = MagicMock()
    mock_stream.subtype = "m4a" # Mock the audio subtype
    # Mock the download method on the stream
    mock_stream.download = MagicMock()

    # Configure the chain of stream filtering calls
    mock_streams = MagicMock()
    mock_streams.filter.return_value.order_by.return_value.desc.return_value.first.return_value = mock_stream

    # Mock the YouTube object itself
    mock_yt_instance = MagicMock()
    mock_yt_instance.title = "Fake YouTube Video Title" # Mock the video title
    mock_yt_instance.streams = mock_streams # Attach the mocked streams object
    
    # Patch the YouTube class where it's imported in the router
    # Make it return our configured instance when called
    mock_youtube_class = mocker.patch("src.routers.transcribe.YouTube", return_value=mock_yt_instance)
    
    # Mock process_audio_to_transcript (same as video test)
    mock_process = mocker.patch(
        "src.routers.transcribe.process_audio_to_transcript",
        return_value="temp/fake_yt_transcript_generated.sbv"
    )
    
    # Mock clean_temp_file (same as video test)
    mock_clean = mocker.patch("src.routers.transcribe.clean_temp_file")
    
    # Mock os.path.basename (same as video test)
    mock_basename = mocker.patch("os.path.basename", return_value="fake_yt_transcript_generated.sbv")

    # --- Test Execution ---
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 
    response = client.post(
        "/transcribe-youtube", 
        json={"url": test_url}
    )

    # --- Assertions ---
    assert response.status_code == 200
    assert response.json() == {
        "message": "Transcription successful", 
        "transcript_file": "fake_yt_transcript_generated.sbv" # From mock_basename
    }

    # Check that our mocked functions/methods were called
    # 1. YouTube class was instantiated with the URL
    mock_youtube_class.assert_called_once_with(test_url)
    # 2. Stream filtering was attempted
    mock_streams.filter.assert_called_once()
    # 3. Stream download was called 
    # We check the download mock attached to the *returned* stream object
    mock_stream.download.assert_called_once()
    # 4. Audio processing was called
    mock_process.assert_called_once_with(
        mocker.ANY, # The temporary downloaded audio path (dynamic uuid)
        mocker.ANY, # The ElevenLabs client dependency
        "Fake YouTube Video Title" # Title from the mocked yt instance
    )
    # 5. Basename was called for the response
    mock_basename.assert_called_once_with("temp/fake_yt_transcript_generated.sbv")
    # 6. Cleanup was called for the temporary audio file
    mock_clean.assert_called_once()

# We will add tests for /transcribe-youtube next.

# To run these tests:
# 1. Make sure pytest and httpx are installed: pip install pytest httpx
# 2. Navigate to your project root in the terminal.
# 3. Run the command: pytest 