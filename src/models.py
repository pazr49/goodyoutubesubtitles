from pydantic import BaseModel, Field, HttpUrl

# --- Models ---
class TranscriptionResponse(BaseModel):
    message: str
    transcript_file: str

class YouTubeRequest(BaseModel):
    url: HttpUrl = Field(..., description="YouTube video URL to transcribe") 