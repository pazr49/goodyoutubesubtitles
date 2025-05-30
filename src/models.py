from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

# --- Models ---
class TranscriptionResponse(BaseModel):
    message: str
    transcript_file: str
    translated_files: Optional[List[str]] = None

class YouTubeRequest(BaseModel):
    url: HttpUrl = Field(..., description="YouTube video URL to transcribe")
    target_languages: Optional[List[str]] = Field(default=None, description="List of target languages for translation (e.g., ['spanish', 'french'])")

class VideoUploadRequest(BaseModel):
    target_languages: Optional[List[str]] = Field(default=None, description="List of target languages for translation (e.g., ['spanish', 'french'])") 