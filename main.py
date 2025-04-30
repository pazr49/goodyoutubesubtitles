import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import config and routers
from src.config import Config, YouTube # Import necessary config items
from src.routers import transcribe # Import the router

# --- Logging Setup ---
# Configure logging (could also be moved to a dedicated logging setup function/module)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log") # Log file in the root directory
    ]
)
logger = logging.getLogger(__name__)

# --- Create FastAPI App ---
def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    app = FastAPI(
        title=Config.API_TITLE,
        version=Config.API_VERSION,
        description="An API for generating YouTube subtitles from video or audio content"
    )

    # Define allowed origins for development and production
    # TODO: Add your *actual* production frontend URL(s) here later
    origins = [
        # Development
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:5173",
        # Production API URL (allows frontend hosted elsewhere to call API)
        "https://past-oven-production.up.railway.app", 
    ]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"], # Allow all methods (GET, POST, etc.)
        allow_headers=["*"], # Allow all headers
    )

    # Include routers
    app.include_router(transcribe.router, prefix="", tags=["Transcription"]) # Add prefix if desired

    logger.info("FastAPI application created successfully.")
    return app

app = create_app() # Create the app instance

# --- Main Execution ---
if __name__ == "__main__":
    # Perform startup checks
    logger.info("Starting application...")
    
    # Check if ElevenLabs API key is configured (already logged in config.py)
    if not Config.ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY not found. Transcription features may fail.")
        
    # Check if pytubefix is available
    if not YouTube:
        logger.warning("pytubefix library not found. YouTube functionality will be disabled.")
        
    # Start the server
    logger.info(f"Running Uvicorn server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 