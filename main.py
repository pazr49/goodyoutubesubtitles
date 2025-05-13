import logging
import uvicorn
import asyncio # Add asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import config and routers
from src.config import Config, YouTube # Import necessary config items
from src.routers import transcribe # Import the router
from src.utils import cleanup_old_sbv_files # Import the cleanup function

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
        # Production frontend URL
        "https://ezsubtitles.vercel.app",
        # New production frontend URLs
        "https://youtubesubtitlegenerator.com",
        "https://www.youtubesubtitlegenerator.com",
        # Production API URL (allows API docs served from FastAPI to work, or calls from same domain)
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

    # Start background cleanup task
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(60) # Run every 60 seconds
            try:
                cleanup_old_sbv_files(max_age_seconds=300) # 5 minutes
            except Exception as e:
                logger.error(f"Error in periodic cleanup task: {e}", exc_info=True)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting background task for periodic cleanup of .sbv files.")
        asyncio.create_task(periodic_cleanup())
        # Perform other startup checks if any (moved from if __name__ == "__main__")
        logger.info("Performing application startup checks...")
        if not Config.ELEVENLABS_API_KEY:
            logger.warning("ELEVENLABS_API_KEY not found. Transcription features may fail.")
        if not YouTube:
            logger.warning("pytubefix library not found. YouTube functionality will be disabled.")
        logger.info("Application startup checks complete.")

    logger.info("FastAPI application created successfully.")
    return app

app = create_app() # Create the app instance

# --- Main Execution ---
if __name__ == "__main__":
    # Start the server
    logger.info(f"Running Uvicorn server on http://0.0.0.0:8000") # Default port for local
    # Note: Railway will use the port specified in its settings or Dockerfile (e.g., 8080)
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT if hasattr(Config, 'PORT') else 8000) 