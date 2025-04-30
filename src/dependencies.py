import logging
from fastapi import HTTPException

# Import the initialized client from config
from .config import elevenlabs_client

logger = logging.getLogger(__name__)

async def get_elevenlabs_client():
    """Dependency to inject the ElevenLabs client.
    
    Raises:
        HTTPException: 503 Service Unavailable if the client is not configured.
    """
    if not elevenlabs_client:
        logger.error("Dependency check failed: ElevenLabs client is not initialized.")
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Speech-to-text service not available. API key may be missing or invalid."
        )
    return elevenlabs_client 