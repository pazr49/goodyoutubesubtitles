import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# In-memory task progress store
# Keyed by task_id, each value is a dict with status, stage, message, and optional filename or error
task_progress_store: Dict[str, Dict[str, Any]] = {}

# Queue store for SSE subscribers, keyed by task_id
event_queue_store: Dict[str, asyncio.Queue] = {}

# Helper to update progress_store and notify SSE clients
def update_task_progress(task_id: str, progress_update: Dict[str, Any]):
    """Updates the task progress and notifies SSE subscribers."""
    logger.info(f"[{task_id}] UPDATING PROGRESS: {progress_update.get('status', 'N/A')} - {progress_update.get('stage', 'N/A')}")
    task_progress_store[task_id] = progress_update
    queue = event_queue_store.get(task_id)
    if queue:
        queue.put_nowait(progress_update.copy()) # Use .copy() to avoid issues if dict is modified later
        logger.info(f"[{task_id}] QUEUED UPDATE: {progress_update.get('status', 'N/A')} - {progress_update.get('stage', 'N/A')}")
    else:
        logger.warning(f"[{task_id}] NO QUEUE FOUND when trying to update progress: {progress_update.get('status', 'N/A')} - {progress_update.get('stage', 'N/A')}")

# Function to get a task's progress queue (creates if not exists for SSE endpoint)
def get_or_create_task_event_queue(task_id: str) -> asyncio.Queue:
    """Gets or creates an asyncio.Queue for a given task_id."""
    return event_queue_store.setdefault(task_id, asyncio.Queue())

# Function to get current task progress state
def get_task_current_progress(task_id: str) -> Optional[Dict[str, Any]]:
    """Returns the current progress for a task_id if it exists."""
    return task_progress_store.get(task_id)

# Function to remove a task's event queue (e.g., after SSE stream ends)
def remove_task_event_queue(task_id: str):
    """Removes the event queue for a given task_id."""
    event_queue_store.pop(task_id, None)
    logger.info(f"[{task_id}] Event queue removed.") 