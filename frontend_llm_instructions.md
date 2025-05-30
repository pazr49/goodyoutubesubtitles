# Instructions for Frontend LLM Interacting with Subtitle API

This document outlines how to interact with the YouTube Subtitle Generator backend API.

## Base URL

The production API is hosted at: `https://past-oven-production.up.railway.app`
(For local development, use `http://localhost:8000` or your configured local address)

## Authentication

Currently, there is **no authentication** required to access these endpoints.

## Endpoints

### 1. Root Info

*   **Path:** `/`
*   **Method:** `GET`
*   **Purpose:** Get basic information about the API.
*   **Success Response (200 OK):**
    ```json
    {
      "api": "YouTube Subtitles Generator",
      "version": "1.0.0", // Version might change
      "status": "online"
    }
    ```

### 2. Health Check

*   **Path:** `/ping`
*   **Method:** `GET`
*   **Purpose:** Check if the API service is running and healthy.
*   **Success Response (200 OK):**
    ```json
    {
      "status": "healthy",
      "message": "Service is running"
    }
    ```

### 3. Transcribe Video File (Updated)

*   **Path:** `/transcribe-video`
*   **Method:** `POST`
*   **Purpose:** Upload a video file to **initiate** an SBV subtitle generation task. The actual transcription happens in the background.
*   **Request:**
    *   Content-Type: `multipart/form-data`
    *   Body: Form data containing:
        *   `video_file`: File attached to this key (required)
        *   `target_languages`: Comma-separated list of target languages for translation (optional, e.g., "spanish,french,german")
*   **Success Response (200 OK - Task Queued):**
    ```json
    {
      "task_id": "your_unique_task_id_string"
    }
    ```
    *   **Important:** This response indicates the task has been **queued**. You must use the `task_id` to track progress via the `/progress/{task_id}` endpoint (see below).
*   **Error Responses (for initial request):**
    *   `400 Bad Request`: Invalid file type. Response body: `{"detail": "Invalid file type '...' Allowed: ..."}`
    *   `422 Unprocessable Entity`: If the request is malformed.
    *   `503 Service Unavailable`: If translation is requested but Gemini service is not available.

### 4. Transcribe YouTube URL (Updated)

*   **Path:** `/transcribe-youtube`
*   **Method:** `POST`
*   **Purpose:** Provide a YouTube URL to **initiate** an SBV subtitle generation task. The actual transcription happens in the background.
*   **Request:**
    *   Content-Type: `application/json`
    *   Body:
        ```json
        {
          "url": "https://www.youtube.com/watch?v=your_video_id",
          "target_languages": ["spanish", "french", "german"]
        }
        ```
        *   `url`: YouTube video URL (required)
        *   `target_languages`: Array of target languages for translation (optional)
*   **Success Response (200 OK - Task Queued):**
    ```json
    {
      "task_id": "your_unique_task_id_string"
    }
    ```
    *   **Important:** This response indicates the task has been **queued**. You must use the `task_id` to track progress via the `/progress/{task_id}` endpoint (see below).
*   **Error Responses (for initial request):**
    *   `422 Unprocessable Entity`: Invalid YouTube URL format or missing URL.
    *   `501 Not Implemented`: If `pytubefix` library is missing on the server (should be rare).
    *   `503 Service Unavailable`: If translation is requested but Gemini service is not available.

### 5. Track Transcription Progress (New)

*   **Path:** `/progress/{task_id}`
*   **Method:** `GET`
*   **Purpose:** Stream real-time progress updates for a given transcription task using Server-Sent Events (SSE).
*   **Request:**
    *   Replace `{task_id}` in the path with the actual `task_id` obtained from a successful `/transcribe-video` or `/transcribe-youtube` response.
    *   Example: `GET /progress/your_unique_task_id_string`
    *   The client should establish an SSE connection (`EventSource` in JavaScript).
*   **Event Stream:**
    *   The server will send a stream of JSON objects. Each event will have a `data` field containing a JSON string.
    *   **Progress Event Structure:**
        *   Each SSE message data contains JSON with these potential fields:
            *   `status`: "queued", "processing", "complete", "error", or "cancelled"
            *   `stage`: A descriptive string (e.g., "uploaded", "download", "transcribing_chunk_1_of_4", "stitching", "finished")
            *   `message`: Human-readable status message.
            *   `current_chunk` and `total_chunks`: Available during chunk processing (optional).
            *   `filename`: The name of the generated SBV file (only present on completion with `status: "complete"`).
            *   `translated_files`: Array of translated SBV filenames (only present if translations were requested and completed).
            *   `current_batch` and `total_batches`: Available during translation processing (optional).
    *   **Frontend Implementation:**
        *   Use the `EventSource` API in JavaScript to connect.
        *   Listen for `onmessage` events to receive progress updates.
        *   Parse `event.data` as JSON.
        *   Update the UI based on the `status`, `stage`, and `message`.
        *   Close the `EventSource` when it's no longer needed or when a terminal status is received.
        *   Handle both original transcript and translated files when they're available.

### 6. Download Transcript File (Unchanged, but used after "complete" status)

*   **Path:** `/download/{filename}`
*   **Method:** `GET`
*   **Purpose:** Download a previously generated SBV subtitle file.
*   **Request:**
    *   Replace `{filename}` in the path with the actual `filename` value obtained from a "complete" progress event from the `/progress/{task_id}` endpoint.
    *   Example: `GET /download/MyVideoTitle_abc123ef.sbv`
*   **Success Response (200 OK):**
    *   The response **body is the raw content of the SBV file**.
    *   The `Content-Type` header will likely be `text/plain`.
    *   The `Content-Disposition` header will be set, suggesting the original filename to the browser for download.
*   **Error Responses:** (Same as before)
    *   `400 Bad Request`: Invalid filename format.
    *   `403 Forbidden`: Access forbidden.
    *   `404 Not Found`: File not found.
    *   `500 Internal Server Error`: Server error during retrieval.

## New Transcription Workflow (Updated)

To get a subtitle file, the frontend application must now follow these steps:

1.  **Initiate Transcription:**
    *   Send a `POST` request to either `/transcribe-video`