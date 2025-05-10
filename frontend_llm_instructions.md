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
    *   Body: Form data containing a file attached to the key `video_file`.
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

### 4. Transcribe YouTube URL (Updated)

*   **Path:** `/transcribe-youtube`
*   **Method:** `POST`
*   **Purpose:** Provide a YouTube URL to **initiate** an SBV subtitle generation task. The actual transcription happens in the background.
*   **Request:**
    *   Content-Type: `application/json`
    *   Body:
        ```json
        {
          "url": "https://www.youtube.com/watch?v=your_video_id"
        }
        ```
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
    *   **Example Progress Event Data (parsed JSON):**
        ```json
        {
          "status": "processing", // "queued", "processing", "complete", "error", "cancelled"
          "stage": "download",    // "uploaded", "initialized", "download", "downloaded", "extract", "extracted", "transcribing", "chunk_X_of_Y", "stitching", "finished", "failed"
          "message": "Downloading audio from YouTube...", // User-friendly message
          // Optional fields depending on stage:
          // "current_chunk": 1,
          // "total_chunks": 5,
          "filename": null // Will contain the SBV filename when status is "complete"
        }
        ```
    *   **Example Completion Event Data (parsed JSON):**
        ```json
        {
          "status": "complete",
          "stage": "finished",
          "message": "Transcription complete.",
          "filename": "generated_filename.sbv"
        }
        ```
    *   **Example Error Event Data (parsed JSON):**
        ```json
        {
          "status": "error",
          "stage": "failed", // or a more specific stage where error occurred
          "message": "Error details from the server."
        }
        ```
    *   **Example Task Not Found Event Data (parsed JSON):** (This is sent once if the task_id is invalid, then the connection closes)
        ```json
        {
          "status": "not_found",
          "message": "Task ID not found."
        }
        ```
    *   The stream will close automatically once the task reaches a "complete", "error", or "cancelled" status, or if the client disconnects.
*   **Frontend Implementation:**
    *   Use the `EventSource` API in JavaScript to connect.
    *   Listen for `onmessage` events to receive progress updates.
    *   Parse `event.data` as JSON.
    *   Update the UI based on the `status`, `stage`, and `message`.
    *   Close the `EventSource` when it's no longer needed or when a terminal status is received.

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
    *   Send a `POST` request to either `/transcribe-video` (with file data) or `/transcribe-youtube` (with JSON body containing the URL).
    *   The API will respond immediately with a `200 OK` and a JSON body containing a `task_id`:
        ```json
        { "task_id": "your_unique_task_id_string" }
        ```
    *   Store this `task_id`.

2.  **Track Progress via Server-Sent Events (SSE):**
    *   Immediately after receiving the `task_id`, establish an SSE connection to `/progress/{task_id}` using the browser's `EventSource` API.
    *   Listen for `message` events on the `EventSource` object.
    *   Each message's `data` field will be a JSON string. Parse it to get a progress object (see details in "Track Transcription Progress" endpoint section).
    *   Update the UI with the information from the progress object (e.g., display `message`, show current `stage`, potentially a progress bar if `current_chunk` and `total_chunks` are available).

3.  **Handle Completion or Errors:**
    *   Continue listening to SSE events.
    *   If a progress event with `status: "complete"` is received:
        *   The event data will include the `filename` of the generated SBV file.
        *   The SSE connection will close shortly after this event.
        *   Proceed to step 4 to download the file.
    *   If a progress event with `status: "error"` (or "cancelled") is received:
        *   Display the error `message` to the user.
        *   The SSE connection will close. The process is finished.
    *   If the `EventSource` emits an `onerror` event, handle the connection failure (e.g., retry connection a few times or inform the user).

4.  **Download Transcript File:**
    *   Once a `status: "complete"` event is received with a valid `filename`:
        *   Construct the download URL: `BaseURL/download/{filename}`.
        *   Initiate a `GET` request to this URL. This can be done by setting `window.location.href` or by using `fetch` and creating a Blob to trigger a download.

## Important Notes (Updated)

*   **CORS:** Ensure your frontend domain is allowed by the backend's CORS policy.
*   **`EventSource` API:** Familiarize yourself with the `EventSource` API on MDN for implementing the SSE client.
*   **Loading Indicators:** Use the detailed progress messages from SSE to provide rich feedback to the user instead of a generic spinner.
*   **File Persistence:** Generated files are temporary. Download them promptly once the "complete" status is received.
*   **Task ID Uniqueness:** `task_id` values are UUIDs and are unique for each transcription request.

This updated guide should provide a clear path for the frontend developer to integrate with the new asynchronous backend. 