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
        *   `target_languages`: A single comma-separated string of target languages for translation (optional). Example: `"spanish,french,german"`
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
        *   `target_languages`: JSON array of target language strings (optional)
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
            *   `stage`: A descriptive string indicating the current step in the lifecycle.
            *   `message`: Human-readable status message.
            *   Other fields like `current_chunk`, `total_chunks`, etc., may appear depending on the stage.
    *   **Complete Lifecycle of Stages:**
        *   The frontend should be prepared to handle the following `stage` values in this approximate order. Note that some stages are conditional (e.g., download vs. extract).
        1.  `initialized`: (For YouTube links) The task is created and ready to start.
        2.  `uploaded`: (For file uploads) The video file has been received by the server.
        3.  `download`: (For YouTube links) The audio is being downloaded from the URL.
        4.  `downloaded`: (For YouTube links) The audio has been successfully downloaded.
        5.  `extract`: (For file uploads) The server is extracting audio from the video file.
        6.  `extracted`: (For file uploads) Audio extraction is complete.
        7.  `transcribing_queued`: The audio file is in the queue for transcription.
        8.  `transcribing_chunk_X_of_Y`: The transcription service is actively processing an audio chunk.
        9.  `chunk_X_of_Y_complete`: A specific audio chunk has been transcribed.
        10. `stitching`: All transcribed chunks are being combined into a single transcript.
        11. `translating_to_{language}`: The transcript is being translated into a specific language (e.g., `translating_to_spanish`). This stage will appear once for each requested language.
        12. `finished`: All processing is complete, and the files are ready. This is the final stage before the `complete` status.
        13. `failed`: This stage is sent if an unrecoverable error occurs.
    *   **Completion Event (status: "complete"):**
        *   When the task is finished, a `complete` status event is sent with `stage: "finished"` and crucial file information.
        *   `filename`: A **base filename** used for downloading.
        *   `processed_file`: The full filename of the final `.sbv` transcript.
        *   `raw_file`: The full filename of the raw word-level `.json` transcript.
        *   `translated_files`: An array of full filenames for each translated `.sbv` file.
        *   **Example Completion Event Payload:**
            ```json
            {
              "status": "complete",
              "stage": "finished",
              "message": "Transcription for 'My Video.mp4' complete. 2 translations created.",
              "filename": "My_Video",
              "processed_file": "My_Video_a1b2c3d4.sbv",
              "raw_file": "My_Video_a1b2c3d4_raw.json",
              "translated_files": [
                "My_Video_a1b2c3d4_spanish.sbv",
                "My_Video_a1b2c3d4_french.sbv"
              ]
            }
            ```
    *   **Frontend Implementation:**
        *   Use the `EventSource` API in JavaScript to connect.
        *   Listen for `onmessage` events to receive progress updates.
        *   Parse `event.data` as JSON.
        *   Update the UI based on the `status`, `stage`, and `message`.
        *   When the `complete` event is received, store the `filename`, `processed_file`, `raw_file`, and `translated_files` to enable downloads.
        *   Close the `EventSource` when a terminal status ("complete", "error") is received.

### 6. Downloading Transcript Files (IMPORTANT UPDATE)

*   **Path:** `/download/{base_filename}`
*   **Method:** `GET`
*   **Purpose:** Download the various transcript files generated by a task.
*   **Request:**
    *   Replace `{base_filename}` in the path with the `filename` value obtained from the "complete" progress event.
    *   Use the `file_type` query parameter to specify which file format you want.
*   **File Types:**
    *   `file_type=processed` (Default): Downloads the final `.sbv` subtitle file.
    *   `file_type=raw`: Downloads the raw, word-level timestamp `.json` file.
    *   `file_type=zip`: Downloads a `.zip` archive containing the processed `.sbv`, the raw `.json`, and all translated `.sbv` files.
*   **Examples:**
    *   To get the `.sbv` file for `filename: "My_Video"`:
        `GET /download/My_Video?file_type=processed`
    *   To get the raw `.json` file:
        `GET /download/My_Video?file_type=raw`
    *   To get the zip file with all transcripts:
        `GET /download/My_Video?file_type=zip`
*   **Downloading Translated Files:**
    *   Translated files can be downloaded individually if needed, but the recommended approach is to use `file_type=zip` to get all files at once. The filenames inside the zip are simplified for clarity (e.g., `My_Video_spanish.sbv`).
*   **Error Responses:**
    *   `404 Not Found`: File not found for the given base filename.
    *   `500 Internal Server Error`: Server error during retrieval.

## New Transcription Workflow (Updated)

To get a subtitle file, the frontend application must now follow these steps:

1.  **Initiate Transcription:**
    *   Send a `POST` request to either `/transcribe-video` (with a comma-separated `target_languages` string) or `/transcribe-youtube` (with a JSON array of `target_languages`).
2.  **Get Task ID:**
    *   From the response, save the `task_id`.
3.  **Monitor Progress:**
    *   Open an `EventSource` connection to `/progress/{task_id}`.
    *   Update the UI with progress events as they arrive.
4.  **Handle Completion:**
    *   When an event with `status: "complete"` is received, parse its data.
    *   Store the `filename` (base name for downloads), `processed_file`, `raw_file`, and the array of `translated_files`.
5.  **Enable Downloads:**
    *   Use the stored `filename` to construct download links. Provide options for the primary transcript (`processed`), the raw data (`raw`), and a complete package (`zip`).
    *   Example for a complete download link: `/download/{filename}?file_type=zip`
6.  **Close Connection:**
    *   Close the `EventSource` connection.