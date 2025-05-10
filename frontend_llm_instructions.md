# Instructions for Frontend LLM Interacting with Subtitle API

This document outlines how to interact with the YouTube Subtitle Generator backend API.

## Base URL

The production API is hosted at: `https://past-oven-production.up.railway.app`

## Authentication

Currently, there is **no authentication** required to access these endpoints.

## Endpoints

### 1. Root Info

*   **Path:** `/`
*   **Method:** `GET`
*   **Purpose:** Get basic information about the API.
*   **Request:** No parameters or body needed.
*   **Success Response (200 OK):**
    ```json
    {
      "api": "YouTube Subtitles Generator",
      "version": "1.0.0",
      "status": "online"
    }
    ```
*   **Error Responses:** Standard HTTP errors possible, but unlikely for this endpoint.

### 2. Health Check

*   **Path:** `/ping`
*   **Method:** `GET`
*   **Purpose:** Check if the API service is running and healthy.
*   **Request:** No parameters or body needed.
*   **Success Response (200 OK):**
    ```json
    {
      "status": "healthy",
      "message": "Service is running"
    }
    ```
*   **Error Responses:** Standard HTTP errors possible, but unlikely for this endpoint.

### 3. Transcribe Video File

*   **Path:** `/transcribe-video`
*   **Method:** `POST`
*   **Purpose:** Upload a video file to generate an SBV subtitle file.
*   **Request:** 
    *   Content-Type: `multipart/form-data`
    *   Body: Form data containing a file attached to the key `video_file`.
*   **Success Response (200 OK):**
    ```json
    {
      "message": "Transcription successful",
      "transcript_file": "generated_filename.sbv" 
    }
    ```
    *   **Important:** The `transcript_file` value is **only the filename** of the generated file stored on the server. It is *not* the file content.
*   **Error Responses:**
    *   `400 Bad Request`: Invalid file type. Response body: `{"detail": "Invalid file type '...' Allowed: ..."}`
    *   `500 Internal Server Error`: Error during processing (audio extraction, transcription). Response body: `{"detail": "Processing error: ..."}` or `{"detail": "An unexpected error occurred..."}`
    *   `503 Service Unavailable`: Transcription service (ElevenLabs) not configured or unavailable. Response body: `{"detail": "Speech-to-text service not available..."}`

### 4. Transcribe YouTube URL

*   **Path:** `/transcribe-youtube`
*   **Method:** `POST`
*   **Purpose:** Provide a YouTube URL to generate an SBV subtitle file.
*   **Request:**
    *   Content-Type: `application/json`
    *   Body:
        ```json
        {
          "url": "https://www.youtube.com/watch?v=your_video_id"
        }
        ```
*   **Success Response (200 OK):**
    ```json
    {
      "message": "Transcription successful",
      "transcript_file": "generated_filename.sbv"
    }
    ```
    *   **Important:** The `transcript_file` value is **only the filename** of the generated file stored on the server. It is *not* the file content.
*   **Error Responses:**
    *   `400 Bad Request`: Invalid YouTube URL or error during YouTube processing. Response body: `{"detail": "Error processing YouTube URL..."}`
    *   `404 Not Found`: No suitable audio stream found for the YouTube video. Response body: `{"detail": "No suitable audio stream found..."}`
    *   `500 Internal Server Error`: Error during processing (transcription). Response body: `{"detail": "Processing error: ..."}` or `{"detail": "An unexpected error occurred..."}`
    *   `501 Not Implemented`: pytubefix library is missing on the server (unlikely with correct deployment). Response body: `{"detail": "YouTube processing dependency... not installed..."}`
    *   `503 Service Unavailable`: Transcription service (ElevenLabs) not configured or unavailable. Response body: `{"detail": "Speech-to-text service not available..."}`

### 5. Download Transcript File

*   **Path:** `/download/{filename}`
*   **Method:** `GET`
*   **Purpose:** Download a previously generated SBV subtitle file.
*   **Request:**
    *   Replace `{filename}` in the path with the actual `transcript_file` value obtained from a successful `/transcribe-video` or `/transcribe-youtube` response.
    *   Example: `GET /download/MyVideoTitle_abc123ef.sbv`
*   **Success Response (200 OK):**
    *   The response **body is the raw content of the SBV file**.
    *   The `Content-Type` header will likely be `text/plain`.
    *   The `Content-Disposition` header will be set, suggesting the original filename to the browser for download.
    *   The frontend should handle this response by triggering a file download (e.g., creating a Blob and an anchor link). 
*   **Error Responses:**
    *   `400 Bad Request`: Invalid filename format (e.g., contains `../`). Response body: `{"detail": "Invalid filename format."}`
    *   `403 Forbidden`: Attempt to access file outside the allowed directory. Response body: `{"detail": "Access forbidden."}`
    *   `404 Not Found`: The requested filename does not exist on the server (it might have been cleaned up or never generated). Response body: `{"detail": "File not found."}`
    *   `500 Internal Server Error`: Server error occurred while trying to read or send the file. Response body: `{"detail": "Internal server error while retrieving file."}`

## Transcription Workflow

To get a subtitle file, the frontend application must follow these steps:

1.  **Initiate Transcription:** Send a POST request to either `/transcribe-video` (with file data) or `/transcribe-youtube` (with JSON body containing the URL).
2.  **Handle Potential Delay:** These transcription endpoints can take time to process. The frontend should display a loading indicator to the user while waiting for the response.
3.  **Receive Response:** 
    *   If the response status code is `200 OK`, parse the JSON body to get the `transcript_file` filename value.
    *   If the response status code is not `200`, handle the error appropriately based on the status code and the `detail` message in the JSON body.
4.  **Construct Download URL:** Create the full download URL by appending `/download/` and the received `transcript_file` filename to the Base URL. (e.g., `https://past-oven-production.up.railway.app/download/generated_filename.sbv`).
5.  **Initiate Download:** Make a GET request to the constructed download URL. Handle the response as a file download.

## Important Notes

*   **CORS:** The backend is configured to allow requests from `http://localhost` (various ports) and `https://past-oven-production.up.railway.app`. If the frontend is deployed to a different domain, that domain **must** be added to the backend's CORS configuration.
*   **Processing Time:** Transcription can be slow. Provide user feedback (loading states).
*   **File Persistence:** Generated files are stored temporarily on the server. Assume they might be deleted after some time (currently no automatic cleanup is implemented, but it might be added later). Do not rely on download links working indefinitely. 