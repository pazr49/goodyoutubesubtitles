import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from .config import Config

logger = logging.getLogger(__name__)

def parse_sbv_file(sbv_path: str) -> List[Dict[str, Any]]:
    """Parse an SBV file and return a list of subtitle segments.
    
    Args:
        sbv_path: Path to the SBV file
        
    Returns:
        List of dictionaries with 'timestamp', 'text', and 'start_time'/'end_time' keys
    """
    segments = []
    
    try:
        with open(sbv_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
        # Split by double newlines to separate segments
        segment_blocks = re.split(r'\n\s*\n', content)
        
        for block in segment_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                # First line is timestamp
                timestamp_line = lines[0]
                # Remaining lines are text
                text = '\n'.join(lines[1:])
                
                # Parse timestamp (format: 00:00:00.079,00:00:05.719)
                timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}),(\d{2}:\d{2}:\d{2}\.\d{3})', timestamp_line)
                if timestamp_match:
                    start_time = timestamp_match.group(1)
                    end_time = timestamp_match.group(2)
                    
                    segments.append({
                        'timestamp': timestamp_line,
                        'text': text,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                    
    except Exception as e:
        logger.error(f"Error parsing SBV file {sbv_path}: {e}")
        raise
        
    return segments

def write_sbv_file(segments: List[Dict[str, Any]], output_path: str) -> None:
    """Write segments back to an SBV file.
    
    Args:
        segments: List of segment dictionaries
        output_path: Path where to write the SBV file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            for i, segment in enumerate(segments):
                if i > 0:
                    file.write('\n')  # Empty line between segments
                file.write(f"{segment['timestamp']}\n")
                file.write(f"{segment['text']}\n")
                
    except Exception as e:
        logger.error(f"Error writing SBV file {output_path}: {e}")
        raise

async def detect_and_translate_segments(
    segments: List[Dict[str, Any]], 
    target_language: str, 
    gemini_client: Any,
    task_id: str,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """Detect language and translate segments that are not in the target language.
    
    Args:
        segments: List of subtitle segments
        target_language: Target language for translation
        gemini_client: Gemini client instance
        task_id: Task ID for logging
        progress_callback: Optional progress callback function
        
    Returns:
        List of segments with translated text where needed
    """
    translated_segments = []
    total_segments = len(segments)
    
    # Process segments in batches to avoid overwhelming the API (smaller size for reliability)
    batch_size = 4
    
    for batch_start in range(0, total_segments, batch_size):
        batch_end = min(batch_start + batch_size, total_segments)
        batch_segments = segments[batch_start:batch_end]
        
        # Prepare batch text for language detection and translation
        batch_texts = [segment['text'] for segment in batch_segments]
        
        logger.info(f"[{task_id}] Processing translation batch {batch_start//batch_size + 1}/{(total_segments + batch_size - 1)//batch_size} for {target_language}")
        
        if progress_callback:
            progress_callback({
                "status": "processing",
                "stage": f"translating_to_{target_language}",
                "message": f"Translating batch {batch_start//batch_size + 1} of {(total_segments + batch_size - 1)//batch_size} to {target_language}...",
                "current_batch": batch_start//batch_size + 1,
                "total_batches": (total_segments + batch_size - 1)//batch_size
            })
        
        try:
            # Create prompt for Gemini – translate EVERY segment unconditionally so no
            # language-detection ambiguity occurs. Keep numbering so we can map back.
            prompt = f"""
You are a professional translator. Translate EACH of the following subtitle segments into {target_language}. 
Keep the same numbering and do NOT merge or split lines.

Return ONLY the translated segments in this exact format:
Segment 1: <translated text>
Segment 2: <translated text>
...

Do NOT add explanations or extra text.

Here are the segments:

"""
            
            for i, text in enumerate(batch_texts):
                prompt += f"Segment {i+1}: {text}\n"
                
            # Call Gemini API
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=Config.GEMINI_MODEL,
                contents=prompt
            )
            
            # Parse response
            response_text = response.text.strip()
            translated_texts = []
            
            # Extract translated segments from response
            for line in response_text.split('\n'):
                if line.strip() and line.startswith('Segment '):
                    # Extract text after "Segment X: "
                    match = re.match(r'Segment \d+: (.+)', line)
                    if match:
                        translated_texts.append(match.group(1))
            
            # If we don't get the expected number of translations, fall back to original
            if len(translated_texts) != len(batch_texts):
                logger.warning(f"[{task_id}] Expected {len(batch_texts)} translations but got {len(translated_texts)}. Using original text.")
                translated_texts = batch_texts
            
            # Create translated segments
            for i, segment in enumerate(batch_segments):
                translated_segment = segment.copy()
                translated_segment['text'] = translated_texts[i] if i < len(translated_texts) else segment['text']
                translated_segments.append(translated_segment)
                
        except Exception as e:
            logger.error(f"[{task_id}] Error translating batch to {target_language}: {e}")
            # Fall back to original segments
            translated_segments.extend(batch_segments)
        
        # Small delay between batches to be respectful to the API
        await asyncio.sleep(0.5)
    
    return translated_segments

async def translate_sbv_file(
    original_sbv_path: str,
    target_language: str,
    gemini_client: Any,
    task_id: str,
    original_name: str,
    progress_callback: Optional[callable] = None
) -> str:
    """Translate an SBV file to a target language.
    
    Args:
        original_sbv_path: Path to the original SBV file
        target_language: Target language for translation
        gemini_client: Gemini client instance
        task_id: Task ID for logging
        original_name: Original video/audio name for logging
        progress_callback: Optional progress callback function
        
    Returns:
        Path to the translated SBV file
    """
    logger.info(f"[{task_id}] Starting translation of '{original_name}' to {target_language}")
    
    try:
        # Parse original SBV file
        segments = parse_sbv_file(original_sbv_path)
        logger.info(f"[{task_id}] Parsed {len(segments)} segments from original SBV")
        
        if progress_callback:
            progress_callback({
                "status": "processing",
                "stage": f"starting_translation_to_{target_language}",
                "message": f"Starting translation to {target_language}...",
                "total_segments": len(segments)
            })
        
        # Translate segments
        translated_segments = await detect_and_translate_segments(
            segments, target_language, gemini_client, task_id, progress_callback
        )
        
        # Generate output filename
        original_filename = Path(original_sbv_path).stem
        translated_filename = f"{original_filename}_{target_language}.sbv"
        translated_path = os.path.join(Config.TEMP_DIR, translated_filename)
        
        # Write translated SBV file
        write_sbv_file(translated_segments, translated_path)
        
        logger.info(f"[{task_id}] Translation to {target_language} complete: {translated_path}")
        
        if progress_callback:
            progress_callback({
                "status": "processing",
                "stage": f"translation_to_{target_language}_complete",
                "message": f"Translation to {target_language} completed",
                "translated_file": os.path.basename(translated_path)
            })
            
        return translated_path
        
    except Exception as e:
        logger.error(f"[{task_id}] Error translating SBV to {target_language}: {e}")
        raise 