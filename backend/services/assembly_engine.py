"""
Assembly Engine
FFmpeg-based video assembly: stitch clips, add audio, subtitles, final export.
"""
import os
import subprocess
import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from models import now_utc

logger = logging.getLogger(__name__)

OUTPUT_DIR = "/app/backend/assembled"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def assemble_video(
    timeline: Dict[str, Any],
    audio_filepath: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble final video from timeline clips using FFmpeg.
    1. Concatenate all video clips in order
    2. Overlay audio track if provided
    3. Export final video
    """
    clips = timeline.get("clips", [])
    ready_clips = [c for c in clips if c.get("video_url") and c.get("status") == "ready"]

    if not ready_clips:
        raise ValueError("No ready clips in timeline")

    assembly_id = str(uuid.uuid4())
    out_name = output_filename or f"qaivid_{assembly_id}.mp4"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    # Build FFmpeg concat file
    concat_path = os.path.join(OUTPUT_DIR, f"concat_{assembly_id}.txt")
    with open(concat_path, "w") as f:
        for clip in ready_clips:
            # Convert API URL to filepath
            video_url = clip.get("video_url", "")
            if video_url.startswith("/api/videos/"):
                filepath = os.path.join("/app/backend/generated_videos", video_url.split("/")[-1])
            else:
                filepath = video_url
            if os.path.exists(filepath):
                f.write(f"file '{filepath}'\n")

    # Step 1: Concatenate clips
    concat_output = os.path.join(OUTPUT_DIR, f"concat_{assembly_id}.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_path,
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p",
        concat_output
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"FFmpeg concat failed: {result.stderr[:500]}")
            raise ValueError(f"FFmpeg concat failed: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        raise ValueError("FFmpeg assembly timed out")

    # Step 2: Add audio if provided
    if audio_filepath and os.path.exists(audio_filepath):
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", concat_output,
            "-i", audio_filepath,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ]
        try:
            result = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.warning(f"Audio overlay failed: {result.stderr[:200]}, using video only")
                os.rename(concat_output, output_path)
        except subprocess.TimeoutExpired:
            os.rename(concat_output, output_path)
    else:
        os.rename(concat_output, output_path)

    # Clean up temp files
    if os.path.exists(concat_path):
        os.remove(concat_path)
    temp_concat = os.path.join(OUTPUT_DIR, f"concat_{assembly_id}.mp4")
    if os.path.exists(temp_concat) and temp_concat != output_path:
        os.remove(temp_concat)

    # Get duration
    duration = _get_video_duration(output_path)

    return {
        "id": assembly_id,
        "filepath": output_path,
        "filename": out_name,
        "video_url": f"/api/exports/{out_name}",
        "duration_sec": duration,
        "total_clips": len(ready_clips),
        "has_audio": audio_filepath is not None and os.path.exists(audio_filepath) if audio_filepath else False,
        "assembled_at": now_utc(),
    }


def _get_video_duration(filepath: str) -> float:
    """Get video duration using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0.0
