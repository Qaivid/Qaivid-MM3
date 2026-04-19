"""
Audio Transcription — Dual-Engine Pipeline (ported from Qaivid 1.0)
Primary: Gemini (understands Punjabi/Hindi/Urdu lyrics natively, returns structured segments)
Secondary: Whisper (word-level timing only, text discarded)
Merge: Gemini text + Whisper timestamps via proportional mapping
"""
import os
import re
import json
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from openai import AsyncOpenAI
from google import genai
from models import now_utc

logger = logging.getLogger(__name__)

UPLOAD_DIR = "/app/backend/uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

WHISPER_TIMEOUT_SEC = 300  # 5 min


# ─── Timestamp Helpers ────────────────────────────────────

def fmt_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = round((seconds - int(seconds)) * 1000)
    return f"{m}:{str(s).zfill(2)}.{str(ms).zfill(3)}"


def parse_ts(ts: str) -> float:
    clean = re.sub(r'[\[\]]', '', ts).strip()
    m = re.match(r'^(\d+):(\d{2})(?:[.,](\d+))?$', clean)
    if not m:
        return 0.0
    mins = int(m.group(1))
    secs = int(m.group(2))
    millis = int((m.group(3) or '0').ljust(3, '0')[:3])
    return mins * 60 + secs + millis / 1000


def _mime_from_ext(filename: str) -> str:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'mp3'
    return {
        'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'm4a': 'audio/mp4',
        'ogg': 'audio/ogg', 'flac': 'audio/flac', 'aac': 'audio/aac',
        'webm': 'audio/webm',
    }.get(ext, 'audio/mpeg')


# ─── JSON Repair (ported from Qaivid 1.0) ────────────────

def repair_and_parse_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    text = text.strip()

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    text = re.sub(r'([{,\[])\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:', r'\1 "\2":', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r"'([^'\\]*)'", r'"\1"', text)
        return json.loads(text)


# ─── Gemini Transcription Prompt (ported from Qaivid 1.0) ─

def get_transcription_prompt(exact_duration: Optional[float] = None) -> str:
    duration_constraint = ""
    if exact_duration and exact_duration > 0:
        mins = int(exact_duration // 60)
        secs = int(exact_duration % 60)
        threshold = exact_duration * 0.8
        duration_constraint = f"""
AUDIO LENGTH CONSTRAINT (this is exact — measured from the file):
- This audio track is EXACTLY {exact_duration:.3f} seconds ({mins}m {str(secs).zfill(2)}s) long
- totalDurationSeconds MUST equal {exact_duration:.3f}
- Your last lyric timestamp MUST be within the final 20% of the track (i.e. after {threshold:.1f}s)
"""

    return f"""You are a professional audio transcriptionist specializing in music lyrics. An audio file is attached.

Listen to the ENTIRE audio track from start to finish and transcribe every lyric line with accurate timestamps. Also identify the vocal characteristics of the LEAD singer by listening to their voice.
{duration_constraint}
Return ONLY this JSON structure:
{{
  "lines": [
    {{
      "timestamp": "<M:SS.mmm format, e.g. '0:14.140'>",
      "line": "<exact lyric text sung at this timestamp>",
      "segmentHint": "<one of: 'intro', 'verse', 'pre-chorus', 'chorus', 'bridge', 'outro', 'instrumental'>"
    }}
  ],
  "totalDurationSeconds": <ACTUAL total track length in seconds>,
  "language": "<e.g. 'English', 'Punjabi', 'Hindi', 'Spanish'>",
  "isInstrumental": <true if no lyrics, false otherwise>,
  "vocalGender": "<one of: 'male', 'female', 'mixed', 'unknown' — based on the LEAD vocalist's voice>",
  "vocalAgeRange": "<one of: 'young', 'middle', 'elder', 'unknown' — based on the LEAD vocalist's voice>"
}}

CRITICAL RULES:
- Listen all the way to the end before returning totalDurationSeconds
- totalDurationSeconds must be the true full length of the audio file
- Timestamps must be spread across the FULL duration — if the track is 3 minutes, lyrics near the end must have timestamps near 3:00, not 2:00
- Do NOT compress or squish timestamps toward the beginning
- Use M:SS.mmm format (e.g. "1:23.450")
- If instrumental, return an empty lines array and set isInstrumental to true
- Include ALL lyrics — every verse, chorus, bridge, ad-lib, and hook
- One object per lyric line — do not group multiple lines
- Return ONLY the JSON, no markdown, no explanation"""


# ─── Step: Measure Exact Duration ─────────────────────────

def get_audio_duration(filepath: str) -> Optional[float]:
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(filepath)
        if audio and audio.info:
            return audio.info.length
    except Exception as e:
        logger.warning(f"Could not read audio duration: {e}")
    return None


# ─── Step: Whisper Word-Level Timing ──────────────────────

async def get_whisper_timings(filepath: str, openai_key: str) -> List[Dict[str, float]]:
    client = AsyncOpenAI(api_key=openai_key)
    with open(filepath, 'rb') as f:
        response = await client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = []
    raw_words = getattr(response, 'words', None)
    if raw_words and isinstance(raw_words, list):
        for w in raw_words:
            start = getattr(w, 'start', None) or (w.get('start') if isinstance(w, dict) else None)
            end = getattr(w, 'end', None) or (w.get('end') if isinstance(w, dict) else None)
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                words.append({"start": start, "end": end})

    return words


# ─── Step: Gemini Transcription ───────────────────────────

async def run_gemini_transcription(
    filepath: str,
    gemini_key: str,
    exact_duration: Optional[float] = None,
    model: str = "gemini-2.5-flash",
) -> dict:
    client = genai.Client(api_key=gemini_key)
    mime_type = _mime_from_ext(filepath)

    with open(filepath, 'rb') as f:
        audio_bytes = f.read()

    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

    prompt = get_transcription_prompt(exact_duration)

    full_text = ""
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                            {"text": prompt},
                        ],
                    }
                ],
                config={
                    "max_output_tokens": 32768,
                    "response_mime_type": "application/json",
                },
            )
            full_text = response.text or ""
            break
        except Exception as e:
            last_error = e
            delay = 1.0 * (2 ** (attempt + 1))
            logger.warning(f"Gemini attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)
            full_text = ""

    if not full_text:
        raise ValueError(f"Gemini transcription failed after 3 attempts: {last_error}")

    return repair_and_parse_json(full_text)


# ─── Step: Timestamp Compression Correction (from 1.0) ────

def correct_timestamps(parsed: dict, exact_duration: float) -> dict:
    if parsed.get("isInstrumental") or not parsed.get("lines"):
        return {**parsed, "totalDurationSeconds": exact_duration}

    lines = parsed["lines"]
    last_ts = parse_ts(lines[-1].get("timestamp", "0:00.000"))

    if last_ts > 0 and last_ts >= exact_duration * 0.95:
        return {**parsed, "totalDurationSeconds": exact_duration}

    denominator = last_ts if last_ts > 5 else (parsed.get("totalDurationSeconds", 0) if parsed.get("totalDurationSeconds", 0) > 0 else None)
    if not denominator:
        return {**parsed, "totalDurationSeconds": exact_duration}

    scale = exact_duration / denominator
    logger.info(f"[transcription] Rescaling x{scale:.3f} (last lyric at {last_ts:.1f}s, actual {exact_duration:.1f}s)")

    corrected_lines = []
    for line in lines:
        old_ts = parse_ts(line.get("timestamp", "0:00.000"))
        corrected_lines.append({
            **line,
            "timestamp": fmt_timestamp(old_ts * scale),
        })

    return {
        **parsed,
        "totalDurationSeconds": exact_duration,
        "lines": corrected_lines,
    }


# ─── Step: Calibrate Gemini Timestamps to Whisper Anchors ─

def merge_with_whisper_timings(
    gemini_lines: List[dict],
    whisper_words: List[Dict[str, float]],
) -> Tuple[List[dict], float]:
    """
    Calibrate Gemini's per-line timestamps using Whisper anchors.

    Gemini "hears" the song and knows the RELATIVE pacing of each line
    (instrumental gaps, sustained notes, rapid hooks). Whisper provides
    PRECISE first-sung-word and last-sung-word times in seconds.

    Strategy (Qaivid 1.0 lineage):
      1. Use Whisper's first word start and last word end as anchors.
      2. Take Gemini's first and last line timestamps as the source range.
      3. Linearly rescale every Gemini line timestamp into the Whisper anchor
         range — preserving Gemini's pacing pattern while locking the
         absolute start/end to the audio truth.

    This avoids the previous proportional-by-line-index bug, which
    overwrote Gemini's good pacing with evenly-spaced fake timestamps.
    """
    if not whisper_words or not gemini_lines:
        return gemini_lines, 0.0

    whisper_start = whisper_words[0]["start"]
    whisper_end = whisper_words[-1]["end"]

    # Source range from Gemini's own timestamps
    gemini_starts = [parse_ts(ln.get("timestamp", "0:00.000")) for ln in gemini_lines]
    g_first = gemini_starts[0]
    g_last = gemini_starts[-1] if len(gemini_starts) > 1 else g_first
    g_span = g_last - g_first

    # If Gemini gave us no usable spread, fall back to even distribution
    # across the whisper sung-window (better than collapsing everything to 0)
    if g_span <= 0.5:
        sung_span = max(whisper_end - whisper_start, 0.001)
        merged = []
        for i, line in enumerate(gemini_lines):
            ratio = i / max(len(gemini_lines) - 1, 1)
            ts = whisper_start + ratio * sung_span
            merged.append({
                "timestamp": fmt_timestamp(ts),
                "line": line.get("line", ""),
                "segmentHint": line.get("segmentHint", "verse"),
            })
        return merged, whisper_end

    # Linear calibration of Gemini timestamps onto whisper anchors
    target_span = max(whisper_end - whisper_start, 0.001)
    scale = target_span / g_span
    merged = []
    for i, line in enumerate(gemini_lines):
        calibrated = whisper_start + (gemini_starts[i] - g_first) * scale
        merged.append({
            "timestamp": fmt_timestamp(max(0.0, calibrated)),
            "line": line.get("line", ""),
            "segmentHint": line.get("segmentHint", "verse"),
        })

    return merged, whisper_end


# ─── Main Pipeline ────────────────────────────────────────

async def transcribe_audio(
    filepath: str,
    gemini_key: str,
    openai_key: str,
    language: str = "auto",
) -> dict:
    """
    Dual-engine transcription pipeline (ported from Qaivid 1.0):
    1. Measure exact audio duration
    2. Gemini: primary transcription (text, lyrics, segment hints, language)
    3. Whisper: word-level timing only (text discarded)
    4. Correct Gemini timestamp compression
    5. Merge: Gemini text + Whisper timing via proportional mapping
    """
    if not gemini_key:
        raise ValueError("Gemini API key not configured. Go to Settings to add your key.")

    # Step 1: Measure exact duration
    exact_duration = get_audio_duration(filepath)
    if exact_duration:
        logger.info(f"[transcription] Audio duration: {exact_duration:.3f}s")

    # Step 2 + 3: Run Gemini and Whisper in parallel
    gemini_task = run_gemini_transcription(filepath, gemini_key, exact_duration)

    whisper_task = None
    if openai_key:
        whisper_task = get_whisper_timings(filepath, openai_key)

    # Await Gemini
    parsed = await gemini_task
    logger.info(f"[transcription] Gemini returned {len(parsed.get('lines', []))} lines, language={parsed.get('language', '?')}")

    if parsed.get("isInstrumental") or not parsed.get("lines"):
        logger.info("[transcription] Track is instrumental — no lyrics")
        return {
            "id": str(uuid.uuid4()),
            "text": "",
            "language": parsed.get("language", "unknown"),
            "segments": [],
            "lines": [],
            "total_duration": exact_duration or parsed.get("totalDurationSeconds", 0),
            "segment_count": 0,
            "is_instrumental": True,
            "vocal_gender": parsed.get("vocalGender", "unknown"),
            "vocal_age_range": parsed.get("vocalAgeRange", "unknown"),
            "transcribed_at": now_utc(),
        }

    # Step 4: Correct timestamp compression
    if exact_duration:
        parsed = correct_timestamps(parsed, exact_duration)

    # Step 5: Merge with Whisper timings
    final_lines = parsed["lines"]
    final_duration = exact_duration or parsed.get("totalDurationSeconds", 0)

    if whisper_task:
        try:
            whisper_words = await asyncio.wait_for(whisper_task, timeout=WHISPER_TIMEOUT_SEC)
            if whisper_words and len(whisper_words) > 0:
                logger.info(f"[transcription] Whisper returned {len(whisper_words)} words — merging proportionally")
                final_lines, final_duration = merge_with_whisper_timings(parsed["lines"], whisper_words)
            else:
                logger.warning("[transcription] Whisper returned no words — keeping corrected Gemini timestamps")
        except asyncio.TimeoutError:
            logger.warning("[transcription] Whisper timed out — keeping corrected Gemini timestamps")
        except Exception as e:
            logger.error(f"[transcription] Whisper failed — keeping Gemini timestamps: {e}")

    # Build segments from lines (for compatibility with the rest of the pipeline)
    full_text = "\n".join(line.get("line", "") for line in final_lines)

    segments = []
    for i, line in enumerate(final_lines):
        ts = parse_ts(line.get("timestamp", "0:00.000"))
        next_ts = parse_ts(final_lines[i + 1]["timestamp"]) if i + 1 < len(final_lines) else final_duration
        segments.append({
            "start": ts,
            "end": next_ts,
            "text": line.get("line", "").strip(),
            "segmentHint": line.get("segmentHint", "verse"),
        })

    return {
        "id": str(uuid.uuid4()),
        "text": full_text,
        "language": parsed.get("language", language),
        "segments": segments,
        "lines": final_lines,
        "total_duration": final_duration,
        "segment_count": len(segments),
        "is_instrumental": False,
        "vocal_gender": parsed.get("vocalGender", "unknown"),
        "vocal_age_range": parsed.get("vocalAgeRange", "unknown"),
        "transcribed_at": now_utc(),
    }
