"""Storyboard Engine v3 — Story → Scenes → Shots → Multishot, in 3 LLM calls.

ROLE
----
Replace v2's "scenes + valid_realizations" intent layer with a fully timed
shot list produced by an LLM that has full story awareness.

Why three calls (not one):
    • Specialization beats generalization for creative work.
    • Failure isolation: if Call 3 produces a weak action sequence we retry
      Call 3 alone — the story (Call 1) and shot timing (Call 2) stay intact.
    • Smaller per-call output → fewer JSON / timestamp errors.

Calls
-----
Call 1 — STORY + SCENES
    Input:  audio (BPM, sections, intensity, duration), timed lyrics,
            context, narrative, style, imagination
    Output: { story: {...}, scenes: [{scene_id, time_window_start,
             time_window_end, purpose, narrative_phase, ...,
             valid_realizations: [...]}] }

Call 2 — SHOTS WITHIN SCENES
    Input:  Call 1 output + same musical map
    Output: { shots: [{shot_id, scene_id, start_time, end_time,
             duration (2-15s), lyric_text, action_intent}] }
    Constraints: shots tile [0, audio_end] with no gaps / overlaps; each
    shot's window is inside its parent scene's window; duration ∈ [2, 15].

Call 3 — MULTISHOT EXPANSION (only on shots > 8s)
    Input:  shots from Call 2 with duration > 8s
    Output: per-shot actions[] of {order, duration, action_text}
    Rule:   8 < d ≤ 10 → 2 sub-actions
            10 < d ≤ 15 → 3+ sub-actions
            sum(sub-action durations) == shot duration
            All sub-actions share the same Frame 0 (same starting still),
            different actions, in continuity.

Output schema (storyboard_packet v3)
------------------------------------
{
  "schema_version":   3,
  "story":            { "arc": "...", "summary": "...",
                        "central_conflict": "..." },
  "scenes":           [scene_v3, ...],   # extends v2 scene with time windows
  "shots":            [shot_v3, ...],    # NEW — flat timed shot list
  "used_fallback_v3": bool,
  "style_preset":     str
}

scene_v3 keeps every v2 field (so the existing brief still reads
valid_realizations) and adds:
  - time_window_start, time_window_end  (seconds)

shot_v3:
  {
    "shot_id":        "shot_1",
    "scene_id":       "s1",
    "start_time":     0.0,
    "end_time":       7.0,
    "duration":       7,
    "lyric_text":     "...",        # joined timed-lyric lines that fall in window
    "action_intent":  "wide establishing shot, slow push toward horizon",
    "actions":        []            # populated by Call 3 if duration > 8
  }

Each entry in actions[]:
  { "order": 1, "duration": 4, "action_text": "..." }

Async because it does OpenAI Chat calls.  Never raises — falls back to a
deterministic per-section storyboard if any call exhausts its retries.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ── Model & retry ────────────────────────────────────────────────────────
_MODEL = "gpt-4o-mini"
_MAX_RETRIES_PER_CALL = 0   # repair pass handles known issues; no retry needed

# ── Duration rules ───────────────────────────────────────────────────────
_MIN_SHOT_DURATION = 2
_MAX_SHOT_DURATION = 15
_MULTISHOT_THRESHOLD = 8    # shots > 8s become multishot

# ── Enum sets (mirrors v2 for scene-shape compatibility) ─────────────────
_NARRATIVE_PHASES = {"intro", "build", "peak", "breakdown", "resolution"}
_INTENSITIES      = {"low", "medium", "high", "peak"}
_PRESENCE         = {"full", "partial", "absent", "object_focus", "silhouette",
                     "hands_only", "memory_presence"}
_MOTION_DENSITY   = {"low", "medium", "high"}
_TIMELINE_POS     = {"present", "memory", "fragmented", "future", "dream"}

_MIN_REALIZATIONS = 3
_MAX_REALIZATIONS = 6


# =====================================================================
# Format helpers — turn raw inputs into compact prompt blocks
# =====================================================================
def _format_musical_map(audio_data: Dict[str, Any],
                        input_structure: Dict[str, Any]) -> str:
    """BPM + section structure + intensity curve + audio duration."""
    ad = audio_data or {}
    bpm = ad.get("bpm") or 120
    bpb = ad.get("beats_per_bar") or 4
    dur = float(ad.get("duration_seconds") or 0)
    intensity_curve = ad.get("intensity_curve") or []

    # Compact intensity curve — sample up to 12 points
    if isinstance(intensity_curve, list) and intensity_curve:
        step = max(1, len(intensity_curve) // 12)
        sampled = []
        for i in range(0, len(intensity_curve), step):
            it = intensity_curve[i]
            if isinstance(it, dict):
                t = it.get("time") or it.get("t") or i
                v = it.get("intensity") or it.get("value") or 0.5
            else:
                t = i
                v = it
            sampled.append(f"{round(float(t), 1)}s={round(float(v), 2)}")
        ic_str = ", ".join(sampled[:12])
    else:
        ic_str = "n/a"

    sections = (input_structure or {}).get("sections") or []
    units    = (input_structure or {}).get("units")    or []
    sec_lines: List[str] = []
    for sec in sections:
        sid    = sec.get("id") or "?"
        stype  = sec.get("type") or "section"
        label  = sec.get("label") or stype
        u_ids  = sec.get("unit_ids") or []
        # Find time window of this section from its units
        sec_starts: List[float] = []
        sec_ends:   List[float] = []
        for u in units:
            if isinstance(u, dict) and u.get("id") in u_ids:
                if u.get("start_time") is not None:
                    sec_starts.append(float(u["start_time"]))
                if u.get("end_time") is not None:
                    sec_ends.append(float(u["end_time"]))
        win = ""
        if sec_starts and sec_ends:
            win = f" [{min(sec_starts):.1f}s → {max(sec_ends):.1f}s]"
        sec_lines.append(f"  [{sid}] {stype} — {label}{win} ({len(u_ids)} units)")

    return (
        f"  audio_duration: {dur:.1f}s\n"
        f"  bpm: {bpm}, beats_per_bar: {bpb}\n"
        f"  intensity_curve_samples: {ic_str}\n"
        f"  sections:\n" + ("\n".join(sec_lines) if sec_lines else "    (none)")
    )


def _format_timed_lyrics(lyrics_timed: List[Dict[str, Any]],
                         max_lines: int = 80) -> str:
    """Compact timed lyric block: 'idx [start–end] text'."""
    if not lyrics_timed:
        return "  (no timed lyrics available)"
    lines: List[str] = []
    for i, ly in enumerate(lyrics_timed[:max_lines]):
        if not isinstance(ly, dict):
            continue
        try:
            s = float(ly.get("start") or ly.get("start_time") or 0)
            e = float(ly.get("end")   or ly.get("end_time")   or 0)
        except (TypeError, ValueError):
            s, e = 0.0, 0.0
        text = (ly.get("text") or ly.get("line") or "").strip()
        lines.append(f"  {i + 1}. [{s:.2f}–{e:.2f}] {text[:120]}")
    if len(lyrics_timed) > max_lines:
        lines.append(f"  … (+{len(lyrics_timed) - max_lines} more lines)")
    return "\n".join(lines)


def _format_context(context_packet: Dict[str, Any]) -> str:
    spk        = context_packet.get("speaker") or {}
    motivation = context_packet.get("motivation") or {}
    world      = context_packet.get("world_assumptions") or {}
    payload = {
        "core_theme":       context_packet.get("core_theme"),
        "emotional_world":  context_packet.get("emotional_world"),
        "emotional_arc":    context_packet.get("emotional_arc"),
        "narrative_mode":   context_packet.get("narrative_mode"),
        "speaker_identity": spk.get("identity"),
        "speaker_emotion":  spk.get("emotional_state"),
        "location_dna":     context_packet.get("location_dna"),
        "era":              context_packet.get("era"),
        "motivation":       motivation,
        "world":            world,
        "must_preserve":    context_packet.get("must_preserve"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_style_light(style_profile: Dict[str, Any]) -> str:
    cin  = (style_profile or {}).get("cinematic")  or {}
    prod = (style_profile or {}).get("production") or {}
    payload = {
        "cinematic_style":  cin.get("name") or cin.get("id"),
        "production_style": prod.get("name") or prod.get("id"),
        "preset":           (style_profile or {}).get("preset"),
        "tone_keywords":    (style_profile or {}).get("tone_keywords"),
    }
    sp = style_profile or {}
    if sp.get("vibe_label"):
        payload["vibe_preset"] = sp["vibe_label"]
    if sp.get("vibe_storyboard_direction"):
        payload["vibe_storyboard_direction"] = sp["vibe_storyboard_direction"]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_narrative_block(narrative_packet: Dict[str, Any]) -> str:
    if not narrative_packet:
        return ""
    try:
        from narrative_engine import format_for_prompt
        return format_for_prompt(narrative_packet) or ""
    except Exception:
        logger.exception("StoryboardV3: failed to format narrative_packet")
        return ""


def _format_imagination_block(imagination_packet: Dict[str, Any]) -> str:
    if not imagination_packet:
        return ""
    try:
        from imagination_engine import format_imagination_for_prompt
        return format_imagination_for_prompt(imagination_packet) or ""
    except Exception:
        logger.exception("StoryboardV3: failed to format imagination_packet")
        return ""


# =====================================================================
# CALL 1 — Story + Scenes
# =====================================================================
def _call1_system_prompt() -> str:
    return (
        "You are the STORYBOARD DIRECTOR for Qaivid — a music-video pipeline.\n\n"
        "ROLE (Call 1 of 3): Read the audio's musical map (BPM, sections, "
        "intensity curve, timed lyrics) AND the story context, then produce:\n"
        "  • a STORY block (arc, summary, central_conflict)\n"
        "  • a SCENE list, each with a time_window_start/end that tiles the "
        "    full audio duration\n\n"
        "RULES:\n"
        "1. Scene boundaries should align with section boundaries when natural, "
        "   but you MAY split a long section or merge two short ones if the "
        "   emotional arc justifies it.\n"
        "2. Scene time windows MUST tile [0, audio_end] with NO gaps and NO "
        "   overlaps. The first scene starts at 0.0; the last scene ends at "
        "   the audio duration. Use intensity peaks to place act breaks.\n"
        "3. Pre-lyric instrumental sections are REAL scenes — assign them a "
        "   purpose (establish world, introduce character, set mood) and a "
        "   time window. Don't merge instrumental space into the first sung "
        "   scene.\n"
        "4. For EACH scene, also produce 3-6 valid_realizations — short "
        "   one-sentence ideas for how the scene could be expressed visually. "
        "   These feed downstream variety; do not pick a single one.\n"
        "5. Maintain continuity_hooks (subject thread, motifs carried across).\n"
        "6. Repeated sections (chorus N times) MUST EVOLVE — different "
        "   timeline_position or presence_hint or environment each time.\n"
        "7. Style is a LIGHT influence; story + context drive the scene plan.\n\n"
        "FORBIDDEN: locking specific locations, props, characters, lens / "
        "camera shots, or executable prompts. That happens in Call 2 and the "
        "Brief stage.\n\n"
        "Return strict JSON. No prose outside the JSON object."
    )


def _call1_user_prompt(audio_data, input_structure, lyrics_timed,
                       context_packet, narrative_packet, style_profile,
                       project_settings, emotional_mode_packet,
                       imagination_packet) -> str:
    parts: List[str] = []
    if imagination_packet:
        ib = _format_imagination_block(imagination_packet)
        if ib:
            parts.append(ib)
    parts.append("MUSICAL MAP:\n" + _format_musical_map(audio_data, input_structure))
    parts.append("TIMED LYRICS (idx [start-end] text):\n"
                 + _format_timed_lyrics(lyrics_timed))
    parts.append("STORY CONTEXT:\n" + _format_context(context_packet))
    nb = _format_narrative_block(narrative_packet)
    if nb:
        parts.append(nb)
    parts.append("STYLE (light influence only):\n" + _format_style_light(style_profile))
    parts.append("PROJECT SETTINGS:\n"
                 + json.dumps(project_settings or {}, indent=2, ensure_ascii=False))

    parts.append(
        "Return JSON of exactly this shape:\n"
        "{\n"
        '  "story": {\n'
        '    "arc": "one-sentence emotional arc of the song",\n'
        '    "summary": "2-3 sentence narrative summary",\n'
        '    "central_conflict": "the core dramatic tension"\n'
        '  },\n'
        '  "scenes": [\n'
        '    {\n'
        '      "scene_id": "s1",\n'
        '      "source_section": "intro|verse1|...",\n'
        '      "narrative_phase": "intro|build|peak|breakdown|resolution",\n'
        '      "purpose": "What this scene does emotionally",\n'
        '      "emotional_intensity": "low|medium|high|peak",\n'
        '      "presence_hint": "full|partial|absent|object_focus|silhouette|hands_only|memory_presence",\n'
        '      "motion_density": "low|medium|high",\n'
        '      "timeline_position": "present|memory|fragmented|future|dream",\n'
        '      "time_window_start": 0.0,\n'
        '      "time_window_end": 26.5,\n'
        '      "motif_usage": ["motif name", "..."],\n'
        '      "continuity_hooks": {\n'
        '        "subject": "subject continuity note",\n'
        '        "motifs": ["motif to carry"]\n'
        '      },\n'
        '      "valid_realizations": [\n'
        '        "concrete idea 1",\n'
        '        "concrete idea 2",\n'
        '        "concrete idea 3"\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "Scene time windows MUST tile [0, audio_duration] with no gaps or overlaps."
    )
    return "\n\n".join(parts)


def _repair_call1_scenes(data: Dict[str, Any], audio_duration: float) -> None:
    """Mutate scene time windows in place: snap small gaps/overlaps, clamp
    first scene to start at 0 and last scene to end at audio_duration.
    Tolerates ±5s of LLM imprecision; anything larger is left for the
    validator to reject."""
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(scenes, list) or not scenes:
        return
    prev_end = 0.0
    for i, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            continue
        try:
            ws = float(sc.get("time_window_start") or 0)
            we = float(sc.get("time_window_end")   or 0)
        except (TypeError, ValueError):
            continue
        # Snap small gaps/overlaps (<5s) to previous end
        if i == 0:
            ws = 0.0
        elif abs(ws - prev_end) < 5.0:
            ws = prev_end
        # Clamp last scene end to audio_duration when within 5s
        if i == len(scenes) - 1 and audio_duration > 0 and abs(we - audio_duration) < 5.0:
            we = audio_duration
        if we <= ws:
            we = ws + 1.0
        sc["time_window_start"] = round(ws, 3)
        sc["time_window_end"]   = round(we, 3)
        prev_end = we


def _validate_call1(data: Dict[str, Any], audio_duration: float) -> Optional[str]:
    """Return None if valid, else an error string for the retry prompt."""
    if not isinstance(data, dict):
        return "Output is not a JSON object."
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return "scenes must be a non-empty array."
    prev_end = 0.0
    for i, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            return f"scene {i} is not an object."
        try:
            ws = float(sc.get("time_window_start"))
            we = float(sc.get("time_window_end"))
        except (TypeError, ValueError):
            return f"scene {i} missing valid time_window_start/end."
        if we <= ws:
            return f"scene {i} time_window_end ({we}) must be > start ({ws})."
        # Every scene must be wide enough to contain at least one valid shot
        # (≥ _MIN_SHOT_DURATION); else there is no way to tile it cleanly.
        if (we - ws) < _MIN_SHOT_DURATION - 0.05:
            return (f"scene {i} window [{ws}, {we}] is only "
                    f"{we - ws:.2f}s wide; must be ≥ {_MIN_SHOT_DURATION}s "
                    f"so it can hold a valid shot. Merge it with a neighbour.")
        if abs(ws - prev_end) > 0.05:
            return (f"scene {i} time_window_start ({ws}) should equal previous "
                    f"scene's end ({prev_end}). No gaps or overlaps.")
        prev_end = we
    if audio_duration > 0 and abs(prev_end - audio_duration) > 0.5:
        return (f"last scene must end at audio_duration ({audio_duration:.1f}); "
                f"got {prev_end:.1f}.")
    return None


def _coerce_call1_scene(raw: Any, idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    def _enum(v: Any, allowed: set, default: str) -> str:
        s = str(v or "").strip().lower().replace(" ", "_")
        return s if s in allowed else default

    realizations_in = raw.get("valid_realizations") or []
    if not isinstance(realizations_in, list):
        realizations_in = []
    realizations = [str(r).strip()[:300] for r in realizations_in if str(r).strip()][:_MAX_REALIZATIONS]
    while len(realizations) < _MIN_REALIZATIONS:
        realizations.append("alternative framing of the same emotional intent")

    hooks_in = raw.get("continuity_hooks") or {}
    if not isinstance(hooks_in, dict):
        hooks_in = {}
    motifs_in = hooks_in.get("motifs") or []
    if not isinstance(motifs_in, list):
        motifs_in = []
    motif_usage_in = raw.get("motif_usage") or []
    if not isinstance(motif_usage_in, list):
        motif_usage_in = []

    try:
        ws = float(raw.get("time_window_start") or 0)
        we = float(raw.get("time_window_end")   or 0)
    except (TypeError, ValueError):
        ws, we = 0.0, 0.0

    return {
        "scene_id":            str(raw.get("scene_id") or f"s{idx + 1}").strip()[:24],
        "source_section":      str(raw.get("source_section") or "").strip()[:48],
        "narrative_phase":     _enum(raw.get("narrative_phase"), _NARRATIVE_PHASES, "build"),
        "purpose":             str(raw.get("purpose") or "").strip()[:300],
        "emotional_intensity": _enum(raw.get("emotional_intensity"), _INTENSITIES, "medium"),
        "presence_hint":       _enum(raw.get("presence_hint"), _PRESENCE, "full"),
        "motion_density":      _enum(raw.get("motion_density"), _MOTION_DENSITY, "medium"),
        "timeline_position":   _enum(raw.get("timeline_position"), _TIMELINE_POS, "present"),
        "time_window_start":   round(ws, 3),
        "time_window_end":     round(we, 3),
        "motif_usage":         [str(m).strip()[:60] for m in motif_usage_in if str(m).strip()][:8],
        "continuity_hooks": {
            "subject": str(hooks_in.get("subject") or "").strip()[:200],
            "motifs":  [str(m).strip()[:60] for m in motifs_in if str(m).strip()][:8],
        },
        "valid_realizations":  realizations,
    }


async def _run_call1(client: AsyncOpenAI, audio_data, input_structure,
                     lyrics_timed, context_packet, narrative_packet,
                     style_profile, project_settings, emotional_mode_packet,
                     imagination_packet, audio_duration: float
                     ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Run Call 1 with retries. Returns (story, coerced_scenes) or (None, None)."""
    base_user = _call1_user_prompt(
        audio_data, input_structure, lyrics_timed, context_packet,
        narrative_packet, style_profile, project_settings,
        emotional_mode_packet, imagination_packet,
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _call1_system_prompt()},
        {"role": "user",   "content": base_user},
    ]

    for attempt in range(_MAX_RETRIES_PER_CALL + 1):
        try:
            resp = await client.chat.completions.create(
                model=_MODEL,
                response_format={"type": "json_object"},
                temperature=0.8 if attempt == 0 else 0.5,
                messages=messages,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            _repair_call1_scenes(data, audio_duration)
            err = _validate_call1(data, audio_duration)
            if err is None:
                story = data.get("story") if isinstance(data.get("story"), dict) else {}
                scenes_in = data.get("scenes") or []
                coerced: List[Dict[str, Any]] = []
                for i, s in enumerate(scenes_in):
                    cs = _coerce_call1_scene(s, i)
                    if cs:
                        coerced.append(cs)
                if coerced:
                    return story, coerced
                err = "no valid scenes after coercion"
            logger.warning("StoryboardV3 Call 1 attempt %d failed validation: %s",
                           attempt + 1, err)
            if attempt < _MAX_RETRIES_PER_CALL:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": (f"Your previous output failed: {err}. "
                                             "Fix the output and return ONLY the corrected JSON.")})
        except Exception as exc:
            logger.exception("StoryboardV3 Call 1 attempt %d raised: %s", attempt + 1, exc)
    return None, None


# =====================================================================
# CALL 2 — Shots within scenes
# =====================================================================
def _call2_system_prompt() -> str:
    return (
        "You are the STORYBOARD DIRECTOR for Qaivid — Call 2 of 3.\n\n"
        "ROLE: Take the scene plan from Call 1 and break each scene's time "
        "window into SHOTS. Every shot has a precise start_time, end_time, "
        "duration, and a one-line action_intent.\n\n"
        "RULES:\n"
        f"1. Each shot duration MUST be between {_MIN_SHOT_DURATION} and "
        f"{_MAX_SHOT_DURATION} seconds (inclusive).\n"
        "2. Within EACH scene, shots tile that scene's [time_window_start, "
        "   time_window_end] window with NO gaps and NO overlaps.\n"
        "3. Across all scenes, shots tile [0, audio_duration].\n"
        "4. Shot count per scene is YOUR creative decision based on "
        "   narrative density. A peak/chorus needs more rapid shots; an "
        "   introspective bridge can use fewer, longer shots. Do NOT pad.\n"
        "5. Each shot's action_intent is ONE concrete cinematic action "
        "   (verb-driven sentence). Shots within the same scene MUST have "
        "   distinct action_intents — do not repeat.\n"
        "6. Use the timed lyrics to align shot boundaries near lyric line "
        "   starts when possible (snap within 0.5s).\n"
        "7. Pre-lyric instrumental shots get cinematic action_intents that "
        "   establish world / character / mood — not 'no action'.\n"
        "8. Shot IDs are sequential across the entire video: shot_1, shot_2, ...\n\n"
        "Return strict JSON. No prose outside the JSON object."
    )


def _call2_user_prompt(scenes: List[Dict[str, Any]], audio_duration: float,
                       audio_data: Dict[str, Any], lyrics_timed: List[Dict[str, Any]],
                       story: Dict[str, Any]) -> str:
    scenes_block = json.dumps(
        [{
            "scene_id":          s["scene_id"],
            "purpose":           s.get("purpose"),
            "narrative_phase":   s.get("narrative_phase"),
            "emotional_intensity": s.get("emotional_intensity"),
            "motion_density":    s.get("motion_density"),
            "time_window_start": s["time_window_start"],
            "time_window_end":   s["time_window_end"],
        } for s in scenes],
        indent=2, ensure_ascii=False,
    )
    bpm = (audio_data or {}).get("bpm") or 120
    return (
        f"AUDIO DURATION: {audio_duration:.1f}s   BPM: {bpm}\n\n"
        f"STORY:\n{json.dumps(story or {}, indent=2, ensure_ascii=False)}\n\n"
        f"SCENES (from Call 1 — use these time windows exactly):\n{scenes_block}\n\n"
        f"TIMED LYRICS:\n{_format_timed_lyrics(lyrics_timed)}\n\n"
        "Return JSON of exactly this shape:\n"
        "{\n"
        '  "shots": [\n'
        '    {\n'
        '      "shot_id":       "shot_1",\n'
        '      "scene_id":      "s1",\n'
        '      "start_time":    0.0,\n'
        '      "end_time":      7.0,\n'
        '      "duration":      7,\n'
        '      "lyric_text":    "",\n'
        '      "action_intent": "wide establishing shot of the field at dusk, slow push toward the horizon"\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        f"Constraints: every duration ∈ [{_MIN_SHOT_DURATION}, {_MAX_SHOT_DURATION}]; "
        "shots tile [0, audio_duration] with no gaps / overlaps; each shot's "
        "[start_time, end_time] sits inside its scene_id's window from above."
    )


def _partition_window_into_shots(window_start: float, window_end: float,
                                 base_intent: str = "",
                                 base_lyric: str = "") -> List[Dict[str, Any]]:
    """Partition [window_start, window_end] into back-to-back shots, each
    in [MIN, MAX] seconds.  Returns shot stubs (no shot_id / scene_id).

    Guarantees: every shot's duration ∈ [MIN, MAX], shots tile the
    window exactly, the final shot absorbs any rounding residue.
    """
    out: List[Dict[str, Any]] = []
    win = window_end - window_start
    if win < _MIN_SHOT_DURATION:
        # Single sub-min shot (caller must accept; validator tolerance
        # handles a single short residue at the very tail of the audio).
        return [{
            "start_time": round(window_start, 3),
            "end_time":   round(window_end, 3),
            "duration":   max(1, int(round(win))),
            "lyric_text": base_lyric,
            "action_intent": (base_intent or "moment held").strip(),
        }]
    # Choose a target shot count such that the average sits comfortably
    # inside [MIN, MAX].  Using a 7-second target keeps things lyrical.
    target = max(1, int(round(win / 7.0)))
    # Adjust until min and max bounds are satisfiable.
    while target > 1 and (win / target) < _MIN_SHOT_DURATION:
        target -= 1
    while (win / target) > _MAX_SHOT_DURATION:
        target += 1
    base_dur = win / target
    cursor = window_start
    for i in range(target):
        if i == target - 1:
            en = window_end                       # absorb residue
        else:
            en = cursor + base_dur
        dur = en - cursor
        # Clamp (very defensive — base_dur is already in [MIN, MAX])
        if dur > _MAX_SHOT_DURATION:
            en = cursor + _MAX_SHOT_DURATION
            dur = _MAX_SHOT_DURATION
        out.append({
            "start_time": round(cursor, 3),
            "end_time":   round(en, 3),
            "duration":   max(1, int(round(dur))),
            "lyric_text": base_lyric,
            "action_intent": (base_intent or "the moment continues").strip(),
        })
        cursor = en
    return out


def _repair_call2_shots(data: Dict[str, Any], scenes: List[Dict[str, Any]],
                        audio_duration: float) -> None:
    """Mutate shot list in place so every shot sits strictly inside its
    parent scene's window.

    Strategy: process scene-by-scene.  Group LLM shots by their
    LLM-assigned scene_id; for each scene, sequentially tile the
    scene's window using the LLM's durations as creative hints.  If a
    scene has no LLM shots, synthesize them.  If LLM durations overflow
    the scene window, scale them down; if they leave a tail, extend the
    last shot or add a synthetic one.
    """
    if not isinstance(data, dict):
        return
    raw_shots = data.get("shots")
    if not isinstance(raw_shots, list):
        return

    # Group LLM shots by their declared scene_id (preserving order)
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for sh in raw_shots:
        if not isinstance(sh, dict):
            continue
        sid = str(sh.get("scene_id") or "")
        by_scene.setdefault(sid, []).append(sh)

    new_shots: List[Dict[str, Any]] = []
    for scene in scenes:
        sid = scene["scene_id"]
        ws  = float(scene["time_window_start"])
        we  = float(scene["time_window_end"])
        win = we - ws
        scene_shots = by_scene.get(sid, [])

        if not scene_shots:
            # No LLM shots for this scene → synthesize
            stubs = _partition_window_into_shots(
                ws, we,
                base_intent=(scene.get("purpose") or "the moment unfolds"),
                base_lyric="",
            )
            for stub in stubs:
                stub["scene_id"] = sid
            new_shots.extend(stubs)
            continue

        # Trust the LLM's *durations* (creative pacing) but force them
        # to tile [ws, we] exactly.  Scale durations to fit the scene.
        llm_durs: List[float] = []
        for sh in scene_shots:
            try:
                d = float(sh.get("duration") or 0)
                if d <= 0:
                    d = float(sh.get("end_time") or 0) - float(sh.get("start_time") or 0)
            except (TypeError, ValueError):
                d = 0.0
            llm_durs.append(max(_MIN_SHOT_DURATION,
                                min(_MAX_SHOT_DURATION, d if d > 0 else 7.0)))

        total = sum(llm_durs) or 1.0
        # If sum doesn't match win, scale (then re-clamp into [MIN, MAX])
        if abs(total - win) > 0.01:
            scale = win / total
            llm_durs = [max(_MIN_SHOT_DURATION,
                            min(_MAX_SHOT_DURATION, d * scale)) for d in llm_durs]
            total = sum(llm_durs)

        # Drive llm_durs toward sum == win, with every element ∈ [MIN, MAX].
        # Strategy: shrink/grow the last element first; if that pushes it
        # outside [MIN, MAX], add or drop shots as needed.
        def _normalize(durs: List[float], target: float) -> List[float]:
            # Clamp every element first
            durs = [max(_MIN_SHOT_DURATION, min(_MAX_SHOT_DURATION, d)) for d in durs]
            # Loop: grow/shrink/append/drop until sum is within tolerance.
            for _ in range(128):
                cur = sum(durs)
                diff = target - cur
                if abs(diff) <= 0.01:
                    return durs
                if diff > 0:
                    # Need MORE duration — try to grow last shot
                    headroom = _MAX_SHOT_DURATION - durs[-1]
                    if headroom >= diff:
                        durs[-1] += diff
                        continue
                    if headroom > 0:
                        durs[-1] = _MAX_SHOT_DURATION
                        continue
                    # Last shot saturated — try to grow earlier shots first
                    grew = False
                    for k in range(len(durs) - 1, -1, -1):
                        room = _MAX_SHOT_DURATION - durs[k]
                        if room > 0.01:
                            take = min(room, diff)
                            durs[k] += take
                            grew = True
                            break
                    if grew:
                        continue
                    # All shots saturated at MAX — append a new shot
                    chunk = max(_MIN_SHOT_DURATION, min(_MAX_SHOT_DURATION, diff))
                    durs.append(chunk)
                else:
                    # Need LESS duration — shrink last shot
                    overflow = -diff
                    slack = durs[-1] - _MIN_SHOT_DURATION
                    if slack >= overflow:
                        durs[-1] -= overflow
                        continue
                    if slack > 0:
                        durs[-1] = _MIN_SHOT_DURATION
                        continue
                    # Last shot at MIN — try shrinking earlier shots
                    shrunk = False
                    for k in range(len(durs) - 1, -1, -1):
                        slack_k = durs[k] - _MIN_SHOT_DURATION
                        if slack_k > 0.01:
                            take = min(slack_k, overflow)
                            durs[k] -= take
                            shrunk = True
                            break
                    if shrunk:
                        continue
                    # Everyone at MIN — drop the last shot
                    if len(durs) > 1:
                        durs.pop()
                    else:
                        return durs
            return durs

        llm_durs = _normalize(llm_durs, win)
        # Sanity: if normalize couldn't hit target, fall back to clean
        # uniform partition of the scene window so we never leak gaps
        # downstream.
        if abs(sum(llm_durs) - win) > 0.05 or any(
                d < _MIN_SHOT_DURATION - 0.01 or d > _MAX_SHOT_DURATION + 0.01
                for d in llm_durs):
            stubs = _partition_window_into_shots(
                ws, we,
                base_intent=(scene.get("purpose") or "the moment unfolds"),
            )
            for stub in stubs:
                stub["scene_id"] = sid
            new_shots.extend(stubs)
            continue

        # Build shots aligned to [ws, we].  Trust llm_durs for both
        # cursor advancement and the last shot's duration.  Snap the
        # final end to `we` only to absorb sub-second rounding drift.
        cursor = ws
        n_d = len(llm_durs)
        for i, d in enumerate(llm_durs):
            llm_sh = scene_shots[i] if i < len(scene_shots) else {}
            en = cursor + d
            if i == n_d - 1 and abs(en - we) < 0.5:
                en = we                  # absorb tiny rounding drift
            actual_dur = en - cursor
            # Hard guard: never emit a shot exceeding MAX
            if actual_dur > _MAX_SHOT_DURATION:
                en = cursor + _MAX_SHOT_DURATION
                actual_dur = _MAX_SHOT_DURATION
            new_shots.append({
                "scene_id":      sid,
                "start_time":    round(cursor, 3),
                "end_time":      round(en, 3),
                "duration":      max(1, int(round(actual_dur))),
                "lyric_text":    str(llm_sh.get("lyric_text") or "")[:300],
                "action_intent": str(llm_sh.get("action_intent")
                                     or scene.get("purpose")
                                     or "the moment continues")[:400],
            })
            cursor = en

    # Final defensive pass: ensure shots are contiguous and we land on
    # audio_end without ever producing a duration > MAX.
    if new_shots and audio_duration > 0:
        prev_end = 0.0
        for idx, sh in enumerate(new_shots):
            gap = sh["start_time"] - prev_end
            if gap > 0.001:
                # There's a gap.  First try to extend the previous shot's
                # end forward (preserves both scene_ids, no shot grows).
                if idx > 0:
                    prev = new_shots[idx - 1]
                    new_prev_dur = sh["start_time"] - prev["start_time"]
                    if new_prev_dur <= _MAX_SHOT_DURATION + 0.05:
                        prev["end_time"] = sh["start_time"]
                        prev["duration"] = max(1, int(round(new_prev_dur)))
                        prev_end = prev["end_time"]
                    # else: extending prev would break MAX.  Leave the
                    # gap; the post-build validator will catch it and
                    # we'll fall back to a clean partition.  We refuse
                    # to mutate `sh` across its scene_id boundary.
                else:
                    sh["start_time"] = round(prev_end, 3)
                    new_dur = sh["end_time"] - sh["start_time"]
                    if new_dur > _MAX_SHOT_DURATION:
                        sh["end_time"] = round(sh["start_time"] + _MAX_SHOT_DURATION, 3)
                        new_dur = _MAX_SHOT_DURATION
                    sh["duration"] = max(1, int(round(new_dur)))
            elif gap < -0.001:
                # Overlap (shouldn't happen but be safe) — push start fwd
                sh["start_time"] = round(prev_end, 3)
                sh["duration"]   = max(1, int(round(
                    sh["end_time"] - sh["start_time"])))
            prev_end = sh["end_time"]
        # Snap final shot end to audio_duration if within tolerance and
        # the resulting duration is still inside [MIN, MAX].
        last = new_shots[-1]
        gap = audio_duration - last["end_time"]
        if abs(gap) <= 1.0:
            new_last_dur = audio_duration - last["start_time"]
            if _MIN_SHOT_DURATION - 0.05 <= new_last_dur <= _MAX_SHOT_DURATION + 0.05:
                last["end_time"] = round(audio_duration, 3)
                last["duration"] = max(1, int(round(new_last_dur)))

    # Assign sequential shot_ids
    for i, sh in enumerate(new_shots):
        sh["shot_id"] = f"shot_{i + 1}"

    data["shots"] = new_shots


def _validate_call2(data: Dict[str, Any], scenes: List[Dict[str, Any]],
                    audio_duration: float) -> Optional[str]:
    if not isinstance(data, dict):
        return "Output is not a JSON object."
    shots = data.get("shots")
    if not isinstance(shots, list) or not shots:
        return "shots must be a non-empty array."
    scene_windows = {s["scene_id"]: (s["time_window_start"], s["time_window_end"])
                     for s in scenes}
    prev_end = 0.0
    for i, sh in enumerate(shots):
        if not isinstance(sh, dict):
            return f"shot {i} is not an object."
        try:
            st  = float(sh.get("start_time"))
            en  = float(sh.get("end_time"))
            dur = float(sh.get("duration") or (en - st))
        except (TypeError, ValueError):
            return f"shot {i} missing numeric start_time / end_time / duration."
        # Validate against the actual time-span (end-start), not just the
        # rounded `duration` field — the field can lie, the timeline can't.
        actual_span = en - st
        if abs(actual_span - dur) > 0.6:
            return (f"shot {i} duration field ({dur}) disagrees with "
                    f"end_time - start_time ({actual_span:.2f}).")
        if (actual_span < _MIN_SHOT_DURATION - 0.05
                or actual_span > _MAX_SHOT_DURATION + 0.05):
            return (f"shot {i} actual duration {actual_span:.2f} outside "
                    f"[{_MIN_SHOT_DURATION}, {_MAX_SHOT_DURATION}].")
        if abs(st - prev_end) > 0.05:
            return (f"shot {i} start_time {st} should equal previous "
                    f"end {prev_end}. No gaps or overlaps allowed.")
        if en <= st:
            return f"shot {i} end_time must be > start_time."
        sid = str(sh.get("scene_id") or "")
        if sid not in scene_windows:
            return f"shot {i} scene_id '{sid}' does not exist in Call 1 scenes."
        # Shots must sit inside their tagged scene's window (the repair
        # pass tiles per-scene to enforce this).
        ws, we = scene_windows[sid]
        if st < ws - 0.05 or en > we + 0.05:
            return (f"shot {i} window [{st}, {en}] is outside its scene "
                    f"'{sid}' window [{ws}, {we}].")
        prev_end = en
    if audio_duration > 0 and abs(prev_end - audio_duration) > 0.5:
        return (f"last shot must end near audio_duration ({audio_duration:.1f}); "
                f"got {prev_end:.1f}.")
    return None


def _coerce_call2_shot(raw: Dict[str, Any], idx: int,
                       lyrics_timed: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        st  = float(raw.get("start_time") or 0)
        en  = float(raw.get("end_time")   or 0)
        dur = float(raw.get("duration")   or (en - st))
    except (TypeError, ValueError):
        st, en, dur = 0.0, 0.0, 0.0
    # Compute lyric_text from timed lyrics that fall in this shot's window
    lyric_text = (raw.get("lyric_text") or "").strip()
    if not lyric_text and lyrics_timed:
        parts: List[str] = []
        for ly in lyrics_timed:
            try:
                ls = float(ly.get("start") or ly.get("start_time") or 0)
                le = float(ly.get("end")   or ly.get("end_time")   or 0)
            except (TypeError, ValueError):
                continue
            if ls >= st - 0.1 and le <= en + 0.1:
                t = (ly.get("text") or ly.get("line") or "").strip()
                if t:
                    parts.append(t)
        lyric_text = " / ".join(parts)
    return {
        "shot_id":       str(raw.get("shot_id") or f"shot_{idx + 1}").strip()[:32],
        "scene_id":      str(raw.get("scene_id") or "").strip()[:24],
        "start_time":    round(st, 3),
        "end_time":      round(en, 3),
        "duration":      int(round(dur)),
        "lyric_text":    lyric_text[:600],
        "action_intent": str(raw.get("action_intent") or "").strip()[:400],
        "actions":       [],   # filled by Call 3
    }


async def _run_call2(client: AsyncOpenAI, scenes: List[Dict[str, Any]],
                     audio_duration: float, audio_data: Dict[str, Any],
                     lyrics_timed: List[Dict[str, Any]], story: Dict[str, Any]
                     ) -> Optional[List[Dict[str, Any]]]:
    base_user = _call2_user_prompt(scenes, audio_duration, audio_data,
                                   lyrics_timed, story)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _call2_system_prompt()},
        {"role": "user",   "content": base_user},
    ]
    for attempt in range(_MAX_RETRIES_PER_CALL + 1):
        try:
            resp = await client.chat.completions.create(
                model=_MODEL,
                response_format={"type": "json_object"},
                temperature=0.7 if attempt == 0 else 0.4,
                messages=messages,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            _repair_call2_shots(data, scenes, audio_duration)
            err = _validate_call2(data, scenes, audio_duration)
            if err is None:
                shots_in = data.get("shots") or []
                return [_coerce_call2_shot(sh, i, lyrics_timed)
                        for i, sh in enumerate(shots_in)]
            logger.warning("StoryboardV3 Call 2 attempt %d failed validation: %s",
                           attempt + 1, err)
            if attempt < _MAX_RETRIES_PER_CALL:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": (f"Your previous output failed: {err}. "
                                             "Fix it and return ONLY the corrected JSON.")})
        except Exception as exc:
            logger.exception("StoryboardV3 Call 2 attempt %d raised: %s", attempt + 1, exc)
    return None


# =====================================================================
# CALL 3 — Multishot expansion (only on shots > 8s)
# =====================================================================
def _call3_system_prompt() -> str:
    return (
        "You are the ACTION CHOREOGRAPHER for Qaivid — Call 3 of 3.\n\n"
        "ROLE: For each long shot you receive, design a sequence of "
        "sub-actions ('multishot') that ALL share the SAME starting frame "
        "(Frame 0). Each sub-action is a continuous physical action that "
        "flows naturally from the previous one — the character / camera / "
        "world stays continuous; only the action progresses.\n\n"
        "RULES:\n"
        "1. Sub-action count is determined by shot duration (NOT your choice):\n"
        f"   • duration ∈ ({_MULTISHOT_THRESHOLD}, 10]  → exactly 2 sub-actions\n"
        "   • duration ∈ (10, 12]                  → exactly 3 sub-actions\n"
        "   • duration ∈ (12, 15]                  → 3 or 4 sub-actions\n"
        "2. Sub-action durations MUST sum to the shot's total duration.\n"
        "3. All sub-actions share the SAME Frame 0 image — they must be "
        "   physically achievable starting from that one still frame, in a "
        "   continuous performance (no cuts, no scene changes).\n"
        "4. Sub-actions must be DIFFERENT from each other (each one a new "
        "   beat in the choreography) but in CONTINUITY (each one begins "
        "   where the previous ended).\n"
        "5. action_text is ONE clear sentence describing the cinematic "
        "   action. Verb-driven. No camera-language jargon ('dolly in', "
        "   'tilt up') — describe what the SUBJECT does.\n\n"
        "Return strict JSON. No prose outside the JSON object."
    )


def _call3_user_prompt(long_shots: List[Dict[str, Any]],
                       scenes_by_id: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for sh in long_shots:
        sc = scenes_by_id.get(sh.get("scene_id") or "", {})
        rows.append({
            "shot_id":       sh["shot_id"],
            "duration":      sh["duration"],
            "action_intent": sh.get("action_intent"),
            "scene_purpose": sc.get("purpose"),
            "scene_phase":   sc.get("narrative_phase"),
            "lyric_text":    sh.get("lyric_text"),
        })
    return (
        f"LONG SHOTS TO EXPAND (each duration > {_MULTISHOT_THRESHOLD}s):\n"
        + json.dumps(rows, indent=2, ensure_ascii=False)
        + "\n\nReturn JSON of exactly this shape:\n"
          "{\n"
          '  "multishot_expansions": [\n'
          "    {\n"
          '      "shot_id": "shot_1",\n'
          '      "actions": [\n'
          '        {"order": 1, "duration": 4, "action_text": "She turns toward the window, lifting her dupatta over her shoulder."},\n'
          '        {"order": 2, "duration": 3, "action_text": "She steps forward and rests her hand on the windowsill."}\n'
          "      ]\n"
          "    }\n"
          "  ]\n"
          "}\n\n"
          "Sub-action durations MUST sum to the shot's total duration. "
          "All sub-actions share the same Frame 0 — design them as one "
          "continuous performance from a single starting still."
    )


def _expected_subaction_count(duration: int) -> int:
    if duration <= _MULTISHOT_THRESHOLD:
        return 1
    if duration <= 10:
        return 2
    if duration <= 12:
        return 3
    return 3   # 12 < d ≤ 15 → minimum 3 (LLM may produce 4)


def _expected_subaction_range(dur: int) -> Tuple[int, int]:
    """Return (min_count, max_count) of sub-actions for a shot duration.

    Spec:  d ∈ (8, 10]  → exactly 2
           d ∈ (10, 12] → exactly 3
           d ∈ (12, 15] → 3 or 4
    """
    if dur <= _MULTISHOT_THRESHOLD:
        return (0, 0)            # not multishot
    if dur <= 10:
        return (2, 2)
    if dur <= 12:
        return (3, 3)
    return (3, 4)                # 12 < d ≤ 15


def _validate_call3(data: Dict[str, Any],
                    long_shots: List[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(data, dict):
        return "Output is not a JSON object."
    expansions = data.get("multishot_expansions")
    if not isinstance(expansions, list):
        return "multishot_expansions must be an array."
    by_id = {e.get("shot_id"): e for e in expansions if isinstance(e, dict)}
    for sh in long_shots:
        ex = by_id.get(sh["shot_id"])
        if not ex:
            return f"missing expansion for shot_id {sh['shot_id']}."
        actions = ex.get("actions")
        if not isinstance(actions, list) or not actions:
            return f"shot {sh['shot_id']} actions must be a non-empty array."
        lo, hi = _expected_subaction_range(int(sh["duration"]))
        if len(actions) < lo:
            return (f"shot {sh['shot_id']} has {len(actions)} sub-actions; "
                    f"expected {lo}{('-' + str(hi)) if hi != lo else ''} "
                    f"for duration {sh['duration']}.")
        if len(actions) > hi:
            return (f"shot {sh['shot_id']} has {len(actions)} sub-actions; "
                    f"max allowed is {hi} for duration {sh['duration']}.")
        try:
            tot = sum(float(a.get("duration") or 0) for a in actions)
        except (TypeError, ValueError):
            return f"shot {sh['shot_id']} has non-numeric sub-action duration."
        if abs(tot - float(sh["duration"])) > 0.5:
            return (f"shot {sh['shot_id']} sub-action durations sum to {tot}, "
                    f"expected {sh['duration']}.")
    return None


def _repair_call3_expansions(data: Dict[str, Any],
                             long_shots: List[Dict[str, Any]]) -> None:
    """Mutate Call 3 output to absorb LLM imprecision: split too-few
    sub-actions, merge too-many, and rescale durations to sum to the
    parent shot's duration."""
    if not isinstance(data, dict):
        return
    expansions = data.get("multishot_expansions")
    if not isinstance(expansions, list):
        return
    by_id = {e.get("shot_id"): e
             for e in expansions if isinstance(e, dict)}

    for sh in long_shots:
        ex = by_id.get(sh["shot_id"])
        if not isinstance(ex, dict):
            continue
        actions = ex.get("actions")
        if not isinstance(actions, list):
            continue
        target_dur = int(sh["duration"])
        lo, hi = _expected_subaction_range(target_dur)
        target_n = lo

        # Split the longest action until we have enough sub-actions
        while len(actions) < target_n:
            longest_i = max(range(len(actions)),
                            key=lambda k: float(actions[k].get("duration") or 0))
            longest = actions[longest_i]
            ld = float(longest.get("duration") or 0)
            half = max(1.0, ld / 2.0)
            longest["duration"] = half
            new_act = {
                "order": longest_i + 2,
                "duration": ld - half,
                "action_text": (str(longest.get("action_text") or "") + " (continued)"),
            }
            actions.insert(longest_i + 1, new_act)

        # Enforce strict max: validator now rejects > hi.  Merge shortest
        # into its neighbour until count is in [lo, hi].
        while len(actions) > hi:
            shortest_i = min(range(len(actions)),
                             key=lambda k: float(actions[k].get("duration") or 0))
            sh_a = actions[shortest_i]
            # merge into neighbour (right if exists, else left)
            nbr_i = shortest_i + 1 if shortest_i + 1 < len(actions) else shortest_i - 1
            nbr = actions[nbr_i]
            nbr["duration"] = float(nbr.get("duration") or 0) + float(sh_a.get("duration") or 0)
            actions.pop(shortest_i)

        # Rescale durations to sum exactly to target_dur (integer rounding)
        cur_total = sum(float(a.get("duration") or 0) for a in actions)
        if cur_total <= 0:
            # fallback: equal split
            seg = max(1, target_dur // len(actions))
            for a in actions:
                a["duration"] = seg
            cur_total = seg * len(actions)
        if abs(cur_total - target_dur) > 0.01:
            scale = target_dur / cur_total
            scaled = [max(1.0, float(a.get("duration") or 0) * scale) for a in actions]
            # round to ints, then fix the residual on the last action
            ints = [int(round(x)) for x in scaled]
            diff = target_dur - sum(ints)
            if ints:
                ints[-1] += diff
                if ints[-1] < 1:
                    # spread negative residual over earlier actions
                    deficit = 1 - ints[-1]
                    ints[-1] = 1
                    for k in range(len(ints) - 2, -1, -1):
                        take = min(deficit, ints[k] - 1)
                        ints[k] -= take
                        deficit -= take
                        if deficit <= 0:
                            break
            for a, n in zip(actions, ints):
                a["duration"] = n

        # Reorder
        for i, a in enumerate(actions):
            a["order"] = i + 1


def _coerce_call3_action(raw: Any, order: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"order": order, "duration": 0, "action_text": ""}
    try:
        d = float(raw.get("duration") or 0)
    except (TypeError, ValueError):
        d = 0.0
    return {
        "order":       int(raw.get("order") or order),
        "duration":    int(round(d)),
        "action_text": str(raw.get("action_text") or "").strip()[:400],
    }


async def _run_call3(client: AsyncOpenAI, shots: List[Dict[str, Any]],
                     scenes: List[Dict[str, Any]]
                     ) -> List[Dict[str, Any]]:
    """Mutates shots in place — adds actions[] to shots > MULTISHOT_THRESHOLD.
    Returns the (same) shots list. If Call 3 fails, long shots get an
    auto-split fallback so multishot is still populated."""
    long_shots = [sh for sh in shots if sh["duration"] > _MULTISHOT_THRESHOLD]
    if not long_shots:
        return shots

    scenes_by_id = {s["scene_id"]: s for s in scenes}
    base_user = _call3_user_prompt(long_shots, scenes_by_id)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _call3_system_prompt()},
        {"role": "user",   "content": base_user},
    ]

    for attempt in range(_MAX_RETRIES_PER_CALL + 1):
        try:
            resp = await client.chat.completions.create(
                model=_MODEL,
                response_format={"type": "json_object"},
                temperature=0.7 if attempt == 0 else 0.4,
                messages=messages,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            _repair_call3_expansions(data, long_shots)
            err = _validate_call3(data, long_shots)
            if err is None:
                expansions_by_id = {e["shot_id"]: e
                                    for e in data["multishot_expansions"]}
                for sh in long_shots:
                    actions_in = expansions_by_id[sh["shot_id"]].get("actions") or []
                    sh["actions"] = [_coerce_call3_action(a, i + 1)
                                     for i, a in enumerate(actions_in)]
                return shots
            logger.warning("StoryboardV3 Call 3 attempt %d failed validation: %s",
                           attempt + 1, err)
            if attempt < _MAX_RETRIES_PER_CALL:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": (f"Your previous output failed: {err}. "
                                             "Fix it and return ONLY the corrected JSON.")})
        except Exception as exc:
            logger.exception("StoryboardV3 Call 3 attempt %d raised: %s", attempt + 1, exc)

    # Fallback: auto-split each long shot into evenly-sized sub-actions
    # using a generic action_text derived from the action_intent.
    logger.warning("StoryboardV3 Call 3 exhausted retries — using auto-split fallback for %d long shots",
                   len(long_shots))
    for sh in long_shots:
        n = _expected_subaction_count(int(sh["duration"]))
        seg = max(1, int(sh["duration"]) // n)
        intent = sh.get("action_intent") or "the subject continues their action"
        sh["actions"] = [
            {"order": i + 1, "duration": seg,
             "action_text": f"{intent} (beat {i + 1} of {n})"}
            for i in range(n)
        ]
        # Adjust last segment so durations sum to total
        total = sum(a["duration"] for a in sh["actions"])
        if total != int(sh["duration"]) and sh["actions"]:
            sh["actions"][-1]["duration"] += int(sh["duration"]) - total
    return shots


# =====================================================================
# Deterministic fallback (when Call 1 or 2 fail outright)
# =====================================================================
def _fallback_storyboard(input_structure: Dict[str, Any],
                         audio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Same shape as a successful v3 packet — produced from sections only.
    Used when LLM calls fail. Each scene gets one shot per ~6 seconds."""
    sections = (input_structure or {}).get("sections") or []
    units    = (input_structure or {}).get("units") or []
    audio_dur = float((audio_data or {}).get("duration_seconds") or 0)

    # Build scene list — one per section, time windows derived from units
    scenes: List[Dict[str, Any]] = []
    cursor = 0.0
    n = max(len(sections), 1)
    for i, sec in enumerate(sections or [{"id": "all", "label": "Scene"}]):
        sid = f"s{i + 1}"
        u_ids = sec.get("unit_ids") or []
        starts: List[float] = []
        ends:   List[float] = []
        for u in units:
            if isinstance(u, dict) and u.get("id") in u_ids:
                if u.get("start_time") is not None:
                    starts.append(float(u["start_time"]))
                if u.get("end_time") is not None:
                    ends.append(float(u["end_time"]))
        if starts and ends:
            ws, we = min(starts), max(ends)
        elif audio_dur > 0:
            step = audio_dur / n
            ws, we = i * step, (i + 1) * step
        else:
            ws, we = cursor, cursor + 8.0
        # Patch first scene to start at 0 and last to end at audio_dur
        if i == 0:
            ws = 0.0
        if i == n - 1 and audio_dur > 0:
            we = audio_dur
        cursor = we
        progress = i / max(n - 1, 1) if n > 1 else 0
        if   progress < 0.20: phase = "intro"
        elif progress < 0.55: phase = "build"
        elif progress < 0.75: phase = "peak"
        elif progress < 0.90: phase = "breakdown"
        else:                 phase = "resolution"
        scenes.append({
            "scene_id":            sid,
            "source_section":      str(sec.get("label") or sec.get("type") or sid)[:48],
            "narrative_phase":     phase,
            "purpose":             "Express the emotional intent of this section.",
            "emotional_intensity": "medium",
            "presence_hint":       "full",
            "motion_density":      "medium",
            "timeline_position":   "present",
            "time_window_start":   round(ws, 3),
            "time_window_end":     round(we, 3),
            "motif_usage":         [],
            "continuity_hooks":    {"subject": "same speaker", "motifs": []},
            "valid_realizations":  [
                "subject in the primary setting expressing the emotion directly",
                "environment-focused beat with the subject partially present",
                "object or motif beat that echoes the subject's inner state",
            ],
        })

    # Build flat shot list — divide each scene into ~6s shots
    shots: List[Dict[str, Any]] = []
    shot_idx = 0
    for sc in scenes:
        scene_dur = sc["time_window_end"] - sc["time_window_start"]
        n_shots = max(1, int(round(scene_dur / 6.0)))
        seg = scene_dur / n_shots
        for k in range(n_shots):
            st = sc["time_window_start"] + k * seg
            en = sc["time_window_start"] + (k + 1) * seg if k < n_shots - 1 \
                 else sc["time_window_end"]
            dur = max(_MIN_SHOT_DURATION,
                      min(_MAX_SHOT_DURATION, int(round(en - st))))
            shot_idx += 1
            shots.append({
                "shot_id":       f"shot_{shot_idx}",
                "scene_id":      sc["scene_id"],
                "start_time":    round(st, 3),
                "end_time":      round(en, 3),
                "duration":      dur,
                "lyric_text":    "",
                "action_intent": f"the subject expresses the scene's purpose ({sc['purpose']})",
                "actions":       [],
            })

    return {
        "story": {
            "arc": "fallback — no LLM-derived story available",
            "summary": "",
            "central_conflict": "",
        },
        "scenes": scenes,
        "shots":  shots,
    }


# =====================================================================
# Public API
# =====================================================================
async def generate_storyboard_v3(
    api_key: str,
    audio_data: Dict[str, Any],
    input_structure: Dict[str, Any],
    lyrics_timed: List[Dict[str, Any]],
    context_packet: Dict[str, Any],
    narrative_packet: Optional[Dict[str, Any]] = None,
    style_profile: Optional[Dict[str, Any]] = None,
    project_settings: Optional[Dict[str, Any]] = None,
    emotional_mode_packet: Optional[Dict[str, Any]] = None,
    imagination_packet: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Three-call storyboard pipeline.

    Returns (storyboard_packet_v3, used_fallback).  Never raises — falls
    back to a deterministic per-section storyboard if either Call 1 or
    Call 2 exhausts its retries.
    """
    style_profile         = style_profile         or {}
    project_settings      = project_settings      or {}
    emotional_mode_packet = emotional_mode_packet or {}
    imagination_packet    = imagination_packet    or {}
    audio_duration = float((audio_data or {}).get("duration_seconds") or 0)
    style_preset = style_profile.get("preset") or "cinematic_natural"

    client = AsyncOpenAI(api_key=api_key)

    # ── Call 1 ─────────────────────────────────────────────────────────
    story, scenes = await _run_call1(
        client, audio_data, input_structure, lyrics_timed,
        context_packet, narrative_packet or {}, style_profile,
        project_settings, emotional_mode_packet, imagination_packet,
        audio_duration,
    )
    if not scenes:
        logger.warning("StoryboardV3: Call 1 failed — using fallback")
        packet = _fallback_storyboard(input_structure, audio_data)
        packet.update({
            "schema_version": 3,
            "used_fallback_v3": True,
            "style_preset":     style_preset,
        })
        return packet, True
    logger.info("StoryboardV3 Call 1 OK: %d scenes spanning %.1fs",
                len(scenes), audio_duration)

    # ── Call 2 ─────────────────────────────────────────────────────────
    shots = await _run_call2(client, scenes, audio_duration,
                             audio_data, lyrics_timed, story or {})
    if not shots:
        logger.warning("StoryboardV3: Call 2 failed — using fallback shots within Call 1 scenes")
        # Call 1 succeeded; build deterministic shots within those scenes
        fb = _fallback_storyboard(input_structure, audio_data)
        # Replace fb scenes with our LLM-derived ones, regenerate shots within them
        shot_idx = 0
        shots = []
        for sc in scenes:
            scene_dur = sc["time_window_end"] - sc["time_window_start"]
            n_shots = max(1, int(round(scene_dur / 6.0)))
            seg = scene_dur / n_shots
            for k in range(n_shots):
                st = sc["time_window_start"] + k * seg
                en = sc["time_window_start"] + (k + 1) * seg if k < n_shots - 1 \
                     else sc["time_window_end"]
                dur = max(_MIN_SHOT_DURATION,
                          min(_MAX_SHOT_DURATION, int(round(en - st))))
                shot_idx += 1
                shots.append({
                    "shot_id":       f"shot_{shot_idx}",
                    "scene_id":      sc["scene_id"],
                    "start_time":    round(st, 3),
                    "end_time":      round(en, 3),
                    "duration":      dur,
                    "lyric_text":    "",
                    "action_intent": f"the subject expresses the scene's purpose ({sc['purpose']})",
                    "actions":       [],
                })
        used_fallback = True
    else:
        used_fallback = False
        logger.info("StoryboardV3 Call 2 OK: %d shots (max duration %ds)",
                    len(shots), max((s["duration"] for s in shots), default=0))

    # ── Call 3 ─────────────────────────────────────────────────────────
    shots = await _run_call3(client, shots, scenes)
    n_multi = sum(1 for s in shots if s.get("actions"))
    logger.info("StoryboardV3 Call 3 OK: %d multishots out of %d shots",
                n_multi, len(shots))

    return ({
        "schema_version":   3,
        "story":            story or {},
        "scenes":           scenes,
        "shots":            shots,
        "used_fallback_v3": used_fallback,
        "style_preset":     style_preset,
    }, used_fallback)


# =====================================================================
# CLI test runner — load a real project's brain inputs and run the engine
# =====================================================================
def _cli_main() -> None:
    """Usage: python storyboard_engine_v3.py <project_id>

    Loads the named project's brain inputs from the database and runs
    Storyboard V3 against them, then prints the output as JSON. Does
    NOT modify any database state. For visual review only.
    """
    import asyncio
    import os
    import sys
    import psycopg
    from psycopg.rows import dict_row

    if len(sys.argv) < 2:
        print("usage: python storyboard_engine_v3.py <project_id>", file=sys.stderr)
        sys.exit(2)
    pid = sys.argv[1]

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    conn = psycopg.connect(db_url, row_factory=dict_row)
    try:
        from project_brain import ProjectBrain
        brain = ProjectBrain.load(pid, conn)
        with conn.cursor() as cur:
            cur.execute("SELECT audio_data, lyrics_timed FROM projects WHERE id=%s", (pid,))
            row = cur.fetchone() or {}
        audio_data    = dict(row.get("audio_data") or {})
        audio_data.pop("_pre_analysis", None)
        lyrics_timed  = list(row.get("lyrics_timed") or [])
        ctx           = brain.read("context_packet")    or {}
        narrative     = brain.read("narrative_packet")  or {}
        style_profile = brain.read("style_packet")      or {}
        input_struct  = brain.read("input_structure")   or {}
        settings      = brain.read("project_settings")  or {}
        emotional     = brain.read("emotional_mode_packet") if brain.is_populated("emotional_mode_packet") else {}
        imagination   = brain.read("imagination_packet")   if brain.is_populated("imagination_packet")   else {}
    finally:
        conn.close()

    packet, used_fallback = asyncio.run(generate_storyboard_v3(
        api_key=api_key,
        audio_data=audio_data,
        input_structure=input_struct,
        lyrics_timed=lyrics_timed,
        context_packet=ctx,
        narrative_packet=narrative,
        style_profile=style_profile,
        project_settings=settings,
        emotional_mode_packet=emotional,
        imagination_packet=imagination,
    ))

    print(json.dumps({
        "used_fallback": used_fallback,
        "scene_count":   len(packet.get("scenes") or []),
        "shot_count":    len(packet.get("shots") or []),
        "multishot_count": sum(1 for s in (packet.get("shots") or []) if s.get("actions")),
        "max_shot_duration": max((s["duration"] for s in (packet.get("shots") or [])), default=0),
        "audio_duration":    float(audio_data.get("duration_seconds") or 0),
        "story":             packet.get("story"),
        "scenes":            packet.get("scenes"),
        "shots":             packet.get("shots"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli_main()
