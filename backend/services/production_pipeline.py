"""
Production Pipeline Services
Context-to-Visual translation layer.

The context engine extracts rich structured intelligence (speaker_model,
entity_map, cultural_setting, line_meanings, etc.). This module is the
bridge that turns that intelligence into concrete, visually grounded
image and video prompts.

Rule: SUBJECT FIRST. Every still prompt must name WHO is in the shot
before describing WHAT they're doing, WHERE they are, and HOW it's lit.
Abstract filmmaker language (held breath, ambient quiet conveying X)
never reaches the image model — it gets translated into visual posture.
"""
from typing import Dict, List, Any, Optional
import uuid
from models import now_utc


# ─── Visual Translation Helpers ───────────────────────────

def _derive_visual_subject(
    context_packet: Dict[str, Any],
    creative_brief: Dict[str, Any],
) -> str:
    """
    Build a concrete, culturally-specific visual character description.
    Priority: brief cast/characters → speaker_model + cultural_setting.
    """
    brief = creative_brief or {}

    # 1. Try brief cast
    cast = brief.get("cast", []) or brief.get("characters", [])
    if cast:
        c = cast[0]
        desc = c.get("physical_description") or c.get("physicalDescription") or c.get("appearance", "")
        name = c.get("name", "the protagonist")
        if desc:
            return f"{name} — {desc}"
        return name

    # 2. Derive from speaker_model + cultural_setting
    speaker = context_packet.get("speaker_model", {})
    cultural = context_packet.get("cultural_setting", {})
    pack = cultural.get("culture_pack", "").lower()

    gender = speaker.get("gender", "")
    age_range = speaker.get("age_range", "")

    age_map = {
        "young":   "in her early 20s",
        "middle":  "in her mid-30s",
        "old":     "in her 50s",
        "elderly": "in her 60s-70s",
    }
    age_str = age_map.get(age_range, "")

    # Cultural placement only — appearance is owned by the Character Materializer.
    # Do not prescribe complexion, wardrobe, or grooming here.
    gender_word = "woman" if gender == "female" else "man" if gender == "male" else "person"
    age_clause = f" {age_str}" if age_str else ""
    return f"a {gender_word}{age_clause}"


def _translate_action_to_visual(
    subject_action: str,
    emotional_state: str,
    expression_mode: str,
    scene_objects: List[str],
) -> str:
    """
    Convert abstract filmmaker/director notes into concrete visual descriptions
    that an image model can render.

    Director writes: "stillness in courtyard — held breath, no movement, ambient quiet conveying resignation"
    Visual output: "sitting alone, hands folded in lap, gaze lowered, utterly still"
    """
    emotional_visual = {
        "resignation":    "sitting alone, hands folded in her lap, gaze lowered, utterly still",
        "longing":        "gazing into the middle distance, one hand resting near her chest, lost in thought",
        "grief":          "head slightly bowed, shoulders carrying quiet weight, eyes glistening",
        "anger":          "standing rigidly, jaw set, hands clenched at her sides",
        "joy":            "face lifted upward, soft unguarded smile, warmth in her eyes",
        "nostalgia":      "cradling something in both hands, gaze soft and inward, lost in memory",
        "hope":           "face turned toward a window or horizon, posture lifting slightly",
        "solitude":       "alone in a wide empty space, small against the surroundings",
        "despair":        "seated on the ground, back against a wall, knees drawn up, head bowed",
        "yearning":       "reaching toward something just beyond the frame, gesture suspended",
        "melancholy":     "seated quietly, hands resting open in lap, eyes unfocused",
        "sorrow":         "seated quietly, one hand pressed to her chest, eyes closed briefly",
        "contemplation":  "still and upright, eyes focused on something small and meaningful",
        "peace":          "seated with quiet composure, hands resting, breathing visible",
        "absence":        "standing in an empty room, the space around her amplified",
        "numbness":       "seated motionless, eyes open but seeing nothing, utterly hollow",
        "anguish":        "doubled slightly inward, one hand pressed to her mouth, silent",
        "pride":          "standing tall, chin lifted, composure held despite inner weight",
        "regret":         "looking down at her hands, still, as if reading something written there",
        "wistfulness":    "half-turned toward a window, face soft, caught between past and present",
        "devastation":    "sitting very still, gaze fixed on nothing, world collapsed inward",
        "acceptance":     "seated upright, hands open in lap, quiet resolve on her face",
        "loneliness":     "small figure in a large empty space, surroundings dwarfing her",
        "love":           "hands cradling something precious, face tender and open",
        "pain":           "shoulders drawn in, arms wrapped around herself, enduring quietly",
        "loss":           "looking at an empty space — a chair, a door — where someone used to be",
        "remembrance":    "eyes closed, face tilted slightly upward, holding a memory",
        "waiting":        "seated facing a door or window, very still, listening",
        "ache":           "seated still, hand resting on her chest, inward weight visible on her face",
        "held emotion":   "expression tightly contained, a tremble at the edge of composure",
        "suppressed":     "body taut, expression held — grief contained but unmistakable",
        "inner conflict": "mid-gesture, arrested — caught between two impulses, face conflicted",
        "quiet":          "seated, hands still, face composed and inward",
        "stillness":      "seated motionless, breath slow, a figure carved from the moment",
        "burden":         "shoulders slightly bowed under invisible weight, gaze heavy",
        "isolation":      "lone figure in a wide empty space, surrounded by silence",
        "tender":         "hands gentle, expression open and unguarded, soft light on her face",
        "heartbreak":     "seated, arms wrapped around herself, face turned slightly away",
        "release":        "eyes closed, a long exhale — something let go, posture softening",
    }

    expression_visual = {
        "silence":         "the stillness itself carrying weight, no gesture breaking it",
        "face":            "her expression the entire story, eyes doing what words cannot",
        "body_posture":    "her posture and body language conveying everything",
        "environment":     "the environment pressing around her, she almost disappears into it",
        "object":          "her attention fixed on a meaningful object she holds or touches",
        "memory_warmth":   "warm soft light falling across her, as if memory itself is the light source",
        "absence":         "empty space framing her, the silence of what is missing filling the shot",
    }

    # Find matching emotional visual
    visual_action = ""
    em_lower = emotional_state.lower()
    for key, visual in emotional_visual.items():
        if key in em_lower:
            visual_action = visual
            break

    if not visual_action:
        # Strip camera-direction language — these describe what the camera does,
        # not what the character does, and must not reach the image model
        camera_prefixes = [
            "close on subject's face", "close on", "cut to", "wide on",
            "we see", "camera finds", "shot of",
            "no person in frame", "no theatrical crying unless intensity is HIGH",
            "no theatrical", "small involuntary movement (a swallow, a blink)",
            "small involuntary movement", "(a swallow, a blink)",
        ]
        cleaned = subject_action
        for prefix in camera_prefixes:
            cleaned = cleaned.replace(prefix, " ")

        # Strip abstract filmmaker language that an image model can't render
        for filler in [
            "conveying", "ambient quiet", "held breath", "no movement",
            "—", " — ", "  ", "cinematic", "symbolic", "narrative tension",
            "character is", "subject is",
        ]:
            cleaned = cleaned.replace(filler, " ")

        cleaned = " ".join(cleaned.split()).strip(" ;,.-")

        # If what's left is still a location description (not an action), fall back to default
        location_words = {
            "courtyard", "room", "home", "field", "rooftop", "interior",
            "exterior", "or", "single-room", "single", "village",
        }
        is_just_location = all(
            w.lower() in location_words or len(w) <= 3
            for w in cleaned.split() if w
        )
        if is_just_location or len(cleaned) < 10:
            # Fall back to emotion-derived default
            visual_action = "seated in quiet contemplation, hands in lap, gaze lowered"
        else:
            visual_action = cleaned

    # Layer in expression mode
    expr_layer = expression_visual.get(expression_mode, "")
    if expr_layer and expr_layer not in visual_action:
        visual_action = f"{visual_action}, {expr_layer}"

    # Inject meaningful scene objects naturally
    meaningful_objects = [o for o in (scene_objects or []) if o not in (
        "roof beams", "ceiling", "walls", "floor"
    )][:2]
    if meaningful_objects:
        visual_action = f"{visual_action}, {', '.join(meaningful_objects)} nearby"

    return visual_action


def _build_cultural_aesthetic(context_packet: Dict[str, Any], creative_brief: Dict[str, Any]) -> str:
    """Derive a visual aesthetic string from cultural context + brief."""
    brief = creative_brief or {}
    brief_style = brief.get("visual_aesthetic", {}) or {}
    if brief_style.get("style"):
        return brief_style["style"]

    cultural = context_packet.get("cultural_setting", {})
    pack = cultural.get("culture_pack", "").lower()

    if "punjabi" in pack:
        return (
            "cinematic realism, Punjabi cultural aesthetics, warm amber and deep blue tones, "
            "dignified and emotionally grounded, not poverty tourism, film grain, 35mm feel"
        )
    elif any(x in pack for x in ["urdu", "desi", "south_asian"]):
        return "cinematic realism, South Asian aesthetics, warm cinematic tones, dignified"
    return "cinematic realism, naturalistic lighting, film grain"


# ─── Reference Image Prompts ──────────────────────────────

def build_reference_prompts(
    creative_brief: Dict[str, Any],
    context_packet: Dict[str, Any],
    vibe_preset: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build character + environment reference prompts.
    Falls back to deriving characters from speaker_model + cultural_setting
    when the creative brief has no cast.
    """
    brief = creative_brief or {}
    prompts = []
    restrictions = context_packet.get("restrictions", [])
    cultural = context_packet.get("cultural_setting", {})
    pack = cultural.get("culture_pack", "").lower()
    pack_restrictions = cultural.get("restrictions", [])
    entity_map = context_packet.get("entity_map", {})

    all_restrictions = restrictions + pack_restrictions
    avoid_clause = f". Avoid: {', '.join(all_restrictions)}" if all_restrictions else ""

    vibe_ref_dir = ""
    if vibe_preset:
        vibe_ref_dir = f". Style direction: {vibe_preset.get('reference_direction', '')}"

    # ── Characters ──────────────────────────────────────────
    characters = brief.get("characters", []) or brief.get("cast", [])

    if not characters:
        # Derive from speaker_model + cultural context
        speaker = context_packet.get("speaker_model", {})
        gender = speaker.get("gender", "")
        age_range = speaker.get("age_range", "")
        role = speaker.get("social_role", "protagonist")
        emotional = speaker.get("emotional_state", "sorrowful")

        age_map = {"young": "early 20s", "middle": "mid-30s", "old": "50s", "elderly": "60s"}
        age_str = age_map.get(age_range, "30s")

        # Cultural placement only — complexion, wardrobe, and grooming are
        # owned by the Character Materializer. Do not prescribe them here.
        gender_word = "woman" if gender == "female" else "man" if gender == "male" else "person"
        appearance = f"{gender_word}, {age_str}" if age_str else gender_word
        wardrobe = ""

        characters = [{
            "name": "The Speaker",
            "role": role,
            "physical_description": appearance,
            "wardrobe": wardrobe,
            "emotional_note": emotional,
        }]

        # Add the addressee if entity_map shows two characters
        entity_chars = entity_map.get("characters", [])
        if len(entity_chars) > 1:
            addressee_gender = "male" if gender == "female" else "female"
            addr_appearance = f"{addressee_gender}, {age_str}" if age_str else addressee_gender

            characters.append({
                "name": "The Beloved",
                "role": "departed lover — seen in memory, not in present",
                "physical_description": addr_appearance,
                "wardrobe": "",
                "emotional_note": "idealized in memory, present only as feeling",
            })

    for char in characters:
        desc = (
            char.get("physical_description")
            or char.get("physicalDescription")
            or char.get("appearance", "")
        )
        prompt_text = (
            f"Character reference portrait: {char.get('name', 'Character')}, {char.get('role', '')}. "
            f"{desc}. "
            f"Wardrobe: {char.get('wardrobe', '')}. "
            f"Full-body reference shot. Clear, consistent facial features. "
            f"Cinematic lighting, dignified framing. No background clutter."
            f"{vibe_ref_dir}{avoid_clause}"
        )
        prompts.append({
            "id": str(uuid.uuid4()),
            "type": "character",
            "name": char.get("name", "Character"),
            "role": char.get("role", ""),
            "prompt": prompt_text.strip(),
            "status": "pending",
            "image_url": None,
        })

    # ── Environments ────────────────────────────────────────
    locations = brief.get("locations", [])

    if not locations:
        entity_locs = entity_map.get("locations", [])
        narrative_mode = context_packet.get("narrative_mode", "symbolic")

        for loc_name in entity_locs[:3]:
            loc_style = "cinematic, atmospheric, faithful to the song's cultural world"

            locations.append({
                "name": loc_name.title(),
                "description": f"A {loc_name} in the song's cultural world.",
                "visual_details": loc_style,
                "time_of_day": "as determined by song context",
                "mood": context_packet.get("core_theme", "emotionally resonant"),
            })

    for loc in locations:
        prompt_text = (
            f"Cinematic environment reference: {loc.get('name', 'Location')}. "
            f"{loc.get('description', '')} "
            f"{loc.get('visual_details', loc.get('visualDetails', ''))}. "
            f"Time of day: {loc.get('time_of_day', loc.get('timeOfDay', 'golden hour'))}. "
            f"Mood: {loc.get('mood', 'melancholic')}. "
            f"Wide establishing shot. No people. Atmospheric, cinematic, emotionally resonant."
            f"{vibe_ref_dir}{avoid_clause}"
        )
        prompts.append({
            "id": str(uuid.uuid4()),
            "type": "environment",
            "name": loc.get("name", "Location"),
            "prompt": prompt_text.strip(),
            "status": "pending",
            "image_url": None,
        })

    return prompts


# ─── Shot Still Prompts ───────────────────────────────────

def build_still_prompts(
    shots: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    context_packet: Dict[str, Any],
    creative_brief: Dict[str, Any],
    vibe_preset: Optional[Dict[str, Any]] = None,
    reference_images: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build shot still prompts — SUBJECT FIRST, then visual action, setting, light, style.

    The context engine's speaker_model, entity_map, and cultural_setting
    are the primary sources of truth. Abstract filmmaker language in
    subject_action is translated into concrete visual descriptions.
    """
    scene_map = {s["id"]: s for s in scenes}
    cultural = context_packet.get("cultural_setting", {})
    palette = cultural.get("visual_palette", [])
    entity_map = context_packet.get("entity_map", {})
    scene_objects = entity_map.get("objects", [])
    entity_locations = entity_map.get("locations", [])

    restrictions = (
        context_packet.get("restrictions", [])
        + cultural.get("restrictions", [])
    )

    # Build visual subject ONCE — reused across all shots
    visual_subject = _derive_visual_subject(context_packet, creative_brief or {})

    # Aesthetic from brief or cultural pack
    aesthetic = _build_cultural_aesthetic(context_packet, creative_brief or {})

    vibe_shot_dir = vibe_preset.get("shot_direction", "") if vibe_preset else ""
    vibe_avoid = vibe_preset.get("avoid", []) if vibe_preset else []

    # Collect all completed reference image IDs
    ref_ids = []
    if reference_images:
        for ref in reference_images:
            if ref.get("image_url") and ref.get("status") == "completed":
                ref_ids.append(ref["id"])

    still_prompts = []
    for shot in shots:
        scene = scene_map.get(shot.get("scene_id", ""), {})
        expression_mode = scene.get("scene_expression_mode", "")
        raw_action = shot.get("subject_action", "")

        # Detect director-specified environment-only shots
        is_environment_shot = (
            raw_action.lower().startswith("no person in frame")
            or expression_mode == "absence"
        )

        # 1. Translate filmmaker language → concrete visual action
        visual_action = _translate_action_to_visual(
            subject_action=raw_action,
            emotional_state=shot.get("emotional_micro_state", ""),
            expression_mode=expression_mode,
            scene_objects=scene_objects,
        )

        # 2. Resolve location
        location = scene.get("location", "")
        if not location or location == "unspecified":
            location = entity_locations[0] if entity_locations else "interior"

        # 3. Shot framing
        shot_type = shot.get("shot_type", "medium")
        camera_height = shot.get("camera_height", "eye-level")
        framing = f"{shot_type} shot, {camera_height}"

        # 4. Lighting
        lighting = shot.get("light_description", "") or "cinematic natural light"

        # 5. Compose — environment shots skip the character; character shots lead with subject
        if is_environment_shot:
            # Environment shot: describe the space, light, atmosphere — no character
            emotional_state_lower = shot.get("emotional_micro_state", "").lower()
            env_mood_map = {
                "longing":     "empty and still, charged with the weight of absence",
                "grief":       "heavy and quiet, shadows longer than they should be",
                "nostalgia":   "warm and golden, soft light touching familiar surfaces",
                "resignation": "still and flat, light neither hopeful nor harsh",
                "ache":        "empty and aching, every object untouched and waiting",
                "remembrance": "soft and hazy, as if seen through memory",
                "absence":     "conspicuously empty, the space amplifying what is missing",
                "loneliness":  "wide and desolate, dwarfing any human scale",
            }
            env_mood = next(
                (desc for key, desc in env_mood_map.items() if key in emotional_state_lower),
                "atmospheric and emotionally charged, no people",
            )
            parts = [
                f"Cinematic establishing shot of {location}",
                env_mood,
                framing,
                lighting,
            ]
        else:
            parts = [
                visual_subject,
                visual_action,
                f"in {location}",
                framing,
                lighting,
            ]

        if palette:
            parts.append(f"{', '.join(palette[:3])} palette")

        parts.append(aesthetic)

        if vibe_shot_dir:
            parts.append(vibe_shot_dir)

        positive_prompt = ", ".join(p.strip() for p in parts if p and p.strip())

        # 6. Negative prompt
        neg_parts = list(shot.get("negative_constraints", []))
        neg_parts.extend(r for r in restrictions[:3] if r not in neg_parts)
        neg_parts.extend(a for a in vibe_avoid[:3] if a not in neg_parts)

        still_prompts.append({
            "id": str(uuid.uuid4()),
            "shot_id": shot.get("id", ""),
            "scene_id": shot.get("scene_id", ""),
            "shot_number": shot.get("shot_number", 0),
            "positive_prompt": positive_prompt[:650],
            "negative_prompt": ", ".join(list(dict.fromkeys(neg_parts)))[:300],
            "reference_image_ids": ref_ids[:3],
            "aspect_ratio": "16:9",
            "status": "pending",
            "image_url": None,
        })

    return still_prompts


# ─── Video Render Plan ───────────────────────────────────

def build_render_plan(
    shots: List[Dict[str, Any]],
    still_images: List[Dict[str, Any]],
    model: str = "wan_2_6",
) -> List[Dict[str, Any]]:
    still_map = {s.get("shot_id", ""): s for s in still_images if s.get("image_url")}

    render_jobs = []
    for shot in shots:
        still = still_map.get(shot.get("id", ""))
        render_jobs.append({
            "id": str(uuid.uuid4()),
            "shot_id": shot.get("id", ""),
            "scene_id": shot.get("scene_id", ""),
            "shot_number": shot.get("shot_number", 0),
            "input_image_url": still.get("image_url") if still else None,
            "motion_prompt": (
                f"{shot.get('camera_behavior', 'static')} camera. "
                f"{shot.get('subject_action', '')}. "
                f"Duration: {shot.get('duration_hint', 3.0)}s."
            ),
            "duration_sec": shot.get("duration_hint", 3.0),
            "model": model,
            "provider": "atlas_cloud",
            "status": "pending",
            "output_video_url": None,
        })

    return render_jobs


# ─── Assembly Timeline ───────────────────────────────────

def build_timeline(
    scenes: List[Dict[str, Any]],
    shots: List[Dict[str, Any]],
    render_jobs: List[Dict[str, Any]],
    audio_url: Optional[str] = None,
    audio_duration: Optional[float] = None,
) -> Dict[str, Any]:
    render_map = {r.get("shot_id", ""): r for r in render_jobs if r.get("output_video_url")}

    clips = []
    current_time = 0.0
    sorted_shots = sorted(shots, key=lambda s: s.get("shot_number", 0))
    has_real_timing = any(s.get("start_time") is not None for s in sorted_shots)

    for shot in sorted_shots:
        render = render_map.get(shot.get("id", ""))
        duration = shot.get("duration_hint", 3.0)

        if has_real_timing and shot.get("start_time") is not None:
            start_t = shot.get("start_time")
            end_t = shot.get("end_time") if shot.get("end_time") is not None else start_t + duration
            duration = round(end_t - start_t, 3)
        else:
            start_t = current_time
            end_t = current_time + duration
            current_time = end_t

        clips.append({
            "clip_id": str(uuid.uuid4()),
            "shot_id": shot.get("id", ""),
            "scene_id": shot.get("scene_id", ""),
            "shot_number": shot.get("shot_number", 0),
            "video_url": render.get("output_video_url") if render else None,
            "start_time": start_t,
            "end_time": end_t,
            "duration_sec": duration,
            "transition_in": "cut",
            "transition_out": "cut",
            "status": "ready" if render and render.get("output_video_url") else "pending",
        })

    total_duration = (
        max((c["end_time"] for c in clips), default=0.0)
        if has_real_timing else current_time
    )

    return {
        "id": str(uuid.uuid4()),
        "total_duration_sec": total_duration,
        "total_clips": len(clips),
        "clips": clips,
        "audio_url": audio_url,
        "audio_duration_sec": audio_duration,
        "sync_mode": "auto",
        "status": "draft",
        "created_at": now_utc(),
    }
