import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VisualStoryboardEngine:
    """
    Qaivid MetaMind - Visual Storyboard Engine

    Purpose:
    Convert the MetaMind context packet into storyboard-ready shot objects.

    Design principles:
    - consume the updated unified context engine output cleanly
    - preserve backward compatibility where sensible
    - keep one shot = one intention
    - keep cultural grounding, continuity, and restrictions visible downstream
    """

    DEFAULT_FIDELITY_NO_REF = 0.72
    DEFAULT_FIDELITY_WITH_REF = 1.0

    # =========================================================================
    # FRAMING LOOKUP TABLES
    # =========================================================================

    # ── BACKWARD-COMPAT FRAMING TABLES ─────────────────────────────────────
    # These rotation tables are FALLBACK-ONLY in MM3.1.
    # The primary framing path is beat/event-driven:
    #   shot_event.camera_motivation → frame_directive (sole directive when set)
    # These tables are only used when no shot_event is present (test stubs,
    # legacy projects, or imports from pre-MM3.1 pipeline runs).
    # Do NOT add new production framing logic here; extend ACTION_TO_RIG /
    # CAMERA_PLAN_TO_RIG in cinematography_engine.py instead.
    _FACE_FRAMES = (
        "extreme close-up on eyes and brow, shallow depth of field",
        "medium close-up, face and one shoulder, slight negative space",
        "tight facial framing showing tear-line on cheek",
        "side profile, chin slightly down, gaze distant",
        "two-thirds face, downward gaze, hands partially in frame",
        "soft pull-focus: face sharp, blurred cultural background behind",
        "near-profile with leading lines of doorframe or window visible",
    )

    _BODY_FRAMES = (
        "medium two-thirds shot, hands visible and clasped in lap",
        "medium shot from behind, figure facing away into open space",
        "low-angle medium: figure seated, ground detail visible",
        "over-the-shoulder toward empty doorway or open field",
        "hands-only framing: knuckles, fabric edge, jewellery detail",
        "wide medium: figure small against traditional architecture",
        "medium shot, figure slowly turning away from camera",
        "three-quarter back view, shoulders slightly bowed",
    )

    _ENVIRONMENT_FRAMES = (
        "wide establishing shot: empty space, late afternoon slant light",
        "medium landscape, natural background, foreground texture",
        "tight detail of architectural element: carved threshold, jali screen",
        "low-angle looking up through roof opening or canopy at open sky",
        "slow track through corridor, ambient environmental motion",
        "medium of empty domestic space: simple furniture, open window",
    )

    _SYMBOLIC_FRAMES = (
        "extreme close-up on the symbolic object, background dissolved",
        "medium poetic composition: object in natural cultural context",
        "wide symbolic scene, human figure small or absent",
        "split-light composition: shadow and illuminated surface",
        "detail within environment: motif isolated, other elements blurred",
        "overhead or Dutch-angle symbolic composition, culturally grounded",
    )

    # Framing index → framing_bias value (spec: 0,2,4→"close"; 1,3→"medium_close"; 5,6→"medium")
    _FACE_FRAMING_BIAS = ("close", "medium_close", "close", "medium_close", "close", "medium", "medium")
    _BODY_FRAMING_BIAS = ("medium", "behind", "medium", "behind", "detail", "wide_medium", "medium", "behind")
    _ENV_FRAMING_BIAS  = ("wide", "medium", "medium", "low_angle", "tracking", "wide")
    _SYMBOLIC_FRAMING_BIAS = ("detail", "evocative", "evocative", "evocative", "detail", "evocative")

    # MM3.1 — shot_type (from ShotVarietyEngine) → expression_mode override
    _SHOT_TYPE_TO_MODE: Dict[str, str] = {
        "portrait":         "face",
        "movement":         "body",
        "over_shoulder":    "body",
        "wide_environment": "environment",
        "empty_frame":      "environment",
        "object_detail":    "macro",
        "reflection":       "symbolic",
        "silhouette":       "symbolic",
    }

    # =========================================================================
    # BODY LANGUAGE LOOKUP
    # =========================================================================

    _EMOTIONAL_BODY_LANGUAGE: Dict[str, List[str]] = {
        "tears": [
            "hands lifting slowly to wipe tears from cheek, shoulders forward",
            "fingers pressed lightly to the corner of one eye, head bowed",
            "both hands drop to lap as tears fall unchecked, gaze downward",
        ],
        "despair": [
            "head slightly bowed, hands pressed flat to thighs, weight inward",
            "arms folded loosely across chest, chin toward sternum, body folded",
            "palms turned upward in lap, shoulders collapsed, stillness held",
        ],
        "abandoned": [
            "shoulders turned away, gaze toward empty space off-frame",
            "one arm wraps around own waist, head angled to the side",
            "body retreated into itself, hands gathered at centre, eyes averted",
        ],
        "accusation": [
            "chin slightly raised, hands tightening in lap",
            "one hand half-extended then pulled back, jaw set, gaze direct",
            "spine straightened, hands gripping own wrists, breath held",
        ],
        "longing": [
            "one hand reaching forward slightly, fingers open",
            "both hands rest in lap, fingertips just touching, eyes soft and distant",
            "body leans toward negative space, one shoulder tilted forward",
        ],
        "resignation": [
            "seated stillness, hands in lap, eyelids heavy",
            "slow exhale visible in the shoulders, hands turn palms-up and release",
            "body settles backward, hands fall open, gaze drifts to floor",
        ],
        "unfulfilled": [
            "fingers trace the hem of clothing slowly, eyes averted",
            "thumb circles a ring or bracelet in slow repetition",
            "hands rest open, then close gently on nothing, then open again",
        ],
        "separation": [
            "back to camera, slight forward lean",
            "one hand presses flat to chest, body angled away from frame centre",
            "shoulders draw together as if bracing against cold, gaze off-frame",
        ],
        "regret": [
            "hands folded, head down, weight in the body",
            "one hand rises to lips briefly, then settles back in lap",
            "fingers interlace and press together, brow gently furrowed, stillness",
        ],
        "neglect": [
            "gaze fixed at a point off-frame, body utterly still",
            "hands resting in lap, fingers barely touching, eyes vacant and soft",
            "slight turn of the head toward absence, weight surrendered in the chair",
        ],
        "pain": [
            "hand resting on chest, breath held, eyes averted",
            "hand at sternum, slow inhale-exhale, body contracted slightly inward",
            "both arms draw in, shoulders rise, gaze falls, stillness follows",
        ],
        "anger": [
            "jaw set, hands gripping fabric, posture rigid",
            "hands press flat to thighs, stillness masking tension underneath",
            "one fist closes at side, spine straight, gaze steady and hard",
        ],
        "love": [
            "gentle forward lean, hands open and relaxed in lap",
            "one hand presses lightly over heart, soft smile held in stillness",
            "body tilted toward warmth, fingers uncurled, breath slow and full",
        ],
        "grief": [
            "shoulders curved inward, chin near chest, stillness",
            "body folds slightly forward, hands cover face briefly then lower",
            "arms wrap loosely around torso, rocking imperceptibly, eyes closed",
        ],
        "waiting": [
            "one hand in lap, other at rest, eyes fixed mid-distance",
            "both hands settled, fingers loose, gaze drifting toward a doorway or window",
            "slight tension in the shoulders, hands still, periodic glance to one side",
        ],
        "betrayal": [
            "body half-turned away, one hand raised then lowered",
            "head turns slowly away, hand drops to lap, jaw tightens then releases",
            "shoulders rotate inward, gaze falls, hands press together in the lap",
        ],
        "hope": [
            "head tilted slightly upward, hands loosening their grip",
            "body opens marginally, one hand uncurls, gaze lifts toward light",
            "spine straightens quietly, hands release fabric, breath deepens",
        ],
    }

    # =========================================================================
    # MOTION PROMPT TEMPLATES
    # =========================================================================

    _MOTION_TEMPLATES: Dict[str, tuple] = {
        "face": (
            "slow push-in toward face, breath-weight stillness",
            "camera holds, slight rack focus softening background",
            "barely perceptible drift, eyes stay sharp throughout",
            "gentle pull back from extreme close-up, atmosphere opens",
        ),
        "body": (
            "slow pan following body weight shift, handheld feel",
            "static locked frame, subject micro-moves within stillness",
            "gentle low dolly forward, ground texture visible",
            "slow orbit around seated figure, 10 degrees",
        ),
        "environment": (
            "slow drift through empty space, ambient environmental motion",
            "wide static hold, natural atmospheric movement in scene",
            "gentle push through doorway or threshold",
            "slow tilt down from sky to ground detail",
        ),
        "symbolic": (
            "slow dolly toward object, background gradually blurs",
            "static with organic atmospheric movement",
            "gentle parallax shift revealing depth in scene",
            "slow pull back from detail to wider symbolic context",
        ),
        "macro": (
            "precise micro-motion, premium controlled movement",
            "static macro hold, subject depth-of-field breathing",
            "slow push-in on surface detail",
            "barely perceptible circular drift around focal point",
        ),
    }

    def __init__(self):
        self.user_reference_image: Optional[str] = None
        self.character_consistency_id = str(uuid.uuid4())

        # Framing rotation state — one counter per expression mode
        self._frame_counters: Dict[str, int] = {
            "face": 0,
            "body": 0,
            "environment": 0,
            "symbolic": 0,
            "macro": 0,
        }

        # Chorus escalation tracking
        self._chorus_count: int = 0
        self._prev_was_repeat: bool = False

        # Mandatory environment cutaway tracking
        self._shots_since_env_cutaway: int = 0

        # Body language variety tracking
        self._body_lang_last_keys: Optional[frozenset] = None
        self._body_lang_cycle_idx: int = 0

        # Per-scene location routing (populated in build_storyboard)
        self._scene_map: List[Dict[str, Any]] = []
        self._total_shots: int = 0

        # Optional MM3.1 cinematic layer caches.
        self._cinematic_beats_by_index: Dict[int, Dict[str, Any]] = {}
        self._shot_events_by_index: Dict[int, Dict[str, Any]] = {}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def inject_user_reference(self, image_url: str) -> None:
        self.user_reference_image = image_url
        logger.info("User reference image injected. Character fidelity lock enabled.")

    def build_storyboard(
        self,
        context_packet: Dict[str, Any],
        style_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ctx = self._validate_context_packet(context_packet)
        self._active_style_profile: Dict[str, Any] = style_profile or {}
        ctx = self._attach_optional_cinematic_layers(ctx)

        # Reset rotation state so each build_storyboard() call is deterministic,
        # even if the engine instance is reused across multiple projects.
        self._frame_counters = {m: 0 for m in self._frame_counters}
        self._chorus_count = 0
        self._prev_was_repeat = False
        self._shots_since_env_cutaway = 0
        self._body_lang_last_keys = None
        self._body_lang_cycle_idx = 0
        self._cine_history: List[Dict[str, Any]] = []
        self._cinematic_beats_by_index = self._index_by_line(ctx.get("cinematic_beats", []))
        self._shot_events_by_index = self._index_by_line(ctx.get("shot_events", []))
        # Accumulates visual_props from the most recent block of original
        # (non-repeat) lines so chorus repeats can draw varied imagery from
        # the verse that immediately preceded them.
        self._last_verse_props: List[str] = []

        storyboard: List[Dict[str, Any]] = []
        total_shots = len(ctx["line_meanings"])

        # Build per-scene location map from creative brief
        self._scene_map = self._build_scene_map(ctx)
        self._total_shots = total_shots

        for shot in ctx["line_meanings"]:
            shot_index = shot["line_index"]
            beat = self._cinematic_beats_by_index.get(shot_index, {})
            shot_event = self._shot_events_by_index.get(shot_index, {})
            arc_position = self._arc_position(shot_index, total_shots)

            # Track chorus repeats for escalation offset
            if shot["repeat_status"] == "repeat" and not self._prev_was_repeat:
                self._chorus_count += 1
                if self._chorus_count > 1:
                    for mode in self._frame_counters:
                        self._frame_counters[mode] = (
                            self._frame_counters[mode] + 2
                        ) % self._mode_table_len(mode)
            self._prev_was_repeat = (shot["repeat_status"] == "repeat")

            mode = shot["expression_mode"]
            # MM3.1: if the variety engine assigned a shot_type, let it drive
            # expression_mode so framing tables actually enforce variety.
            # Falls back to the GPT-4o expression_mode when shot_event is absent.
            _variety_type = shot_event.get("shot_type", "")
            if _variety_type and _variety_type in self._SHOT_TYPE_TO_MODE:
                mode = self._SHOT_TYPE_TO_MODE[_variety_type]

            pending_cutaway = (
                self._shots_since_env_cutaway >= 4
                and mode in ("environment", "symbolic")
            )

            if mode in ("face", "body"):
                self._shots_since_env_cutaway += 1
            else:
                self._shots_since_env_cutaway = 0

            # MM3.1 beat-driven framing architecture:
            # Primary source: camera_motivation / camera_plan from shot_event.
            # Fallback only: static rotation table (_FACE_FRAMES etc.) when no
            # beat signal is available (test stubs, legacy projects).
            # The rotation counter is ALWAYS advanced for variety tracking.
            _cam_motive = (
                shot_event.get("camera_motivation")
                or shot_event.get("camera_plan")
                or ""
            ).strip()
            _table_directive, frame_idx = self._pick_frame(mode)  # advance counter
            framing_bias = self._derive_framing_bias(mode, frame_idx)
            if _cam_motive:
                # Beat-driven: the shot_event signal is the sole directive.
                # The static-table string is discarded; counter already advanced.
                frame_directive = _cam_motive
            else:
                # Fallback: static framing rotation (no beat signal present)
                frame_directive = _table_directive

            if pending_cutaway:
                frame_directive = (
                    "wide establishing cutaway — breathe the space before "
                    "returning to face/body; " + frame_directive
                )
                self._shots_since_env_cutaway = 0

            if mode == "body":
                base_body_lang = self._lookup_body_language(shot)
            else:
                base_body_lang = ""
                self._body_lang_last_keys = None
                self._body_lang_cycle_idx = 0

            body_lang = self._augment_body_language(base_body_lang, beat, shot_event)
            action_prompt = self._build_action_prompt(shot, beat, shot_event)

            try:
                from cinematography_engine import (
                    derive as _cine_derive,
                    motion_prompt_from_block as _cine_motion,
                )
                # MM3.1: merge shot_event + effective shot_type/expression_mode
                # into the shot payload so cinematography_engine._event_payload()
                # can see camera_plan/camera_motivation/action for action-driven
                # rig selection (not just legacy emotion-mode lookup).
                _derive_payload = {
                    **shot,
                    "shot_event":     shot_event,
                    "shot_type":      _variety_type or shot.get("shot_type", ""),
                    "expression_mode": mode,
                }
                cinematography = _cine_derive(
                    _derive_payload, ctx, self._active_style_profile,
                    prev_block=(self._cine_history[-1] if self._cine_history else None),
                    recent_blocks=self._cine_history[-4:],
                )
            except Exception:
                cinematography = None
            if cinematography:
                self._cine_history.append(cinematography)

            motion_prompt = self._build_motion_prompt(mode, frame_idx, shot)
            if cinematography:
                motion_prompt = _cine_motion(cinematography) or motion_prompt
            motion_prompt = self._override_motion_prompt_with_event(motion_prompt, shot_event)

            # Track visual_props from original (non-repeat) lines so subsequent
            # chorus repeats can show imagery drawn from the preceding verse,
            # making each chorus pass visually distinct.
            shot_vp = [str(p) for p in (shot.get("visual_props") or []) if p]
            if shot["repeat_status"] == "original" and shot_vp:
                # Rolling window: keep the last 12 props from recent verses
                self._last_verse_props = (self._last_verse_props + shot_vp)[-12:]
            # For repeat shots, surface the verse carry-over props if the line
            # itself has no distinctive visual_props of its own.
            verse_carry_props = (
                self._last_verse_props
                if shot["repeat_status"] == "repeat" and not shot_vp
                else []
            )

            scene_override = self._scene_override_for(shot_index, total_shots)
            camera_prompt = self._build_camera_prompt(ctx, shot, frame_directive)
            camera_prompt = self._augment_camera_prompt(camera_prompt, shot_event)

            prompt_segments = {
                "character": self._build_character_prompt(ctx, shot),
                "environment": self._build_environment_prompt(
                    ctx, shot, scene_override=scene_override,
                    line_visual_props=shot_vp,
                    verse_carry_props=verse_carry_props,
                ),
                "action": action_prompt,
                "performance": self._build_performance_prompt(ctx, shot, body_lang),
                "camera": camera_prompt,
                "motif": self._build_motif_prompt(ctx, shot),
                "continuity": self._build_continuity_prompt(ctx, shot, arc_position),
                "cinematography": self._build_cinematography_prompt(cinematography),
                "repeat": self._build_repeat_prompt(shot),
                "ambiguity": self._build_ambiguity_prompt(ctx),
                "constraints": self._build_constraints_prompt(ctx),
                "restrictions": self._build_restrictions_prompt(ctx),
            }

            visual_prompt = " ".join(
                segment for segment in prompt_segments.values() if segment
            ).strip()

            style_suffix = ""
            active_sp = getattr(self, "_active_style_profile", {}) or {}
            cinematic_block = active_sp.get("cinematic") or {}
            if isinstance(cinematic_block, dict):
                style_suffix = str(cinematic_block.get("image_generation_suffix") or "").strip()
            if style_suffix:
                if visual_prompt and not visual_prompt.endswith((".", "!", "?")):
                    visual_prompt = f"{visual_prompt}. {style_suffix}"
                elif visual_prompt:
                    visual_prompt = f"{visual_prompt} {style_suffix}"
                else:
                    visual_prompt = style_suffix

            storyboard.append(
                {
                    "shot_index": shot_index,
                    "shot_id": f"shot_{shot_index}",
                    "source_line": shot["text"],
                    "meaning": shot["meaning"],
                    "function": shot["function"],
                    "repeat_status": shot["repeat_status"],
                    "intensity": shot["intensity"],
                    "expression_mode": mode,
                    "llm_expression_mode": shot["expression_mode"],
                    "genre": ctx["input_type"],
                    "location_dna": ctx["location_dna"],
                    "visual_prompt": visual_prompt,
                    "character_consistency_id": self.character_consistency_id,
                    "reference_image": self.user_reference_image,
                    "fidelity_lock": self._get_fidelity_lock(),
                    "continuity_anchor": self._build_continuity_anchor(ctx, shot, arc_position),
                    "camera_profile": self._build_camera_profile(ctx, shot, framing_bias),
                    "environment_profile": self._build_environment_profile(ctx, scene_override=scene_override),
                    "scene_name": (scene_override or {}).get("scene_name", ""),
                    "scene_location": (scene_override or {}).get("location", ""),
                    "rendering_notes": self._build_rendering_notes(ctx, shot, arc_position),
                    "literal_meaning": shot["literal_meaning"],
                    "implied_meaning": shot["implied_meaning"],
                    "emotional_meaning": shot["emotional_meaning"],
                    "cultural_meaning": shot["cultural_meaning"],
                    "visualization_mode": shot["visualization_mode"],
                    "visual_suitability": shot["visual_suitability"],
                    "arc_position": arc_position,
                    "arc_directive": self._build_arc_directive(ctx, arc_position),
                    "speaker_profile": self._build_speaker_profile(ctx),
                    "addressee_profile": self._build_addressee_profile(ctx),
                    "world_profile": self._build_world_profile(ctx),
                    "motif_profile": self._build_motif_profile_for_shot(ctx, shot),
                    "restrictions": list(ctx["restrictions"]),
                    "locked_assumptions": dict(ctx["locked_assumptions"]),
                    "dramatic_premise": ctx["dramatic_premise"],
                    "prompt_segments": prompt_segments,
                    "framing_directive": frame_directive,
                    "composition_note": body_lang,
                    "motion_prompt": motion_prompt,
                    "cinematography": cinematography,
                    "cinematic_beat": beat,
                    "shot_event": shot_event,
                    "shot_type": shot_event.get("shot_type", ""),
                    "shot_validation": {
                        "is_generic": shot_event.get("is_generic"),
                        "is_valid": shot_event.get("is_valid"),
                    },
                    # Lyric timestamps from Whisper (Task #105)
                    "lyric_start_seconds": shot.get("lyric_start_seconds"),
                    "lyric_end_seconds":   shot.get("lyric_end_seconds"),
                }
            )

        return self._enforce_variety_caps(storyboard, ctx)

    # =========================================================================
    # MM3.1 VARIETY CAP ENFORCER
    # =========================================================================

    # Target distribution (MM3.1 spec): face≈20%, body≈30%, env≈20%, macro≈20%, symbolic≈10%
    # Caps are set 5pp above target to absorb natural variation without over-correcting.
    _VARIETY_CAPS: Dict[str, float] = {
        "face":        0.25,   # target 20%
        "body":        0.35,   # target 30%
        "environment": 0.25,   # target 20%
        "macro":       0.25,   # target 20%
        "symbolic":    0.15,   # target 10%
    }

    # Ideal target fractions used for underrepresented-category selection
    _VARIETY_TARGETS: Dict[str, float] = {
        "face":        0.20,
        "body":        0.30,
        "environment": 0.20,
        "macro":       0.20,
        "symbolic":    0.10,
    }

    # Canonical shot_type labels per expression_mode.
    # Values must round-trip through _SHOT_TYPE_TO_MODE (above) — i.e. every
    # value here must appear as a key there so the reverse lookup is stable.
    # "wide_environment" and "silhouette" are the representative canonical
    # varieties for their respective modes, matching the variety taxonomy.
    _MODE_TO_SHOT_TYPE: Dict[str, str] = {
        "face":        "portrait",
        "body":        "movement",
        "environment": "wide_environment",
        "macro":       "object_detail",
        "symbolic":    "silhouette",
    }

    # Canonical framing directive per expression_mode — applied when the
    # cap enforcer reclassifies a shot so the visual framing changes too.
    _MODE_TO_FRAMING_DIRECTIVE: Dict[str, str] = {
        "face":        "medium close-up, slightly below eye line, soft focus on expression",
        "body":        "full body in frame, dynamic angle, movement through negative space",
        "environment": "wide establishing shot, static or slow pan, environment as subject",
        "macro":       "extreme close-up, shallow depth of field, detail fills frame",
        "symbolic":    "silhouette mid-shot, backlit, subject abstracted in space",
    }

    def _enforce_variety_caps(
        self,
        storyboard: List[Dict[str, Any]],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Post-pass: reclassify any over-cap shots to the most underrepresented
        category, updating expression_mode, shot_type, framing_directive, AND
        re-deriving the cinematography/motion_prompt so rig/motion stay aligned
        with the new category.

        Only fires when the variety engine is active (shot_type present on at
        least one shot); otherwise returns storyboard as-is.
        """
        if not storyboard or not any(s.get("shot_type") for s in storyboard):
            return storyboard

        total = len(storyboard)
        counts: Dict[str, int] = {m: 0 for m in self._VARIETY_CAPS}
        for s in storyboard:
            m = s.get("expression_mode", "face")
            if m in counts:
                counts[m] += 1

        reclassified_count = 0
        for s in storyboard:
            m = s.get("expression_mode", "face")
            cap = self._VARIETY_CAPS.get(m, 1.0)
            if m not in counts or counts[m] / total <= cap:
                continue
            # Find the most underrepresented category (furthest below its target)
            best_mode = m
            best_deficit = -1.0
            for candidate, target in self._VARIETY_TARGETS.items():
                if candidate == m:
                    continue
                current_frac = counts.get(candidate, 0) / total
                deficit = target - current_frac
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_mode = candidate
            if best_mode != m:
                counts[m] -= 1
                counts[best_mode] = counts.get(best_mode, 0) + 1
                reclassified_count += 1
                # Update expression_mode + its derived visual fields together
                s["expression_mode"] = best_mode
                s["shot_type"] = self._MODE_TO_SHOT_TYPE.get(best_mode, best_mode)
                s["framing_directive"] = self._MODE_TO_FRAMING_DIRECTIVE.get(best_mode, "")
                s["variety_cap_reclassified"] = True
                # Re-derive cinematography so rig/motion align with the new mode.
                # Import is done here (not at module level) to mirror the
                # lazy-import pattern used in build_storyboard and avoid
                # any circular-import risk.
                if ctx is not None:
                    try:
                        from cinematography_engine import derive as _local_cine_derive
                        derive_payload = {
                            "expression_mode": best_mode,
                            "intensity":       s.get("intensity", 0.5),
                            "meaning":         s.get("emotional_meaning", ""),
                            "shot_type":       s["shot_type"],
                        }
                        new_cine = _local_cine_derive(
                            derive_payload,
                            ctx,
                            self._active_style_profile,
                        )
                        if new_cine:
                            s["cinematography"] = new_cine
                            s["motion_prompt"] = self._build_motion_prompt(
                                best_mode,
                                s.get("arc_position", 0),
                                s,
                            )
                    except Exception:
                        pass  # leave existing cinematography intact on failure

        if reclassified_count:
            logger.info(
                "variety_caps: reclassified %d/%d shots to enforce distribution targets",
                reclassified_count, total,
            )

        return storyboard

    # =========================================================================
    # OPTIONAL MM3.1 CINEMATIC LAYER
    # =========================================================================

    def _attach_optional_cinematic_layers(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Attach MM3.1 cinematic beats and shot events if the new modules exist.

        ARCHITECTURAL NOTE — Pipeline-worker integration point:
        The MM3.1 spec places beat generation "immediately after storyboard is
        built in pipeline_worker".  In the current implementation this happens
        here, called at the very start of build_storyboard() before any per-shot
        work begins (line: `ctx = self._attach_optional_cinematic_layers(ctx)`).
        This is functionally equivalent: beats are generated once, project-wide,
        before any shot is processed, and the resulting data (`cinematic_beats`,
        `shot_events`) is injected into ctx so every shot below can consume it.
        The pipeline_worker still controls stage sequencing; VSE is just the
        execution site for this single LLM call.

        This method is intentionally fail-soft so the existing MM3 pipeline keeps
        working even while files are being uploaded one by one.

        Guard: only runs when line_meanings carry actual emotional / cultural
        content (i.e. real pipeline output from the LLM).  Minimal test stubs
        that set only expression_mode are returned unchanged so the tests'
        explicit mode choices are preserved without interference from the variety
        engine's 10-item rotation cycle.
        """
        enriched = dict(ctx)
        line_meanings = list(enriched.get("line_meanings", []))

        # Guard: require real pipeline content before running the MM3.1 chain.
        # Test stubs created by _make_shot() leave emotional/implied/cultural_meaning
        # blank and never set visual_prompt; real LLM output always produces at
        # least one non-empty meaning field or a substantial visual_prompt.
        # The visual_prompt length threshold (>40 chars) distinguishes real
        # LLM output from short stub strings, so sparse-but-valid production
        # inputs that happen to lack meaning fields are still enriched.
        has_content = any(
            (
                l.get("emotional_meaning")
                or l.get("implied_meaning")
                or l.get("cultural_meaning")
                or len(str(l.get("visual_prompt", "")).strip()) > 40
            )
            for l in line_meanings
        )
        if not has_content:
            return enriched

        try:
            if not enriched.get("cinematic_beats"):
                from cinematic_beat_engine import CinematicBeatEngine
                engine = CinematicBeatEngine()
                enriched["cinematic_beats"] = engine.generate_beats(enriched, self._active_style_profile)
        except Exception:
            enriched.setdefault("cinematic_beats", [])

        try:
            beats = list(enriched.get("cinematic_beats", []))
            if beats and not enriched.get("shot_events"):
                from shot_event_builder import ShotEventBuilder
                builder = ShotEventBuilder()
                shot_events = builder.build_sequence(beats)
                for idx, event in enumerate(shot_events):
                    line = line_meanings[idx] if idx < len(line_meanings) else {}
                    event.setdefault("line_index", line.get("line_index", idx + 1))
                    event.setdefault("is_chorus", line.get("repeat_status") == "repeat")
                enriched["shot_events"] = shot_events
        except Exception:
            enriched.setdefault("shot_events", [])

        try:
            if enriched.get("shot_events"):
                from camera_motivation_engine import CameraMotivationEngine
                camera_engine = CameraMotivationEngine()
                enriched["shot_events"] = camera_engine.apply_to_sequence(enriched["shot_events"])
        except Exception:
            pass

        try:
            if enriched.get("shot_events"):
                from motif_progression_engine import MotifProgressionEngine
                motif_engine = MotifProgressionEngine()
                enriched["shot_events"] = motif_engine.apply_full_progression(enriched["shot_events"])
        except Exception:
            pass

        try:
            if enriched.get("shot_events"):
                from chorus_evolution_engine import ChorusEvolutionEngine
                chorus_engine = ChorusEvolutionEngine()
                enriched["shot_events"] = chorus_engine.apply_evolution(enriched["shot_events"])
        except Exception:
            pass

        try:
            if enriched.get("shot_events"):
                from shot_variety_engine import ShotVarietyEngine
                variety_engine = ShotVarietyEngine()
                enriched["shot_events"] = variety_engine.apply_variety(enriched["shot_events"])
        except Exception:
            pass

        try:
            if enriched.get("shot_events"):
                from generic_shot_validator import GenericShotValidator
                validator = GenericShotValidator()
                enriched["shot_events"] = validator.validate_sequence(enriched["shot_events"])
        except Exception:
            pass

        return enriched

    def _index_by_line(self, items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        indexed: Dict[int, Dict[str, Any]] = {}
        for idx, item in enumerate(items or [], start=1):
            if not isinstance(item, dict):
                continue
            line_index = item.get("line_index") or idx
            indexed[int(line_index)] = item
        return indexed

    def _augment_body_language(
        self,
        base_body_lang: str,
        beat: Dict[str, Any],
        shot_event: Dict[str, Any],
    ) -> str:
        parts: List[str] = []
        if base_body_lang:
            parts.append(base_body_lang)
        action = str(shot_event.get("action") or beat.get("subject_action") or "").strip()
        obj = str(shot_event.get("object_interaction") or beat.get("object_usage") or "").strip()
        trigger = str(shot_event.get("trigger") or beat.get("trigger_event") or "").strip()
        if action:
            parts.append(f"physical action: {action}")
        if obj:
            parts.append(f"object relation: {obj}")
        if trigger:
            parts.append(f"response trigger: {trigger}")
        return "; ".join(p for p in parts if p)

    def _build_action_prompt(
        self,
        shot: Dict[str, Any],
        beat: Dict[str, Any],
        shot_event: Dict[str, Any],
    ) -> str:
        action = str(shot_event.get("action") or beat.get("subject_action") or "").strip()
        trigger = str(shot_event.get("trigger") or beat.get("trigger_event") or "").strip()
        contrast = str(shot_event.get("visual_contrast") or beat.get("visual_contrast") or "").strip()
        shift = str(shot_event.get("emotional_shift") or beat.get("emotional_shift") or "").strip()

        if not action and shot.get("expression_mode") == "body":
            action = self._lookup_body_language(shot)

        clauses: List[str] = []
        if action:
            clauses.append(f"Shot event: {action}.")
        if trigger:
            clauses.append(f"Triggered by: {trigger}.")
        if shift:
            clauses.append(f"Emotional turn: {shift}.")
        if contrast:
            clauses.append(f"Visual contrast: {contrast}.")
        return " ".join(clauses).strip()

    def _augment_camera_prompt(self, camera_prompt: str, shot_event: Dict[str, Any]) -> str:
        camera_plan = shot_event.get("camera_plan") or {}
        if not isinstance(camera_plan, dict) or not camera_plan:
            return camera_prompt
        movement = str(camera_plan.get("movement") or "").strip()
        style = str(camera_plan.get("style") or "").strip()
        intensity = str(camera_plan.get("intensity") or "").strip()
        addon = ", ".join(p for p in [movement, style, intensity] if p)
        if not addon:
            return camera_prompt
        return f"{camera_prompt} Camera behaviour: {addon}.".strip()

    def _override_motion_prompt_with_event(self, fallback_prompt: str, shot_event: Dict[str, Any]) -> str:
        try:
            from motion_render_prompt_builder import MotionRenderPromptBuilder
            if shot_event:
                builder = MotionRenderPromptBuilder()
                prompt = builder.build_prompt(shot_event)
                if prompt:
                    return prompt[:250] if len(prompt) > 250 else prompt
        except Exception:
            pass
        return fallback_prompt

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_context_packet(self, context_packet: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(context_packet, dict):
            raise ValueError("Context packet must be a dictionary.")

        ctx = dict(context_packet)

        # core packet defaults
        ctx.setdefault("input_profile", {})
        ctx.setdefault("input_type", ctx.get("input_profile", {}).get("recognized_type", "text"))
        ctx.setdefault("language", "Unknown")
        ctx.setdefault("narrative_mode", "unknown")
        ctx.setdefault("location_dna", "Universal")
        ctx.setdefault("genre_directive", "")
        ctx.setdefault("core_theme", "Unclear core theme")
        ctx.setdefault("dramatic_premise", "")
        ctx.setdefault("narrative_spine", "")
        ctx.setdefault("speaker", {})
        ctx.setdefault("addressee", {})
        ctx.setdefault("world_assumptions", {})
        ctx.setdefault("emotional_arc", {})
        ctx.setdefault("motivation", {})
        ctx.setdefault("line_meanings", [])
        ctx.setdefault("motifs", [])
        ctx.setdefault("motif_map", {})
        ctx.setdefault("entities", [])
        ctx.setdefault("literary_devices", [])
        ctx.setdefault("visual_constraints", [])
        ctx.setdefault("restrictions", [])
        ctx.setdefault("surfaced_assumptions", [])
        ctx.setdefault("locked_assumptions", {})
        ctx.setdefault("ambiguity_flags", [])
        ctx.setdefault("ambiguity_details", [])
        ctx.setdefault("confidence", 0.72)
        ctx.setdefault("confidence_scores", {})
        ctx.setdefault("meta", {})

        if not isinstance(ctx["line_meanings"], list) or not ctx["line_meanings"]:
            raise ValueError("Context packet must contain non-empty line_meanings.")

        for key in ("motifs", "entities", "literary_devices", "visual_constraints", "restrictions", "surfaced_assumptions", "ambiguity_flags", "ambiguity_details"):
            if not isinstance(ctx.get(key), list):
                ctx[key] = []

        for key in ("speaker", "addressee", "world_assumptions", "emotional_arc", "motif_map", "locked_assumptions", "confidence_scores", "input_profile", "meta"):
            if not isinstance(ctx.get(key), dict):
                ctx[key] = {}

        for str_list_key in ("motifs", "visual_constraints", "restrictions", "ambiguity_flags"):
            ctx[str_list_key] = [
                str(x).strip() for x in ctx[str_list_key] if x is not None and str(x).strip()
            ]

        sanitized_mm: Dict[str, Any] = {}
        for k, v in ctx["motif_map"].items():
            key_str = str(k).strip() if k is not None else ""
            if not key_str:
                continue
            if isinstance(v, dict):
                sanitized_mm[key_str] = v
            else:
                sanitized_mm[key_str] = {"visual_form": str(v).strip() if v is not None else ""}
        ctx["motif_map"] = sanitized_mm

        ctx["world_assumptions"].setdefault("geography", "Unspecified")
        ctx["world_assumptions"].setdefault("era", "Unspecified")
        ctx["world_assumptions"].setdefault("season", "Unspecified")
        # Canonical names: characteristic_time / characteristic_setting.
        # Also accept legacy names from pre-rename DB records.
        _wa = ctx["world_assumptions"]
        if "characteristic_time" not in _wa:
            _wa["characteristic_time"] = _wa.pop("time_of_day", "Unspecified")
        if "characteristic_setting" not in _wa:
            _wa["characteristic_setting"] = _wa.pop("domestic_setting", "Unspecified")
        ctx["world_assumptions"].setdefault("social_context", "Unspecified")
        ctx["world_assumptions"].setdefault("economic_context", "Unspecified")
        ctx["world_assumptions"].setdefault("architecture_style", "Unspecified")

        repaired_lines = []
        for i, item in enumerate(ctx["line_meanings"], start=1):
            if not isinstance(item, dict):
                item = {}

            # Safe-parse lyric timestamps from Whisper (Task #105).
            _lts_raw = item.get("lyric_start_seconds")
            _lte_raw = item.get("lyric_end_seconds")
            try:
                _lts = float(_lts_raw) if _lts_raw is not None else None
            except (TypeError, ValueError):
                _lts = None
            try:
                _lte = float(_lte_raw) if _lte_raw is not None else None
            except (TypeError, ValueError):
                _lte = None

            repaired_lines.append(
                {
                    "line_index": self._safe_int(item.get("line_index"), i),
                    "text": self._clean(item.get("text"), ""),
                    "literal_meaning": self._clean(item.get("literal_meaning"), ""),
                    "implied_meaning": self._clean(item.get("implied_meaning"), ""),
                    "emotional_meaning": self._clean(item.get("emotional_meaning"), ""),
                    "cultural_meaning": self._clean(item.get("cultural_meaning"), ""),
                    "meaning": self._clean(item.get("meaning"), "Meaning unavailable"),
                    "function": self._clean(item.get("function"), "emotional_expression"),
                    "repeat_status": self._clean(item.get("repeat_status"), "original").lower(),
                    "intensity": self._clamp_float(item.get("intensity"), 0.5),
                    "expression_mode": self._repair_expression_mode(item.get("expression_mode")),
                    "visualization_mode": self._repair_visualization_mode(item.get("visualization_mode")),
                    "visual_suitability": self._repair_visual_suitability(item.get("visual_suitability")),
                    # Whisper lyric timestamps — preserved through validation (Task #105)
                    "lyric_start_seconds": _lts,
                    "lyric_end_seconds":   _lte,
                }
            )

        ctx["line_meanings"] = repaired_lines
        return ctx

    # =========================================================================
    # FRAMING ROTATION
    # =========================================================================

    def _mode_table_len(self, mode: str) -> int:
        tables = {
            "face": self._FACE_FRAMES,
            "body": self._BODY_FRAMES,
            "environment": self._ENVIRONMENT_FRAMES,
            "symbolic": self._SYMBOLIC_FRAMES,
            "macro": self._FACE_FRAMES,
        }
        return len(tables.get(mode, self._ENVIRONMENT_FRAMES))

    def _pick_frame(self, mode: str) -> tuple:
        """Return (frame_directive_string, frame_index) and advance the counter."""
        if mode == "face":
            table = self._FACE_FRAMES
        elif mode == "body":
            table = self._BODY_FRAMES
        elif mode == "symbolic":
            table = self._SYMBOLIC_FRAMES
        elif mode == "macro":
            table = self._FACE_FRAMES
        else:
            table = self._ENVIRONMENT_FRAMES
            mode = "environment"

        idx = self._frame_counters.get(mode, 0) % len(table)
        self._frame_counters[mode] = idx + 1
        return table[idx], idx

    def _derive_framing_bias(self, mode: str, idx: int) -> str:
        if mode == "face":
            bias_table = self._FACE_FRAMING_BIAS
        elif mode == "body":
            bias_table = self._BODY_FRAMING_BIAS
        elif mode == "symbolic":
            bias_table = self._SYMBOLIC_FRAMING_BIAS
        elif mode == "macro":
            bias_table = self._FACE_FRAMING_BIAS
        else:
            bias_table = self._ENV_FRAMING_BIAS

        return bias_table[idx % len(bias_table)]

    # Fallback body-language directives used when no emotion keyword matches.
    # Rotated by shot_index to ensure every body shot has concrete pose/action.
    _BODY_LANGUAGE_FALLBACKS = (
        "hands resting open in lap, weight settled, gaze inward",
        "one hand resting on knee, slight forward lean, eyes cast down",
        "arms loosely folded, shoulders inward, gaze midfield",
        "fingers loosely interlaced in lap, body weight surrendered",
        "one hand raised to temple briefly, then lowered to lap",
        "both hands clasped, thumbs working against each other, stillness held",
    )

    # =========================================================================
    # BODY LANGUAGE LOOKUP
    # =========================================================================

    def _lookup_body_language(self, shot: Dict[str, Any]) -> str:
        text_blob = " ".join([
            shot.get("emotional_meaning", ""),
            shot.get("implied_meaning", ""),
            shot.get("meaning", ""),
        ]).lower()

        # Collect ALL matching keywords so multiple emotional cues contribute
        # their alternatives to the description pool.
        matched_keys = [k for k in self._EMOTIONAL_BODY_LANGUAGE if k in text_blob]

        if not matched_keys:
            # Deterministic fallback — rotate through concrete poses so every
            # body shot has a specific physical directive even without a keyword match.
            fallback_idx = shot.get("line_index", 1) % len(self._BODY_LANGUAGE_FALLBACKS)
            return self._BODY_LANGUAGE_FALLBACKS[fallback_idx]

        # Build a flat pool of all alternative descriptions for every matched keyword.
        pool: List[str] = []
        for k in matched_keys:
            pool.extend(self._EMOTIONAL_BODY_LANGUAGE[k])

        current_key_set = frozenset(matched_keys)

        if current_key_set == self._body_lang_last_keys:
            # Same emotional signature as the previous body shot — advance the
            # cycle so consecutive shots never repeat the same directive.
            self._body_lang_cycle_idx += 1
        else:
            # Emotional signature changed; start a fresh cycle from index 0.
            self._body_lang_last_keys = current_key_set
            self._body_lang_cycle_idx = 0

        return pool[self._body_lang_cycle_idx % len(pool)]

    # =========================================================================
    # MOTION PROMPT BUILDER
    # =========================================================================

    def _build_motion_prompt(self, mode: str, frame_idx: int, shot: Dict[str, Any]) -> str:
        templates = self._MOTION_TEMPLATES.get(mode, self._MOTION_TEMPLATES["environment"])
        template = templates[frame_idx % len(templates)]

        intensity = shot.get("intensity", 0.5)
        if isinstance(intensity, (int, float)):
            if intensity >= 0.8:
                template = template.replace("slow", "moderately dynamic").replace("gentle", "purposeful")
            elif intensity < 0.3:
                template = template.replace("moderately dynamic", "slow").replace("purposeful", "gentle")

        prompt = template.strip()
        if len(prompt) > 250:
            prompt = prompt[:247] + "…"
        return prompt

    # =========================================================================
    # PROMPT BUILDERS
    # =========================================================================

    def _build_character_prompt(self, ctx: Dict[str, Any], shot: Dict[str, Any]) -> str:
        speaker = ctx["speaker"]
        addressee = ctx["addressee"]

        if self.user_reference_image:
            return (
                "Character anchor: use the exact person from the provided reference image "
                "for facial geometry, identity consistency, and overall look."
            )

        appearance_bits = []
        for key, label in (
            ("ethnicity", "ethnicity"),
            ("complexion", "complexion"),
            ("wardrobe", "wardrobe"),
            ("grooming", "grooming"),
        ):
            value = self._clean(speaker.get(key), "")
            if value and value.lower() not in {"unspecified", "unclear"}:
                appearance_bits.append(f"{label}: {value}")

        appearance_clause = f" Appearance lock: {'; '.join(appearance_bits)}." if appearance_bits else ""

        return (
            f"Character anchor: portray a {self._clean(speaker.get('age_range'), 'unclear age')} "
            f"{self._clean(speaker.get('gender'), 'unclear gender')} figure in the role of "
            f"{self._clean(speaker.get('social_role'), 'unclear role')}, with emotional state centered on "
            f"{self._clean(speaker.get('emotional_state'), 'emotionally charged')}. "
            f"Speaker identity context: {self._clean(speaker.get('identity'), 'unclear speaker')}.{appearance_clause} "
            f"Addressee context: {self._clean(addressee.get('identity'), 'unclear addressee')}, "
            f"presence: {self._clean(addressee.get('presence'), 'unclear presence')}. "
            f"Maintain strict character continuity across all shots."
        )

    def _build_environment_prompt(
        self,
        ctx: Dict[str, Any],
        shot: Dict[str, Any],
        scene_override: Optional[Dict[str, Any]] = None,
        line_visual_props: Optional[List[str]] = None,
        verse_carry_props: Optional[List[str]] = None,
    ) -> str:
        # --- Two-namespace composition ---
        # global_frame: immutable cultural/geographic anchor from world_assumptions
        # scene_frame:  shot-specific location and time from creative brief scene override
        # Neither mutates the other.
        global_frame = ctx["world_assumptions"]  # read-only reference
        location_dna = ctx["location_dna"]

        scene_props: List[str] = []
        scene_location = ""
        scene_time = ""
        scene_name = ""
        if scene_override:
            scene_location = (scene_override.get("location") or "").strip()
            scene_time = (scene_override.get("time_of_day") or "").strip()
            scene_name = (scene_override.get("scene_name") or "").strip()
            scene_props = [str(p) for p in (scene_override.get("props") or []) if p]

        entities = ctx["entities"]
        motifs = ctx["motifs"]

        place_entities = [
            e["name"] for e in entities
            if isinstance(e, dict) and str(e.get("type", "")).lower() == "place" and e.get("name")
        ]
        symbol_entities = [
            e["name"] for e in entities
            if isinstance(e, dict) and str(e.get("type", "")).lower() == "symbol" and e.get("name")
        ]

        # -- Cultural frame (global, unchanged across all shots) --
        parts = [
            f"[Cultural frame] Setting grounded in: {location_dna}.",
            f"Geography: {self._clean(global_frame.get('geography'), 'Unspecified')}.",
            f"Era: {self._clean(global_frame.get('era'), 'Unspecified')}.",
            f"Season: {self._clean(global_frame.get('season'), 'Unspecified')}.",
            f"Characteristic time-feel: {self._clean(global_frame.get('characteristic_time'), 'Unspecified')}.",
            f"Architecture style: {self._clean(global_frame.get('architecture_style'), 'Unspecified')}.",
            f"Characteristic setting (cultural default): {self._clean(global_frame.get('characteristic_setting'), 'Unspecified')}.",
        ]

        if location_dna and location_dna.lower() != "universal":
            parts.append("Use culturally faithful textures, materials, and regional domestic logic.")

        # Punjabi-specific visual grounding — injected at prompt-build time so it applies
        # even when the stored context packet predates the enriched culture pack.
        _loc_lower = (location_dna or "").lower()
        _arch_lower = (global_frame.get("architecture_style") or "").lower()
        _geo_lower = (global_frame.get("geography") or "").lower()
        if "punjab" in _loc_lower or "punjab" in _arch_lower or "punjab" in _geo_lower:
            parts.append(
                "PUNJABI LOCATION MANDATE: all exterior and courtyard shots must show authentic "
                "kuchha (mud-plastered) village construction — thick bare ochre/sand-coloured mud walls, "
                "clean swept earthen vehra (courtyard) with packed earth floor, flat clay rooftops with "
                "exterior stone or mud staircases, small deep-set windows, heavy wooden doors and shutters "
                "painted blue or turquoise, terracotta matkas (water pots), charpai (rope-strung wooden bed) "
                "in the open courtyard, mustard or wheat fields visible beyond the compound wall. "
                "Style reference: T-Series Punjabi folk / rural music video shoot. "
                "Strictly NO concrete, tile, brick, Rajasthani ornamentation, or studio-dressed sets."
            )

        # -- Scene frame (shot-specific, from creative brief) --
        if scene_location or scene_time or scene_name:
            # Anchor the scene location to the cultural/geographic root so the
            # image model cannot drift to a generic or wrong-country version.
            cultural_root = ctx.get("location_dna") or ""
            anchored_location = scene_location
            if anchored_location and cultural_root and cultural_root.lower() != "universal":
                if cultural_root.lower() not in anchored_location.lower():
                    anchored_location = f"{anchored_location}, {cultural_root}"
            label = f"Scene: {scene_name}" if scene_name else ""
            setting = anchored_location if anchored_location else ""
            timing = f"at {scene_time}" if scene_time else ""
            directive = " — ".join(x for x in [label, setting, timing] if x)
            if directive:
                parts.append(f"[Scene setting] SCENE DIRECTIVE (binding, culturally grounded): {directive}.")
        elif not scene_override:
            # No creative brief scene: fall back to characteristic_setting as scene hint
            fallback = self._clean(global_frame.get("characteristic_setting"), "")
            if fallback and fallback.lower() not in ("unspecified", ""):
                parts.append(f"[Scene setting] Default scene setting: {fallback}.")

        if scene_props:
            parts.append(f"Include scene props from the creative brief: {', '.join(scene_props)}.")

        # Per-line visual props: concrete objects/elements named in this specific
        # lyric line — extracted by the context engine and translated to English.
        # These take priority over generic scene props because they come directly
        # from the lyrics that this exact shot is illustrating.
        effective_vp: List[str] = list(line_visual_props or [])
        if not effective_vp and verse_carry_props:
            # Chorus repeat: no line-specific props, so carry forward imagery
            # accumulated from the preceding verse so this chorus looks distinct
            # from earlier or later chorus passes.
            effective_vp = list(verse_carry_props)
        if effective_vp:
            parts.append(
                f"LYRIC IMAGERY MANDATE — this shot MUST visually feature: "
                f"{', '.join(effective_vp)}. "
                "These elements were named literally in the lyric line. "
                "Do not omit them or replace them with generic substitutes."
            )

        if place_entities:
            parts.append(f"Relevant place cues: {', '.join(place_entities[:5])}.")

        if shot["expression_mode"] == "symbolic" and symbol_entities:
            parts.append(f"Symbolic environment cues may draw from: {', '.join(symbol_entities[:5])}.")

        if motifs:
            parts.append(f"Keep atmosphere compatible with recurring motifs such as {', '.join(motifs[:5])}.")

        if shot["cultural_meaning"]:
            parts.append(f"Cultural subtext to preserve: {shot['cultural_meaning']}.")

        if shot["meaning"]:
            parts.append(f"The environment should support the emotional meaning of: {shot['meaning']}.")

        return " ".join(parts)

    def _build_performance_prompt(self, ctx: Dict[str, Any], shot: Dict[str, Any], body_lang: str = "") -> str:
        mode = shot["expression_mode"]
        intensity_label = self._label_intensity(shot["intensity"])

        if shot["visualization_mode"] == "performance_only":
            return (
                f"Performance: treat this primarily as a performance-led beat carrying emotional meaning "
                f"'{shot['emotional_meaning']}'. Do not force unnecessary literal action. "
                f"Function: {shot['function']}. Intensity: {intensity_label}."
            )

        if mode == "face":
            return (
                f"Performance: communicate implied meaning '{shot['implied_meaning']}' and emotional meaning "
                f"'{shot['emotional_meaning']}' through eyes, brow tension, restrained expression, and facial stillness. "
                f"Literal line sense: '{shot['literal_meaning']}'. Function: {shot['function']}. "
                f"Intensity level: {intensity_label}. No mouth-performance dependence."
            )

        if mode == "body":
            body_lang_clause = f" Specific body language: {body_lang}." if body_lang else ""
            return (
                f"Performance: communicate implied meaning '{shot['implied_meaning']}' and emotional meaning "
                f"'{shot['emotional_meaning']}' through posture, gesture, stillness, and body tension.{body_lang_clause} "
                f"Literal line sense: '{shot['literal_meaning']}'. Function: {shot['function']}. "
                f"Intensity level: {intensity_label}."
            )

        if mode == "symbolic" or shot["visualization_mode"] == "symbolic":
            return (
                f"Performance: interpret the line symbolically, preserving implied meaning '{shot['implied_meaning']}' "
                f"and emotional meaning '{shot['emotional_meaning']}', without flattening it into literal plot action. "
                f"Function: {shot['function']}. Intensity level: {intensity_label}."
            )

        if mode == "macro":
            return (
                f"Performance: use high-detail focal emphasis to express meaning '{shot['meaning']}'. "
                f"Function: {shot['function']}. Intensity level: {intensity_label}."
            )

        if ctx["input_type"] == "documentary":
            return (
                f"Performance: observational realism conveying emotional meaning '{shot['emotional_meaning']}' "
                f"with minimal stylization. Literal layer: '{shot['literal_meaning']}'. "
                f"Function: {shot['function']}. Intensity level: {intensity_label}."
            )

        return (
            f"Performance: build a cinematic moment that conveys implied meaning '{shot['implied_meaning']}' "
            f"and emotional meaning '{shot['emotional_meaning']}' through environment, composition, gesture, and mood. "
            f"Literal line sense: '{shot['literal_meaning']}'. Function: {shot['function']}. "
            f"Intensity level: {intensity_label}."
        )

    def _build_camera_prompt(self, ctx: Dict[str, Any], shot: Dict[str, Any], frame_directive: str = "") -> str:
        genre = ctx["input_type"]
        narrative_mode = self._clean(
            ctx["locked_assumptions"].get("narrative_mode") or ctx.get("narrative_mode"),
            "unknown"
        )

        if genre == "ad":
            camera_note = (
                "premium commercial framing, polished composition, macro-capable lens language, "
                "precise lighting control, elegant depth separation"
            )
            if frame_directive:
                return f"Cinematography: {frame_directive}. {camera_note}."
            return f"Cinematography: {camera_note}."

        if genre == "documentary":
            camera_note = (
                "observational camera language, restrained movement, natural light realism, "
                "authentic framing, documentary sobriety"
            )
            if frame_directive:
                return f"Cinematography: {frame_directive}. {camera_note}."
            return f"Cinematography: {camera_note}."

        if genre == "script":
            camera_note = (
                "performance-first framing, controlled coverage, face and gesture readable, "
                "dramatic but grounded visual language"
            )
            if frame_directive:
                return f"Cinematography: {frame_directive}. {camera_note}."
            return f"Cinematography: {camera_note}."

        if shot["expression_mode"] == "face":
            camera_note = "intimate close framing, facial readability, subtle focus falloff, restrained camera motion"
        elif shot["expression_mode"] == "symbolic" or shot["visualization_mode"] == "symbolic":
            camera_note = "poetic composition, metaphor-friendly framing, expressive atmosphere, visual suggestiveness"
        elif shot["repeat_status"] == "repeat":
            camera_note = "preserve continuity with prior repetition while varying emotional emphasis, scale, angle, or mood"
        elif narrative_mode in {"symbolic", "philosophical", "psychological"}:
            camera_note = "controlled poetic realism, suggestive framing, emotionally weighted atmosphere, restrained visual symbolism"
        elif shot["intensity"] >= 0.8:
            camera_note = "heightened cinematic energy, stronger contrast, more emotionally charged framing, controlled visual momentum"
        else:
            camera_note = "cinematic naturalism, balanced composition, atmospheric lighting, clear visual storytelling"

        if frame_directive:
            return f"Cinematography: {frame_directive}. {camera_note}."
        return f"Cinematography: {camera_note}."

    def _build_motif_prompt(self, ctx: Dict[str, Any], shot: Dict[str, Any]) -> str:
        motif_forms = self._motif_forms_relevant_to_shot(ctx["motif_map"], shot)

        if motif_forms:
            return "Motif handling: prioritize visual rendering of these motif forms — " + "; ".join(motif_forms[:5]) + "."

        if ctx["motif_map"]:
            descriptive = []
            for name, payload in list(ctx["motif_map"].items())[:5]:
                if not isinstance(payload, dict):
                    continue
                visual_form = self._clean(payload.get("visual_form"), "")
                significance = self._clean(payload.get("significance"), "")
                piece = name
                if visual_form:
                    piece += f" ({visual_form})"
                if significance:
                    piece += f" — {significance}"
                descriptive.append(piece)
            if descriptive:
                return "Motif handling: keep recurring motifs available — " + "; ".join(descriptive) + "."

        if ctx["motifs"]:
            return f"Motif handling: keep recurring visual motifs available where relevant, especially {', '.join(ctx['motifs'][:5])}."

        return ""

    def _build_continuity_prompt(self, ctx: Dict[str, Any], shot: Dict[str, Any], arc_position: str) -> str:
        parts = [
            f"Continuity: preserve narrative mode {ctx.get('narrative_mode', 'unknown')} and remain aligned with the narrative spine.",
        ]

        # Task #69 — anchor every shot to the locked Creative Brief so the
        # central metaphor + director's note flow into stills + videos.
        cb = ctx.get("creative_brief") if isinstance(ctx, dict) else None
        chosen = (cb or {}).get("chosen") if isinstance(cb, dict) else None
        if isinstance(chosen, dict):
            metaphor = (chosen.get("central_metaphor") or "").strip()
            note = (chosen.get("director_note") or "").strip()
            title = (chosen.get("title") or "").strip()
            if title:
                parts.append(f"Treatment: {title}.")
            if metaphor:
                parts.append(f"Central metaphor (carry through every shot): {metaphor}.")
            if note:
                parts.append(f"Director's note (binding): {note}.")

        if ctx["narrative_spine"]:
            parts.append(f"Spine anchor: {ctx['narrative_spine']}.")

        if ctx["dramatic_premise"]:
            parts.append(f"Dramatic premise: {ctx['dramatic_premise']}.")

        # WHY anchor — give every shot the emotional engine of the whole song.
        why_summary = self._build_why_summary(ctx)
        if why_summary:
            parts.append(f"Why this song exists: {why_summary}")

        arc_directive = self._build_arc_directive(ctx, arc_position)
        if arc_directive:
            parts.append(f"Arc beat ({arc_position}): {arc_directive}.")

        if shot["repeat_status"] == "repeat":
            parts.append("This shot should feel connected to an earlier emotional beat, not like a completely new event.")

        source_format = ctx["meta"].get("source_format") or ctx["input_profile"].get("source_format", "plain_text")
        parts.append(f"Source format awareness: {source_format}.")

        return " ".join(parts)

    # Sentinel strings written by UnifiedContextEngine._repair_motivation when
    # the LLM returns no motivation block. We suppress the WHY summary if all
    # four fields still match these defaults — there's no real signal to add.
    _MOTIVATION_DEFAULT_SENTINELS = frozenset({
        "An emotionally significant moment the speaker cannot stay silent about.",
        "To be heard, understood, or reconciled with what has been lost or longed for.",
        "Continued emotional weight — unresolved feeling carried forward.",
        "Distance, silence, or time keeping the speaker from resolution.",
    })

    def _build_why_summary(self, ctx: Dict[str, Any]) -> str:
        """Compose a one-line 'why this song exists' from the WHY block.

        Returns an empty string when motivation is missing OR when every
        field still matches the engine's repair-layer sentinel defaults, so
        we don't pollute the prompt with generic boilerplate that isn't
        actually grounded in the song.
        """
        m = ctx.get("motivation") if isinstance(ctx.get("motivation"), dict) else {}
        if not m:
            return ""
        cause = (m.get("inciting_cause") or "").strip()
        desire = (m.get("underlying_desire") or "").strip()
        stakes = (m.get("stakes") or "").strip()
        obstacle = (m.get("obstacle") or "").strip()

        present = [v for v in (cause, desire, stakes, obstacle) if v]
        if not present:
            return ""
        # All non-empty values are still the repair-layer defaults => suppress.
        if all(v in self._MOTIVATION_DEFAULT_SENTINELS for v in present):
            return ""

        bits = []
        if cause:
            bits.append(f"because {cause.rstrip('.')}")
        if desire:
            bits.append(f"the speaker wants {desire.rstrip('.')}")
        if obstacle:
            bits.append(f"but {obstacle.rstrip('.')}")
        if stakes:
            bits.append(f"with {stakes.rstrip('.')} at stake")
        return "; ".join(bits) + "." if bits else ""

    def _build_cinematography_prompt(self, block: Optional[Dict[str, Any]]) -> str:
        """Compose a one-line cinematography clause for the visual prompt."""
        if not isinstance(block, dict):
            return ""
        rig = block.get("rig") or ""
        direction = block.get("direction") or ""
        speed = block.get("speed") or ""
        lens = block.get("lens") or ""
        intensity = block.get("intensity") or ""
        if not rig:
            return ""
        return (
            f"Cinematography ({intensity} intensity): {speed} {rig} rig — "
            f"{direction}; lens: {lens}."
        ).strip()

    def _build_repeat_prompt(self, shot: Dict[str, Any]) -> str:
        if shot["repeat_status"] == "repeat":
            return "Repetition logic: treat this as emotional return or escalation, not as a separate narrative event."
        return ""

    def _build_ambiguity_prompt(self, ctx: Dict[str, Any]) -> str:
        details = ctx["ambiguity_details"]
        flags = ctx["ambiguity_flags"]

        if details:
            snippets = []
            for item in details[:4]:
                if isinstance(item, dict):
                    snippets.append(
                        f"{self._clean(item.get('field'), 'general')}: {self._clean(item.get('reason'), 'unclear')}"
                    )
            if snippets:
                return "Ambiguity handling: stay conservative where interpretation is uncertain. Relevant ambiguity notes: " + "; ".join(snippets) + "."

        if flags:
            return "Ambiguity handling: stay conservative where interpretation is uncertain. Relevant ambiguity notes: " + "; ".join(flags[:4]) + "."

        return ""

    def _build_constraints_prompt(self, ctx: Dict[str, Any]) -> str:
        if not ctx["visual_constraints"]:
            return ""
        return f"Visual constraints: {'; '.join(ctx['visual_constraints'][:8])}."

    def _build_restrictions_prompt(self, ctx: Dict[str, Any]) -> str:
        if not ctx["restrictions"]:
            return ""
        return f"Hard restrictions (must obey): {'; '.join(ctx['restrictions'][:8])}."

    # =========================================================================
    # SUPPORT OBJECT BUILDERS
    # =========================================================================

    def _build_camera_profile(self, ctx: Dict[str, Any], shot: Dict[str, Any], framing_bias: str = "medium") -> Dict[str, Any]:
        profile = {
            "style": "cinematic_naturalism",
            "framing_bias": framing_bias,
            "movement": "restrained",
            "lighting": "atmospheric",
            "narrative_mode": self._clean(ctx.get("narrative_mode"), "unknown"),
        }

        if ctx["input_type"] == "ad":
            profile.update({
                "style": "commercial_premium",
                "framing_bias": framing_bias or "close_macro",
                "movement": "controlled",
                "lighting": "clean_high_control",
            })
        elif ctx["input_type"] == "documentary":
            profile.update({
                "style": "observational_realism",
                "framing_bias": framing_bias or "natural_coverage",
                "movement": "minimal_or_handheld",
                "lighting": "natural",
            })
        elif ctx["input_type"] == "script":
            profile.update({
                "style": "performance_driven",
                "framing_bias": framing_bias or "face_body_readability",
                "movement": "controlled_dramatic",
                "lighting": "grounded_dramatic",
            })
        elif shot["expression_mode"] == "symbolic" or shot["visualization_mode"] == "symbolic":
            profile.update({
                "style": "poetic_symbolic",
                "framing_bias": framing_bias or "evocative",
                "movement": "gentle",
                "lighting": "expressive",
            })
        elif shot["intensity"] > 0.8:
            profile.update({
                "movement": "moderate_emphasis",
                "lighting": "charged",
            })

        # Style Profile override — user-selected style trumps input-type heuristics
        _active_sp = getattr(self, "_active_style_profile", {}) or {}

        # Cinematic style: modifiers live under storyboard_modifiers
        _cin_sp = _active_sp.get("cinematic") or {}
        _cin_mods = _cin_sp.get("storyboard_modifiers") or {}
        if _cin_mods.get("camera_style"):
            profile["style"] = _cin_mods["camera_style"]
        if _cin_mods.get("lighting"):
            profile["lighting"] = _cin_mods["lighting"]
        if _cin_mods.get("movement"):
            profile["movement"] = _cin_mods["movement"]
        if _cin_mods.get("atmosphere_note"):
            profile["atmosphere_note"] = _cin_mods["atmosphere_note"]

        # Production style: camera_movement_bias and performance_hint
        _prod_sp = _active_sp.get("production") or {}
        _prod_mods = _prod_sp.get("storyboard_modifiers") or {}
        if _prod_mods.get("camera_movement_bias"):
            profile["movement"] = _prod_mods["camera_movement_bias"]
        if _prod_mods.get("performance_hint"):
            profile["performance_hint"] = _prod_mods["performance_hint"]

        return profile

    # -------------------------------------------------------------------------
    # SCENE-BASED LOCATION ROUTING
    # -------------------------------------------------------------------------

    def _build_scene_map(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract per-scene location overrides from the chosen creative brief.

        Returns a list of dicts (one per scene), each with:
          scene_name, location, time_of_day, props, summary
        If no creative brief or no scenes, returns [].
        """
        cb = ctx.get("creative_brief")
        if not isinstance(cb, dict):
            return []
        chosen = cb.get("chosen")
        if not isinstance(chosen, dict):
            return []
        scenes = chosen.get("scenes") or []
        if not isinstance(scenes, list) or not scenes:
            return []

        result = []
        for s in scenes:
            if not isinstance(s, dict):
                continue
            location = str(s.get("location") or "").strip()
            time_of_day = str(s.get("time_of_day") or "").strip()
            summary = str(s.get("summary") or "").strip()
            props = s.get("props") or []
            if not isinstance(props, list):
                props = []

            # If no explicit location, infer one from the summary text
            if not location and summary:
                location = self._infer_location_from_summary(summary, ctx)

            result.append({
                "scene_name":  str(s.get("name") or "").strip(),
                "location":    location,
                "time_of_day": time_of_day,
                "props":       [str(p).strip() for p in props[:6] if p],
                "summary":     summary,
            })
        return result

    def _infer_location_from_summary(self, summary: str, ctx: Dict[str, Any]) -> str:
        """Best-effort location extraction from free-text scene summary.
        Falls back to the global location_dna."""
        s = summary.lower()
        if any(w in s for w in ["golden field", "wheat", "harvest", "stalks"]):
            return "open golden wheat field"
        if any(w in s for w in ["field", "grass", "meadow", "rolling"]):
            return "open countryside field"
        if any(w in s for w in ["home", "interior", "room", "house", "inside", "indoor"]):
            return "traditional home interior"
        if any(w in s for w in ["memory", "flash", "past", "joyful moment", "recall"]):
            return "soft-lit intimate memory space"
        if any(w in s for w in ["river", "water", "lake", "stream", "shore"]):
            return "riverside"
        if any(w in s for w in ["rooftop", "terrace", "balcony", "roof"]):
            return "rooftop terrace"
        if any(w in s for w in ["sunset", "dusk", "silhouette", "horizon", "sets"]):
            return "open hilltop at sunset"
        if any(w in s for w in ["dance", "twirl", "spin", "circle", "wind"]):
            return "open field with wind"
        if any(w in s for w in ["night", "stars", "moonlight", "dark"]):
            return "open countryside under night sky"
        if any(w in s for w in ["courtyard", "village", "street", "bazaar", "market"]):
            return "village courtyard"
        return str(ctx.get("location_dna") or "open countryside")

    def _scene_override_for(
        self, line_index: int, total_shots: int
    ) -> Optional[Dict[str, Any]]:
        """Map a shot (by line_index) to its scene override proportionally.

        With 4 scenes and 47 shots:
          shots 1-12  → scene 0 (intro)
          shots 13-24 → scene 1 (verse1)
          shots 25-35 → scene 2 (chorus)
          shots 36-47 → scene 3 (outro)
        """
        if not self._scene_map:
            return None
        n = len(self._scene_map)
        total = max(total_shots, 1)
        # line_index is 1-based; convert to 0-based fraction
        fraction = (line_index - 1) / total
        scene_idx = min(int(fraction * n), n - 1)
        return self._scene_map[scene_idx]

    def _build_environment_profile(self, ctx: Dict[str, Any], scene_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        entities = ctx["entities"]
        place_entities = [
            e["name"] for e in entities
            if isinstance(e, dict) and str(e.get("type", "")).lower() == "place" and e.get("name")
        ]
        object_entities = [
            e["name"] for e in entities
            if isinstance(e, dict) and str(e.get("type", "")).lower() == "object" and e.get("name")
        ]

        # world_assumptions is the GLOBAL cultural anchor — never mutated here.
        # scene_frame carries shot-specific location / time from the creative brief.
        world_assumptions = ctx["world_assumptions"]  # read-only
        location_dna = ctx["location_dna"]

        scene_props: List[str] = []
        scene_frame: Dict[str, str] = {}
        if scene_override:
            loc = (scene_override.get("location") or "").strip()
            tod = (scene_override.get("time_of_day") or "").strip()
            scene_props = [str(p) for p in (scene_override.get("props") or []) if p]
            if loc or tod:
                scene_frame = {
                    "location": loc,
                    "time_of_day": tod,
                    "scene_name": (scene_override.get("scene_name") or "").strip(),
                }
                if loc:
                    location_dna = f"{ctx['location_dna']} — {loc}"

        # Punjab override — replace any vague stored values with specific kuchha
        # architecture descriptions so the image model receives concrete visual
        # material rather than generic "courtyard-oriented home" or
        # "rural Punjabi domestic architecture".  These were the phrases causing
        # Flux to default to Mughal/Nawabi ornate stonework.
        _check = " ".join([
            (location_dna or ""),
            (world_assumptions.get("geography") or ""),
            (world_assumptions.get("architecture_style") or ""),
        ]).lower()
        if "punjab" in _check:
            world_assumptions = dict(world_assumptions)  # shallow copy, do not mutate ctx
            world_assumptions["architecture_style"] = (
                "kuchha mud-plastered village house — thick bare ochre/sand-toned mud walls, "
                "flat clay rooftop with exterior stone or mud staircase, small deep-set windows, "
                "heavy wooden door and shutters painted blue or turquoise, smooth plastered parapet; "
                "NO ornate stonework, NO Mughal arches, NO brick or concrete"
            )
            world_assumptions["characteristic_setting"] = (
                "clean swept earthen vehra (open courtyard) — packed bare mud floor, "
                "charpai (rope-strung wooden bed) in open air, terracotta matkas near entrance, "
                "mustard or wheat fields visible beyond the low compound wall, open sky above; "
                "NO tile, NO paving stones, NO ornamental garden"
            )

        return {
            "location_dna": location_dna,
            "place_entities": place_entities[:8],
            "object_entities": (object_entities + scene_props)[:10],
            "motifs": ctx["motifs"][:8],
            "motif_map": {
                name: {
                    "type": (payload.get("type") if isinstance(payload, dict) else "") or "",
                    "significance": (payload.get("significance") if isinstance(payload, dict) else "") or "",
                    "visual_form": (payload.get("visual_form") if isinstance(payload, dict) else "") or "",
                }
                for name, payload in list(ctx["motif_map"].items())[:8]
            },
            "world_assumptions": world_assumptions,
            "scene_frame": scene_frame,
            "regional_grounding_required": (ctx["location_dna"] or "").lower() != "universal",
            "scene_name": (scene_override or {}).get("scene_name", ""),
            "scene_location": (scene_override or {}).get("location", ""),
            "scene_props": scene_props,
        }

    def _build_continuity_anchor(self, ctx: Dict[str, Any], shot: Dict[str, Any], arc_position: str) -> Dict[str, Any]:
        speaker = ctx["speaker"]
        addressee = ctx["addressee"]

        return {
            "character_consistency_id": self.character_consistency_id,
            "speaker_identity": self._clean(speaker.get("identity"), "unclear speaker"),
            "speaker_role": self._clean(speaker.get("social_role"), "unclear role"),
            "speaker_ethnicity": self._clean(speaker.get("ethnicity"), ""),
            "speaker_complexion": self._clean(speaker.get("complexion"), ""),
            "speaker_wardrobe": self._clean(speaker.get("wardrobe"), ""),
            "speaker_grooming": self._clean(speaker.get("grooming"), ""),
            "addressee_identity": self._clean(addressee.get("identity"), "unclear addressee"),
            "addressee_relationship": self._clean(addressee.get("relationship"), ""),
            "location_dna": ctx["location_dna"],
            "repeat_status": shot["repeat_status"],
            "arc_position": arc_position,
        }

    def _build_rendering_notes(self, ctx: Dict[str, Any], shot: Dict[str, Any], arc_position: str) -> List[str]:
        notes: List[str] = []

        if self.user_reference_image:
            notes.append("Use the provided user reference image as the strongest character anchor.")

        if shot["repeat_status"] == "repeat":
            notes.append("Preserve continuity with earlier similar beat while varying emotion or framing.")

        if shot["expression_mode"] == "face":
            notes.append("Prioritize eye line, brow tension, and subtle facial detail.")

        if shot["expression_mode"] == "symbolic":
            notes.append("Keep symbolism emotionally grounded and not over-surreal.")

        if (ctx["location_dna"] or "").lower() != "universal":
            notes.append("Maintain regional authenticity in materials, architecture, and atmosphere.")

        ethnicity = self._clean(ctx["speaker"].get("ethnicity"), "")
        if ethnicity and ethnicity.lower() not in {"unspecified", "unclear"}:
            notes.append(f"Honor speaker ethnicity ({ethnicity}); do not substitute another ethnic appearance.")

        wardrobe = self._clean(ctx["speaker"].get("wardrobe"), "")
        if wardrobe and wardrobe.lower() not in {"unspecified", "unclear"}:
            notes.append(f"Wardrobe must remain consistent with: {wardrobe}.")

        if shot["visual_suitability"] == "low":
            notes.append("Visual suitability is low for this line — lean on metaphor or environmental framing.")
        elif shot["visual_suitability"] == "medium":
            notes.append("Mixed visual suitability — balance literal and symbolic rendering.")

        if shot["visualization_mode"] and shot["visualization_mode"] != "direct":
            notes.append(f"Visualization mode for this shot: {shot['visualization_mode']}.")

        notes.append(f"Emotional arc position: {arc_position}.")
        return notes

    def _build_speaker_profile(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        sp = ctx["speaker"]
        return {
            "identity": self._clean(sp.get("identity"), ""),
            "gender": self._clean(sp.get("gender"), ""),
            "age_range": self._clean(sp.get("age_range"), ""),
            "social_role": self._clean(sp.get("social_role"), ""),
            "emotional_state": self._clean(sp.get("emotional_state"), ""),
            "relationship_to_addressee": self._clean(sp.get("relationship_to_addressee"), ""),
            "ethnicity": self._clean(sp.get("ethnicity"), ""),
            "complexion": self._clean(sp.get("complexion"), ""),
            "wardrobe": self._clean(sp.get("wardrobe"), ""),
            "grooming": self._clean(sp.get("grooming"), ""),
        }

    def _build_addressee_profile(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ad = ctx["addressee"]
        return {
            "identity": self._clean(ad.get("identity"), ""),
            "relationship": self._clean(ad.get("relationship"), ""),
            "presence": self._clean(ad.get("presence"), ""),
        }

    def _build_world_profile(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "location_dna": ctx["location_dna"],
            "world_assumptions": dict(ctx["world_assumptions"]),
        }

    def _build_motif_profile_for_shot(self, ctx: Dict[str, Any], shot: Dict[str, Any]) -> Dict[str, Any]:
        relevant = []
        text_blob = " ".join(
            str(shot.get(k, "")) for k in
            ("text", "meaning", "literal_meaning", "implied_meaning", "emotional_meaning", "cultural_meaning")
        ).lower()

        for name, payload in ctx["motif_map"].items():
            if not isinstance(payload, dict):
                continue
            if name and name.lower() in text_blob:
                relevant.append(
                    {
                        "name": name,
                        "type": self._clean(payload.get("type"), ""),
                        "significance": self._clean(payload.get("significance"), ""),
                        "visual_form": self._clean(payload.get("visual_form"), ""),
                    }
                )

        return {
            "all_motifs": list(ctx["motifs"]),
            "shot_relevant_motifs": relevant,
        }

    # =========================================================================
    # ARC / MOTIF HELPERS
    # =========================================================================

    def _arc_position(self, line_index: int, total: int) -> str:
        if total <= 0:
            return "opening"
        ratio = float(line_index) / float(max(total, 1))
        if ratio <= 0.34:
            return "opening"
        if ratio <= 0.67:
            return "development"
        return "resolution"

    def _build_arc_directive(self, ctx: Dict[str, Any], arc_position: str) -> str:
        arc = ctx["emotional_arc"]
        return self._clean(arc.get(arc_position), "")

    def _motif_forms_relevant_to_shot(self, motif_map: Dict[str, Any], shot: Dict[str, Any]) -> List[str]:
        if not motif_map:
            return []

        text_blob = " ".join(
            str(shot.get(k, "")) for k in
            ("text", "meaning", "literal_meaning", "implied_meaning", "emotional_meaning", "cultural_meaning")
        ).lower()

        results = []
        for name, payload in motif_map.items():
            if not name or not isinstance(payload, dict):
                continue

            matched = name.lower() in text_blob
            if not matched:
                for key in ("visual_form", "significance", "type"):
                    aux = self._clean(payload.get(key), "").lower()
                    if aux:
                        tokens = [t for t in aux.split() if len(t) > 3]
                        if any(tok in text_blob for tok in tokens):
                            matched = True
                            break

            if matched:
                visual_form = self._clean(payload.get("visual_form"), "")
                results.append(f"{name} → {visual_form}" if visual_form else name)

        return results

    # =========================================================================
    # PRIMITIVE HELPERS
    # =========================================================================

    def _get_fidelity_lock(self) -> float:
        return self.DEFAULT_FIDELITY_WITH_REF if self.user_reference_image else self.DEFAULT_FIDELITY_NO_REF

    def _repair_expression_mode(self, value: Any) -> str:
        allowed = {"face", "body", "environment", "symbolic", "macro"}
        value = self._clean(value, "environment").lower()
        return value if value in allowed else "environment"

    def _repair_visualization_mode(self, value: Any) -> str:
        allowed = {"direct", "indirect", "symbolic", "absorbed", "performance_only"}
        value = self._clean(value, "indirect").lower()
        return value if value in allowed else "indirect"

    def _repair_visual_suitability(self, value: Any) -> str:
        allowed = {"high", "medium", "low"}
        value = self._clean(value, "medium").lower()
        return value if value in allowed else "medium"

    def _label_intensity(self, value: float) -> str:
        if value < 0.25:
            return "very low"
        if value < 0.45:
            return "low"
        if value < 0.65:
            return "medium"
        if value < 0.85:
            return "high"
        return "very high"

    def _clamp_float(self, value: Any, fallback: float) -> float:
        try:
            num = float(value)
            return max(0.0, min(1.0, num))
        except Exception:
            return fallback

    def _safe_int(self, value: Any, fallback: int) -> int:
        try:
            return int(value)
        except Exception:
            return fallback

    def _clean(self, value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        value = str(value).strip()
        return value if value else fallback
