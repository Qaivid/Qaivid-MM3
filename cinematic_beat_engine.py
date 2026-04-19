import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CinematicBeatEngine:
    """
    Qaivid MM3.1 — Cinematic Beat Engine

    Purpose:
    Convert semantic line meanings into shot-worthy cinematic beat objects.

    Design goals:
    - preserve backward compatibility with MM3 context packets
    - stay deterministic by default
    - generate behaviour, event, and camera motivation hints
    - remain safe to add before wiring into existing orchestrators

    This module does not replace the storyboard engine by itself.
    It prepares richer intermediate beat objects so downstream systems can
    stop building generic portrait prompts from raw emotion labels alone.
    """

    DEFAULT_BEAT_TYPE = "narrative"
    DEFAULT_SHOT_FUNCTION = "emotional_expression"
    DEFAULT_CAMERA_MOTIVE = "observe_without_intrusion"
    DEFAULT_VISUAL_CONTRAST = "inner emotion against outer stillness"

    _EMOTION_BEHAVIOUR_MAP: Dict[str, List[str]] = {
        "longing": [
            "pauses mid-task after hearing something outside the frame",
            "waits at a threshold where someone used to arrive",
            "checks a meaningful object with restrained hope",
            "leans subtly toward absence before pulling back",
        ],
        "waiting": [
            "holds a routine while attention drifts toward an entrance",
            "stops to listen, then resumes without satisfaction",
            "sits in stillness but reacts to small sounds",
            "keeps a place or object ready for someone who does not come",
        ],
        "grief": [
            "touches an object linked to memory and then withdraws",
            "sits after unfinished movement, unable to continue",
            "folds inward before regaining composure",
            "holds still while the environment carries the emotion",
        ],
        "betrayal": [
            "begins to engage and then turns away before completion",
            "tightens grip on fabric or an object before releasing it",
            "refuses eye line with the source of pain",
            "stops a habitual action with visible restraint",
        ],
        "regret": [
            "revisits an object or place but does not fully enter the memory",
            "starts to reach, then lets the hand fall",
            "replays a gesture with hesitation and self-interruption",
            "stays near evidence of the past without disturbing it",
        ],
        "hope": [
            "lifts attention toward a possible arrival or sign",
            "loosens body tension as if something may change",
            "moves one step forward before checking the feeling",
            "holds a pause that almost becomes a smile",
        ],
        "pain": [
            "tries to continue routine while the body betrays the feeling",
            "touches chest, throat, or hands as emotion rises",
            "halts movement at the exact point it becomes difficult",
            "lets the frame stay still while discomfort settles in",
        ],
        "anger": [
            "contains the emotion through forced stillness",
            "redirects energy into an object or gesture instead of speech",
            "cuts off movement sharply rather than finishing it smoothly",
            "holds posture rigid while the eyes do the work",
        ],
        "love": [
            "protects or preserves an object linked to the beloved",
            "moves gently through a familiar ritual once shared",
            "lingers in a place charged with warmth",
            "lets tenderness appear through careful handling of detail",
        ],
        "resignation": [
            "completes the routine mechanically without emotional resistance",
            "stops expecting interruption and settles into stillness",
            "allows an object or room to remain untouched",
            "sits with absence rather than reacting to it",
        ],
        "despair": [
            "fails to complete a simple action",
            "sinks into the environment after a moment of effort",
            "lets the frame hold after hope has fully collapsed",
            "releases an object with no attempt to recover it",
        ],
        "nostalgia": [
            "retraces a known movement with softened attention",
            "touches surfaces as if they still hold the past",
            "re-enters a routine now emptied of its original warmth",
            "smiles faintly before the feeling turns inward",
        ],
    }

    _FUNCTION_EVENT_MAP: Dict[str, List[str]] = {
        "emotional_expression": [
            "private emotional shift during a simple routine",
            "inner feeling revealed through small physical hesitation",
            "emotion carried by silence rather than overt action",
        ],
        "memory_recall": [
            "present action triggers a memory-charged pause",
            "an ordinary object opens a hidden emotional layer",
            "the body repeats something the heart remembers",
        ],
        "accusation": [
            "emotion turns outward but stops short of confrontation",
            "the subject reacts to absence as if it were present",
            "blame appears through restraint rather than aggression",
        ],
        "separation": [
            "space itself emphasises distance and lack of return",
            "the subject behaves as if someone should be there but is not",
            "a threshold or pathway becomes emotionally active",
        ],
        "realisation": [
            "the shot captures the instant hope changes shape",
            "a small detail confirms what the subject already feared",
            "the body understands before the face fully does",
        ],
        "release": [
            "tension softens and something internal finally gives way",
            "an object or gesture carries the emotional letting-go",
            "the frame opens after prolonged compression",
        ],
    }

    _MODE_BEAT_TYPE: Dict[str, str] = {
        "face": "performance",
        "body": "narrative",
        "environment": "atmospheric",
        "symbolic": "symbolic",
        "macro": "detail",
    }

    _MODE_CAMERA_MOTIVE: Dict[str, str] = {
        "face": "read_the_micro_expression",
        "body": "follow_the_weight_shift",
        "environment": "let_space_carry_meaning",
        "symbolic": "elevate_the_motif",
        "macro": "reward_small_detail",
    }

    def generate_beats(
        self,
        context_packet: Dict[str, Any],
        style_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build one cinematic beat per line_meanings entry."""
        ctx = self._validate_context_packet(context_packet)
        style_profile = style_profile or {}

        lines = ctx["line_meanings"]
        total = len(lines)
        beats: List[Dict[str, Any]] = []

        for idx, line in enumerate(lines):
            previous_line = lines[idx - 1] if idx > 0 else None
            next_line = lines[idx + 1] if idx + 1 < total else None
            beat = self._build_beat(
                ctx=ctx,
                line=line,
                index=idx,
                total=total,
                previous_line=previous_line,
                next_line=next_line,
                style_profile=style_profile,
            )
            beats.append(beat)

        return beats

    def attach_beats(
        self,
        context_packet: Dict[str, Any],
        style_profile: Optional[Dict[str, Any]] = None,
        field_name: str = "cinematic_beats",
    ) -> Dict[str, Any]:
        """Return a copied context packet with generated beats attached."""
        ctx = deepcopy(context_packet or {})
        ctx[field_name] = self.generate_beats(ctx, style_profile=style_profile)
        return ctx

    def _build_beat(
        self,
        ctx: Dict[str, Any],
        line: Dict[str, Any],
        index: int,
        total: int,
        previous_line: Optional[Dict[str, Any]],
        next_line: Optional[Dict[str, Any]],
        style_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        emotion = self._primary_emotion(line)
        function = self._clean(line.get("function"), self.DEFAULT_SHOT_FUNCTION)
        expression_mode = self._clean(line.get("expression_mode"), "body").lower()
        intensity = self._clamp_float(line.get("intensity"), 0.5)
        repeat_status = self._clean(line.get("repeat_status"), "original").lower()
        beat_type = self._MODE_BEAT_TYPE.get(expression_mode, self.DEFAULT_BEAT_TYPE)

        subject_action = self._choose_subject_action(
            emotion=emotion,
            function=function,
            line=line,
            expression_mode=expression_mode,
            index=index,
            repeat_status=repeat_status,
        )
        trigger_event = self._infer_trigger_event(line, previous_line, next_line, emotion)
        emotional_shift = self._infer_emotional_shift(
            line=line,
            previous_line=previous_line,
            next_line=next_line,
            repeat_status=repeat_status,
        )
        object_usage = self._infer_object_usage(ctx, line, emotion, expression_mode)
        environment_usage = self._infer_environment_usage(ctx, line, expression_mode)
        visual_contrast = self._infer_visual_contrast(
            line=line,
            emotion=emotion,
            expression_mode=expression_mode,
            repeat_status=repeat_status,
        )
        camera_motive = self._infer_camera_motive(
            expression_mode=expression_mode,
            subject_action=subject_action,
            trigger_event=trigger_event,
            intensity=intensity,
        )
        beat_function = self._infer_shot_function(function, emotion, repeat_status)
        continuity_focus = self._infer_continuity_focus(ctx, line, expression_mode)
        freshness_tags = self._build_freshness_tags(
            line=line,
            expression_mode=expression_mode,
            repeat_status=repeat_status,
            emotion=emotion,
        )

        lyric_relation_type = self._infer_lyric_relation_type(
            line=line,
            expression_mode=expression_mode,
            repeat_status=repeat_status,
            function=function,
            visual_contrast=visual_contrast,
            intensity=intensity,
        )

        return {
            "beat_id": f"beat_{self._safe_int(line.get('line_index'), index + 1)}",
            "shot_index": self._safe_int(line.get("line_index"), index + 1),
            "source_line": self._clean(line.get("text"), ""),
            "meaning": self._clean(line.get("meaning"), ""),
            "function": function,
            "expression_mode": expression_mode,
            "repeat_status": repeat_status,
            "intensity": intensity,
            "arc_position": self._arc_position(index=index, total=total),
            "beat_type": beat_type,
            "shot_function": beat_function,
            "subject_action": subject_action,
            "trigger_event": trigger_event,
            "emotional_shift": emotional_shift,
            "object_usage": object_usage,
            "environment_usage": environment_usage,
            "camera_motive": camera_motive,
            "visual_contrast": visual_contrast,
            "continuity_focus": continuity_focus,
            "freshness_tags": freshness_tags,
            "style_hints": self._extract_style_hints(style_profile),
            "lyric_relation_type": lyric_relation_type,
            "confidence": self._estimate_confidence(
                line=line,
                emotion=emotion,
                expression_mode=expression_mode,
                subject_action=subject_action,
                trigger_event=trigger_event,
            ),
        }

    def _validate_context_packet(self, context_packet: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(context_packet, dict):
            raise ValueError("Context packet must be a dictionary.")

        ctx = dict(context_packet)
        line_meanings = ctx.get("line_meanings")
        if not isinstance(line_meanings, list) or not line_meanings:
            raise ValueError("Context packet must contain non-empty line_meanings.")

        ctx.setdefault("world_assumptions", {})
        ctx.setdefault("speaker", {})
        ctx.setdefault("addressee", {})
        ctx.setdefault("motifs", [])
        ctx.setdefault("motif_map", {})
        ctx.setdefault("creative_brief", {})
        return ctx

    def _primary_emotion(self, line: Dict[str, Any]) -> str:
        candidates = [
            line.get("emotional_meaning"),
            line.get("function"),
            line.get("meaning"),
            line.get("implied_meaning"),
        ]
        text = " ".join(self._clean(c, "") for c in candidates).lower()

        emotion_keywords = [
            "longing", "waiting", "grief", "pain", "betrayal", "regret",
            "hope", "love", "anger", "resignation", "despair", "nostalgia",
            "separation",
        ]
        for keyword in emotion_keywords:
            if keyword in text:
                return "longing" if keyword == "separation" else keyword
        return "longing" if "absence" in text else "pain"

    def _choose_subject_action(
        self,
        emotion: str,
        function: str,
        line: Dict[str, Any],
        expression_mode: str,
        index: int,
        repeat_status: str,
    ) -> str:
        behaviour_pool = list(self._EMOTION_BEHAVIOUR_MAP.get(emotion, []))
        behaviour_pool.extend(self._FUNCTION_EVENT_MAP.get(function, []))

        if expression_mode == "environment":
            behaviour_pool = [
                "the subject is absent but their emotional trace remains in the space",
                "the room or landscape holds the aftermath of a human feeling",
                "the environment behaves like a witness rather than a backdrop",
            ] + behaviour_pool
        elif expression_mode == "symbolic":
            behaviour_pool = [
                "the human action is displaced into a charged symbolic object",
                "meaning is carried through an indirect visual substitute",
            ] + behaviour_pool
        elif expression_mode == "macro":
            behaviour_pool = [
                "a tiny physical detail becomes the emotional event",
                "the action is reduced to one tactile or material change",
            ] + behaviour_pool

        if not behaviour_pool:
            return "holds a restrained emotional pause within an ordinary moment"

        offset = 1 if repeat_status == "repeat" else 0
        choice_index = (index + offset) % len(behaviour_pool)
        return behaviour_pool[choice_index]

    def _infer_trigger_event(
        self,
        line: Dict[str, Any],
        previous_line: Optional[Dict[str, Any]],
        next_line: Optional[Dict[str, Any]],
        emotion: str,
    ) -> str:
        text = " ".join(
            self._clean(x, "")
            for x in [
                line.get("text"),
                line.get("meaning"),
                line.get("implied_meaning"),
                line.get("cultural_meaning"),
            ]
        ).lower()

        trigger_patterns = [
            ("door", "a doorway or threshold becomes emotionally charged"),
            ("window", "the frame catches attention at the edge of the room"),
            ("road", "something beyond the immediate space disturbs the inner state"),
            ("rain", "weather externalises the emotional pressure"),
            ("phone", "a device or signal creates expectation and disappointment"),
            ("call", "the possibility of contact changes the body before words do"),
            ("letter", "a preserved message or trace reactivates memory"),
            ("name", "the mention or memory of someone shifts the emotional ground"),
            ("return", "the thought of return changes how the body inhabits space"),
            ("footstep", "a small sound creates false anticipation"),
        ]
        for needle, phrase in trigger_patterns:
            if needle in text:
                return phrase

        if previous_line and self._clean(previous_line.get("meaning"), "") != self._clean(line.get("meaning"), ""):
            return "the emotional temperature changes from the previous beat"
        if next_line and self._clean(next_line.get("meaning"), "") != self._clean(line.get("meaning"), ""):
            return "this moment prepares the turn into the next emotional beat"

        generic = {
            "longing": "a small off-screen cue awakens hope that will not fully resolve",
            "waiting": "time passing becomes the silent trigger",
            "grief": "memory rises without needing a visible external cause",
            "pain": "the body itself becomes the trigger point",
            "betrayal": "a remembered wound reactivates the present moment",
            "hope": "the possibility of change briefly opens the frame",
            "resignation": "nothing happens, and that absence becomes the event",
        }
        return generic.get(emotion, "an internal emotional shift quietly activates the shot")

    def _infer_emotional_shift(
        self,
        line: Dict[str, Any],
        previous_line: Optional[Dict[str, Any]],
        next_line: Optional[Dict[str, Any]],
        repeat_status: str,
    ) -> str:
        current = self._clean(line.get("emotional_meaning"), "emotion held")
        previous = self._clean((previous_line or {}).get("emotional_meaning"), "")
        upcoming = self._clean((next_line or {}).get("emotional_meaning"), "")

        if repeat_status == "repeat":
            return "same lyric, deeper emotional colouring than before"
        if previous and current and previous.lower() != current.lower():
            return f"{previous} -> {current}"
        if current and upcoming and current.lower() != upcoming.lower():
            return f"{current} -> {upcoming}"
        return current or "emotion sustained without overt external release"

    def _infer_object_usage(
        self,
        ctx: Dict[str, Any],
        line: Dict[str, Any],
        emotion: str,
        expression_mode: str,
    ) -> str:
        motifs = ctx.get("motifs") or []
        motif_map = ctx.get("motif_map") or {}
        line_text = " ".join(
            self._clean(x, "")
            for x in [line.get("text"), line.get("meaning"), line.get("implied_meaning")]
        ).lower()

        for motif in motifs:
            motif_str = self._clean(motif, "")
            if motif_str and motif_str.lower() in line_text:
                motif_info = motif_map.get(motif_str) if isinstance(motif_map, dict) else None
                visual_form = self._clean((motif_info or {}).get("visual_form"), motif_str)
                return f"the motif '{visual_form}' becomes the emotional carrier in the frame"

        if expression_mode == "macro":
            return "surface, texture, or touch becomes the meaningful object event"

        defaults = {
            "longing": "a held or half-used personal object carries the missing connection",
            "waiting": "an everyday object remains ready for someone who does not arrive",
            "grief": "a remembered belonging becomes too heavy to ignore",
            "betrayal": "fabric, jewellery, or a shared object absorbs restrained tension",
            "hope": "a small object becomes a possible sign of change",
            "resignation": "the object is left untouched, and that stillness carries meaning",
        }
        return defaults.get(emotion, "an ordinary object quietly holds emotional residue")

    def _infer_environment_usage(
        self,
        ctx: Dict[str, Any],
        line: Dict[str, Any],
        expression_mode: str,
    ) -> str:
        world = ctx.get("world_assumptions") or {}
        geography = self._clean(world.get("geography"), "unspecified geography")
        setting = self._clean(
            world.get("characteristic_setting") or world.get("domestic_setting"),
            "lived-in setting",
        )
        architecture = self._clean(world.get("architecture_style"), "")

        if expression_mode == "environment":
            return f"the {setting} in {geography} is allowed to carry the feeling without relying on the face"
        if expression_mode == "symbolic":
            return f"the environment frames the symbol within the logic of {setting} and {architecture or geography}"
        return f"the action should feel native to {setting} within {geography}, not staged for the camera"

    def _infer_visual_contrast(
        self,
        line: Dict[str, Any],
        emotion: str,
        expression_mode: str,
        repeat_status: str,
    ) -> str:
        if expression_mode == "environment":
            base = "outer stillness against human absence"
        elif expression_mode == "symbolic":
            base = "literal reality against displaced symbolic meaning"
        elif expression_mode == "macro":
            base = "small physical detail against large emotional weight"
        else:
            contrast_map = {
                "longing": "ordinary routine against private hope",
                "waiting": "passing time against suspended expectation",
                "grief": "composure against inward collapse",
                "betrayal": "surface stillness against tightened feeling",
                "hope": "fragile openness against uncertainty",
                "resignation": "habit against emotional finality",
                "pain": "body control against emotional pressure",
            }
            base = contrast_map.get(emotion, self.DEFAULT_VISUAL_CONTRAST)

        if repeat_status == "repeat":
            return f"{base}, but with evolved emotional pressure from the earlier recurrence"
        return base

    def _infer_lyric_relation_type(
        self,
        line: Dict[str, Any],
        expression_mode: str,
        repeat_status: str,
        function: str,
        visual_contrast: str,
        intensity: float,
    ) -> str:
        """Classify how the visual strategy relates to the lyric text.

        Canonical MM3.1 enum:
          literal     — visual depicts what the lyric describes directly
          indirect    — interpretive / subtext-driven treatment
          symbolic    — shot uses abstract / displaced visual language
          contrast    — visual provides emotional counterpoint to the lyric
          memory      — repeated section echoing earlier imagery
          performance — high-intensity amplification of the lyric's emotion
        """
        if repeat_status == "repeat":
            return "memory"
        if expression_mode == "symbolic":
            return "symbolic"
        if visual_contrast and any(
            kw in visual_contrast for kw in ("against", "contrast", "opposite")
        ):
            return "contrast"
        if intensity >= 0.75 or function in (
            "chorus_reveal", "emotional_climax", "build_release"
        ):
            return "performance"
        if function in ("narrative_turn", "metaphor_development", "thematic_anchor"):
            return "indirect"
        return "literal"

    def _infer_camera_motive(
        self,
        expression_mode: str,
        subject_action: str,
        trigger_event: str,
        intensity: float,
    ) -> str:
        base = self._MODE_CAMERA_MOTIVE.get(expression_mode, self.DEFAULT_CAMERA_MOTIVE)
        intensity_note = "restrained" if intensity < 0.4 else "measured" if intensity < 0.75 else "heightened"

        if "pause" in subject_action or "still" in subject_action:
            motion_logic = "favour a patient frame that lets the interruption register"
        elif "turn" in subject_action or "reach" in subject_action or "move" in subject_action:
            motion_logic = "let the camera respond softly to the subject's directional change"
        elif expression_mode in ("environment", "symbolic"):
            motion_logic = "use motion only to reveal emotional meaning already present in the space"
        else:
            motion_logic = "keep the camera subordinate to the behavioural beat"

        return f"{base}; {motion_logic}; {intensity_note} emotional handling"

    def _infer_shot_function(self, function: str, emotion: str, repeat_status: str) -> str:
        if repeat_status == "repeat":
            return "motif_recurrence_with_escalation"
        function_map = {
            "memory_recall": "memory_activation",
            "separation": "distance_reveal",
            "accusation": "contained_outward_charge",
            "realisation": "private_realisation",
            "release": "emotional_release",
        }
        if function in function_map:
            return function_map[function]

        emotion_map = {
            "longing": "false_anticipation",
            "waiting": "suspended_attention",
            "grief": "private_breakdown",
            "betrayal": "emotional_recoil",
            "regret": "hesitant_return",
            "hope": "fragile_opening",
            "resignation": "settled_absence",
        }
        return emotion_map.get(emotion, function or self.DEFAULT_SHOT_FUNCTION)

    def _infer_continuity_focus(
        self,
        ctx: Dict[str, Any],
        line: Dict[str, Any],
        expression_mode: str,
    ) -> Dict[str, str]:
        speaker = ctx.get("speaker") or {}
        world = ctx.get("world_assumptions") or {}
        return {
            "character": self._clean(speaker.get("identity"), "same primary subject"),
            "wardrobe": "preserve wardrobe and grooming continuity unless arc change is explicit",
            "world": self._clean(
                world.get("characteristic_setting") or world.get("domestic_setting"),
                "same world logic",
            ),
            "mode": expression_mode,
            "emotion_anchor": self._clean(line.get("emotional_meaning"), "same emotional arc branch"),
        }

    def _build_freshness_tags(
        self,
        line: Dict[str, Any],
        expression_mode: str,
        repeat_status: str,
        emotion: str,
    ) -> List[str]:
        tags: List[str] = []
        if expression_mode in ("environment", "symbolic", "macro"):
            tags.append("non_portrait_bias")
        else:
            tags.append("behaviour_first")

        if repeat_status == "repeat":
            tags.append("recurrence_needs_variation")
        if emotion in ("longing", "waiting", "grief"):
            tags.append("micro_drama")
        if self._clean(line.get("text"), ""):
            tags.append("lyric_linked")
        return tags

    def _extract_style_hints(self, style_profile: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(style_profile, dict):
            return {}
        cinematic = style_profile.get("cinematic") or {}
        if not isinstance(cinematic, dict):
            return {}
        allowed_keys = {
            "camera_movement_bias",
            "pacing_bias",
            "performance_intimacy",
            "visual_density",
            "image_generation_suffix",
        }
        return {
            key: cinematic[key]
            for key in cinematic.keys()
            if key in allowed_keys and cinematic.get(key) is not None
        }

    def _estimate_confidence(
        self,
        line: Dict[str, Any],
        emotion: str,
        expression_mode: str,
        subject_action: str,
        trigger_event: str,
    ) -> float:
        score = 0.55
        if emotion:
            score += 0.10
        if expression_mode in self._MODE_BEAT_TYPE:
            score += 0.10
        if subject_action and len(subject_action) > 20:
            score += 0.10
        if trigger_event and len(trigger_event) > 20:
            score += 0.05
        if self._clean(line.get("text"), ""):
            score += 0.05
        if self._clean(line.get("meaning"), ""):
            score += 0.05
        return round(min(score, 0.95), 2)

    def _arc_position(self, index: int, total: int) -> str:
        if total <= 1:
            return "opening"
        fraction = index / max(total - 1, 1)
        if fraction <= 0.20:
            return "opening"
        if fraction <= 0.50:
            return "development"
        if fraction <= 0.80:
            return "climax"
        return "resolution"

    @staticmethod
    def _clean(value: Any, default: str = "") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _clamp_float(value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except Exception:
            numeric = default
        return max(0.0, min(1.0, numeric))
