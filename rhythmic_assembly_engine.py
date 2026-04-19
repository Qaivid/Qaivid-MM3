import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class RhythmicAssemblyEngine:
    """
    Qaivid Rhythmic Assembly Engine
    """

    DEFAULT_FPS = 24
    DEFAULT_BPM = 120.0
    DEFAULT_BEATS_PER_BAR = 4
    DEFAULT_MIN_SHOT_DURATION = 1.5
    DEFAULT_MAX_SHOT_DURATION = 6.0
    DEFAULT_INTENSITY = 0.5

    def __init__(self):
        self.default_fps = self.DEFAULT_FPS
        self.min_shot_duration = self.DEFAULT_MIN_SHOT_DURATION
        self.max_shot_duration = self.DEFAULT_MAX_SHOT_DURATION

    def assemble_timeline(
        self,
        storyboard: List[Dict[str, Any]],
        audio_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        validated_storyboard = self._validate_storyboard(storyboard)
        validated_audio = self._validate_audio_data(audio_data)

        bpm = validated_audio["bpm"]
        beats_per_bar = validated_audio["beats_per_bar"]
        intensity_curve = validated_audio["intensity_curve"]

        beat_duration = 60.0 / bpm
        bar_duration = beat_duration * beats_per_bar

        timeline: List[Dict[str, Any]] = []
        current_timestamp = 0.0

        for i, shot in enumerate(validated_storyboard):
            shot_intensity = self._clamp_float(
                shot.get("intensity", self.DEFAULT_INTENSITY),
                self.DEFAULT_INTENSITY,
            )

            audio_intensity = self._get_audio_intensity_for_index(intensity_curve, i)
            blended_intensity = self._blend_intensity(shot_intensity, audio_intensity)

            raw_duration = self._calculate_duration(
                intensity=blended_intensity,
                beat_duration=beat_duration,
                repeat_status=shot.get("repeat_status", "original"),
                expression_mode=shot.get("expression_mode", "environment"),
            )

            synced_duration = self._snap_duration_to_beat(raw_duration, beat_duration)
            synced_duration = self._clamp_duration(synced_duration)

            motion_scale = self._get_motion_scale(
                intensity=blended_intensity,
                expression_mode=shot.get("expression_mode", "environment"),
                repeat_status=shot.get("repeat_status", "original"),
                shot_index=i,
            )

            transition = self._get_transition_type(
                current_shot=shot,
                previous_shot=validated_storyboard[i - 1] if i > 0 else None,
                intensity=blended_intensity,
            )

            beat_start = round(current_timestamp / beat_duration)
            bar_index = int(current_timestamp // bar_duration) + 1

            timeline.append(
                {
                    "timeline_index": i + 1,
                    "shot_index": shot["shot_index"],
                    "shot_id": shot.get("shot_id", f"shot_{shot['shot_index']}"),
                    "start_time": round(current_timestamp, 3),
                    "duration": round(synced_duration, 3),
                    "end_time": round(current_timestamp + synced_duration, 3),
                    "start_beat": beat_start,
                    "bar_index": bar_index,
                    "visual_prompt": self._build_timed_visual_prompt(shot, motion_scale, transition),
                    "meaning": shot.get("meaning", ""),
                    "function": shot.get("function", ""),
                    "repeat_status": shot.get("repeat_status", "original"),
                    "intensity": round(blended_intensity, 3),
                    "raw_shot_intensity": round(shot_intensity, 3),
                    "audio_intensity": round(audio_intensity, 3),
                    "motion_scale": motion_scale,
                    "transition": transition,
                    "expression_mode": shot.get("expression_mode", "environment"),
                    "reference_image": shot.get("reference_image"),
                    "fidelity_lock": shot.get("fidelity_lock"),
                    "character_consistency_id": shot.get("character_consistency_id"),
                    "camera_profile": shot.get("camera_profile", {}),
                    "environment_profile": shot.get("environment_profile", {}),
                    "continuity_anchor": shot.get("continuity_anchor", {}),
                    "rendering_notes": shot.get("rendering_notes", []),
                    # Cinematic variety fields (Task #50)
                    "motion_prompt": shot.get("motion_prompt", ""),
                    "framing_directive": shot.get("framing_directive", ""),
                    "composition_note": shot.get("composition_note", ""),
                    # Cinematography rig block (Task #69) — preserve so
                    # downstream stages (style grading, UI badges, video
                    # generator) can read the structured rig/lens metadata.
                    "cinematography": shot.get("cinematography"),
                }
            )

            current_timestamp += synced_duration

        return timeline

    def _validate_storyboard(self, storyboard: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(storyboard, list) or not storyboard:
            raise ValueError("Storyboard must be a non-empty list.")

        repaired: List[Dict[str, Any]] = []
        for i, shot in enumerate(storyboard, start=1):
            if not isinstance(shot, dict):
                raise ValueError(f"Storyboard shot at position {i} must be a dictionary.")

            repaired.append(
                {
                    "shot_index": shot.get("shot_index", i),
                    "shot_id": shot.get("shot_id", f"shot_{i}"),
                    "visual_prompt": str(shot.get("visual_prompt", "")).strip(),
                    "meaning": str(shot.get("meaning", "")).strip(),
                    "function": str(shot.get("function", "emotional_expression")).strip(),
                    "repeat_status": str(shot.get("repeat_status", "original")).strip().lower(),
                    "intensity": self._clamp_float(
                        shot.get("intensity", self.DEFAULT_INTENSITY),
                        self.DEFAULT_INTENSITY,
                    ),
                    "expression_mode": self._repair_expression_mode(
                        shot.get("expression_mode", "environment")
                    ),
                    "reference_image": shot.get("reference_image"),
                    "fidelity_lock": self._clamp_float(shot.get("fidelity_lock", 0.72), 0.72),
                    "character_consistency_id": shot.get("character_consistency_id"),
                    "camera_profile": shot.get("camera_profile", {}),
                    "environment_profile": shot.get("environment_profile", {}),
                    "continuity_anchor": shot.get("continuity_anchor", {}),
                    "rendering_notes": shot.get("rendering_notes", []),
                    # Cinematic variety fields (Task #50)
                    "motion_prompt": str(shot.get("motion_prompt", "")).strip(),
                    "framing_directive": str(shot.get("framing_directive", "")).strip(),
                    "composition_note": str(shot.get("composition_note", "")).strip(),
                    # Cinematography rig block (Task #69) — preserve through
                    # storyboard validation so it survives into the timeline.
                    "cinematography": shot.get("cinematography"),
                }
            )

        return repaired

    def _validate_audio_data(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(audio_data, dict):
            audio_data = {}

        bpm = self._clamp_float(audio_data.get("bpm", self.DEFAULT_BPM), self.DEFAULT_BPM)
        if bpm <= 0:
            bpm = self.DEFAULT_BPM

        beats_per_bar = audio_data.get("beats_per_bar", self.DEFAULT_BEATS_PER_BAR)
        try:
            beats_per_bar = int(beats_per_bar)
        except Exception:
            beats_per_bar = self.DEFAULT_BEATS_PER_BAR
        if beats_per_bar <= 0:
            beats_per_bar = self.DEFAULT_BEATS_PER_BAR

        intensity_curve = audio_data.get("intensity_curve", [])
        if not isinstance(intensity_curve, list):
            intensity_curve = []

        cleaned_curve = []
        for item in intensity_curve:
            cleaned_curve.append(self._extract_curve_intensity(item))

        return {
            "bpm": bpm,
            "beats_per_bar": beats_per_bar,
            "intensity_curve": cleaned_curve,
        }

    def _calculate_duration(
        self,
        intensity: float,
        beat_duration: float,
        repeat_status: str,
        expression_mode: str,
    ) -> float:
        if intensity >= 0.85:
            beats = 2
        elif intensity >= 0.70:
            beats = 3
        elif intensity >= 0.50:
            beats = 4
        elif intensity >= 0.30:
            beats = 6
        else:
            beats = 8

        if repeat_status == "repeat":
            beats = max(2, beats - 1)

        if expression_mode in {"face", "body"} and intensity < 0.75:
            beats += 1

        return beat_duration * beats

    def _snap_duration_to_beat(self, duration: float, beat_duration: float) -> float:
        if beat_duration <= 0:
            return duration

        beats = max(1, round(duration / beat_duration))
        return beats * beat_duration

    def _clamp_duration(self, duration: float) -> float:
        return max(self.min_shot_duration, min(self.max_shot_duration, duration))

    def _get_audio_intensity_for_index(self, intensity_curve: List[float], index: int) -> float:
        if not intensity_curve:
            return self.DEFAULT_INTENSITY

        if index < len(intensity_curve):
            return self._clamp_float(intensity_curve[index], self.DEFAULT_INTENSITY)

        return self._clamp_float(intensity_curve[-1], self.DEFAULT_INTENSITY)

    def _blend_intensity(self, shot_intensity: float, audio_intensity: float) -> float:
        blended = (shot_intensity * 0.6) + (audio_intensity * 0.4)
        return max(0.0, min(1.0, blended))

    _FACE_MOTION_VARIANTS = (
        "minimal motion, intimate hold",
        "barely perceptible drift toward face",
        "static hold with breath-weight stillness",
        "subtle rack focus shift during hold",
    )

    _FACE_HIGH_INTENSITY_VARIANTS = (
        "subtle but emotionally charged push-in",
        "tight hold with controlled rack focus, tension rising",
        "slow push toward eyes, shallow depth tightening",
        "barely perceptible pull-back then hold, heightened stillness",
    )

    _BODY_MOTION_VARIANTS = (
        "gentle cinematic drift",
        "slow body-weight follow",
        "static frame, subject carries motion",
        "slow drift across body plane",
    )

    _BODY_HIGH_INTENSITY_VARIANTS = (
        "controlled body-follow motion",
        "purposeful pan tracking weight shift",
        "tight handheld follow of gesture",
        "slow orbit at heightened emotional beat",
    )

    _ENV_MOTION_VARIANTS = (
        "slow environmental drift",
        "wide static hold, scene breathes naturally",
        "gentle push through open space",
        "slow tilt across landscape",
    )

    def _get_motion_scale(
        self,
        intensity: float,
        expression_mode: str,
        repeat_status: str,
        shot_index: int = 0,
    ) -> str:
        if expression_mode == "face":
            if intensity > 0.8:
                return self._FACE_HIGH_INTENSITY_VARIANTS[shot_index % len(self._FACE_HIGH_INTENSITY_VARIANTS)]
            return self._FACE_MOTION_VARIANTS[shot_index % len(self._FACE_MOTION_VARIANTS)]

        if expression_mode == "body":
            if intensity > 0.75:
                return self._BODY_HIGH_INTENSITY_VARIANTS[shot_index % len(self._BODY_HIGH_INTENSITY_VARIANTS)]
            return self._BODY_MOTION_VARIANTS[shot_index % len(self._BODY_MOTION_VARIANTS)]

        if expression_mode == "macro":
            return "precise micro-motion and premium controlled movement"

        if repeat_status == "repeat":
            if intensity > 0.75:
                return "echoed motion with escalated emphasis"
            return "familiar motion language with slight variation"

        if intensity > 0.8:
            return "dynamic and fast-paced"
        if intensity < 0.3:
            return "static or very slow movement"
        return self._ENV_MOTION_VARIANTS[shot_index % len(self._ENV_MOTION_VARIANTS)]

    def _get_transition_type(
        self,
        current_shot: Dict[str, Any],
        previous_shot: Dict[str, Any],
        intensity: float,
    ) -> str:
        if previous_shot is None:
            return "opening_hold"

        current_repeat = current_shot.get("repeat_status", "original")
        previous_repeat = previous_shot.get("repeat_status", "original")

        if current_repeat == "repeat":
            return "match_emotional_return"

        if current_shot.get("expression_mode") == "symbolic":
            return "poetic_dissolve"

        if intensity > 0.82:
            return "hard_cut"

        if previous_repeat == "repeat" and current_repeat == "original":
            return "release_cut"

        return "straight_cut"

    def _build_timed_visual_prompt(
        self,
        shot: Dict[str, Any],
        motion_scale: str,
        transition: str,
    ) -> str:
        base_prompt = str(shot.get("visual_prompt", "")).strip()
        extras = [
            f"Motion scale: {motion_scale}.",
            f"Transition behavior: {transition}.",
        ]
        return f"{base_prompt} {' '.join(extras)}".strip()

    def _extract_curve_intensity(self, item: Any) -> float:
        if isinstance(item, dict):
            if "intensity" in item:
                return self._clamp_float(item["intensity"], self.DEFAULT_INTENSITY)
            if "value" in item:
                return self._clamp_float(item["value"], self.DEFAULT_INTENSITY)

        return self._clamp_float(item, self.DEFAULT_INTENSITY)

    def _repair_expression_mode(self, value: Any) -> str:
        allowed = {"face", "body", "environment", "symbolic", "macro"}
        if isinstance(value, str) and value.strip().lower() in allowed:
            return value.strip().lower()
        return "environment"

    def _clamp_float(self, value: Any, fallback: float) -> float:
        try:
            number = float(value)
            return max(0.0, min(1.0 if fallback <= 1.0 else number, number))
        except Exception:
            return fallback
