import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AssetExportModule:
    """
    Qaivid Asset Export Module
    """

    DEFAULT_FPS = 24
    DEFAULT_ASPECT_RATIO = "16:9"

    def __init__(self, production_name: str = "Qaivid_Project", fps: int = DEFAULT_FPS):
        self.production_name = production_name or "Qaivid_Project"
        self.fps = fps if isinstance(fps, int) and fps > 0 else self.DEFAULT_FPS

    def export_to_json(
        self,
        final_timeline: List[Dict[str, Any]],
        include_summary: bool = True,
    ) -> str:
        timeline = self._validate_timeline(final_timeline)

        export_data = {
            "project_name": self.production_name,
            "version": "Qaivid_Core_4.0",
            "fps": self.fps,
            "aspect_ratio": self.DEFAULT_ASPECT_RATIO,
            "frames_total": self._calculate_total_frames(timeline),
            "shots_total": len(timeline),
            "shots": timeline,
        }

        if include_summary:
            export_data["summary"] = self._build_summary(timeline)

        return json.dumps(export_data, indent=4, ensure_ascii=False)

    def generate_api_payloads(
        self,
        final_timeline: List[Dict[str, Any]],
        target_api: str = "generic",
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    ) -> List[Dict[str, Any]]:
        timeline = self._validate_timeline(final_timeline)
        target_api = (target_api or "generic").strip().lower()

        payloads = []
        for shot in timeline:
            prompt = shot.get("styled_visual_prompt") or shot.get("visual_prompt") or ""
            duration = shot.get("duration", 2.0)

            base_payload = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
            }

            if shot.get("reference_image"):
                base_payload["image_prompt"] = shot["reference_image"]
                base_payload["image_fidelity_weight"] = shot.get("fidelity_lock", 0.72)

            metadata = {
                "shot_id": shot.get("shot_id"),
                "shot_index": shot.get("shot_index"),
                "timeline_index": shot.get("timeline_index"),
                "meaning": shot.get("meaning", ""),
                "repeat_status": shot.get("repeat_status", "original"),
                "motion_scale": shot.get("motion_scale", ""),
                "transition": shot.get("transition", ""),
                "style_preset": shot.get("style_preset", ""),
                "style_strength": shot.get("style_strength", 0.0),
                "character_consistency_id": shot.get("character_consistency_id"),
            }

            formatted_payload = self._format_api_payload(
                base_payload=base_payload,
                shot=shot,
                metadata=metadata,
                target_api=target_api,
            )

            payloads.append(
                {
                    "shot_id": shot.get("shot_id"),
                    "target_api": target_api,
                    "payload": formatted_payload,
                    "metadata": metadata,
                }
            )

        return payloads

    def generate_edl(self, final_timeline: List[Dict[str, Any]]) -> str:
        timeline = self._validate_timeline(final_timeline)

        edl_lines = [
            f"TITLE: {self.production_name}",
            "FCM: NON-DROP FRAME",
            "",
        ]

        current_time = 0.0

        for i, shot in enumerate(timeline, start=1):
            start = self._seconds_to_timecode(current_time)
            end = self._seconds_to_timecode(current_time + shot["duration"])

            shot_id = shot.get("shot_id", f"shot_{i}")
            meaning = (shot.get("meaning", "") or "").replace("\n", " ").strip()
            meaning_preview = meaning[:80] + ("..." if len(meaning) > 80 else "")

            edl_lines.append(
                f"{str(i).zfill(3)}  AX       V     C        {start} {end} {start} {end}"
            )
            edl_lines.append(f"* FROM CLIP NAME: {shot_id}.mp4")
            edl_lines.append(f"* MEANING: {meaning_preview}")
            edl_lines.append("")

            current_time += shot["duration"]

        return "\n".join(edl_lines)

    def export_master_package(
        self,
        final_timeline: List[Dict[str, Any]],
        target_api: str = "generic",
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    ) -> Dict[str, Any]:
        timeline = self._validate_timeline(final_timeline)

        return {
            "project_name": self.production_name,
            "fps": self.fps,
            "aspect_ratio": aspect_ratio,
            "summary": self._build_summary(timeline),
            "json_export": json.loads(self.export_to_json(timeline, include_summary=True)),
            "api_payloads": self.generate_api_payloads(
                final_timeline=timeline,
                target_api=target_api,
                aspect_ratio=aspect_ratio,
            ),
            "edl": self.generate_edl(timeline),
        }

    def _validate_timeline(self, final_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(final_timeline, list) or not final_timeline:
            raise ValueError("Final timeline must be a non-empty list.")

        repaired = []
        for i, shot in enumerate(final_timeline, start=1):
            if not isinstance(shot, dict):
                raise ValueError(f"Timeline shot at position {i} must be a dictionary.")

            repaired.append(
                {
                    "timeline_index": shot.get("timeline_index", i),
                    "shot_index": shot.get("shot_index", i),
                    "shot_id": shot.get("shot_id", f"shot_{i}"),
                    "start_time": self._coerce_float(shot.get("start_time", 0.0), 0.0),
                    "duration": self._coerce_float(shot.get("duration", 2.0), 2.0),
                    "end_time": self._coerce_float(shot.get("end_time", 2.0), 2.0),
                    "visual_prompt": str(shot.get("visual_prompt", "")).strip(),
                    "styled_visual_prompt": str(shot.get("styled_visual_prompt", "")).strip(),
                    "meaning": str(shot.get("meaning", "")).strip(),
                    "function": str(shot.get("function", "emotional_expression")).strip(),
                    "repeat_status": str(shot.get("repeat_status", "original")).strip().lower(),
                    "intensity": self._clamp_01(shot.get("intensity", 0.5), 0.5),
                    "motion_scale": str(shot.get("motion_scale", "")).strip(),
                    "transition": str(shot.get("transition", "")).strip(),
                    "expression_mode": str(shot.get("expression_mode", "environment")).strip().lower(),
                    "reference_image": shot.get("reference_image"),
                    "fidelity_lock": self._clamp_01(shot.get("fidelity_lock", 0.72), 0.72),
                    "character_consistency_id": shot.get("character_consistency_id"),
                    "camera_profile": shot.get("camera_profile", {}),
                    "environment_profile": shot.get("environment_profile", {}),
                    "continuity_anchor": shot.get("continuity_anchor", {}),
                    "rendering_notes": shot.get("rendering_notes", []),
                    "style_preset": str(shot.get("style_preset", "")).strip(),
                    "style_strength": self._clamp_01(shot.get("style_strength", 0.0), 0.0),
                    "color_palette": str(shot.get("color_palette", "")).strip(),
                    "lighting_style": str(shot.get("lighting_style", "")).strip(),
                    "contrast_profile": str(shot.get("contrast_profile", "")).strip(),
                    "texture_profile": str(shot.get("texture_profile", "")).strip(),
                    "atmosphere_profile": str(shot.get("atmosphere_profile", "")).strip(),
                    "lens_feel": str(shot.get("lens_feel", "")).strip(),
                    "style_notes": shot.get("style_notes", []),
                }
            )

        return repaired

    def _format_api_payload(
        self,
        base_payload: Dict[str, Any],
        shot: Dict[str, Any],
        metadata: Dict[str, Any],
        target_api: str,
    ) -> Dict[str, Any]:
        if target_api == "runway":
            payload = {
                "promptText": base_payload["prompt"],
                "duration": base_payload["duration"],
                "ratio": base_payload["aspect_ratio"],
            }
            if "image_prompt" in base_payload:
                payload["imagePrompt"] = base_payload["image_prompt"]
                payload["imageFidelity"] = base_payload.get("image_fidelity_weight", 0.72)
            return payload

        if target_api == "kling":
            payload = {
                "prompt": base_payload["prompt"],
                "duration": base_payload["duration"],
                "aspect_ratio": base_payload["aspect_ratio"],
                "motion_strength": shot.get("motion_scale", ""),
            }
            if "image_prompt" in base_payload:
                payload["reference_image"] = base_payload["image_prompt"]
                payload["reference_strength"] = base_payload.get("image_fidelity_weight", 0.72)
            return payload

        if target_api == "luma":
            payload = {
                "prompt": base_payload["prompt"],
                "duration": base_payload["duration"],
                "aspect_ratio": base_payload["aspect_ratio"],
            }
            if "image_prompt" in base_payload:
                payload["keyframes"] = [
                    {
                        "type": "image",
                        "url": base_payload["image_prompt"],
                        "weight": base_payload.get("image_fidelity_weight", 0.72),
                    }
                ]
            return payload

        return dict(base_payload)

    def _build_summary(self, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_duration = round(sum(shot["duration"] for shot in timeline), 3)
        total_frames = self._calculate_total_frames(timeline)

        return {
            "project_name": self.production_name,
            "shots_total": len(timeline),
            "total_duration_seconds": total_duration,
            "frames_total": total_frames,
            "fps": self.fps,
            "has_reference_images": any(bool(shot.get("reference_image")) for shot in timeline),
            "styled_shots_total": sum(
                1 for shot in timeline if shot.get("styled_visual_prompt")
            ),
        }

    def _calculate_total_frames(self, timeline: List[Dict[str, Any]]) -> int:
        return sum(int(round(shot["duration"] * self.fps)) for shot in timeline)

    def _seconds_to_timecode(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int(round((seconds % 1) * self.fps))

        if frames >= self.fps:
            frames = 0
            secs += 1
            if secs >= 60:
                secs = 0
                mins += 1
                if mins >= 60:
                    mins = 0
                    hrs += 1

        return f"{hrs:02}:{mins:02}:{secs:02}:{frames:02}"

    def _coerce_float(self, value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

    def _clamp_01(self, value: Any, fallback: float) -> float:
        try:
            num = float(value)
            return max(0.0, min(1.0, num))
        except Exception:
            return fallback
