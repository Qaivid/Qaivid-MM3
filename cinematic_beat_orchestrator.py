"""
cinematic_beat_orchestrator.py

MM3.1 cinematic enrichment layer — standalone debug / test utility.
- Runs the 10 new cinematic modules against an already-built storyboard.
- Safe: fails gracefully when any module is missing.
- NOT the main ProductionOrchestrator (production_orchestrator.py) —
  that file handles the full context→storyboard→timeline pipeline.
"""

from typing import Dict, Any, List, Optional

# Optional imports (fail-safe)
def _opt_import(name):
    try:
        return __import__(name, fromlist=['*'])
    except Exception:
        return None

cinematic_beat_engine = _opt_import("cinematic_beat_engine")
behaviour_mapper = _opt_import("behaviour_mapper")
shot_event_builder = _opt_import("shot_event_builder")
shot_variety_engine = _opt_import("shot_variety_engine")
generic_shot_validator = _opt_import("generic_shot_validator")
camera_motivation_engine = _opt_import("camera_motivation_engine")
still_builder = _opt_import("still_keyframe_prompt_builder")
motion_builder = _opt_import("motion_render_prompt_builder")
motif_engine = _opt_import("motif_progression_engine")
chorus_engine = _opt_import("chorus_evolution_engine")


class CinematicBeatOrchestrator:
    def __init__(self, emotional_mode_packet: Optional[Dict[str, Any]] = None):
        self.cine = cinematic_beat_engine.CinematicBeatEngine() if cinematic_beat_engine else None
        self.behaviour = behaviour_mapper.BehaviourMapper() if behaviour_mapper else None
        self.event_builder = shot_event_builder.ShotEventBuilder() if shot_event_builder else None
        # MM3.1: inject emotional_mode_packet so variety cycle is mode-aware.
        self.variety = (
            shot_variety_engine.ShotVarietyEngine(emotional_mode_packet=emotional_mode_packet or {})
            if shot_variety_engine else None
        )
        self.validator = generic_shot_validator.GenericShotValidator() if generic_shot_validator else None
        self.camera = camera_motivation_engine.CameraMotivationEngine() if camera_motivation_engine else None
        self.still = still_builder.StillKeyframePromptBuilder() if still_builder else None
        self.motion = motion_builder.MotionRenderPromptBuilder() if motion_builder else None
        self.motif = motif_engine.MotifProgressionEngine() if motif_engine else None
        self.chorus = chorus_engine.ChorusEvolutionEngine() if chorus_engine else None

    def run(self, context_packet: Dict[str, Any], shots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main orchestration entry.
        Accepts:
            context_packet (from context engine)
            shots (from storyboard engine)
        Returns enriched shots with prompts.
        """

        # 1. Cinematic beats
        if self.cine:
            beats = self.cine.generate_beats(context_packet)
        else:
            beats = [{} for _ in shots]

        # 2. Attach beats to shots
        for i, shot in enumerate(shots):
            shot["cinematic_beat"] = beats[i] if i < len(beats) else {}

        # 3. Build events
        if self.event_builder:
            events = self.event_builder.build_sequence([s.get("cinematic_beat", {}) for s in shots])
            for s, e in zip(shots, events):
                s["shot_event"] = e

        # 4. Motif progression
        if self.motif:
            shots = self.motif.apply_full_progression(shots)

        # 5. Chorus evolution
        if self.chorus:
            shots = self.chorus.apply_evolution(shots)

        # 6. Camera motivation
        if self.camera:
            shots = self.camera.apply_to_sequence(shots)

        # 7. Variety
        if self.variety:
            shots = self.variety.apply_variety(shots)

        # 8. Validation
        if self.validator:
            shots = self.validator.validate_sequence(shots)

        # 9. Prompt generation
        for shot in shots:
            if self.still:
                shot["still_prompt"] = self.still.build_prompt(shot.get("shot_event", shot))
            if self.motion:
                shot["motion_prompt"] = self.motion.build_prompt(shot.get("shot_event", shot))

        return {
            "context": context_packet,
            "shots": shots
        }
