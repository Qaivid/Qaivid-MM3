"""Regression: cinematography block survives storyboard → timeline → styled.

Task #69 — the structured cinematography rig block must propagate from the
VisualStoryboardEngine through RhythmicAssemblyEngine and StyleGradingEngine
into the final styled_timeline. Earlier validation whitelists silently dropped
unknown fields; this test guards against that regression.
"""

from rhythmic_assembly_engine import RhythmicAssemblyEngine
from style_grading_engine import StyleGradingEngine


_CINE = {
    "rig": "dolly",
    "direction": "slow push-in",
    "speed": "slow",
    "lens": "85mm portrait, shallow depth of field",
    "intensity": "medium",
    "justification": "intimate emotion match",
}


def _storyboard():
    return [
        {
            "shot_index": i,
            "shot_id": f"shot_{i}",
            "visual_prompt": f"a quiet frame {i}",
            "meaning": "longing",
            "function": "emotional_expression",
            "repeat_status": "original",
            "intensity": 0.6,
            "expression_mode": "face",
            "motion_prompt": "slow dolly — push-in, 85mm",
            "framing_directive": "medium close-up",
            "composition_note": "centered subject",
            "cinematography": dict(_CINE),
        }
        for i in range(1, 4)
    ]


def test_cinematography_survives_assembly_and_grading():
    storyboard = _storyboard()
    timeline = RhythmicAssemblyEngine().assemble_timeline(storyboard, audio_data=None)
    assert timeline, "assembly returned empty"
    for shot in timeline:
        assert isinstance(shot.get("cinematography"), dict), \
            "cinematography dropped by RhythmicAssemblyEngine"
        assert shot["cinematography"]["rig"] == "dolly"
        assert shot["cinematography"]["lens"].startswith("85mm")

    styled = StyleGradingEngine().apply_style(
        timeline=timeline,
        style_profile={"preset": "cinematic_natural"},
    )
    assert styled, "grading returned empty"
    for shot in styled:
        assert isinstance(shot.get("cinematography"), dict), \
            "cinematography dropped by StyleGradingEngine"
        assert shot["cinematography"]["rig"] == "dolly"
