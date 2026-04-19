"""
behaviour_mapper.py

Qaivid MM3.1
------------
Maps abstract emotional states into human visual behaviour patterns.

Purpose:
- convert emotion labels into filmable behaviour
- support the CinematicBeatEngine
- remain standalone and safe to add without changing current MM3 flow

Design notes:
- deterministic first; easy to expand later
- culture/world aware in a lightweight way
- returns structured behaviour candidates, not finished prompts
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import copy
import re


@dataclass
class BehaviourCandidate:
    emotion: str
    behaviour_type: str
    subject_action: str
    trigger_event: str
    object_usage: str
    environment_usage: str
    emotional_shift: str
    visual_contrast: str
    intensity: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BehaviourMapper:
    """
    Converts emotion signals into filmable behaviour candidates.

    Main methods:
    - map_emotion(...)
    - best_for_context(...)
    """

    _EMOTION_LIBRARY: Dict[str, List[Dict[str, str]]] = {
        "longing": [
            {
                "behaviour_type": "ritualised_waiting",
                "subject_action": "waits in a place where someone used to arrive",
                "trigger_event": "a distant sound suggests a possible return",
                "object_usage": "holds an everyday object without actually using it",
                "environment_usage": "the threshold or doorway becomes the emotional focal point",
                "emotional_shift": "hope → restraint",
                "visual_contrast": "ordinary routine versus private expectation",
                "intensity": "medium",
            },
            {
                "behaviour_type": "false_anticipation",
                "subject_action": "pauses mid-task and listens before pretending not to react",
                "trigger_event": "a passing vehicle, footstep, ringtone, or gate sound",
                "object_usage": "the item in hand slips, stalls, or hangs unfinished",
                "environment_usage": "a wall, curtain, or courtyard edge blocks clear view",
                "emotional_shift": "hope → collapse",
                "visual_contrast": "sudden alertness versus immediate disappointment",
                "intensity": "high",
            },
            {
                "behaviour_type": "repetitive_checking",
                "subject_action": "checks for a sign of contact again with less hope than before",
                "trigger_event": "a quiet interval invites the habit to return",
                "object_usage": "revisits the phone, letter, gift, or keepsake",
                "environment_usage": "the surrounding silence amplifies the action",
                "emotional_shift": "expectation → emptiness",
                "visual_contrast": "small repeated action versus growing emotional damage",
                "intensity": "medium",
            },
        ],
        "sorrow": [
            {
                "behaviour_type": "unfinished_routine",
                "subject_action": "continues a routine slowly and stops before finishing it",
                "trigger_event": "a memory or sensory reminder interrupts concentration",
                "object_usage": "a domestic or personal object remains half-used",
                "environment_usage": "the familiar room feels too large or too quiet",
                "emotional_shift": "containment → ache",
                "visual_contrast": "normal routine versus inner collapse",
                "intensity": "medium",
            },
            {
                "behaviour_type": "private_breakdown",
                "subject_action": "turns away from others or from the open space to conceal emotion",
                "trigger_event": "the effort to remain composed finally fails",
                "object_usage": "grips fabric, furniture, or a keepsake too tightly",
                "environment_usage": "a corner, doorway, mirror, or shadowed edge offers cover",
                "emotional_shift": "control → release",
                "visual_contrast": "outward restraint versus inward overflow",
                "intensity": "high",
            },
        ],
        "despair": [
            {
                "behaviour_type": "collapse_of_will",
                "subject_action": "sits or lowers themselves as though the body has given up first",
                "trigger_event": "the last remaining hope clearly disappears",
                "object_usage": "lets an important object fall or drift aside",
                "environment_usage": "the empty surrounding space swallows the figure",
                "emotional_shift": "strain → surrender",
                "visual_contrast": "previous effort versus total depletion",
                "intensity": "high",
            },
            {
                "behaviour_type": "numb_repetition",
                "subject_action": "repeats a meaningless routine with no expectation of change",
                "trigger_event": "habit continues even after hope has ended",
                "object_usage": "handles the same item mechanically",
                "environment_usage": "time feels stagnant and the setting barely changes",
                "emotional_shift": "pain → numbness",
                "visual_contrast": "motion still present but feeling nearly absent",
                "intensity": "high",
            },
        ],
        "nostalgia": [
            {
                "behaviour_type": "retrace_memory",
                "subject_action": "recreates a small past gesture without anyone else being there",
                "trigger_event": "a place, object, or sound reactivates memory",
                "object_usage": "touches or repositions an object tied to the past",
                "environment_usage": "the present-day space overlays with remembered emotional presence",
                "emotional_shift": "distance → brief nearness",
                "visual_contrast": "present emptiness versus remembered intimacy",
                "intensity": "medium",
            },
            {
                "behaviour_type": "lingering_touch",
                "subject_action": "touches a surface or keepsake longer than necessary",
                "trigger_event": "the object quietly carries emotional residue",
                "object_usage": "uses the object as a substitute for direct contact",
                "environment_usage": "stillness in the room allows memory to dominate",
                "emotional_shift": "composure → soft ache",
                "visual_contrast": "small physical contact versus large emotional return",
                "intensity": "low",
            },
        ],
        "betrayal": [
            {
                "behaviour_type": "recoil",
                "subject_action": "pulls back from an object, place, or gesture once associated with trust",
                "trigger_event": "a reminder makes old certainty feel contaminated",
                "object_usage": "rejects, drops, hides, or turns away from the keepsake",
                "environment_usage": "the setting itself feels altered by broken trust",
                "emotional_shift": "recognition → rejection",
                "visual_contrast": "past closeness versus present refusal",
                "intensity": "high",
            },
            {
                "behaviour_type": "controlled_anger",
                "subject_action": "contains anger through precise, almost overly controlled movement",
                "trigger_event": "emotion rises but is not allowed open release",
                "object_usage": "folds, tears, crushes, or aligns something with deliberate force",
                "environment_usage": "the room becomes rigid and formal rather than intimate",
                "emotional_shift": "hurt → control",
                "visual_contrast": "calm surface versus violent emotional pressure",
                "intensity": "high",
            },
        ],
        "yearning": [
            {
                "behaviour_type": "toward_then_stop",
                "subject_action": "moves toward a possibility and stops before reaching it",
                "trigger_event": "hope rises faster than certainty",
                "object_usage": "extends a hand, then withdraws before contact",
                "environment_usage": "distance in the frame becomes emotionally visible",
                "emotional_shift": "impulse → restraint",
                "visual_contrast": "desire to close distance versus inability to complete the motion",
                "intensity": "medium",
            }
        ],
        "hope": [
            {
                "behaviour_type": "careful_revival",
                "subject_action": "allows a small hopeful action but protects against disappointment",
                "trigger_event": "a sign appears that might mean return or change",
                "object_usage": "straightens, reopens, or prepares something that had been abandoned",
                "environment_usage": "light, doorway, or open space feels newly active",
                "emotional_shift": "doubt → cautious openness",
                "visual_contrast": "fragility of hope versus desire to believe it",
                "intensity": "medium",
            }
        ],
        "love": [
            {
                "behaviour_type": "quiet_intimacy",
                "subject_action": "handles a shared object or remembered gesture with tenderness",
                "trigger_event": "an intimate memory quietly resurfaces",
                "object_usage": "uses an object as proof of emotional continuity",
                "environment_usage": "the environment softens around the intimate action",
                "emotional_shift": "distance → warmth",
                "visual_contrast": "physical absence versus emotional closeness",
                "intensity": "low",
            }
        ],
        "regret": [
            {
                "behaviour_type": "hesitating_return",
                "subject_action": "revisits a decision or gesture too late to change it easily",
                "trigger_event": "memory reframes a past action as mistake",
                "object_usage": "reopens, rereads, or restores something once rejected",
                "environment_usage": "the same place now feels morally heavier",
                "emotional_shift": "defensiveness → remorse",
                "visual_contrast": "the wish to undo versus the fact of lateness",
                "intensity": "medium",
            }
        ],
    }

    _ALIASES: Dict[str, str] = {
        "ache": "sorrow",
        "aching": "sorrow",
        "heartbreak": "sorrow",
        "pain": "sorrow",
        "grief": "sorrow",
        "melancholy": "nostalgia",
        "sadness": "sorrow",
        "loneliness": "longing",
        "isolation": "longing",
        "missing": "longing",
        "loss": "despair",
        "devastation": "despair",
    }

    _WORLD_OBJECT_OVERRIDES: Dict[str, List[str]] = {
        "phone": ["phone", "screen", "call log", "unread message"],
        "letter": ["letter", "envelope", "paper note", "creased page"],
        "rural": ["dupatta", "metal cup", "charpai edge", "cloth bundle"],
        "domestic": ["cup", "fabric", "chair back", "door latch"],
        "wedding": ["bangle", "veil edge", "ornament box", "flower thread"],
        "music": ["instrument case", "earring", "scarf", "microphone cord"],
    }

    def normalize_emotion(self, emotion: Optional[str]) -> str:
        if not emotion:
            return "longing"
        e = re.sub(r"[^a-zA-Z_ -]", "", str(emotion)).strip().lower()
        e = e.replace("-", " ")
        if e in self._EMOTION_LIBRARY:
            return e
        if e in self._ALIASES:
            return self._ALIASES[e]
        # fuzzy fallback by token
        for token in e.split():
            if token in self._EMOTION_LIBRARY:
                return token
            if token in self._ALIASES:
                return self._ALIASES[token]
        return "longing"

    def map_emotion(
        self,
        emotion: Optional[str],
        *,
        world_hint: str = "",
        motif_hint: str = "",
        intensity_hint: str = "",
    ) -> List[Dict[str, Any]]:
        normalized = self.normalize_emotion(emotion)
        templates = copy.deepcopy(self._EMOTION_LIBRARY.get(normalized, self._EMOTION_LIBRARY["longing"]))

        enriched: List[Dict[str, Any]] = []
        for template in templates:
            candidate = BehaviourCandidate(
                emotion=normalized,
                behaviour_type=template["behaviour_type"],
                subject_action=template["subject_action"],
                trigger_event=template["trigger_event"],
                object_usage=self._select_object_usage(template["object_usage"], world_hint, motif_hint),
                environment_usage=self._select_environment_usage(template["environment_usage"], world_hint),
                emotional_shift=template["emotional_shift"],
                visual_contrast=template["visual_contrast"],
                intensity=intensity_hint or template.get("intensity", "medium"),
            )
            enriched.append(candidate.to_dict())
        return enriched

    def best_for_context(
        self,
        emotion: Optional[str],
        *,
        world_hint: str = "",
        motif_hint: str = "",
        intensity_hint: str = "",
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        candidates = self.map_emotion(
            emotion,
            world_hint=world_hint,
            motif_hint=motif_hint,
            intensity_hint=intensity_hint,
        )
        return candidates[:max_results]

    def _select_object_usage(self, base_text: str, world_hint: str, motif_hint: str) -> str:
        motif = (motif_hint or "").strip().lower()
        world = (world_hint or "").strip().lower()

        if motif:
            return f"{base_text}; the key emotional object is the {motif}"
        for key, objects in self._WORLD_OBJECT_OVERRIDES.items():
            if key in world:
                return f"{base_text}; suitable object options include {', '.join(objects)}"
        return base_text

    def _select_environment_usage(self, base_text: str, world_hint: str) -> str:
        world = (world_hint or "").strip().lower()
        if "courtyard" in world:
            return f"{base_text}; the courtyard boundary and open sky should matter visually"
        if "village" in world or "rural" in world:
            return f"{base_text}; the rural space should feel lived-in and emotionally observant"
        if "city" in world or "urban" in world:
            return f"{base_text}; nearby structures, windows, traffic, or reflected light should influence the scene"
        if "wedding" in world:
            return f"{base_text}; traces of celebration should now feel emotionally displaced"
        return base_text


__all__ = ["BehaviourMapper", "BehaviourCandidate"]
