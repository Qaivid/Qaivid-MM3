"""
Project Brain — shared intelligence state for a single Qaivid project.

Pipeline  = execution order (who runs when)
Brain     = source of truth (what every stage knows)

Architecture rule:
    Each stage WRITES to its own namespace only.
    Each stage may READ from any namespace above it.
    The brain is append-only — no stage edits another stage's work.

Namespaces (in pipeline order):
    raw_input          Stage 0  — original text/audio/genre submitted by user
    project_settings   Stage 0  — genre, duration, platform, style_preset (user/system config)
    input_structure    Stage 1  — InputProcessor output (sections, units, repetition map…)
    context_packet     Stage 2  — Context Engine output (meaning, world, speaker…)
    narrative_packet   Stage 3  — Narrative Engine output (motifs, arc, story logic…)
    style_packet       Stage 4  — Style Engine output (visual language, palette…)
    storyboard_packet  Stage 5  — Storyboard intent (scenes + valid_realizations) — POSSIBILITIES
    creative_briefs    Stage 6  — Creative Brief — ONE locked direction per scene (SELECTION)
    character_bible    Stage 7  — Materializer character anchors
    location_bible     Stage 7  — Materializer location anchors
    reference_assets   Stage 8  — Reference image metadata
    shot_plan          Stage 9  — Final shot asset plan
    continuity_state   Cross    — Running continuity rules (updated by multiple stages)
    render_outputs     Stage 10/11 — Final video / assembly outputs
    validation_notes   Cross    — Warnings and flags accumulated across stages
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NAMESPACES: List[str] = [
    "raw_input",
    "project_settings",
    "input_structure",
    "context_packet",
    "narrative_packet",
    "style_packet",
    "storyboard_packet",
    "creative_briefs",
    "materializer_packet",
    "character_bible",
    "location_bible",
    "reference_assets",
    "shot_plan",
    "continuity_state",
    "render_outputs",
    "validation_notes",
]

NAMESPACE_OWNERS: Dict[str, str] = {
    "raw_input":            "stage_0",
    "project_settings":     "stage_0",
    "input_structure":      "stage_1_input",
    "context_packet":       "stage_2_context",
    "narrative_packet":     "stage_3_narrative",
    "style_packet":         "stage_4_style",
    "storyboard_packet":    "stage_5_storyboard",
    "creative_briefs":      "stage_6_brief",
    # Stage 7 — Materializer: materializer_packet is the authoritative output.
    # character_bible and location_bible are mirrors kept for backward compat
    # (older stages that read brain.character_bible still work unchanged).
    "materializer_packet":  "stage_7_materializer",
    "character_bible":      "stage_7_materializer",
    "location_bible":       "stage_7_materializer",
    "reference_assets":     "stage_8_refs",
    "shot_plan":            "stage_9_stills",
    "continuity_state":     "cross_stage",
    "render_outputs":       "stage_10_assembly",
    "validation_notes":     "cross_stage",
}


class ProjectBrain:
    """
    Per-project intelligence state.

    Load from DB → read/write namespaces → save back to DB.

    Usage in a pipeline stage:
        brain = ProjectBrain.load(project_id, conn)
        input_structure = brain.read("input_structure")
        context = run_context_engine(input_structure)
        brain.write("context_packet", context)
        brain.save(conn)
    """

    def __init__(self, project_id: str, data: Optional[Dict[str, Any]] = None):
        self.project_id = project_id
        self._data: Dict[str, Any] = {ns: {} for ns in NAMESPACES}
        self._data["validation_notes"] = []
        if data:
            for ns in NAMESPACES:
                if ns in data:
                    self._data[ns] = data[ns]

    @classmethod
    def load(cls, project_id: str, conn) -> "ProjectBrain":
        """Load brain from the project_brain JSONB column."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT project_brain FROM projects WHERE id = %s",
                (project_id,),
            )
            row = cur.fetchone()
        if not row:
            raise RuntimeError(f"Project {project_id} not found.")
        raw = row["project_brain"] if row["project_brain"] else {}
        return cls(project_id, raw)

    def read(self, namespace: str) -> Any:
        """Read any namespace. Returns empty dict/list if not yet populated."""
        if namespace not in NAMESPACES:
            raise ValueError(f"Unknown brain namespace: {namespace!r}")
        return self._data.get(namespace, {} if namespace != "validation_notes" else [])

    def write(self, namespace: str, data: Any) -> None:
        """
        Write to a namespace.
        Logs a warning if called out of pipeline order (namespace already has data
        being overwritten) — does not block, just makes violations visible.
        """
        if namespace not in NAMESPACES:
            raise ValueError(f"Unknown brain namespace: {namespace!r}")
        existing = self._data.get(namespace)
        if existing and existing != {} and existing != []:
            logger.warning(
                "ProjectBrain: overwriting namespace %r for project %s — "
                "check pipeline order.",
                namespace,
                self.project_id,
            )
        self._data[namespace] = data

    def add_validation_note(self, stage: str, code: str, detail: str) -> None:
        """Append a validation note without overwriting existing notes."""
        notes = self._data.get("validation_notes")
        if not isinstance(notes, list):
            self._data["validation_notes"] = []
        self._data["validation_notes"].append({
            "stage": stage,
            "code": code,
            "detail": detail,
        })

    def update_continuity(self, patch: Dict[str, Any]) -> None:
        """Merge a patch into continuity_state (append-safe for lists)."""
        cs = self._data.get("continuity_state")
        if not isinstance(cs, dict):
            self._data["continuity_state"] = {}
            cs = self._data["continuity_state"]
        for key, value in patch.items():
            if isinstance(value, list) and isinstance(cs.get(key), list):
                cs[key] = cs[key] + value
            else:
                cs[key] = value

    def save(self, conn) -> None:
        """Persist brain to DB. Uses JSONB merge so partial writes are safe."""
        from psycopg.types.json import Json
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE projects
                SET project_brain = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (Json(self._data), self.project_id),
            )

    def get_full(self) -> Dict[str, Any]:
        """Return a copy of the full brain dict."""
        return dict(self._data)

    def is_populated(self, namespace: str) -> bool:
        """Return True if a namespace has been written to."""
        val = self._data.get(namespace)
        return bool(val)

    def summary(self) -> str:
        """Human-readable one-liner of what's in the brain."""
        parts = []
        for ns in NAMESPACES:
            val = self._data.get(ns)
            if val and val != {} and val != []:
                parts.append(ns)
        return f"ProjectBrain[{self.project_id}] populated: {', '.join(parts) or 'empty'}"
