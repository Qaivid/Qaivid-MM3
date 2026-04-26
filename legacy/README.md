# V1 Legacy Engines — Quarantined (Task #163)

These files are **retired V1 pipeline engines** preserved here for reference and
back-compat only.  No new code should import from this directory except as noted
below.

## Files

| File | Status | Notes |
|------|--------|-------|
| `production_orchestrator.py` | **Partially active** — `run_context_only()` still called by one path in `pipeline_worker.py` pending Task #165. `run_to_timeline()` is fully retired. | |
| `visual_storyboard_engine.py` | **Retired** — replaced by `storyboard_engine_v2.py` (pure intent layer) | Do not call |
| `rhythmic_assembly_engine.py` | **Retired** — replaced by `timeline_builder_v2.py` (BPM timing) | Do not call |

## V2 Replacements

| Retired V1 | V2 replacement | Stage |
|------------|----------------|-------|
| `ProductionOrchestrator.run_to_timeline()` | `timeline_builder_v2.build_timeline_from_brief()` | After Creative Brief (Stage 6) |
| `VisualStoryboardEngine.build_storyboard()` | `storyboard_engine_v2.generate_storyboard_v2()` | Storyboard Stage 5 |
| `RhythmicAssemblyEngine.assemble_timeline()` | `timeline_builder_v2` (BPM/lyric timing) | After Creative Brief (Stage 6) |
| `StyleGradingEngine` (called with `{"preset": ...}`) | `timeline_builder_v2._apply_style_grading()` with full `style_packet` | After Creative Brief (Stage 6) |

## Migration path

- Task #165: Remove the last `ProductionOrchestrator.run_context_only()` usage and
  call `UnifiedContextEngine` directly so this entire directory becomes inactive.
