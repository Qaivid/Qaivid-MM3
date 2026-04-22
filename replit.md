# Qaivid MetaMind 3.1

A web-based SaaS that turns lyrics, poems, scripts, and stories into beat-synced cinematic music videos, powered by a multi-stage user-gated pipeline (audio analysis → context extraction → visual storyboard → rhythmic timeline → style grading → image/video generation).

## Architecture

- **Backend**: Flask (`app.py`) running on port 5000
- **Database**: Replit PostgreSQL via `psycopg` (connection via `DATABASE_URL`)
- **Frontend**: Server-rendered Jinja2 templates + plain CSS (no build step)
- **Core pipeline engines**:
  - `audio_processor.py` - extracts BPM, beats, segments, energy from audio
  - `unified_context_engine_master.py` - GPT-4o context extraction (theme, speaker, location)
  - `visual_storyboard_engine.py` - generates visual prompts per beat; MM3.1 adds cinematic beat enrichment via `_attach_optional_cinematic_layers()`
  - `rhythmic_assembly_engine.py` - assembles a beat-synced timeline; normalizes shot durations to match `audio_data.duration_seconds` (Task #104 fix — ensures timeline covers full audio, not just lyric-line subset)
  - `style_grading_engine.py` - applies cinematic style profiles; passes through MM3.1 fields
  - `asset_export_module.py` - exports JSON for downstream tooling
  - `production_orchestrator.py` - coordinates all engines end-to-end (context→storyboard→timeline)
- **MM3.1 Cinematic Beat Engine** (all fail-safe, loaded optionally inside VSE):
  - `cinematic_beat_engine.py` - emotion → behaviour → beat objects (fixes "static portrait" problem)
  - `behaviour_mapper.py` - emotion label → concrete physical behaviour library
  - `shot_event_builder.py` - beat → shot event (action, trigger, contrast, shift, environment)
  - `shot_variety_engine.py` - cycles shot_type across 8 types (portrait/movement/wide_env/etc.)
  - `generic_shot_validator.py` - flags and filters generic shots
  - `camera_motivation_engine.py` - subject action → camera_plan (movement, style, intensity)
  - `still_keyframe_prompt_builder.py` - event → still image prompt
  - `motion_render_prompt_builder.py` - event → motion/video prompt
  - `motif_progression_engine.py` - tracks motif stages (introduction→resolution) across sequence
  - `chorus_evolution_engine.py` - evolves repeated chorus shots (hope→numbness arc)
  - `cinematic_beat_orchestrator.py` - standalone debug/test utility for the 10-module chain
- **Updated core modules** (MM3.1 backward-compatible replacements):
  - `cinematography_engine.py` - adds ACTION_TO_RIG, CAMERA_PLAN_TO_RIG, event-signal rig selection
  - `shot_prompt_composer.py` - event lead sentence now preferred over raw meaning in prompts
- **Auth**: `auth.py` - login_required + admin_required decorators
- **Web layer**: `app.py`, `templates/`, `static/`

## Required Secrets

- `OPENAI_API_KEY` - GPT-4o context engine
- `FAL_API_KEY` - FAL image generation (FLUX schnell + FLUX-PuLID for face-lock)
- `DATABASE_URL` - Provisioned automatically (Replit Postgres)
- Optional: `ATLAS_CLOUD_API_KEY`, `GEMINI_API_KEY` (for upcoming video phase)

## Database Schema

`users` table:
- `id SERIAL PRIMARY KEY`, `email TEXT UNIQUE NOT NULL`, `password_hash TEXT NOT NULL`
- `is_admin BOOLEAN DEFAULT FALSE`, `created_at`

`projects` table:
- `id TEXT PRIMARY KEY` (12-char hex)
- `user_id INTEGER REFERENCES users(id) ON DELETE CASCADE` (indexed)
- `name`, `genre`, `text`, `audio_filename`, `status`, `error`
- `summary JSONB`, `context_packet JSONB`, `styled_timeline JSONB`, `progress JSONB`
- `export_path TEXT`, `created_at`, `updated_at`

`refs` table — character + environment reference images
- `(project_id, role)` unique; `role` in `character|environment`
- `source` in `uploaded|generated`, `status`, `file_path`, `prompt`, `error`

`shot_assets` table — per-shot stills
- `(project_id, shot_index)` unique; `status`, `file_path`, `prompt`, `error`

`characters` table — materialized from context_packet after each pipeline run
- `project_id FK → projects(cascade)`, `name`, `role`, `entity_type` (speaker|named_entity)
- `appearance`, `age_range`, `cultural_notes`, `emotional_notes`, `metadata JSONB`
- Populated by `character_materializer.py`; idempotent (skips if rows exist)

`locations` table — materialized from context_packet after each pipeline run
- `project_id FK → projects(cascade)`, `name`, `description`, `time_of_day`, `mood`
- `cultural_notes`, `visual_details`, `entity_type` (world_dna|named_place), `metadata JSONB`
- Populated by `location_materializer.py`; idempotent

## Entity Materializers

After `_store_pipeline_result()` in `pipeline_worker.py`, the context_packet's:
- `speaker` dict → primary character row (`entity_type=speaker`)
- `entities[type=person]` → secondary character rows (`entity_type=named_entity`)
- `location_dna` → primary location row (`entity_type=world_dna`)
- `entities[type=place]` → named location rows (`entity_type=named_place`)

API: `GET /project/<id>/entities` (login_required + ownership) → `{characters:[], locations:[]}`.
Result page fetches entities on load and on pipeline completion, renders styled cards.
Materializer failures are non-fatal (caught, logged, pipeline continues).

## Auth

`auth.py` owns user CRUD, password hashing (`werkzeug.security`), session glue,
and `@login_required`. Sessions are Flask signed cookies storing only `user_id`;
every request re-fetches the user row so deleted users are kicked immediately.

Routes:
- `GET/POST /signup`, `GET/POST /login`, `POST /logout`
- All `/generate` and `/project/...` routes are gated by `@login_required` and
  scope queries to `current_user().id`. Cross-tenant access returns 404.

`bootstrap_admin()` runs at app startup: if `ADMIN_EMAIL` + `ADMIN_PASSWORD` env
secrets are set, it creates (or promotes) that user with `is_admin=TRUE`.
`FLASK_SECRET_KEY` should be set in production; falls back to a dev default.

## Image Pipeline

`pipeline_worker.py` runs the production in a background thread pool:
1. Audio features (if audio uploaded)
2. **Pass 1** — `run_context_only` to get `speaker` + `location_dna`
3. Reference images: use uploads if present, else generate via FAL flux-schnell
4. **Pass 2** — `run_full_production` with `user_image_url` set to the character ref
5. Per-shot stills:
   - `flux-pulid` face-lock when a character ref exists AND the shot has human focus
   - `flux/dev/image-to-image` against the environment ref for env-only shots
   - plain `flux/dev` text-to-image otherwise
6. Status streamed into `projects.progress` + `shot_assets.status`

Frontend polls `/project/<id>/status` every 2.5s and renders refs + shots live
(including beat timing, expression mode, transition, and the styled prompt).
Retry endpoints: `/project/<id>/retry/ref/<role>` and `/.../retry/shot/<idx>`.

## On-disk Layout

Every project owns one folder so cleanup is a single `rmtree`:

```
projects/<project_id>/
  uploads/    # uploaded audio
  refs/       # uploaded + generated reference images
  shots/      # per-shot stills
  exports/    # final JSON export
```

Assets are served read-only at `/p/<project_id>/<sub>/<filename>` with a
whitelist of subfolders (`refs|shots|uploads`). `pipeline_worker.ensure_schema()`
runs at app startup so a fresh DB works with no manual SQL.
`POST /project/<id>/delete` soft-deletes the project (sets `deleted_at = NOW()`).
`POST /project/<id>/restore` restores from bin.
`POST /project/<id>/delete/permanent` hard-deletes + cleans up assets.
`GET /bin` — shows all soft-deleted projects for the current user.

## Build Roadmap

- T001 (DONE): Flask + Postgres + pipeline wiring
- T006 (DONE): Reference images + per-shot stills (face-locked) via FAL
- Next: Video generation per shot, ffmpeg assembly + audio mux

## Run

`bash start.sh` (workflow "Start application") starts Flask on `0.0.0.0:5000`.
