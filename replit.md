# Qaivid MetaMind 3.0

A web-based SaaS that turns lyrics, poems, scripts, and stories into beat-synced cinematic music videos. Powered by an 8-module Python pipeline and a 5-stage user-gated workflow.

## Architecture

- **Backend**: Flask (`app.py`) on port 5000
- **Database**: Replit PostgreSQL via `psycopg` (`DATABASE_URL`)
- **Frontend**: Server-rendered Jinja2 templates + plain CSS (no build step)
- **Storage**: Cloudflare R2 (all generated assets) accessed via authenticated boto3; served to browser via `/r2proxy` presigned redirect
- **AI**: OpenAI GPT-4o (context engine) + FAL AI (image + video generation)

## Pipeline Engines (do NOT modify without explicit permission)

- `audio_processor.py` — BPM, beats, energy extraction; Whisper transcription
- `unified_context_engine_master.py` — 5W context analysis (who/where/when/why/what)
- `visual_storyboard_engine.py` — per-segment visual prompts
- `rhythmic_assembly_engine.py` — beat-synced timeline assembly
- `style_grading_engine.py` — cinematic style grading
- `asset_export_module.py` — JSON export
- `production_orchestrator.py` — end-to-end coordinator (use `run_context_only`, `run_to_timeline`, NOT `run_full_production`)

## Staged Pipeline Flow (5 stages, each user-gated)

User uploads lyrics + optional audio → pipeline advances one stage at a time with a review screen between each:

| Stage | Action | Engine | Review Screen |
|-------|---------|--------|---------------|
| 0 | Audio features + Whisper | `audio_processor.py` | BPM, transcript, energy |
| 1 | 5W Context Audit | `run_context_only` → `unified_context_engine_master.py` | WHO/WHERE/WHAT with overrides |
| 2 | Storyboard + Style | `run_to_timeline` + `style_grading_engine.py` | Shot list table |
| 3 | Stills | FAL FLUX (env i2i or text-to-image) | Gallery with per-shot retry |
| 4 | Video Clips | FAL Kling image-to-video | Video player grid |

Stage state machine: `new → running_0 → audio_review → running_1 → context_review → running_2 → storyboard_review → running_3 → stills_review → running_4 → videos_review`

Failed projects land on `stage=failed`. Each stage advance is a `POST /project/<id>/advance/<1..4>`.

## R2 Storage + Proxy

R2 bucket is **private** (no public access). Assets are served to the browser through:

```
GET /r2proxy?url=<R2 URL>  →  presigned redirect  →  browser downloads asset
```

`_asset_url()` in `app.py` wraps every file_path through this route. All templates use `shot.url` / `vid.url` (never `file_path`) to get the proxied URL.

For FAL models (image + video generation) that need to download R2 assets:
- `_fal_accessible_url()` in `image_generator.py` and `video_generator.py`
- Checks if R2 URL is publicly accessible (HEAD request)
- If not: downloads via boto3, uploads to FAL's transient CDN, returns FAL CDN URL

## Database Schema

**`users`**: `id SERIAL PK`, `email UNIQUE`, `password_hash`, `is_admin`, `created_at`

**`projects`**: `id TEXT PK` (12-char hex), `user_id FK→users`, `name`, `genre`, `text`, `audio_filename`, `status`, `stage TEXT`, `audio_data JSONB`, `transcript TEXT`, `context_packet JSONB`, `styled_timeline JSONB`, `summary JSONB`, `progress JSONB`, `export_path`, `error`, `created_at`, `updated_at`

**`refs`**: per-project character/environment reference images; `role` in `character|environment`, `source` in `uploaded|generated`

**`shot_assets`**: per-shot stills; unique on `(project_id, shot_index)`; `status` in `pending|rendering|ready|failed`

**`video_assets`**: per-shot video clips; unique on `(project_id, shot_index)`;  `status` in `pending|rendering|ready|failed`

**`characters`** / **`locations`**: materialized from context_packet by `character_materializer.py` / `location_materializer.py`

## Auth + Admin

`auth.py` owns user CRUD, `werkzeug.security` password hashing, `@login_required`.
`bootstrap_admin()` creates/promotes the `ADMIN_EMAIL` account at startup.
Admin dashboard at `/admin` — user management, project control, system health.

## Key Routes

- `GET/POST /signup`, `GET/POST /login`, `POST /logout`
- `POST /generate` — submit new project (kicks Stage 0)
- `GET /project/<id>` — routes to correct stage template based on `project.stage`
- `GET /project/<id>/status` — JSON API (stage, status, shots, videos, refs)
- `POST /project/<id>/advance/<1..4>` — user-gate advance between stages
- `POST /project/<id>/retry/shot/<idx>` — retry a failed still
- `POST /project/<id>/retry/video/<idx>` — retry a failed video clip
- `GET /r2proxy?url=<url>` — presigned redirect for private R2 assets
- `GET /admin` — admin dashboard (`is_admin=True` required)

## Image + Video Generation Models

| Use | FAL Model |
|-----|-----------|
| Character reference portrait | `fal-ai/flux/schnell` |
| Environment reference plate | `fal-ai/flux/schnell` |
| Face/body shots | `fal-ai/flux-pulid` (PuLID face-lock) |
| Environment shots | `fal-ai/flux/dev/image-to-image` (env ref as guide) |
| Plain text-to-image shots | `fal-ai/flux/dev` |
| Video clips | `fal-ai/kling-video/v1/standard/image-to-video` |

## Required Secrets

- `OPENAI_API_KEY` — GPT-4o context engine
- `FAL_API_KEY` — all image + video generation
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`, `R2_PUBLIC_URL` — Cloudflare R2
- `DATABASE_URL` — Replit PostgreSQL (auto-provisioned)
- `ADMIN_EMAIL`, `ADMIN_PASSWORD` — bootstrap admin account
- `FLASK_SECRET_KEY` — Flask session signing (generate a random secret for production)

## Run

`bash start.sh` (workflow "Start application") — starts Flask on `0.0.0.0:5000`.
