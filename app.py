import os
import uuid
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from psycopg.rows import dict_row
from psycopg.types.json import Json
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect

from pipeline_worker import (
    cleanup_project_assets,
    ensure_schema,
    generate_all_videos,
    kick_pipeline,
    kick_stage_0,
    kick_stage_1,
    kick_stage_style,
    kick_stage_2,
    kick_stage_brief,
    kick_stage_3,
    kick_stage_4,
    kick_stage_5,
    kick_stage_refs,
    kick_quick_video,
    regenerate_entity_plate,
    retry_all_failed_refs,
    retry_all_failed_shots,
    retry_ref,
    retry_shot,
    retry_video,
    set_entity_uploaded_plate,
    seed_shot_rows_with_prompts,
    update_shot_prompt,
    set_shot_uploaded_image,
    kick_single_shot,
    kick_all_pending_shots,
)
from auth import (
    DuplicateEmailError,
    admin_required,
    bootstrap_admin,
    count_user_projects,
    create_user,
    current_user,
    delete_user,
    get_user_by_email,
    get_user_password_hash,
    login_required,
    login_user,
    logout_user,
    update_user_password,
    update_user_plan,
    verify_password,
)
import r2_storage
from billing import billing_bp, FREE_PROJECT_LIMIT


load_dotenv()

ROOT = Path(__file__).parent
PROJECTS_ROOT = ROOT / "projects"
PROJECTS_ROOT.mkdir(exist_ok=True)

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 80 * 1024 * 1024


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 30  # 30 days

_secret = os.getenv("FLASK_SECRET_KEY")
if not _secret:
    if os.getenv("FLASK_ENV") == "development" or os.getenv("REPL_ID"):
        import secrets as _sec
        _secret = _sec.token_hex(32)
        print("[WARN] FLASK_SECRET_KEY not set — generated an ephemeral one. "
              "Set it as a secret before deploying or sessions will reset on restart.")
    else:
        raise RuntimeError(
            "FLASK_SECRET_KEY is not set. Refusing to start with an insecure "
            "default key in production."
        )
app.secret_key = _secret

csrf = CSRFProtect(app)

app.register_blueprint(billing_bp)
# Exempt the Stripe webhook from CSRF — we verify via Stripe-Signature header instead
csrf.exempt(app.view_functions["billing.webhook"])

DATABASE_URL = os.environ["DATABASE_URL"]

ensure_schema()
bootstrap_admin()


def _recover_stalled_jobs():
    """On startup, find projects whose background threads died mid-run and re-queue them.

    Any project stuck in running_0..5 with status='running' had its worker thread
    killed when the server restarted. Reset stale asset rows and re-kick the worker.
    Stages 0 and 2 require additional args we no longer have at restart time, so
    those are marked failed for re-entry from the UI instead.
    """
    import logging
    log = logging.getLogger("startup_recovery")

    stage_kickers = {
        "running_0": kick_stage_0,
        "running_style": kick_stage_style,
        "running_1": kick_stage_1,
        "running_brief": kick_stage_brief,
        "running_2": kick_stage_2,
        "running_refs": kick_stage_refs,
        "running_3": kick_stage_3,
        "running_4": kick_stage_4,
        "running_5": kick_stage_5,
    }

    try:
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, stage FROM projects WHERE stage LIKE 'running_%' AND status='running'"
                )
                stalled = cur.fetchall()

            if not stalled:
                return

            for proj in stalled:
                pid = proj["id"]
                stage = proj["stage"]
                log.warning("Recovering stalled project %s (stage=%s)", pid, stage)
                try:
                    with conn.cursor() as cur:
                        # Reset rendering shots → pending so stage_3 re-generates them
                        cur.execute(
                            "UPDATE shot_assets SET status='pending', error=NULL WHERE project_id=%s AND status='rendering'",
                            (pid,),
                        )
                        # Reset rendering videos → pending so stage_4 re-generates them
                        cur.execute(
                            "UPDATE video_assets SET status='pending', error=NULL WHERE project_id=%s AND status='rendering'",
                            (pid,),
                        )
                    conn.commit()

                    kicker = stage_kickers.get(stage)
                    if not kicker:
                        log.error("Unknown stage %s for project %s — cannot re-kick", stage, pid)
                        continue

                    # Stages 0, 2, and brief require extra args we don't have
                    # on hand at restart (audio_path/text/genre, overrides for
                    # variant generation). Mark them failed so the user can
                    # re-enter via the UI rather than crashing the loop and
                    # starving later stages.
                    if stage in ("running_0", "running_2", "running_brief"):
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE projects SET status='failed', "
                                "       error='Worker restarted mid-run; please retry from the UI.', "
                                "       updated_at=NOW() WHERE id=%s",
                                (pid,),
                            )
                        conn.commit()
                        log.warning("Marked %s failed for re-entry: %s", pid, stage)
                        continue

                    kicker(pid)
                    log.warning("Re-kicked %s for project %s", stage, pid)
                except Exception:
                    log.exception("Recovery failed for project %s (stage=%s); continuing scan", pid, stage)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

    except Exception as exc:
        import logging as _log
        _log.getLogger("startup_recovery").error("Recovery scan failed: %s", exc)


_recover_stalled_jobs()


@app.context_processor
def _inject_user():
    from flask import session as _sess
    return {
        "current_user": current_user(),
        "is_impersonating": bool(_sess.get("original_admin_id")),
    }


_SITE_DEFAULTS: dict[str, str] = {
    "site_header_tagline":    "Lyrics · Scripts · Poems · Stories · End-to-End Production",
    "site_header_film_opacity": "0.40",
    "site_hero_eyebrow":      "AI Cinema Studio · MetaMind 3.1",
    "site_hero_line1":        "Qaivid",
    "site_hero_line2":        "Cinema Studio",
    "site_hero_tagline":      "Powered by MetaMind 3.1 — the Most Advanced AI Director",
    "site_hero_subtitle":     (
        "From a lyric line to a full film script — MetaMind generates creative briefs, "
        "face-locked character stills, beat-synced storyboards, and cinematic exports. "
        "In 40+ languages. No editing skills required."
    ),
    "site_hero_cta_primary":   "Start a Project",
    "site_hero_cta_secondary": "My Projects",
    "site_trust_1": "🌐  40+ languages",
    "site_trust_2": "🎬  8 shot types",
    "site_trust_3": "🎵  Beat-synced",
    "site_trust_4": "🔒  Face-locked stills",
    "site_howitworks_title": "From first word to finished film",
    "site_howitworks_sub":   "— end to end",
    "site_features_title":   "Everything included —",
    "site_features_sub":     "nothing left to chance",
    "site_pricing_title":    "One plan for every",
    "site_pricing_sub":      "level of ambition",
    "site_footer_tagline":   (
        "Turn any lyrics, poem, or script into a\n"
        "beat-synced cinematic music video — in minutes."
    ),
    "site_banner_enabled": "false",
    "site_banner_text":    "",
    "site_banner_color":   "#d4ff3a",
}


@app.context_processor
def _inject_site_settings():
    try:
        from system_config import get_all_site_settings
        saved = get_all_site_settings()
    except Exception:
        saved = {}
    merged = {**_SITE_DEFAULTS, **saved}
    return {"sc": merged}


def db():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def _api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def _fal_set() -> bool:
    return bool(os.getenv("FAL_API_KEY") or os.getenv("FAL_KEY"))


def _list_projects(user_id: int) -> list[dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, genre, status, stage, created_at, updated_at,
                   COALESCE((summary->>'styled_timeline_shot_count')::int, 0) AS shot_count
            FROM projects
            WHERE user_id = %s
            ORDER BY updated_at DESC NULLS LAST, created_at DESC
            LIMIT 100
            """,
            (user_id,),
        )
        return cur.fetchall()


def _get_project(project_id: str, user_id: int) -> dict | None:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM projects WHERE id = %s AND user_id = %s",
            (project_id, user_id),
        )
        return cur.fetchone()


def _get_refs(project_id: str) -> dict:
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM refs WHERE project_id = %s", (project_id,))
        rows = cur.fetchall()
    return {r["role"]: r for r in rows}


def _persist_uploaded_ref(project_id: str, role: str, file_path: str) -> None:
    """Save a user-uploaded ref at project-creation time so stage 3 reuses it
    instead of generating a new one (or asking the user to re-upload)."""
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO refs (project_id, role, source, file_path, status)
            VALUES (%s, %s, 'uploaded', %s, 'ready')
            ON CONFLICT (project_id, role) DO UPDATE
                SET source = 'uploaded', file_path = EXCLUDED.file_path,
                    status = 'ready', error = NULL
            """,
            (project_id, role, file_path),
        )
        conn.commit()


def _get_characters(project_id: str) -> list[dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM characters WHERE project_id = %s ORDER BY entity_type, id",
            (project_id,),
        )
        return cur.fetchall()


def _get_locations(project_id: str) -> list[dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM locations WHERE project_id = %s ORDER BY entity_type, id",
            (project_id,),
        )
        return cur.fetchall()


def _get_character_looks(project_id: str) -> dict[int, list[dict]]:
    """Return {character_id: [look_rows]} for the reference images page."""
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, character_id, cluster_id, cluster_label, wardrobe_text, "
            "       ref_image_url, ref_status, ref_prompt, ref_error "
            "  FROM character_looks "
            " WHERE project_id = %s "
            " ORDER BY character_id, id",
            (project_id,),
        )
        rows = cur.fetchall() or []
    result: dict[int, list[dict]] = {}
    for row in rows:
        cid = row["character_id"]
        result.setdefault(cid, []).append(dict(row))
    return result


def _get_shot_assets(project_id: str) -> list[dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM shot_assets WHERE project_id = %s ORDER BY shot_index ASC",
            (project_id,),
        )
        return cur.fetchall()


def _get_video_assets(project_id: str) -> list[dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM video_assets WHERE project_id = %s ORDER BY shot_index ASC",
            (project_id,),
        )
        return cur.fetchall()


def _asset_url(file_path: str | None) -> str | None:
    """Return a usable URL for an asset.

    For private R2 buckets, routes the URL through /r2proxy which generates a
    short-lived presigned redirect so the browser can display the image.
    Legacy local paths (projects/...) fall back to None.
    """
    if not file_path:
        return None
    if file_path.startswith("http://") or file_path.startswith("https://"):
        from urllib.parse import quote
        return url_for("r2proxy", url=file_path, _external=False)
    return None


def _video_payload(asset: dict, shot: dict) -> dict:
    return {
        "shot_index": asset["shot_index"],
        "status": asset["status"],
        "url": _asset_url(asset.get("file_path")),
        "error": asset.get("error"),
        "duration": shot.get("duration"),
        "start_time": shot.get("start_time"),
        "motion_scale": shot.get("motion_scale"),
        "transition": shot.get("transition"),
        "intensity": shot.get("intensity"),
        "expression_mode": shot.get("expression_mode"),
        "meaning": shot.get("meaning"),
        "prompt": shot.get("styled_visual_prompt") or shot.get("visual_prompt"),
    }


def _insert_pending_project(project_id: str, user_id: int, name: str, genre: str,
                            text: str, audio_filename: str | None) -> None:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO projects
                (id, user_id, name, genre, text, audio_filename, status, stage, progress)
            VALUES
                (%s, %s, %s, %s, %s, %s, 'queued', 'new', %s)
            """,
            (project_id, user_id, name, genre, text, audio_filename,
             Json({"stage": "queued", "label": "Queued..."})),
        )
        conn.commit()


class UploadValidationError(ValueError):
    pass


def _save_upload_to_r2(project_id: str, file_storage, r2_sub: str, prefix: str) -> str | None:
    """Upload a file-like storage object to R2 and return its public URL, or None."""
    if not file_storage or not file_storage.filename:
        return None
    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE:
        raise UploadValidationError(
            f"Unsupported image format: {ext}. Use JPG, PNG, or WEBP."
        )
    r2_key = f"projects/{project_id}/{r2_sub}/{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}
    content_type = mime_map.get(ext, "image/jpeg")
    return r2_storage.upload_fileobj(file_storage.stream, r2_key, content_type=content_type)


def _shot_payload(asset: dict, shot: dict) -> dict:
    idx = asset["shot_index"]
    return {
        "shot_index": idx,
        "status": asset["status"],
        "url": _asset_url(asset.get("file_path")),
        "error": asset.get("error"),
        "meaning": shot.get("meaning"),
        "function": shot.get("function"),
        "repeat_status": shot.get("repeat_status"),
        "start_time": shot.get("start_time"),
        "end_time": shot.get("end_time"),
        "duration": shot.get("duration"),
        "start_beat": shot.get("start_beat"),
        "bar_index": shot.get("bar_index"),
        "intensity": shot.get("intensity"),
        "audio_intensity": shot.get("audio_intensity"),
        "raw_shot_intensity": shot.get("raw_shot_intensity"),
        "motion_scale": shot.get("motion_scale"),
        "expression_mode": shot.get("expression_mode"),
        "transition": shot.get("transition"),
        "fidelity_lock": shot.get("fidelity_lock"),
        "camera_profile": shot.get("camera_profile") or {},
        "environment_profile": shot.get("environment_profile") or {},
        "continuity_anchor": shot.get("continuity_anchor") or {},
        "rendering_notes": shot.get("rendering_notes") or [],
        "prompt": shot.get("styled_visual_prompt") or shot.get("visual_prompt"),
    }


STAGE_ORDER = [
    "audio_review",
    "style_review",
    "context_review",
    "assumptions_review",
    "creative_brief_review",
    "storyboard_review",
    "references_review",
    "stills_control",
    "stills_review",
    "post_production",
    "videos_review",
    "final_review",
]
STAGE_LABELS = {
    "new": "Queued",
    "audio_review": "Acoustic Audit",
    "style_review": "Style Profile",
    "context_review": "Context Engine",
    "assumptions_review": "METAMAN Dialogue",
    "creative_brief_review": "Creative Brief",
    "storyboard_review": "Storyboard",
    "references_review": "References",
    "stills_control": "Stills",
    "stills_review": "Stills (Legacy)",
    "post_production": "Post Production",
    "videos_review": "Video Clips",
    "final_review": "Final Cut",
    "complete": "Complete",
    "failed": "Failed",
}


def _stage_progress(stage: str) -> dict:
    """Return {label, step, total} for a project stage."""
    total = len(STAGE_ORDER)
    if stage in STAGE_ORDER:
        step = STAGE_ORDER.index(stage) + 1
    elif stage == "complete":
        step = total
    else:
        step = 0
    return {"label": STAGE_LABELS.get(stage, stage), "step": step, "total": total}


@app.route("/")
def index():
    return render_template(
        "landing.html",
        api_key_set=bool(_api_key()),
        fal_set=_fal_set(),
    )


@app.route("/projects")
@login_required
def projects():
    user = current_user()
    project_list = _list_projects(user["id"])
    for p in project_list:
        p["progress_meta"] = _stage_progress(p.get("stage") or "new")
    return render_template(
        "projects.html",
        projects=project_list,
        api_key_set=bool(_api_key()),
        fal_set=_fal_set(),
    )


@app.route("/new")
@login_required
def new_project():
    user = current_user()
    if (not user.get("is_admin")
            and user.get("plan", "free") == "free"
            and count_user_projects(user["id"]) >= FREE_PROJECT_LIMIT):
        flash(
            f"Free plan is limited to {FREE_PROJECT_LIMIT} projects. "
            "Upgrade to Pro for unlimited projects.",
            "info",
        )
        return redirect(url_for("billing.pricing"))
    return render_template(
        "new_project.html",
        api_key_set=bool(_api_key()),
        fal_set=_fal_set(),
    )


# --- auth routes ------------------------------------------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user():
        return redirect(url_for("projects"))
    if request.method == "GET":
        return render_template("signup.html")
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    if not email or "@" not in email:
        flash("Please enter a valid email.", "error")
        return render_template("signup.html", email=email), 400
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return render_template("signup.html", email=email), 400
    if get_user_by_email(email):
        flash("An account with that email already exists.", "error")
        return render_template("signup.html", email=email), 400
    try:
        user = create_user(email, password)
    except DuplicateEmailError:
        flash("An account with that email already exists.", "error")
        return render_template("signup.html", email=email), 400
    login_user(user)
    flash("Welcome to Qaivid.", "success")
    return redirect(url_for("projects"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("projects"))
    next_url = request.values.get("next") or ""
    if request.method == "GET":
        return render_template("login.html", next_url=next_url)
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    user = get_user_by_email(email)
    if not user or not verify_password(user, password):
        flash("Invalid email or password.", "error")
        return render_template("login.html", email=email, next_url=next_url), 401
    login_user(user)
    if next_url.startswith("/") and not next_url.startswith("//"):
        return redirect(next_url)
    return redirect(url_for("projects"))


@app.route("/logout", methods=["POST"])
def logout():
    logout_user()
    flash("Signed out.", "success")
    return redirect(url_for("login"))


@app.route("/account")
@login_required
def account():
    user = current_user()
    if user.get("plan") not in ("pro", "studio"):
        user["plan"] = "free"
    if user["plan"] == "free":
        user["project_count"] = count_user_projects(user["id"])
    return render_template("account.html", user=user)


@app.route("/account/change-password", methods=["POST"])
@login_required
def account_change_password():
    user = current_user()
    current_pw = request.form.get("current_password", "")
    new_pw = request.form.get("new_password", "")
    confirm_pw = request.form.get("confirm_password", "")

    pw_hash = get_user_password_hash(user["id"])
    if not pw_hash or not check_password_hash(pw_hash, current_pw):
        flash("Current password is incorrect.", "error")
        return redirect(url_for("account"))
    if len(new_pw) < 8:
        flash("New password must be at least 8 characters.", "error")
        return redirect(url_for("account"))
    if new_pw != confirm_pw:
        flash("New passwords do not match.", "error")
        return redirect(url_for("account"))

    update_user_password(user["id"], new_pw)
    flash("Password updated successfully.", "success")
    return redirect(url_for("account"))


@app.route("/account/delete", methods=["POST"])
@login_required
def account_delete():
    user = current_user()
    password = request.form.get("delete_password", "")

    pw_hash = get_user_password_hash(user["id"])
    if not pw_hash or not check_password_hash(pw_hash, password):
        flash("Incorrect password — account not deleted.", "error")
        return redirect(url_for("account"))

    uid = user["id"]
    logout_user()
    delete_user(uid)
    flash("Your account has been permanently deleted.", "info")
    return redirect(url_for("index"))


@app.route("/generate", methods=["POST"])
@login_required
def generate():
    user = current_user()
    if (not user.get("is_admin")
            and user.get("plan", "free") == "free"
            and count_user_projects(user["id"]) >= FREE_PROJECT_LIMIT):
        flash(
            f"Free plan is limited to {FREE_PROJECT_LIMIT} projects. "
            "Upgrade to Pro for unlimited projects.",
            "info",
        )
        return redirect(url_for("billing.pricing"))

    text = (request.form.get("text") or "").strip()
    genre = (request.form.get("genre") or "song").strip().lower()
    name = (request.form.get("name") or "Untitled").strip()

    audio_provided = bool(request.files.get("audio") and request.files["audio"].filename)
    if not text and not audio_provided:
        flash("Please paste your lyrics or upload an audio file — at least one is required.", "error")
        return redirect(url_for("new_project"))
    if not _api_key():
        flash("OPENAI_API_KEY is not set. Add it in Replit secrets.", "error")
        return redirect(url_for("new_project"))
    if not _fal_set():
        flash("FAL_API_KEY is not set. Add it in Replit secrets to render images.", "error")
        return redirect(url_for("new_project"))

    project_id = uuid.uuid4().hex[:12]

    # Upload audio to R2 if provided
    audio_path: Path | None = None
    audio_file = request.files.get("audio")
    if audio_file and audio_file.filename:
        ext = Path(audio_file.filename).suffix.lower()
        if ext not in ALLOWED_AUDIO:
            flash(f"Unsupported audio format: {ext}. Use MP3, WAV, M4A, OGG, or FLAC.", "error")
            return redirect(url_for("new_project"))
        safe_name = secure_filename(audio_file.filename)
        # Save locally for audio processing (librosa needs a file path)
        local_audio_dir = PROJECTS_ROOT / project_id / "uploads"
        local_audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = local_audio_dir / safe_name
        audio_file.save(audio_path)
        # Also upload to R2 for permanent storage
        try:
            r2_key = f"projects/{project_id}/uploads/{safe_name}"
            r2_storage.upload_file(audio_path, r2_key)
        except Exception:
            pass  # non-fatal; audio_path local copy is used by pipeline

    # Reference images (character + environment plates) are no longer
    # uploaded on the new-project form. They are generated and reviewed in
    # the dedicated References stage of the pipeline, where the user can
    # also upload custom replacements per entity if desired.
    _insert_pending_project(project_id, current_user()["id"], name, genre, text,
                            audio_path.name if audio_path else None)

    kick_stage_0(
        project_id=project_id,
        audio_path=audio_path,
        text=text,
        genre=genre,
    )

    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>")
@login_required
def project_detail(project_id: str):
    user = current_user()
    project = _get_project(project_id, user["id"])
    viewing_as_admin = False
    if not project and user.get("is_admin"):
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
            project = cur.fetchone()
            viewing_as_admin = bool(project)
    if not project:
        abort(404)
    project["summary"] = project.get("summary") or {}
    project["context_packet"] = project.get("context_packet") or {}
    project["styled_timeline"] = project.get("styled_timeline") or []
    project["progress"] = project.get("progress") or {}
    project["audio_data"] = project.get("audio_data") or {}

    # Look up owner email for admin banner
    owner_email = None
    if viewing_as_admin:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT email FROM users WHERE id = %s", (project["user_id"],))
            row = cur.fetchone()
            if row:
                owner_email = row["email"]

    actual_stage = project.get("stage") or "new"
    status = project.get("status") or "queued"

    # When a stage crashes the worker writes stage='failed', which by itself
    # gives us no way to recover. Infer the LAST successfully-completed stage
    # from the data already persisted, so the user can land back on its
    # review screen and re-trigger the failed step.
    if actual_stage == "failed" or status == "failed":
        cp_for_recovery = project.get("context_packet") or {}
        # Pull video/still readiness so we can land late-stage failures on the
        # latest screen the user actually completed (not all the way back at
        # storyboard_review).
        with db() as _rconn, _rconn.cursor() as _rcur:
            _rcur.execute(
                "SELECT COUNT(*) AS n FROM video_assets "
                " WHERE project_id=%s AND status='ready'",
                (project_id,),
            )
            _ready_videos = (_rcur.fetchone() or {}).get("n", 0) or 0
            _rcur.execute(
                "SELECT COUNT(*) AS n FROM shot_assets "
                " WHERE project_id=%s AND status='ready'",
                (project_id,),
            )
            _ready_stills = (_rcur.fetchone() or {}).get("n", 0) or 0

        if _ready_videos > 0:
            # Final assembly (or anything after stills) failed — drop the
            # user back at the Videos review so they can re-trigger Stage 5.
            # Also reset the videos_review status so advance_stage_5's atomic
            # gate will accept the next submit.
            recovery_stage = "videos_review"
            with db() as _rconn, _rconn.cursor() as _rcur:
                _rcur.execute(
                    "UPDATE projects SET stage='videos_review', "
                    "       status='awaiting_review', updated_at=NOW() "
                    " WHERE id=%s AND (stage='failed' OR status='failed')",
                    (project_id,),
                )
                _rconn.commit()
            project["stage"] = "videos_review"
            project["status"] = "awaiting_review"
            actual_stage = "videos_review"
            status = "awaiting_review"
        elif _ready_stills > 0:
            recovery_stage = "stills_review"
        elif project.get("styled_timeline"):
            # If at least one character/location plate was already locked, the
            # references stage was reached — drop the user back there so they
            # can fix the broken plate without restarting the storyboard.
            with db() as _rconn, _rconn.cursor() as _rcur:
                _rcur.execute(
                    "SELECT 1 FROM characters "
                    " WHERE project_id=%s AND ref_status='ready' AND ref_image_url IS NOT NULL "
                    " UNION ALL "
                    "SELECT 1 FROM locations "
                    " WHERE project_id=%s AND ref_status='ready' AND ref_image_url IS NOT NULL "
                    " LIMIT 1",
                    (project_id, project_id),
                )
                _has_plate = _rcur.fetchone() is not None
            recovery_stage = "references_review" if _has_plate else "storyboard_review"
        elif cp_for_recovery.get("locked_assumptions") or cp_for_recovery.get("_pending_overrides"):
            # User already passed through context review; resume at the dialogue.
            # If the Creative Brief variants were already generated (stage
            # failed at running_brief after LLM call succeeded), land them at
            # the brief review so they don't lose the generated variants.
            _brief_variants = ((cp_for_recovery.get("creative_brief") or {}).get("variants") or [])
            if _brief_variants:
                recovery_stage = "creative_brief_review"
                with db() as _rconn, _rconn.cursor() as _rcur:
                    _rcur.execute(
                        "UPDATE projects SET stage='creative_brief_review', "
                        "       status='awaiting_review', error=NULL, updated_at=NOW() "
                        " WHERE id=%s AND status='failed'",
                        (project_id,),
                    )
                    _rconn.commit()
                project["stage"] = "creative_brief_review"
                project["status"] = "awaiting_review"
                actual_stage = "creative_brief_review"
                status = "awaiting_review"
            else:
                recovery_stage = "assumptions_review"
                # Reset the DB row so /advance/2b's atomic gate will accept
                # the next submit (gate requires stage=assumptions_review &
                # status=awaiting_review).
                with db() as _rconn, _rconn.cursor() as _rcur:
                    _rcur.execute(
                        "UPDATE projects SET stage='assumptions_review', "
                        "       status='awaiting_review', error=NULL, updated_at=NOW() "
                        " WHERE id=%s AND status='failed'",
                        (project_id,),
                    )
                    _rconn.commit()
                project["stage"] = "assumptions_review"
                project["status"] = "awaiting_review"
                actual_stage = "assumptions_review"
                status = "awaiting_review"
        elif cp_for_recovery:
            recovery_stage = "context_review"
        elif project.get("style_suggestions"):
            recovery_stage = "style_review"
        elif project.get("audio_data"):
            recovery_stage = "audio_review"
        else:
            recovery_stage = None
    else:
        recovery_stage = None

    # Effective stage used for routing + ?at= gating.
    routing_stage = recovery_stage or actual_stage

    # ?at=<stage> lets the user jump back to any *previously completed* stage's
    # review screen (read-only inspection, all data is still in the DB).
    requested = request.args.get("at")
    routing_idx = STAGE_ORDER.index(routing_stage) if routing_stage in STAGE_ORDER else -1
    if (requested and requested in STAGE_ORDER
            and status not in ("running", "queued")
            and routing_idx >= 0
            and STAGE_ORDER.index(requested) <= routing_idx):
        stage = requested
    else:
        stage = routing_stage

    # If actual_stage is still the raw 'failed' string and a recovery stage
    # was determined, use the recovery stage so the wizard pills get a valid
    # index and render as clickable links rather than dead spans.
    if actual_stage == "failed" and recovery_stage:
        actual_stage = recovery_stage

    project["_viewing_as_admin"] = viewing_as_admin
    project["_owner_email"] = owner_email
    project["_actual_stage"] = actual_stage
    # Human-readable label of the step that was running when it failed.
    # Recovery banner uses this to say "Storyboard step failed" etc.
    project["_actual_stage_label"] = STAGE_LABELS.get(
        project.get("progress", {}).get("stage", "") if isinstance(project.get("progress"), dict) else "",
        STAGE_LABELS.get(actual_stage, "Pipeline")
    )
    project["_viewing_stage"] = stage
    project["_is_review_only"] = (stage != actual_stage)

    # Running or queued — show spinner screen that auto-reloads
    if status in ("running", "queued") or stage.startswith("running_"):
        return render_template("stage_running.html", project=project)

    if stage == "audio_review":
        try:
            with db() as _ac, _ac.cursor(row_factory=dict_row) as _acur:
                _acur.execute(
                    "SELECT lyrics_timed, text, audio_data FROM projects WHERE id=%s",
                    (project_id,),
                )
                _ar = _acur.fetchone()
            _lyrics_timed = list((_ar or {}).get("lyrics_timed") or []) if _ar else []
            if not _lyrics_timed and _ar:
                _dur = ((_ar.get("audio_data") or {}).get("duration_seconds") or 0)
                _txt = (_ar.get("text") or "").strip()
                if _dur > 0 and _txt:
                    _lines = [l for l in _txt.splitlines() if l.strip()]
                    _n = len(_lines)
                    if _n:
                        _step = _dur / _n
                        _lyrics_timed = [
                            {"text": _lines[_i], "start": round(_i * _step, 3),
                             "end": round((_i + 1) * _step, 3)}
                            for _i in range(_n)
                        ]
                        try:
                            with db() as _ac2, _ac2.cursor() as _cur2:
                                _cur2.execute(
                                    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS "
                                    "lyrics_timed JSONB;"
                                )
                                _cur2.execute(
                                    "UPDATE projects SET lyrics_timed=%s WHERE id=%s",
                                    (Json(_lyrics_timed), project_id),
                                )
                                _ac2.commit()
                        except Exception as _pe:
                            logger.warning(
                                "audio_review: could not persist fallback lyrics_timed (%s)", _pe
                            )
        except Exception:
            _lyrics_timed = []
        return render_template("stage_audio.html", project=project,
                               lyrics_timed=_lyrics_timed)

    if stage == "style_review":
        from style_profile_registry import StyleProfileRegistry
        return render_template(
            "stage_style.html",
            project=project,
            all_production_styles=StyleProfileRegistry.all_production_styles(),
            all_cinematic_styles=StyleProfileRegistry.all_cinematic_styles(),
        )

    if stage == "context_review":
        with db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT lyrics_timed, text, audio_data FROM projects WHERE id=%s",
                (project_id,),
            )
            _lrow = cur.fetchone()
        lyrics_timed = list((_lrow or {}).get("lyrics_timed") or []) if _lrow else []

        # Fallback: generate approximate timestamps for projects that were
        # transcribed before Task #105 (Whisper skipped or pre-#105 records).
        # This runs before rendering so the Timed Lyrics panel is visible at
        # the Context review step, not only after re-running the storyboard.
        if not lyrics_timed and _lrow:
            _text = (_lrow.get("text") or "").strip()
            _audio_data = dict(_lrow.get("audio_data") or {})
            _audio_dur = 0.0
            try:
                _audio_dur = float(_audio_data.get("duration_seconds") or 0)
            except (TypeError, ValueError):
                pass
            if _audio_dur > 0 and _text:
                _lines = [l.strip() for l in _text.splitlines() if l.strip()]
                if _lines:
                    _n = len(_lines)
                    _step = _audio_dur / _n
                    lyrics_timed = [
                        {
                            "text": _lines[_i],
                            "start": round(_i * _step, 3),
                            "end": round((_i + 1) * _step, 3),
                        }
                        for _i in range(_n)
                    ]
                    try:
                        with db() as _conn, _conn.cursor() as _cur:
                            _cur.execute(
                                "ALTER TABLE projects ADD COLUMN IF NOT EXISTS "
                                "lyrics_timed JSONB;"
                            )
                            _cur.execute(
                                "UPDATE projects SET lyrics_timed=%s WHERE id=%s",
                                (Json(lyrics_timed), project_id),
                            )
                            _conn.commit()
                    except Exception as _e:
                        import logging as _log
                        _log.getLogger(__name__).warning(
                            "context_review: could not persist fallback lyrics_timed (%s)", _e
                        )

        return render_template("stage_context.html", project=project,
                               lyrics_timed=lyrics_timed)

    if stage == "assumptions_review":
        return render_template("stage_assumptions.html", project=project)

    if stage == "creative_brief_review":
        return render_template("stage_creative_brief.html", project=project)

    if stage == "storyboard_review":
        shot_assets = _get_shot_assets(project_id)
        return render_template("stage_storyboard.html", project=project,
                               shot_assets=shot_assets)

    if stage == "references_review":
        characters = _get_characters(project_id)
        locations = _get_locations(project_id)
        looks_by_char = _get_character_looks(project_id)
        for c in characters:
            c["ref_url"] = _asset_url(c.get("ref_image_url"))
            c["looks"] = [
                {**lk, "ref_url": _asset_url(lk.get("ref_image_url"))}
                for lk in looks_by_char.get(c["id"], [])
            ]
        for l in locations:
            l["ref_url"] = _asset_url(l.get("ref_image_url"))
        return render_template("references_review.html", project=project,
                               characters=characters, locations=locations)

    if stage == "stills_control":
        from pipeline_worker import composed_prompts_for_project
        shot_assets = _get_shot_assets(project_id)
        timeline = project.get("styled_timeline") or []
        # Precompute composed prompts for the whole timeline in one pass
        # (single DB read, single sequential cinematography derivation).
        # Avoids the O(n²) re-derive that would happen if we called the
        # per-shot helper inside this loop.
        try:
            preview_by_idx = composed_prompts_for_project(project_id)
        except Exception:
            preview_by_idx = {}
        shots = []
        for a in shot_assets:
            tl_shot = next(
                (s for s in timeline
                 if (s.get("shot_index") or s.get("timeline_index")) == a["shot_index"]),
                {},
            )
            p = _shot_payload(a, tl_shot)
            # If the user hand-edited the prompt, show their text verbatim.
            # Otherwise show the *real* composed prompt that the model will
            # see (NOT the legacy 4000-char styled_visual_prompt that was
            # seeded into shot_assets.prompt before the composer existed).
            if a.get("prompt_user_edited") and a.get("prompt"):
                p["prompt"] = a["prompt"]
            else:
                p["prompt"] = (preview_by_idx.get(a["shot_index"])
                               or a.get("prompt") or p.get("prompt") or "")
            p["source"] = a.get("source")
            shots.append(p)

        # Compute total stills duration vs audio duration for display
        total_stills_dur = round(sum(s.get("duration") or 0 for s in timeline), 1)
        audio_data = project.get("audio_data") or {}
        # Use duration_seconds directly from audio_data (set by AudioProcessor.extract_features)
        # Fall back to estimate from beat_times if duration_seconds is missing (legacy projects)
        audio_dur_raw = audio_data.get("duration_seconds") or 0.0
        if audio_dur_raw:
            audio_dur = round(float(audio_dur_raw), 1)
        else:
            beat_times = audio_data.get("beat_times") or []
            bpm = audio_data.get("bpm") or 0
            if beat_times and bpm:
                beat_interval = 60.0 / bpm
                audio_dur = round(beat_times[-1] + beat_interval, 1)
            else:
                audio_dur = 0.0
        # Scale factor applied to per-shot durations when generating quick video
        # For new projects (fixed assembly engine) this should be ~1.0
        scale = round(audio_dur / total_stills_dur, 3) if total_stills_dur > 0 and audio_dur > 0 else 1.0

        return render_template(
            "stills_control.html",
            project=project,
            shots=shots,
            total_stills_dur=total_stills_dur,
            audio_dur=audio_dur,
            dur_scale=scale,
        )

    if stage == "stills_review":
        shot_assets = _get_shot_assets(project_id)
        refs_raw = _get_refs(project_id)
        # Route ref images through /r2proxy so private R2 URLs render in <img>.
        refs = {
            role: {**r, "url": _asset_url(r.get("file_path"))}
            for role, r in refs_raw.items()
        }
        timeline = project["styled_timeline"]
        shots = [_shot_payload(a, next((s for s in timeline
                               if (s.get("shot_index") or s.get("timeline_index")) == a["shot_index"]), {}))
                 for a in shot_assets]
        return render_template("stage_stills.html", project=project,
                               shots=shots, refs=refs)

    if stage == "post_production":
        return redirect(url_for("postprod_page", project_id=project_id))

    if stage == "videos_review":
        shot_assets = _get_shot_assets(project_id)
        video_assets = _get_video_assets(project_id)
        timeline = project["styled_timeline"]
        shots = [_shot_payload(a, next((s for s in timeline
                               if (s.get("shot_index") or s.get("timeline_index")) == a["shot_index"]), {}))
                 for a in shot_assets]
        videos = {
            v["shot_index"]: _video_payload(v, next(
                (s for s in timeline
                 if (s.get("shot_index") or s.get("timeline_index")) == v["shot_index"]),
                {}
            ))
            for v in video_assets
        }
        return render_template("stage_videos.html", project=project,
                               shots=shots, videos=videos)

    if stage in ("final_review", "complete"):
        return render_template("stage_final.html", project=project)

    if stage == "failed":
        return render_template("stage_running.html", project=project)

    # Fallback: legacy projects without stages OR complete
    shot_assets = _get_shot_assets(project_id)
    video_assets = _get_video_assets(project_id)
    refs = _get_refs(project_id)
    characters = _get_characters(project_id)
    locations = _get_locations(project_id)
    timeline = project["styled_timeline"]
    shots = [_shot_payload(a, next((s for s in timeline
                           if (s.get("shot_index") or s.get("timeline_index")) == a["shot_index"]), {}))
             for a in shot_assets]
    videos = [_video_payload(v, next((s for s in timeline
                             if (s.get("shot_index") or s.get("timeline_index")) == v["shot_index"]), {}))
              for v in video_assets]
    return render_template("result.html", project=project, shots=shots,
                           videos=videos, refs=refs, characters=characters,
                           locations=locations)


@app.route("/project/<project_id>/status")
@login_required
def project_status(project_id: str):
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    refs = _get_refs(project_id)
    shot_assets = _get_shot_assets(project_id)
    video_assets = _get_video_assets(project_id)
    styled_timeline = project.get("styled_timeline") or []
    timeline_by_idx = {
        (s.get("shot_index") or s.get("timeline_index")): s for s in styled_timeline
    }

    def _ref_payload(role):
        r = refs.get(role)
        if not r:
            return {"status": "absent", "url": None, "source": None, "error": None}
        return {
            "status": r["status"],
            "url": _asset_url(r.get("file_path")),
            "source": r.get("source"),
            "error": r.get("error"),
        }

    asset_by_idx = {a["shot_index"]: a for a in shot_assets}
    enriched_timeline = []
    for s in styled_timeline:
        idx = s.get("shot_index") or s.get("timeline_index")
        a = asset_by_idx.get(idx) or {}
        enriched_timeline.append({
            **s,
            "character_id": a.get("character_id"),
            "location_id":  a.get("location_id"),
        })

    return jsonify({
        "id": project["id"],
        "status": project["status"],
        "stage": project.get("stage") or "",
        "progress": project.get("progress") or {},
        "error": project.get("error"),
        "refs": {
            "character": _ref_payload("character"),
            "environment": _ref_payload("environment"),
        },
        "shot_count": len(styled_timeline),
        "styled_timeline": enriched_timeline,
        "shots": [
            _shot_payload(a, timeline_by_idx.get(a["shot_index"]) or {})
            for a in shot_assets
        ],
        "videos": [
            _video_payload(v, timeline_by_idx.get(v["shot_index"]) or {})
            for v in video_assets
        ],
    })


@app.route("/project/<project_id>/entities")
@login_required
def project_entities(project_id: str):
    if not _get_project(project_id, current_user()["id"]):
        abort(404)
    characters = _get_characters(project_id)
    locations = _get_locations(project_id)

    def _serialize(row: dict) -> dict:
        skip = {"metadata"}
        return {
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in row.items()
            if k not in skip
        }

    return jsonify({
        "characters": [_serialize(c) for c in characters],
        "locations": [_serialize(l) for l in locations],
    })


@app.route("/project/<project_id>/advance/1", methods=["POST"])
@login_required
def advance_stage_1(project_id: str):
    """User approved the audio analysis — apply singer-gender choice, kick Style."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    override = (request.form.get("vocal_gender_override") or "auto").strip().lower()
    if override not in ("auto", "male", "female", "mixed", "instrumental"):
        override = "auto"

    audio_data = dict(project.get("audio_data") or {})
    detected = str(audio_data.get("vocal_gender") or "unknown").lower()
    final_gender = detected if override == "auto" else override

    audio_data["vocal_gender_user_choice"] = override
    audio_data["vocal_gender_final"] = final_gender
    if isinstance(audio_data.get("audio_hints"), dict):
        audio_data["audio_hints"]["vocal_gender_final"] = final_gender
    if isinstance(audio_data.get("_pre_analysis"), dict):
        ah = audio_data["_pre_analysis"].setdefault("audio_hints", {})
        ah["vocal_gender_final"] = final_gender
        ah["vocal_gender"] = detected

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET audio_data=%s, updated_at=NOW() WHERE id=%s",
            (Json(audio_data), project_id),
        )
        conn.commit()

    kick_stage_style(project_id)
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/advance/style", methods=["POST"])
@login_required
def advance_stage_style(project_id: str):
    """User chose a style profile — save it and kick Context Engine (Stage 1).

    Task #59 — if the project already has a styled_timeline (i.e. the user
    came back from storyboard_review via the "Change Style" button), this is
    a re-pick. We skip the audio stage entirely and re-run the
    context → storyboard → timeline chain by kicking Stage 2 directly.
    audio_data is preserved on the row.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    prod_id = request.form.get("production_style_id") or ""
    cin_id = request.form.get("cinematic_style_id") or ""

    from style_profile_registry import StyleProfileRegistry
    style_profile = StyleProfileRegistry.build_style_profile(prod_id, cin_id)

    is_repick = bool(project.get("styled_timeline"))

    with db() as conn, conn.cursor() as cur:
        if is_repick:
            cur.execute(
                "UPDATE projects SET style_profile=%s, status=%s, stage=%s, "
                "       error=NULL, updated_at=NOW() WHERE id=%s",
                (Json(style_profile), "queued", "queued", project_id),
            )
        else:
            cur.execute(
                "UPDATE projects SET style_profile=%s, updated_at=NOW() WHERE id=%s",
                (Json(style_profile), project_id),
            )
        conn.commit()

    if is_repick:
        # Task #73 — preserve the user's previously-approved context edits
        # (speaker name, location, era) so they survive the style re-run.
        cp = dict(project.get("context_packet") or {})
        _speaker = cp.get("speaker")
        overrides = {
            "speaker_name": (_speaker.get("name") if isinstance(_speaker, dict) else None),
            "location":     cp.get("location_dna"),
            "era":          cp.get("era"),
            "style_preset": style_profile.get("preset") or "cinematic_natural",
        }
        name = project.get("name") or "Qaivid_Project"
        kick_stage_2(project_id, name, overrides)
    else:
        kick_stage_1(project_id)
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/restyle", methods=["GET"])
@login_required
def restyle_project(project_id: str):
    """Task #59 — render the Style Profile picker again so the user can swap
    style profiles without restarting the project from audio. The form posts
    to advance_stage_style which detects the re-pick and runs only
    context → storyboard → timeline.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    # Only allow re-pick once a storyboard exists (otherwise just normal flow).
    if not project.get("styled_timeline"):
        return redirect(url_for("project_detail", project_id=project_id))
    from style_profile_registry import StyleProfileRegistry
    return render_template(
        "stage_style.html",
        project=project,
        all_production_styles=StyleProfileRegistry.all_production_styles(),
        all_cinematic_styles=StyleProfileRegistry.all_cinematic_styles(),
        is_repick=True,
    )


@app.route("/project/<project_id>/advance/2", methods=["POST"])
@login_required
def advance_stage_2(project_id: str):
    """User approved the 5W context — open the METAMAN Surfaced-Assumptions Dialogue.

    No worker is kicked here. Free-text overrides + style preset are stashed
    in context_packet so the dialogue stage can finalize them. The actual
    storyboard worker is kicked from advance_stage_2b.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    overrides = {
        "speaker_name": (request.form.get("speaker_name") or "").strip() or None,
        "location":     (request.form.get("location") or "").strip() or None,
        "era":          (request.form.get("era") or "").strip() or None,
        "style_preset": (request.form.get("style_preset") or "cinematic_natural").strip(),
    }
    cp = dict(project.get("context_packet") or {})
    cp["_pending_overrides"] = {k: v for k, v in overrides.items() if v}

    # WHY overrides — persist directly into the motivation block so the
    # storyboard prompt + METAMAN dialogue see the user's edits immediately.
    motivation = dict(cp.get("motivation") or {})
    _why_keys = ("inciting_cause", "underlying_desire", "stakes", "obstacle")
    _changed = False
    for key in _why_keys:
        val = (request.form.get(f"motivation_{key}") or "").strip()
        if val:
            motivation[key] = val
            _changed = True
    if _changed:
        # The user explicitly refined the WHY block — bump confidence so the
        # METAMAN dialogue stops surfacing motivation as low-confidence.
        try:
            _prev_conf = float(motivation.get("confidence") or 0.0)
        except (TypeError, ValueError):
            _prev_conf = 0.0
        motivation["confidence"] = max(_prev_conf, 0.9)
        cp["motivation"] = motivation
        scores = dict(cp.get("confidence_scores") or {})
        scores["motivation"] = motivation["confidence"]
        cp["confidence_scores"] = scores
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET context_packet=%s, stage=%s, status=%s, "
            "error=NULL, updated_at=NOW() WHERE id=%s",
            (Json(cp), "assumptions_review", "awaiting_review", project_id),
        )
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/advance/2b", methods=["POST"])
@login_required
def advance_stage_2b(project_id: str):
    """User completed the METAMAN Dialogue — apply locked_assumptions, kick Storyboard."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    cp = dict(project.get("context_packet") or {})
    surfaced = cp.get("surfaced_assumptions") or []
    locked: dict = dict(cp.get("locked_assumptions") or {})
    flags = list(cp.get("ambiguity_flags") or [])

    # ── WHY panel ── motivation fields have their own dedicated form inputs
    # (why_action_<field> / why_value_<field>) so they are processed here,
    # BEFORE the generic surfaced_assumptions loop.  This prevents the same
    # field from being double-processed if it also appears in surfaced_assumptions.
    _WHY_SUBFIELDS = ("inciting_cause", "underlying_desire", "stakes", "obstacle")
    _WHY_LOCK_KEYS = {f"motivation_{wf}" for wf in _WHY_SUBFIELDS}
    motivation_block = dict(cp.get("motivation") or {})
    for wf in _WHY_SUBFIELDS:
        action = (request.form.get(f"why_action_{wf}") or "").strip()
        if not action:
            continue
        lock_key = f"motivation_{wf}"
        if action == "reject":
            locked.pop(lock_key, None)
            flags.append({
                "field": lock_key,
                "reason": "Marked as ambiguous by user during METAMAN Dialogue.",
                "confidence": 0.0,
            })
        else:
            val = (request.form.get(f"why_value_{wf}") or "").strip()
            if action == "override" and val:
                chosen = val
            else:
                chosen = str(motivation_block.get(wf) or "").strip()
            if chosen:
                locked[lock_key] = chosen

    for i, item in enumerate(surfaced):
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        if not field:
            continue
        # Skip motivation fields already handled by the WHY panel above so
        # the generic action_<i> loop cannot override what the user set there.
        if field in _WHY_LOCK_KEYS:
            continue
        action = (request.form.get(f"action_{i}") or "accept").strip()
        override_value = (request.form.get(f"value_{i}") or "").strip()
        if action == "reject":
            flags.append({
                "field": field,
                "reason": "Marked as ambiguous by user during METAMAN Dialogue.",
                "confidence": 0.0,
            })
            locked.pop(field, None)
            continue
        chosen = override_value if action == "override" and override_value else item.get("value")
        if chosen:
            locked[field] = chosen

    # Apply locks back into the packet so downstream stages see resolved values.
    _apply_locked_assumptions_inplace(cp, locked)
    cp["locked_assumptions"] = locked
    cp["ambiguity_flags"] = flags

    pending = cp.pop("_pending_overrides", {}) or {}

    # Atomic gate: only transition + kick the worker if the project is
    # actually parked at this dialogue stage AND awaiting review. This both
    # blocks crafted POSTs that try to skip the dialogue from prior states,
    # and prevents a rapid double-submit from queueing the storyboard twice.
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET context_packet=%s, status=%s, "
            "       stage=%s, error=NULL, updated_at=NOW() "
            " WHERE id=%s "
            "   AND stage='assumptions_review' "
            "   AND status='awaiting_review'",
            (Json(cp), "queued", "queued", project_id),
        )
        if cur.rowcount != 1:
            flash("This step has already been completed or is not ready yet.", "error")
            return redirect(url_for("project_detail", project_id=project_id))

    overrides = {
        "speaker_name": pending.get("speaker_name"),
        "location":     pending.get("location"),
        "era":          pending.get("era"),
        "style_preset": pending.get("style_preset") or "cinematic_natural",
    }
    # Task #69 — instead of jumping straight to Storyboard, kick the new
    # Creative Brief stage which generates director treatment variants and
    # parks at creative_brief_review for the user to lock one.
    kick_stage_brief(project_id, overrides)
    return redirect(url_for("project_detail", project_id=project_id))


_LOCK_TO_WORLD = ("geography", "era", "season", "characteristic_time", "social_context",
                  "economic_context", "architecture_style", "characteristic_setting")
# Aliases: old field names that may exist in locked_assumptions for existing projects
_LOCK_TO_WORLD_ALIASES = {"time_of_day": "characteristic_time", "domestic_setting": "characteristic_setting"}
_LOCK_TO_SPEAKER = {"speaker_identity": "identity", "speaker_gender": "gender"}
_LOCK_TO_MOTIVATION = ("inciting_cause", "underlying_desire", "stakes", "obstacle")


def _apply_locked_assumptions_inplace(cp: dict, locked: dict) -> None:
    """Mirror UnifiedContextEngine._apply_locked_assumptions on a stored packet."""
    if not isinstance(locked, dict) or not locked:
        return
    world = dict(cp.get("world_assumptions") or {})
    for key in _LOCK_TO_WORLD:
        if key in locked and locked[key]:
            world[key] = str(locked[key]).strip()
    # Also handle old field names stored in existing locked_assumptions
    for old_key, canonical in _LOCK_TO_WORLD_ALIASES.items():
        if old_key in locked and locked[old_key]:
            world[canonical] = str(locked[old_key]).strip()
    cp["world_assumptions"] = world
    if locked.get("location_dna"):
        cp["location_dna"] = str(locked["location_dna"]).strip()
    speaker = dict(cp.get("speaker") or {})
    for lock_key, sk in _LOCK_TO_SPEAKER.items():
        if locked.get(lock_key):
            speaker[sk] = str(locked[lock_key]).strip()
    cp["speaker"] = speaker
    if locked.get("narrative_mode"):
        cp["narrative_mode"] = str(locked["narrative_mode"]).strip()
    # WHY block — mirror motivation_<field> locks back into cp["motivation"]
    # so METAMAN-Dialogue overrides for surfaced motivation assumptions actually
    # show up in the storyboard prompt.
    motivation = dict(cp.get("motivation") or {})
    _changed_mot = False
    for key in _LOCK_TO_MOTIVATION:
        lock_key = f"motivation_{key}"
        if locked.get(lock_key):
            motivation[key] = str(locked[lock_key]).strip()
            _changed_mot = True
    if _changed_mot:
        # User explicitly resolved motivation in the dialogue — bump confidence
        # so the same fields don't get re-surfaced on subsequent passes.
        try:
            motivation["confidence"] = max(float(motivation.get("confidence") or 0.0), 0.9)
        except (TypeError, ValueError):
            motivation["confidence"] = 0.9
        cp["motivation"] = motivation
        scores = dict(cp.get("confidence_scores") or {})
        scores["motivation"] = motivation["confidence"]
        cp["confidence_scores"] = scores


@app.route("/project/<project_id>/shot/<int:shot_index>/motion_prompt", methods=["POST"])
@login_required
def update_shot_motion_prompt(project_id: str, shot_index: int):
    """Save an inline director override for a shot's motion_prompt."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if project.get("stage") != "storyboard_review":
        return {"ok": False, "error": "Not at storyboard review stage"}, 400
    new_value = (request.get_json(silent=True) or {}).get("motion_prompt", "")
    new_value = str(new_value).strip()[:500]
    timeline = list(project.get("styled_timeline") or [])
    updated = False
    for shot in timeline:
        if (shot.get("shot_index") or shot.get("timeline_index")) == shot_index:
            shot["motion_prompt"] = new_value
            updated = True
            break
    if not updated:
        return {"ok": False, "error": "Shot not found"}, 404
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET styled_timeline=%s, updated_at=NOW() WHERE id=%s",
            (Json(timeline), project_id),
        )
        conn.commit()
    return {"ok": True}


@app.route("/project/<project_id>/shot/<int:shot_index>/framing_directive", methods=["POST"])
@login_required
def update_shot_framing_directive(project_id: str, shot_index: int):
    """Save an inline director override for a shot's framing_directive."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if project.get("stage") != "storyboard_review":
        return {"ok": False, "error": "Not at storyboard review stage"}, 400
    new_value = (request.get_json(silent=True) or {}).get("framing_directive", "")
    new_value = str(new_value).strip()[:500]
    timeline = list(project.get("styled_timeline") or [])
    updated = False
    for shot in timeline:
        if (shot.get("shot_index") or shot.get("timeline_index")) == shot_index:
            shot["framing_directive"] = new_value
            updated = True
            break
    if not updated:
        return {"ok": False, "error": "Shot not found"}, 404
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET styled_timeline=%s, updated_at=NOW() WHERE id=%s",
            (Json(timeline), project_id),
        )
        conn.commit()
    return {"ok": True}


@app.route("/project/<project_id>/shot/<int:shot_index>/meaning", methods=["POST"])
@login_required
def update_shot_meaning(project_id: str, shot_index: int):
    """Save an inline director override for a shot's meaning (one-line summary)."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if project.get("stage") != "storyboard_review":
        return {"ok": False, "error": "Not at storyboard review stage"}, 400
    new_value = (request.get_json(silent=True) or {}).get("meaning", "")
    new_value = str(new_value).strip()[:500]
    timeline = list(project.get("styled_timeline") or [])
    updated = False
    for shot in timeline:
        if (shot.get("shot_index") or shot.get("timeline_index")) == shot_index:
            shot["meaning"] = new_value
            updated = True
            break
    if not updated:
        return {"ok": False, "error": "Shot not found"}, 404
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET styled_timeline=%s, updated_at=NOW() WHERE id=%s",
            (Json(timeline), project_id),
        )
        conn.commit()
    return {"ok": True}


@app.route("/project/<project_id>/shot/<int:shot_index>/styled_visual_prompt", methods=["POST"])
@login_required
def update_shot_styled_visual_prompt(project_id: str, shot_index: int):
    """Save an inline director override for a shot's styled_visual_prompt (image model prompt)."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if project.get("stage") != "storyboard_review":
        return {"ok": False, "error": "Not at storyboard review stage"}, 400
    new_value = (request.get_json(silent=True) or {}).get("styled_visual_prompt", "")
    new_value = str(new_value).strip()[:4000]
    timeline = list(project.get("styled_timeline") or [])
    updated = False
    for shot in timeline:
        if (shot.get("shot_index") or shot.get("timeline_index")) == shot_index:
            shot["styled_visual_prompt"] = new_value
            updated = True
            break
    if not updated:
        return {"ok": False, "error": "Shot not found"}, 404
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET styled_timeline=%s, updated_at=NOW() WHERE id=%s",
            (Json(timeline), project_id),
        )
        conn.commit()
    return {"ok": True}


@app.route("/project/<project_id>/advance/brief", methods=["POST"])
@login_required
def advance_brief(project_id: str):
    """Task #69 — User locked one of the Creative Brief variants.

    Persists the chosen variant + director's note + cast roster into
    context_packet.creative_brief.chosen, then kicks the Storyboard worker.
    Atomic gate: only transitions if currently parked at creative_brief_review.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    cp = dict(project.get("context_packet") or {})
    cb = dict(cp.get("creative_brief") or {})
    variants = list(cb.get("variants") or [])

    from creative_brief_engine import coerce_chosen
    chosen_in = coerce_chosen({
        "variant_id":       request.form.get("variant_id"),
        "title":            request.form.get("title"),
        "pitch":            request.form.get("pitch"),
        "treatment":        request.form.get("treatment"),
        "central_metaphor": request.form.get("central_metaphor"),
        "director_note":    request.form.get("director_note"),
        # UI sends a single comma-separated text input; always use scalar
        # so coerce_chosen's comma-split logic fires correctly.
        "cast_roster":      request.form.get("cast_roster"),
    })

    if not chosen_in["variant_id"]:
        flash("Please pick one variant before locking.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    # Validate the submitted variant_id against the generated variants to
    # block crafted POSTs that inject an arbitrary variant_id.
    valid_ids = {str(v.get("id") or "") for v in variants}
    if variants and chosen_in["variant_id"] not in valid_ids:
        flash("Invalid variant selection — please choose one of the listed options.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    # Inherit any unedited fields from the picked variant
    base = next((v for v in variants if v.get("id") == chosen_in["variant_id"]), {}) or {}
    for k in ("title", "pitch", "treatment", "central_metaphor", "justification"):
        if not chosen_in.get(k):
            chosen_in[k] = str(base.get(k) or "")
    # Do NOT auto-inherit cast_roster from base when the submitted value is
    # empty — blank = "no people on screen" is a valid intentional choice that
    # must be preserved for objects-only treatments.

    chosen_in["scenes"] = base.get("scenes") or []
    cb["chosen"] = chosen_in
    pending = cb.pop("_pending_overrides", {}) or {}
    cp["creative_brief"] = cb

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET context_packet=%s, status=%s, "
            "       stage=%s, error=NULL, updated_at=NOW() "
            " WHERE id=%s "
            "   AND stage='creative_brief_review' "
            "   AND status='awaiting_review'",
            (Json(cp), "queued", "queued", project_id),
        )
        if cur.rowcount != 1:
            flash("This step has already been completed or is not ready yet.", "error")
            return redirect(url_for("project_detail", project_id=project_id))

    overrides = {
        "speaker_name": pending.get("speaker_name"),
        "location":     pending.get("location"),
        "era":          pending.get("era"),
        "style_preset": pending.get("style_preset") or "cinematic_natural",
    }
    name = project.get("name") or "Qaivid_Project"
    kick_stage_2(project_id, name, overrides)
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/reset_brief", methods=["POST"])
@login_required
def reset_brief(project_id: str):
    """Task #70 — Return to Creative Brief selection without regenerating variants.

    Clears creative_brief.chosen so the user can pick a different variant (or
    edit the same one differently) and re-lock.  The existing variants are kept
    intact so the page loads instantly.  The project must currently be parked at
    storyboard_review (or creative_brief_review) and awaiting_review.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    allowed_stages = {"storyboard_review", "creative_brief_review"}
    if project.get("stage") not in allowed_stages or project.get("status") != "awaiting_review":
        flash("Cannot change the brief from the current project state.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    cp = dict(project.get("context_packet") or {})
    cb = dict(cp.get("creative_brief") or {})
    cb.pop("chosen", None)
    cp["creative_brief"] = cb

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET context_packet=%s, stage=%s, status=%s, "
            "       styled_timeline=NULL, error=NULL, updated_at=NOW() "
            " WHERE id=%s "
            "   AND stage IN ('storyboard_review', 'creative_brief_review') "
            "   AND status='awaiting_review'",
            (Json(cp), "creative_brief_review", "awaiting_review", project_id),
        )
        if cur.rowcount != 1:
            flash("Project state changed — please refresh and try again.", "error")
            return redirect(url_for("project_detail", project_id=project_id))
        conn.commit()

    flash("Brief reset — pick a different variant and lock when ready.", "info")
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/regenerate_brief", methods=["POST"])
@login_required
def regenerate_brief(project_id: str):
    """Task #70 — Kick _stage_brief_job again to generate a fresh set of variants.

    Preserves audio/context/style work. Re-uses the _pending_overrides stashed
    in creative_brief so the user doesn't have to re-enter speaker/location/era.
    Allowed from storyboard_review or creative_brief_review while awaiting_review.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    allowed_stages = {"storyboard_review", "creative_brief_review"}
    if project.get("stage") not in allowed_stages or project.get("status") != "awaiting_review":
        flash("Cannot regenerate the brief from the current project state.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    cp = dict(project.get("context_packet") or {})
    cb = dict(cp.get("creative_brief") or {})
    pending = cb.get("_pending_overrides") or {}

    overrides = {
        "speaker_name": pending.get("speaker_name"),
        "location":     pending.get("location"),
        "era":          pending.get("era"),
        "style_preset": pending.get("style_preset") or "cinematic_natural",
    }

    # Clear stale chosen state so the new brief review starts clean.
    cb.pop("chosen", None)
    cp["creative_brief"] = cb

    # Atomically flip to queued — prevents double-submission and mirrors
    # the gating pattern used by all other kick routes in this file.
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET context_packet=%s, styled_timeline=NULL, "
            "       stage=%s, status=%s, error=NULL, updated_at=NOW() "
            " WHERE id=%s "
            "   AND stage IN ('storyboard_review', 'creative_brief_review') "
            "   AND status='awaiting_review'",
            (Json(cp), "queued", "queued", project_id),
        )
        if cur.rowcount != 1:
            flash("Project state changed — please refresh and try again.", "error")
            return redirect(url_for("project_detail", project_id=project_id))
        conn.commit()

    kick_stage_brief(project_id, overrides)
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/advance/3", methods=["POST"])
@login_required
def advance_stage_3(project_id: str):
    """User approved the storyboard — kick the Reference Engine (Stage 3a).

    The reference stage generates ONE identity plate per character + ONE
    environment plate per location, parks at `references_review`, and the user
    must approve them before any per-shot still is rendered.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    char_ref_url: str | None = None
    env_ref_url: str | None = None
    try:
        char_ref_url = _save_upload_to_r2(project_id, request.files.get("character_ref"),
                                          "refs", "character")
        env_ref_url = _save_upload_to_r2(project_id, request.files.get("environment_ref"),
                                         "refs", "environment")
    except UploadValidationError as exc:
        flash(str(exc), "error")
        return redirect(url_for("project_detail", project_id=project_id))

    # Atomic gate: only kick refs if project is parked at storyboard_review.
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET status=%s, stage=%s, error=NULL, "
            "       updated_at=NOW() "
            " WHERE id=%s "
            "   AND stage='storyboard_review' "
            "   AND status='awaiting_review'",
            ("queued", "queued", project_id),
        )
        if cur.rowcount != 1:
            flash("This step has already been completed or is not ready yet.", "error")
            return redirect(url_for("project_detail", project_id=project_id))

    kick_stage_refs(project_id, char_ref_url, env_ref_url)
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/references/regenerate/<kind>/<int:entity_id>",
           methods=["POST"])
@login_required
def references_regenerate(project_id: str, kind: str, entity_id: int):
    """Regenerate a single character or location plate (optionally with a custom prompt)."""
    if kind not in ("character", "location"):
        abort(400)
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    # Only allow regenerate while parked at references_review awaiting review.
    # If the project has already advanced to stills/videos/final, the plates
    # are now in active use and must stay locked.
    if (project.get("stage") != "references_review"
            or project.get("status") != "awaiting_review"):
        flash("Reference plates can only be edited at the references review step.", "error")
        return redirect(url_for("project_detail", project_id=project_id))
    cp = project.get("context_packet") or {}
    location_dna = (cp.get("location_dna") or "Universal")
    prompt_override = (request.form.get("prompt") or "").strip() or None
    regenerate_entity_plate(project_id, kind, entity_id,
                            prompt_override=prompt_override,
                            location_dna=location_dna)
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/references/upload/<kind>/<int:entity_id>",
           methods=["POST"])
@login_required
def references_upload(project_id: str, kind: str, entity_id: int):
    """User uploads their own plate for one character or location."""
    if kind not in ("character", "location"):
        abort(400)
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if (project.get("stage") != "references_review"
            or project.get("status") != "awaiting_review"):
        flash("Reference plates can only be edited at the references review step.", "error")
        return redirect(url_for("project_detail", project_id=project_id))
    try:
        file_url = _save_upload_to_r2(project_id, request.files.get("plate"),
                                      "refs", f"{kind}_{entity_id}")
    except UploadValidationError as exc:
        flash(str(exc), "error")
        return redirect(url_for("project_detail", project_id=project_id))
    if not file_url:
        flash("Please choose an image to upload.", "error")
        return redirect(url_for("project_detail", project_id=project_id))
    set_entity_uploaded_plate(project_id, kind, entity_id, file_url)
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/references/approve", methods=["POST"])
@login_required
def references_approve(project_id: str):
    """User approved the reference plates — kick Stills (Stage 3b).

    Refuses to advance if any plate is still pending/rendering/failed so the
    user can't accidentally start the expensive stills run with broken refs.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    # Single atomic gate: flip the project to 'queued' ONLY if it's parked at
    # references_review awaiting review AND every character/location plate is
    # already 'ready'. This closes the TOCTOU window where a regenerate could
    # flip a plate to 'rendering'/'failed' between a separate readiness check
    # and the stage update.
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET status=%s, stage=%s, error=NULL, "
            "       updated_at=NOW() "
            " WHERE id=%s "
            "   AND stage='references_review' "
            "   AND status='awaiting_review' "
            "   AND NOT EXISTS ("
            "         SELECT 1 FROM characters "
            "          WHERE project_id=%s AND ref_status <> 'ready'"
            "   ) "
            "   AND NOT EXISTS ("
            "         SELECT 1 FROM locations "
            "          WHERE project_id=%s AND ref_status <> 'ready'"
            "   )",
            # SET status=%s, stage=%s — order matters! status is the
            # state-machine state ('awaiting_review'), stage is the step
            # name ('stills_control'). Swapping these breaks the
            # _stills_control_guard 403 check downstream.
            ("awaiting_review", "stills_control", project_id, project_id, project_id),
        )
        advanced = cur.rowcount == 1
        if advanced:
            # Seed shot_assets with prompts from the styled_timeline so the
            # stills_control page has editable prompts immediately.
            timeline = project.get("styled_timeline") or []
            if timeline:
                seed_shot_rows_with_prompts(project_id, timeline)
        conn.commit()

    if not advanced:
        # Either someone else moved the stage, or a plate isn't ready. Tell the
        # user what's actually blocking them.
        with db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM characters "
                " WHERE project_id=%s AND ref_status <> 'ready'",
                (project_id,),
            )
            bad_chars = (cur.fetchone() or {}).get("n", 0) or 0
            cur.execute(
                "SELECT COUNT(*) AS n FROM locations "
                " WHERE project_id=%s AND ref_status <> 'ready'",
                (project_id,),
            )
            bad_locs = (cur.fetchone() or {}).get("n", 0) or 0
        if bad_chars or bad_locs:
            flash(
                f"{bad_chars + bad_locs} reference plate(s) are not ready yet. "
                f"Regenerate or upload a replacement before continuing.",
                "error",
            )
        else:
            flash("This step has already been completed or is not ready yet.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/advance/4", methods=["POST"])
@login_required
def advance_stage_4(project_id: str):
    """User approved the stills — kick Video generation (Stage 4)."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    kick_stage_4(project_id)
    return redirect(url_for("project_detail", project_id=project_id))


# ── Stills Control routes ────────────────────────────────────────────────────

_STILLS_EDITABLE_STAGES = (
    "stills_control", "stills_review", "post_production", "videos_review", "complete",
)


def _stills_control_guard(project_id: str):
    """Return project if user is allowed to edit/regenerate individual stills.

    Editing is allowed once shot rows have been seeded (stills_control onward).
    We deliberately keep it open at stills_review / videos_review / complete so
    directors can still iterate on individual shots after the first auto-pass.
    """
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if project.get("stage") not in _STILLS_EDITABLE_STAGES:
        abort(403)
    return project


@app.route("/project/<project_id>/stills/status.json")
@login_required
def stills_status_json(project_id: str):
    """Live status of all shot_assets rows — used by the polling JS."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if project.get("stage") not in ("stills_control", "stills_review"):
        abort(403)
    assets = _get_shot_assets(project_id)
    return jsonify({
        "shots": [
            {
                "shot_index": a["shot_index"],
                "status": a["status"],
                "url": _asset_url(a.get("file_path")),
                "source": a.get("source"),
                "error": a.get("error"),
            }
            for a in assets
        ]
    })


@app.route("/project/<project_id>/stills/generate/<int:shot_index>", methods=["POST"])
@login_required
def stills_generate_one(project_id: str, shot_index: int):
    """Kick generation for a single shot (AJAX — returns JSON)."""
    _stills_control_guard(project_id)
    kick_single_shot(project_id, shot_index)
    return jsonify({"ok": True, "shot_index": shot_index})


@app.route("/project/<project_id>/stills/generate-all", methods=["POST"])
@login_required
def stills_generate_all(project_id: str):
    """Kick generation for shots (AJAX — returns JSON).

    Query/body param ``force=1`` regenerates every shot, including those
    already marked ``ready`` (used by the bulk Regenerate All button).
    Without it, only non-ready shots are queued.
    """
    _stills_control_guard(project_id)
    force_raw = (request.values.get("force") or "").strip().lower()
    force = force_raw in ("1", "true", "yes", "on")
    kick_all_pending_shots(project_id, force=force)
    return jsonify({"ok": True, "force": force})


@app.route("/project/<project_id>/stills/prompt", methods=["POST"])
@login_required
def stills_update_prompt(project_id: str):
    """Save an edited prompt for a shot (AJAX — returns JSON)."""
    _stills_control_guard(project_id)
    data = request.get_json(silent=True) or {}
    shot_index = data.get("shot_index")
    prompt = (data.get("prompt") or "").strip()
    if shot_index is None or not prompt:
        return jsonify({"ok": False, "error": "shot_index and prompt required"}), 400
    update_shot_prompt(project_id, int(shot_index), prompt)
    return jsonify({"ok": True})


@app.route("/project/<project_id>/stills/upload/<int:shot_index>", methods=["POST"])
@login_required
def stills_upload_one(project_id: str, shot_index: int):
    """User uploads their own image for a specific shot."""
    _stills_control_guard(project_id)
    try:
        file_url = _save_upload_to_r2(project_id, request.files.get("image"),
                                      "stills", f"shot_{shot_index}")
    except UploadValidationError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if not file_url:
        return jsonify({"ok": False, "error": "No file received."}), 400
    set_shot_uploaded_image(project_id, shot_index, file_url)
    return jsonify({"ok": True, "url": _asset_url(file_url)})


@app.route("/project/<project_id>/stills/approve", methods=["POST"])
@login_required
def stills_approve(project_id: str):
    """User approved all stills — advance to Post Production stage."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET status='awaiting_review', stage='post_production', "
            "updated_at=NOW() "
            " WHERE id=%s AND stage='stills_control' AND status='awaiting_review'",
            (project_id,),
        )
        advanced = cur.rowcount == 1
        conn.commit()
    if not advanced:
        flash("Cannot advance — project is not at the stills review step.", "error")
        return redirect(url_for("project_detail", project_id=project_id))
    return redirect(url_for("postprod_page", project_id=project_id))


@app.route("/project/<project_id>/advance/5", methods=["POST"])
@login_required
def advance_stage_5(project_id: str):
    """User approved the video clips — kick Final assembly (Stage 5)."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)

    # Atomic gate: only kick assembly when project is parked at videos_review
    # awaiting review. Blocks crafted POSTs from earlier stages and prevents
    # double-submits from queueing two assembly jobs.
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET status=%s, stage=%s, error=NULL, "
            "       updated_at=NOW() "
            " WHERE id=%s "
            "   AND stage='videos_review' "
            "   AND status='awaiting_review'",
            ("queued", "queued", project_id),
        )
        if cur.rowcount != 1:
            flash("This step has already been completed or is not ready yet.", "error")
            return redirect(url_for("project_detail", project_id=project_id))

    kick_stage_5(project_id)
    return redirect(url_for("project_detail", project_id=project_id))


# ── Post Production Routes (Task #100) ───────────────────────────────────────

ALLOWED_LOGO = {".png"}        # spec: PNG-only overlay slots
ALLOWED_SRT  = {".srt", ".vtt"}


@app.route("/project/<project_id>/postprod", methods=["GET"])
@login_required
def postprod_page(project_id: str):
    """Render the Post Production studio page."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if project.get("stage") not in ("post_production", "videos_review", "final_review",
                                    "complete", "stills_review"):
        flash("Post Production is not available yet — approve your stills first.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    # Populate wizard fields so _stage_header.html can render clickable back-steps
    project["_actual_stage"] = project.get("stage") or "post_production"
    project["_viewing_stage"] = project.get("stage") or "post_production"
    project["_is_review_only"] = False
    project["_viewing_as_admin"] = False
    project["_owner_email"] = None
    project["_actual_stage_label"] = STAGE_LABELS.get(project.get("stage", ""), "Post Production")

    shot_assets = _get_shot_assets(project_id)
    timeline = project.get("styled_timeline") or []
    shots = []
    for a in shot_assets:
        tl_shot = next(
            (s for s in timeline
             if (s.get("shot_index") or s.get("timeline_index")) == a["shot_index"]),
            {},
        )
        p = _shot_payload(a, tl_shot)
        p["url"] = _asset_url(a.get("file_path"))
        shots.append(p)

    config = project.get("postprod_config") or {}
    quick_video_url = _asset_url(project.get("quick_video_url")) if project.get("quick_video_url") else None
    total_duration = sum(s.get("duration") or 0 for s in timeline)

    # Construct audio preview URL from the uploaded audio file
    audio_url = None
    audio_filename = project.get("audio_filename")
    if audio_filename:
        try:
            import r2_storage as _r2
            if _r2.r2_available():
                raw_url = _r2.public_url_for(f"projects/{project_id}/uploads/{audio_filename}")
                audio_url = url_for("r2proxy", url=raw_url, _external=False)
        except Exception:
            pass

    return render_template(
        "stage_postprod.html",
        project=project,
        shots=shots,
        config=config,
        quick_video_url=quick_video_url,
        total_duration=round(total_duration, 1),
        audio_url=audio_url,
    )


@app.route("/project/<project_id>/postprod/save", methods=["POST"])
@login_required
def postprod_save(project_id: str):
    """Upsert the postprod_config JSON for this project."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    data = request.get_json(silent=True) or {}
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET postprod_config=%s, updated_at=NOW() WHERE id=%s",
            (Json(data), project_id),
        )
        conn.commit()
    return jsonify({"ok": True})


@app.route("/project/<project_id>/postprod/generate", methods=["POST"])
@login_required
def postprod_generate(project_id: str):
    """Kick async Quick Video assembly."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    settings = request.get_json(silent=True) or {}
    kick_quick_video(project_id, settings)
    return jsonify({"ok": True, "status": "generating"})


@app.route("/project/<project_id>/postprod/status", methods=["GET"])
@login_required
def postprod_status(project_id: str):
    """Poll for quick video URL and error state."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    config = project.get("postprod_config") or {}
    generating = bool(config.get("generating"))
    error = config.get("quick_video_error") or ""
    raw_url = project.get("quick_video_url") or ""
    url = _asset_url(raw_url) if raw_url else ""
    if url:
        generating = False
    return jsonify({"ok": True, "generating": generating, "url": url, "error": error})


@app.route("/project/<project_id>/postprod/advance", methods=["POST"])
@login_required
def postprod_advance(project_id: str):
    """Skip quick video — kick AI render (Stage 4) and go to videos_review."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    stage = project.get("stage")
    if stage not in ("post_production", "stills_review"):
        flash("Cannot advance from this stage.", "error")
        return redirect(url_for("project_detail", project_id=project_id))
    kick_stage_4(project_id)
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<project_id>/postprod/upload_srt", methods=["POST"])
@login_required
def postprod_upload_srt(project_id: str):
    """Upload an SRT/VTT subtitle file and store its R2 key in postprod_config."""
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    f = request.files.get("srt_file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "No file provided"}), 400
    ext = Path(secure_filename(f.filename)).suffix.lower()
    if ext not in ALLOWED_SRT:
        return jsonify({"ok": False, "error": "Only .srt and .vtt files accepted"}), 400
    raw_bytes = f.read()
    if len(raw_bytes) > 2 * 1024 * 1024:
        return jsonify({"ok": False, "error": "File too large (max 2 MB)"}), 400
    r2_key = f"projects/{project_id}/quick/subtitles{ext}"
    if r2_storage.r2_available():
        r2_storage.upload_bytes(raw_bytes, r2_key, content_type="text/plain")
    else:
        local_path = PROJECTS_ROOT / project_id / "quick"
        local_path.mkdir(parents=True, exist_ok=True)
        (local_path / f"subtitles{ext}").write_bytes(raw_bytes)
    config = project.get("postprod_config") or {}
    config["srt_r2_key"] = r2_key
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET postprod_config=%s, updated_at=NOW() WHERE id=%s",
            (Json(config), project_id),
        )
        conn.commit()
    return jsonify({"ok": True, "r2_key": r2_key, "filename": f.filename})


@app.route("/project/<project_id>/postprod/upload_logo/<slot>", methods=["POST"])
@login_required
def postprod_upload_logo(project_id: str, slot: str):
    """Upload a logo PNG for a given slot (top-left, top-right, bottom-left, bottom-right)."""
    if slot not in ("top-left", "top-right", "bottom-left", "bottom-right"):
        return jsonify({"ok": False, "error": "Invalid slot"}), 400
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    f = request.files.get("logo_file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "No file provided"}), 400
    ext = Path(secure_filename(f.filename)).suffix.lower()
    if ext not in ALLOWED_LOGO:
        return jsonify({"ok": False, "error": "Only PNG files accepted for logo overlays"}), 400
    raw_bytes = f.read()
    if len(raw_bytes) > 5 * 1024 * 1024:
        return jsonify({"ok": False, "error": "File too large (max 5 MB)"}), 400
    safe_slot = slot.replace("-", "_")
    r2_key = f"projects/{project_id}/quick/logo_{safe_slot}{ext}"
    if r2_storage.r2_available():
        r2_storage.upload_bytes(raw_bytes, r2_key, content_type=f"image/{ext.lstrip('.')}")
    else:
        local_path = PROJECTS_ROOT / project_id / "quick"
        local_path.mkdir(parents=True, exist_ok=True)
        (local_path / f"logo_{safe_slot}{ext}").write_bytes(raw_bytes)
    config = project.get("postprod_config") or {}
    logos = config.get("logos") or {}
    logos[slot] = {**(logos.get(slot) or {}), "r2_key": r2_key, "filename": f.filename}
    config["logos"] = logos
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET postprod_config=%s, updated_at=NOW() WHERE id=%s",
            (Json(config), project_id),
        )
        conn.commit()
    preview_url = _asset_url(r2_key) if r2_storage.r2_available() else ""
    return jsonify({"ok": True, "r2_key": r2_key, "preview_url": preview_url})


@app.route("/project/<project_id>/retry/all_failed_refs", methods=["POST"])
@login_required
def project_retry_all_failed_refs(project_id: str):
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    try:
        counts = retry_all_failed_refs(project_id)
        return jsonify({"ok": True, **counts})
    except Exception as exc:
        logger.exception("retry_all_failed_refs failed for project=%s", project_id)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/project/<project_id>/retry/all_failed_shots", methods=["POST"])
@login_required
def project_retry_all_failed_shots(project_id: str):
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    try:
        n = retry_all_failed_shots(project_id)
        return jsonify({"ok": True, "queued": n})
    except Exception as exc:
        logger.exception("retry_all_failed_shots failed for project=%s", project_id)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/project/<project_id>/retry/shot/<int:shot_index>", methods=["POST"])
@login_required
def project_retry_shot(project_id: str, shot_index: int):
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    # Gate: per-shot retries are only allowed once stills generation has begun.
    # Otherwise a user could re-trigger _render_shot from the references_review
    # page and bypass the entire reference-approval gate.
    if (project.get("stage") or "") not in (
        "running_3", "stills_review", "running_4", "videos_review",
        "running_5", "final_review", "complete",
    ):
        return jsonify({
            "ok": False,
            "error": "Stills haven't been started yet — approve the reference plates first.",
        }), 409
    try:
        retry_shot(project_id, shot_index)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/project/<project_id>/retry/video/<int:shot_index>", methods=["POST"])
@login_required
def project_retry_video(project_id: str, shot_index: int):
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    try:
        retry_video(project_id, shot_index)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/project/<project_id>/generate_videos", methods=["POST"])
@login_required
def project_generate_videos(project_id: str):
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    if project.get("status") not in {"complete", "failed"}:
        return jsonify({"ok": False, "error": "Project pipeline must complete first."}), 400
    try:
        generate_all_videos(project_id)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/project/<project_id>/retry/ref/<role>", methods=["POST"])
@login_required
def project_retry_ref(project_id: str, role: str):
    if role not in {"character", "environment"}:
        abort(400)
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    try:
        retry_ref(project_id, role)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/project/<project_id>/delete", methods=["POST"])
@login_required
def project_delete(project_id: str):
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    cleanup_project_assets(project_id)
    with db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
        conn.commit()
    flash(f"Project '{project.get('name')}' deleted.", "success")
    return redirect(url_for("projects"))


@app.route("/project/<project_id>/export.json")
@login_required
def project_export(project_id: str):
    project = _get_project(project_id, current_user()["id"])
    if not project:
        abort(404)
    export_path = project.get("export_path") or ""
    # R2 URL — redirect the browser directly to it
    if export_path.startswith("http://") or export_path.startswith("https://"):
        return redirect(export_path)
    abort(404)


@app.route("/admin")
@admin_required
def admin():
    with db() as conn, conn.cursor() as cur:
        # Users with project counts
        cur.execute(
            """
            SELECT u.id, u.email, u.is_admin, u.created_at,
                   u.plan, u.plan_expires_at, u.stripe_customer_id,
                   COUNT(p.id) AS project_count
            FROM users u
            LEFT JOIN projects p ON p.user_id = u.id
            GROUP BY u.id
            ORDER BY u.created_at DESC
            """
        )
        users = cur.fetchall()

        # Projects with owner
        cur.execute(
            """
            SELECT p.id, p.name, p.genre, p.status, p.stage,
                   p.created_at, p.updated_at, u.email AS owner_email,
                   p.error
            FROM projects p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC
            LIMIT 500
            """
        )
        projects = cur.fetchall()

        # Aggregate user stats
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_users,
                COUNT(CASE WHEN plan = 'pro' THEN 1 END) AS pro_count,
                COUNT(CASE WHEN plan = 'studio' THEN 1 END) AS studio_count,
                COUNT(CASE WHEN is_admin THEN 1 END) AS admin_count,
                COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) AS signups_7d,
                COUNT(CASE WHEN created_at >= NOW() - INTERVAL '30 days' THEN 1 END) AS signups_30d
            FROM users
            """
        )
        user_stats = cur.fetchone()

        # Aggregate project stats
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_projects,
                COUNT(CASE WHEN status = 'complete' THEN 1 END) AS complete_count,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) AS failed_count,
                COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) AS projects_7d
            FROM projects
            """
        )
        proj_stats = cur.fetchone()

        cur.execute("SELECT COUNT(*) AS c FROM shot_assets")
        total_shots = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM video_assets")
        total_videos = cur.fetchone()["c"]

        # Status counts
        cur.execute("SELECT status, COUNT(*) AS c FROM projects GROUP BY status")
        status_counts = {r["status"]: r["c"] for r in cur.fetchall()}

        # Recent activity (last 15 signups + last 15 projects)
        cur.execute(
            """
            SELECT email, created_at FROM users
            ORDER BY created_at DESC LIMIT 15
            """
        )
        recent_signups = cur.fetchall()
        cur.execute(
            """
            SELECT p.id, p.name, p.status, p.created_at, u.email AS owner_email
            FROM projects p JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC LIMIT 15
            """
        )
        recent_projects = cur.fetchall()

    me = current_user()
    mrr = (user_stats["pro_count"] * 29) + (user_stats["studio_count"] * 99)
    completion_rate = (
        round(proj_stats["complete_count"] / proj_stats["total_projects"] * 100)
        if proj_stats["total_projects"] > 0 else 0
    )
    try:
        from system_config import get_image_modes
        image_modes = get_image_modes()
    except Exception:
        image_modes = {"ref": "quality", "shot": "quality"}
    stripe_ok = bool(os.getenv("STRIPE_SECRET_KEY"))
    stripe_webhook_ok = bool(os.getenv("STRIPE_WEBHOOK_SECRET"))
    return render_template(
        "admin.html",
        users=users,
        projects=projects,
        user_stats=user_stats,
        proj_stats=proj_stats,
        total_shots=total_shots,
        total_videos=total_videos,
        status_counts=status_counts,
        recent_signups=recent_signups,
        recent_projects=recent_projects,
        mrr=mrr,
        completion_rate=completion_rate,
        r2_ok=r2_storage.r2_available(),
        api_ok=bool(_api_key()),
        fal_ok=_fal_set(),
        stripe_ok=stripe_ok,
        stripe_webhook_ok=stripe_webhook_ok,
        current_user_id=me["id"],
        image_modes=image_modes,
    )


@app.route("/admin/user/<int:user_id>/set_plan", methods=["POST"])
@admin_required
def admin_set_user_plan(user_id: int):
    plan = request.form.get("plan", "free")
    if plan not in ("free", "pro", "studio"):
        flash("Invalid plan value.", "error")
        return redirect(url_for("admin"))
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    if not row:
        flash("User not found.", "error")
        return redirect(url_for("admin"))
    update_user_plan(user_id, plan)
    flash(f"{row['email']} plan set to {plan}.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/user/<int:user_id>/impersonate", methods=["POST"])
@admin_required
def admin_impersonate(user_id: int):
    from flask import session as _sess
    me = current_user()
    if user_id == me["id"]:
        flash("You can't impersonate yourself.", "error")
        return redirect(url_for("admin"))
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
        target = cur.fetchone()
    if not target:
        flash("User not found.", "error")
        return redirect(url_for("admin"))
    _sess["original_admin_id"] = me["id"]
    _sess["user_id"] = user_id
    # Invalidate cached user on g
    from flask import g as _g
    _g.pop("current_user", None)
    flash(f"Now impersonating {target['email']}. Use Exit Impersonation to return.", "info")
    return redirect(url_for("projects"))


@app.route("/admin/impersonate/exit", methods=["POST"])
@login_required
def admin_impersonate_exit():
    from flask import session as _sess
    original_id = _sess.get("original_admin_id")
    if not original_id:
        flash("You are not impersonating anyone.", "info")
        return redirect(url_for("new_project"))
    _sess["user_id"] = original_id
    _sess.pop("original_admin_id", None)
    from flask import g as _g
    _g.pop("current_user", None)
    flash("Returned to admin account.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/user/<int:user_id>/toggle_admin", methods=["POST"])
@admin_required
def admin_toggle_admin(user_id: int):
    me = current_user()
    if user_id == me["id"]:
        flash("You can't change your own admin status.", "error")
        return redirect(url_for("admin"))
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET is_admin = NOT is_admin WHERE id = %s "
            "RETURNING email, is_admin",
            (user_id,),
        )
        row = cur.fetchone()
        conn.commit()
    if not row:
        flash("User not found.", "error")
    else:
        state = "admin" if row["is_admin"] else "regular user"
        flash(f"{row['email']} is now a {state}.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/user/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id: int):
    me = current_user()
    if user_id == me["id"]:
        flash("You can't delete your own account.", "error")
        return redirect(url_for("admin"))
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM projects WHERE user_id = %s", (user_id,))
        project_ids = [r["id"] for r in cur.fetchall()]
        cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
        target = cur.fetchone()
    for pid in project_ids:
        try:
            cleanup_project_assets(pid)
        except Exception:
            pass
    with db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
    if target:
        flash(
            f"Deleted {target['email']} and {len(project_ids)} project(s).",
            "success",
        )
    else:
        flash("User not found.", "error")
    return redirect(url_for("admin"))


@app.route("/admin/project/<project_id>/delete", methods=["POST"])
@admin_required
def admin_delete_project(project_id: str):
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name FROM projects WHERE id = %s", (project_id,)
        )
        row = cur.fetchone()
    if not row:
        flash("Project not found.", "error")
        return redirect(url_for("admin"))
    try:
        cleanup_project_assets(project_id)
    except Exception:
        pass
    with db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
        conn.commit()
    flash(f"Deleted project '{row['name']}'.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/image_modes", methods=["POST"])
@admin_required
def admin_set_image_modes():
    from system_config import set_setting, KEY_REF, KEY_SHOT, invalidate_cache
    ref_mode = request.form.get("ref_mode", "quality")
    shot_mode = request.form.get("shot_mode", "quality")
    set_setting(KEY_REF, ref_mode)
    set_setting(KEY_SHOT, shot_mode)
    invalidate_cache()
    flash(
        f"Image modes updated — References: {ref_mode}, Stills: {shot_mode}.",
        "success",
    )
    return redirect(url_for("admin"))


@app.route("/api/wardrobe/diversify/<project_id>", methods=["POST"])
@login_required
def api_diversify_wardrobe(project_id: str):
    """Re-run the wardrobe engine for a project.

    Called from the storyboard review page (or admin) to re-assign
    scene-appropriate outfits without re-generating any images.
    Safe to call multiple times — idempotent; skips prompt_user_edited shots.
    """
    from pipeline_worker import _db
    # Verify ownership
    with _db() as conn, conn.cursor() as cur:
        cur.execute("SELECT user_id FROM projects WHERE id=%s", (project_id,))
        row = cur.fetchone()
    if not row:
        return {"ok": False, "error": "Project not found"}, 404
    if row["user_id"] != current_user.id and not current_user.is_admin:
        return {"ok": False, "error": "Forbidden"}, 403
    try:
        from wardrobe_engine import diversify_wardrobe
        n = diversify_wardrobe(project_id)
        return {"ok": True, "shots_updated": n}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 500


@app.route("/r2proxy")
@login_required
def r2proxy():
    """Serve a private R2 asset via presigned redirect with ownership check.

    Usage: /r2proxy?url=<full R2 URL stored in DB>
    Extracts the R2 key, verifies the asset belongs to a project the current
    user owns, generates a short-lived presigned URL, and redirects to it.
    """
    from urllib.parse import urlparse
    r2_url = request.args.get("url", "")
    if not r2_url:
        abort(400)

    parsed = urlparse(r2_url)
    raw_path = parsed.path.lstrip("/")
    bucket = os.environ.get("R2_BUCKET_NAME", "")
    if bucket and raw_path.startswith(bucket + "/"):
        r2_key = raw_path[len(bucket) + 1:]
    else:
        r2_key = raw_path

    # Ownership check — R2 keys are always projects/<project_id>/...
    # Admins may access any project; regular users only their own.
    parts = r2_key.split("/")
    if len(parts) < 2 or parts[0] != "projects":
        abort(403)
    project_id = parts[1]
    uid = current_user()["id"]
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM projects WHERE id = %s",
            (project_id,),
        )
        row = cur.fetchone()
    if not row:
        abort(404)
    if row["user_id"] != uid and not current_user().get("is_admin"):
        abort(403)

    try:
        presigned = r2_storage.presigned_url(r2_key, expires_in=900)
        return redirect(presigned)
    except Exception:
        abort(404)


@app.route("/healthz")
def healthz():
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            db_ok = cur.fetchone() is not None
    except Exception:
        db_ok = False
    return {
        "status": "ok",
        "api_key_set": bool(_api_key()),
        "fal_key_set": _fal_set(),
        "r2_set": r2_storage.r2_available(),
        "db_ok": db_ok,
    }


_SITE_ALLOWED_KEYS = frozenset([
    "site_header_tagline", "site_header_film_opacity",
    "site_hero_eyebrow", "site_hero_line1", "site_hero_line2",
    "site_hero_tagline", "site_hero_subtitle",
    "site_hero_cta_primary", "site_hero_cta_secondary",
    "site_trust_1", "site_trust_2", "site_trust_3", "site_trust_4",
    "site_howitworks_title", "site_howitworks_sub",
    "site_features_title", "site_features_sub",
    "site_pricing_title", "site_pricing_sub",
    "site_footer_tagline",
    "site_banner_enabled", "site_banner_text", "site_banner_color",
])


@app.route("/admin/site-settings", methods=["POST"])
@admin_required
def admin_site_settings_save():
    from system_config import set_raw, invalidate_cache
    data = request.get_json(force=True) or {}
    saved = []
    for k, v in data.items():
        if k in _SITE_ALLOWED_KEYS:
            set_raw(k, str(v))
            saved.append(k)
    invalidate_cache()
    return jsonify({"ok": True, "saved": saved})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
