"""
wardrobe_engine.py — Scene-aware wardrobe diversification.

Problem: A music video character previously wore the same outfit in EVERY shot
because `characters.wardrobe` is a single string and the continuity cue locked
the model to copy the reference plate's clothes for all 70 shots. That's not
how real filmmaking works — wardrobe changes between scenes.

Solution: After the storyboard is generated, group shots into scene clusters
(distinct location/setting/mood combinations). Call an LLM once per project to
assign a scene-appropriate outfit to each cluster. Store the per-shot result in
`shot_assets.wardrobe_context`. The prompt composer then uses this instead of
the global character wardrobe.

Architecture:
- world_assumptions (geography, era, social_context, economic_context,
  characteristic_setting) = the cultural anchor for ALL wardrobe decisions.
- scene_frame (location, time_of_day, props) = the per-scene override that
  adjusts formality, colour, and accessory weight within that cultural world.
- LLM derives wardrobe from this two-level context — no heuristic keyword
  matching, no hardcoded ethnicity defaults, no South Asian bias.

Rules:
- Shots within the same scene cluster share ONE outfit (realistic continuity).
- Different scene clusters get distinct outfits (wedding ≠ morning home ≠ field).
- Wardrobe is culturally derived from world_assumptions — not imposed.
- User-edited shot prompts (`prompt_user_edited=TRUE`) are never touched.
- Always writes to DB idempotently — re-running is safe.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# ── Cluster heuristics ────────────────────────────────────────────────────────

def _env_field(env: dict, *keys: str) -> str:
    """Read the first non-empty value from scene_frame, then world_assumptions,
    across a list of candidate key names (supports both new and legacy names)."""
    sf = env.get("scene_frame") or {}
    wa = env.get("world_assumptions") or {}
    for k in keys:
        v = (sf.get(k) or wa.get(k) or "").strip()
        if v:
            return v
    return ""


def _cluster_key(shot: dict, shot_location_ids: dict) -> str:
    """Return a stable string key that groups shots sharing the same
    costume context.  Primary signal: linked location_id (most reliable).
    Fallback: scene location + time_of_day from environment_profile."""
    idx = shot.get("shot_index") or shot.get("timeline_index")
    loc_id = shot_location_ids.get(idx)
    if loc_id:
        return f"loc:{loc_id}"
    env = shot.get("environment_profile") or {}
    # Prefer scene_frame (specific location), fall back to characteristic_setting
    setting = _env_field(env, "location", "characteristic_setting", "domestic_setting").lower()
    tod = _env_field(env, "time_of_day", "characteristic_time").lower()
    place_raw = env.get("place_entities") or []
    place = "_".join(sorted(p.lower() for p in place_raw if p))
    # Combine to produce a stable, descriptive key
    parts = [p for p in (setting, place, tod) if p]
    return ("_".join(parts) or "default").replace(" ", "_")[:80]


def _build_cluster_descriptions(shots: list[dict], shot_location_ids: dict,
                                 location_rows: list[dict]) -> dict[str, dict]:
    """Return {cluster_key: {description, time_of_day, scene_props, ...}}."""
    loc_map = {r["id"]: r for r in location_rows}
    clusters: dict[str, dict] = {}
    for shot in shots:
        key = _cluster_key(shot, shot_location_ids)
        if key not in clusters:
            idx = shot.get("shot_index") or shot.get("timeline_index")
            loc_id = shot_location_ids.get(idx)
            loc_rec = loc_map.get(loc_id) if loc_id else None
            env = shot.get("environment_profile") or {}
            wa = env.get("world_assumptions") or {}
            sf = env.get("scene_frame") or {}
            place_entities = env.get("place_entities") or []
            location_dna = env.get("location_dna") or ""
            # Props from creative brief scene_frame (e.g. "worn letter, candle")
            scene_props = sf.get("props") or []
            if isinstance(scene_props, str):
                scene_props = [p.strip() for p in scene_props.split(",") if p.strip()]
            clusters[key] = {
                "cluster_id": key,
                "location_name": (loc_rec or {}).get("name") or location_dna or key,
                "setting": _env_field(env, "location", "characteristic_setting", "domestic_setting"),
                "time_of_day": _env_field(env, "time_of_day", "characteristic_time"),
                "place_entities": place_entities,
                "location_mood": (loc_rec or {}).get("mood") or "",
                "location_desc": (loc_rec or {}).get("description") or "",
                "scene_name": (sf.get("scene_name") or "").strip(),
                "scene_props": scene_props,
                "era": (wa.get("era") or "").strip(),
                "social_context": (wa.get("social_context") or "").strip(),
                "shot_meanings": [],
                "n_shots": 0,
            }
        # Accumulate shot meanings to give the LLM richer context
        meaning = (shot.get("meaning") or "").strip()
        if meaning and meaning not in clusters[key]["shot_meanings"]:
            clusters[key]["shot_meanings"].append(meaning)
        clusters[key]["n_shots"] += 1
    return clusters


# ── LLM call ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the costume and wardrobe designer for a cinematic music video.
Your job is to assign a specific, visually distinct outfit to each scene cluster
so that the character's look changes authentically between scenes — just like
real filmmaking, where characters wear different clothes in different settings.

You will receive:
- A "Cultural World Context" that anchors ALL wardrobe decisions: geography, era,
  social class, and the broad cultural setting. Every outfit must feel native to
  this world (fabrics, silhouettes, colour traditions, accessory conventions).
- Per-scene details: location, time of day, props present in the scene, mood,
  and the emotional meaning of the shots. These drive formality and styling
  within the cultural world.

Rules:
1. Derive wardrobe FROM the cultural world context — never impose a default or
   generic look that ignores geography/era/social class.
2. Calibrate FORMALITY from the scene: a ceremony or performance merits heavier
   fabric and jewellery; a dawn field scene merits light, worn everyday clothes.
3. Include specific detail: garment names native to the culture, exact colours,
   fabric texture, accessories, hair style.
4. When props are listed (a worn letter, candle, wind-swept cloth), let them
   influence the character's practical wardrobe choices.
5. Two clusters with the same formality can still differ in colour palette and
   accessory weight.
6. Output ONLY a valid JSON array — no markdown, no commentary.
   Format: [{"cluster_id": "...", "wardrobe": "...one detailed English sentence..."}]"""


def _call_llm(character: dict, clusters: dict[str, dict],
              world_assumptions: dict | None = None) -> dict[str, str]:
    """Call GPT to assign per-cluster wardrobes. Returns {cluster_id: wardrobe_string}.

    world_assumptions (geography, era, social_context, economic_context,
    characteristic_setting) is passed as the cultural anchor so the LLM derives
    wardrobe from the song's actual cultural world rather than an imposed default.
    """
    from openai import OpenAI
    client = OpenAI()

    wa = world_assumptions or {}
    name = character.get("name") or "the character"
    # Derive ethnicity from character row; if absent, derive from cultural geography
    # rather than falling back to a hardcoded default
    ethnicity = (character.get("ethnicity") or "").strip()
    base_wardrobe = character.get("wardrobe") or ""
    grooming = character.get("grooming") or ""
    gender = character.get("gender") or ""
    appearance = character.get("appearance") or ""

    # Build Cultural World Context section from world_assumptions
    geography = (wa.get("geography") or "").strip()
    era = (wa.get("era") or "").strip()
    social_context = (wa.get("social_context") or "").strip()
    economic_context = (wa.get("economic_context") or "").strip()
    characteristic_setting = (
        wa.get("characteristic_setting") or wa.get("domestic_setting") or ""
    ).strip()
    architecture = (wa.get("architecture_style") or "").strip()

    # Assemble cultural world context lines (omit empty fields)
    world_lines = []
    if geography:
        world_lines.append(f"Geography: {geography}")
    if era:
        world_lines.append(f"Era: {era}")
    if social_context:
        world_lines.append(f"Social context: {social_context}")
    if economic_context:
        world_lines.append(f"Economic context: {economic_context}")
    if characteristic_setting:
        world_lines.append(f"Characteristic setting: {characteristic_setting}")
    if architecture:
        world_lines.append(f"Architecture / material culture: {architecture}")
    if not world_lines:
        # Soft fallback: use location_dna from any cluster if world_assumptions empty
        for cl in clusters.values():
            loc = cl.get("location_name") or ""
            if loc:
                world_lines.append(f"Location DNA: {loc}")
                break
    world_context_block = "\n".join(world_lines) if world_lines else "Universal / unspecified"

    # Build the cluster list for the LLM
    cluster_lines = []
    for cid, cl in clusters.items():
        meanings = "; ".join(cl["shot_meanings"][:4]) or "unspecified"
        props_str = ", ".join(cl["scene_props"]) if cl.get("scene_props") else ""
        scene_name = cl.get("scene_name") or ""
        lines = [
            f'- cluster_id: "{cid}"',
            f'  Scene: {scene_name or cl["location_name"]}, {cl["setting"]}, {cl["time_of_day"]}',
            f'  Mood: {cl["location_mood"] or "neutral"}',
        ]
        if props_str:
            lines.append(f'  Props present: {props_str}')
        lines.append(f'  Shot meanings: {meanings}')
        lines.append(f'  ({cl["n_shots"]} shots)')
        cluster_lines.append("\n".join(lines))

    # Assemble character description lines
    char_lines = [f"Name: {name}"]
    if gender:
        char_lines.append(f"Gender: {gender}")
    if ethnicity:
        char_lines.append(f"Ethnicity / heritage: {ethnicity}")
    if appearance:
        char_lines.append(f"Appearance: {appearance}")
    if base_wardrobe:
        char_lines.append(f"Base wardrobe (reference plate look): {base_wardrobe}")
    if grooming:
        char_lines.append(f"Grooming base: {grooming}")

    user_content = f"""CULTURAL WORLD CONTEXT (anchor for ALL wardrobe decisions)
{world_context_block}

CHARACTER
{chr(10).join(char_lines)}

SCENE CLUSTERS — assign one culturally grounded, scene-appropriate outfit per cluster:
{chr(10).join(cluster_lines)}

For each cluster, write a SPECIFIC outfit description (one sentence, max 60 words).
Include: garment name (culturally native), colour, fabric detail, accessories, hair.
Derive formality from scene context — do not impose a generic default.
Output ONLY the JSON array."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=1200,
    )

    raw = (response.choices[0].message.content or "").strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("wardrobe_engine: invalid JSON from LLM: %s", raw[:300])
        return {}

    result: dict[str, str] = {}

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            cid = (item.get("cluster_id") or "").strip()
            wardrobe = (item.get("wardrobe") or "").strip()
            if cid and wardrobe:
                result[cid] = wardrobe

    elif isinstance(parsed, dict):
        if "cluster_id" in parsed and "wardrobe" in parsed:
            cid = (parsed.get("cluster_id") or "").strip()
            wardrobe = (parsed.get("wardrobe") or "").strip()
            if cid and wardrobe:
                result[cid] = wardrobe
        else:
            unwrapped = False
            for v in parsed.values():
                if isinstance(v, list):
                    for item in v:
                        if not isinstance(item, dict):
                            continue
                        cid = (item.get("cluster_id") or "").strip()
                        wardrobe = (item.get("wardrobe") or "").strip()
                        if cid and wardrobe:
                            result[cid] = wardrobe
                    unwrapped = True
                    break
            if not unwrapped:
                for k, v in parsed.items():
                    if isinstance(v, str) and k and v.strip():
                        result[k] = v.strip()

    if not result:
        logger.warning("wardrobe_engine: could not extract wardrobes from LLM output: %s", raw[:300])

    return result


# ── Look-plate helpers ────────────────────────────────────────────────────────

def _seed_look_rows(project_id: str, conn, character_id: int,
                    clusters: dict[str, dict],
                    cluster_wardrobes: dict[str, str]) -> None:
    """Insert one character_looks row per cluster that got a wardrobe assignment.

    Uses INSERT … ON CONFLICT DO UPDATE so re-running is idempotent: if the
    wardrobe text changed (e.g. after regeneration) the new text is written and
    ref_status is reset to 'pending' so the plate gets re-rendered.
    """
    with conn.cursor() as cur:
        for cluster_id, wardrobe_text in cluster_wardrobes.items():
            cl = clusters.get(cluster_id, {})
            label = cl.get("scene_name") or cl.get("location_name") or cluster_id
            cur.execute(
                """
                INSERT INTO character_looks
                    (project_id, character_id, cluster_id, cluster_label,
                     wardrobe_text, ref_status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                ON CONFLICT (project_id, character_id, cluster_id) DO UPDATE
                    SET cluster_label = EXCLUDED.cluster_label,
                        wardrobe_text = EXCLUDED.wardrobe_text,
                        ref_status    = CASE
                            WHEN character_looks.wardrobe_text IS DISTINCT FROM EXCLUDED.wardrobe_text
                            THEN 'pending'
                            ELSE character_looks.ref_status
                        END,
                        ref_error     = NULL
                """,
                (project_id, character_id, cluster_id, label, wardrobe_text),
            )
    conn.commit()


def generate_look_plates(project_id: str,
                         location_dna: str = "Universal",
                         style_suffix: Optional[str] = None) -> int:
    """Generate one styled character plate per pending character_looks row.

    Each look plate shows the character in the scene-specific outfit derived
    by the wardrobe engine, using the same image generation pipeline as the
    base identity plates.  Returns the number of plates successfully generated.

    Safe to call multiple times — rows already at 'ready' are skipped.
    """
    from pipeline_worker import _db, _try_acquire, _release
    from image_generator import (
        REF_MODEL, CHARACTER_REF_HINT,
        _run_fal, _extract_image_url, _save_to_r2, _new_ref_key,
        _openai_generate, _resolve_ref_mode, OPENAI_SIZE_PORTRAIT,
        _save_bytes_to_r2,
    )

    # Load pending look rows + the linked character rows.
    # Also fetch the character's base identity plate URL — look plates MUST be
    # generated with face-locking using that plate so all scene looks show the
    # same person, not random faces from a fresh text-to-image pass.
    with _db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT cl.id, cl.cluster_id, cl.cluster_label, cl.wardrobe_text,
                   cl.character_id, cl.ref_prompt AS stored_prompt,
                   c.name, c.gender, c.ethnicity, c.complexion, c.appearance,
                   c.age_range, c.cultural_notes,
                   c.ref_image_url  AS base_plate_url,
                   c.ref_status     AS base_plate_status
              FROM character_looks cl
              JOIN characters c ON c.id = cl.character_id
             WHERE cl.project_id = %s
               AND cl.ref_status IN ('pending', 'failed')
            """,
            (project_id,),
        )
        pending = list(cur.fetchall() or [])

    if not pending:
        logger.info("generate_look_plates: no pending rows for project=%s", project_id)
        return 0

    generated = 0
    for row in pending:
        look_id   = row["id"]
        cluster_id = row["cluster_id"]
        label      = row["cluster_label"] or cluster_id
        wardrobe_text = (row["wardrobe_text"] or "").strip()
        char_name  = row["name"] or "character"

        key = (project_id, f"look:{look_id}")
        if not _try_acquire(key):
            continue

        try:
            # Build prompt: use stored prompt if user edited it, otherwise
            # derive from character identity + scene-specific outfit.
            stored_prompt = (row.get("stored_prompt") or "").strip()
            if stored_prompt:
                prompt = stored_prompt
            else:
                age        = (row["age_range"] or "").strip()
                gender     = (row["gender"] or "").strip()
                ethnicity  = (row["ethnicity"] or "").strip()
                complexion = (row["complexion"] or "").strip()
                appearance = (row["appearance"] or "").strip()
                cultural   = (row["cultural_notes"] or "").strip()

                descriptors = [d for d in [age, gender, ethnicity, complexion]
                               if d and d.lower() not in {"unclear", "unknown", "any"}]
                descriptor_str = ", ".join(descriptors) if descriptors else "adult"

                region = f" Consistent with {location_dna}." if location_dna and location_dna.lower() != "universal" else ""
                outfit_clause = f"Wearing: {wardrobe_text}." if wardrobe_text else ""
                appearance_clause = f"Appearance: {appearance}." if appearance else ""
                cultural_clause = f"Cultural context: {cultural}." if cultural else ""

                prompt = (
                    f"Identity plate of {char_name} — a single {descriptor_str} character "
                    f"in '{label}' scene.{region} "
                    f"{outfit_clause} {appearance_clause} {cultural_clause} "
                    f"{CHARACTER_REF_HINT}."
                ).strip()

                if style_suffix:
                    prompt = f"{prompt} {style_suffix}".strip()

            # Mark as rendering and persist the prompt (so UI can display it)
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE character_looks SET ref_status='rendering', ref_error=NULL, "
                    "ref_prompt=%s WHERE id=%s",
                    (prompt, look_id),
                )
                conn.commit()

            safe_name    = "".join(ch if ch.isalnum() else "_" for ch in char_name)[:20]
            safe_cluster = "".join(ch if ch.isalnum() else "_" for ch in cluster_id)[:30]

            # Face continuity: if the character's base identity plate is ready,
            # use it as a face reference so all look plates show the SAME person.
            # Without this each text-to-image call produces a different face.
            base_url    = (row.get("base_plate_url") or "").strip() or None
            has_base    = bool(base_url and row.get("base_plate_status") == "ready")

            from image_generator import _fal_accessible_url
            PULID_MODEL = "fal-ai/flux-pulid"

            if _resolve_ref_mode() == "cheap":
                if has_base:
                    # img2img from the base plate keeps the face; wardrobe text
                    # in the prompt steers the outfit change.
                    from image_generator import _openai_edit
                    img_bytes = _openai_edit(prompt[:4000], base_url,
                                             size=OPENAI_SIZE_PORTRAIT)
                else:
                    img_bytes = _openai_generate(prompt[:4000], size=OPENAI_SIZE_PORTRAIT)
                r2_key = _new_ref_key(
                    project_id, f"look_{look_id}_{safe_name}_{safe_cluster}", ext="png"
                )
                url = _save_bytes_to_r2(img_bytes, r2_key)
            else:
                if has_base:
                    # PuLID: injects identity embeddings from the base plate into
                    # the diffusion process — proper face transfer, not pixel copy.
                    fal_base_url = _fal_accessible_url(base_url)
                    result = _run_fal(PULID_MODEL, {
                        "prompt": prompt[:1800],
                        "reference_image_url": fal_base_url,
                        "image_size": "portrait_4_3",
                        "num_inference_steps": 20,
                        "guidance_scale": 4.0,
                        "id_weight": 1.0,
                        "true_cfg": 1.0,
                        "num_images": 1,
                        "enable_safety_checker": False,
                    })
                    logger.info("generate_look_plates: used PuLID face-lock for look=%s", look_id)
                else:
                    # No base plate yet — fall back to plain text-to-image.
                    # This look plate will be visually inconsistent but won't block
                    # the pipeline; the director can regenerate after uploading a photo.
                    logger.warning(
                        "generate_look_plates: no ready base plate for character_id=%s "
                        "— look=%s generated without face-lock (text-to-image fallback)",
                        row.get("character_id"), look_id,
                    )
                    result = _run_fal(REF_MODEL, {
                        "prompt": prompt[:1800],
                        "image_size": "portrait_4_3",
                        "num_inference_steps": 8,
                        "num_images": 1,
                        "seed": random.randint(1, 2**32 - 1),
                        "enable_safety_checker": False,
                    })
                fal_url = _extract_image_url(result)
                r2_key  = _new_ref_key(
                    project_id, f"look_{look_id}_{safe_name}_{safe_cluster}"
                )
                url = _save_to_r2(fal_url, r2_key)

            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE character_looks "
                    "   SET ref_status='ready', ref_image_url=%s, ref_prompt=%s, ref_error=NULL "
                    " WHERE id=%s",
                    (url, prompt, look_id),
                )
                conn.commit()

            logger.info("generate_look_plates: rendered look=%s cluster=%s project=%s",
                        look_id, cluster_id, project_id)
            generated += 1

        except Exception as exc:
            logger.exception("generate_look_plates: look=%s failed", look_id)
            with _db() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE character_looks SET ref_status='failed', ref_error=%s WHERE id=%s",
                    (str(exc)[:400], look_id),
                )
                conn.commit()
        finally:
            _release(key)

    logger.info("generate_look_plates: generated %d look plates for project=%s",
                generated, project_id)
    return generated


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_project_data(project_id: str, conn) -> tuple[
    list[dict], list[dict], list[dict], dict, dict, dict[int, Optional[int]]
]:
    """Returns (styled_timeline, characters, locations, shot_location_ids,
    world_assumptions, shot_character_ids).

    shot_character_ids maps shot_index → character.id (integer FK), populated
    by _link_shots_to_entities in the Materializer stage before refs run.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT styled_timeline, context_packet FROM projects WHERE id=%s", (project_id,))
        row = cur.fetchone() or {}
        styled_timeline: list[dict] = row.get("styled_timeline") or []

        # Extract world_assumptions from context_packet (project-level cultural anchor)
        context_packet = dict(row.get("context_packet") or {})
        raw_world = context_packet.get("world_assumptions")
        world_assumptions: dict = raw_world if isinstance(raw_world, dict) else {}

        # If context_packet doesn't have world_assumptions, try the first shot's env_profile
        if not world_assumptions and styled_timeline:
            first_env = (styled_timeline[0] or {}).get("environment_profile") or {}
            world_assumptions = first_env.get("world_assumptions") or {}

        cur.execute(
            "SELECT id, name, role, gender, ethnicity, complexion, wardrobe, grooming, appearance "
            "  FROM characters WHERE project_id=%s ORDER BY id",
            (project_id,),
        )
        characters = list(cur.fetchall() or [])

        cur.execute(
            "SELECT id, name, description, time_of_day, mood FROM locations "
            " WHERE project_id=%s",
            (project_id,),
        )
        locations = list(cur.fetchall() or [])

        cur.execute(
            "SELECT shot_index, location_id, character_id FROM shot_assets "
            " WHERE project_id=%s",
            (project_id,),
        )
        sa_rows = cur.fetchall() or []
        shot_location_ids: dict[int, Optional[int]] = {
            r["shot_index"]: r.get("location_id") for r in sa_rows
        }
        shot_character_ids: dict[int, Optional[int]] = {
            r["shot_index"]: r.get("character_id")
            for r in sa_rows
            if r.get("character_id") is not None
        }
    return styled_timeline, characters, locations, shot_location_ids, world_assumptions, shot_character_ids


def _pick_main_character(characters: list[dict]) -> Optional[dict]:
    """Return the most-detailed character record (most filled fields)."""
    if not characters:
        return None
    _FIELDS = ("gender", "ethnicity", "complexion", "wardrobe", "grooming", "appearance")
    def _score(c):
        return sum(1 for f in _FIELDS if (c.get(f) or "").strip())
    return max(characters, key=_score)


def _save_wardrobe_contexts(project_id: str, conn,
                             shot_wardrobe: dict[int, str],
                             shot_clusters: dict[int, str] | None = None) -> None:
    """Write wardrobe_context (and optionally look_cluster_id) to shot_assets."""
    if not shot_wardrobe:
        return
    shot_clusters = shot_clusters or {}
    with conn.cursor() as cur:
        for shot_index, wardrobe_text in shot_wardrobe.items():
            cluster_id = shot_clusters.get(shot_index)
            cur.execute(
                """UPDATE shot_assets
                      SET wardrobe_context  = %s,
                          look_cluster_id   = COALESCE(%s, look_cluster_id),
                          updated_at        = NOW()
                    WHERE project_id = %s AND shot_index = %s
                      AND (prompt_user_edited IS NOT TRUE)""",
                (wardrobe_text, cluster_id, project_id, shot_index),
            )
    conn.commit()


# ── Public API ────────────────────────────────────────────────────────────────

def diversify_wardrobe(project_id: str) -> int:
    """Assign scene-appropriate per-shot wardrobe contexts for a project.

    Wardrobe is derived from the song's cultural world (world_assumptions) and
    the creative brief's scene-level context (scene_frame), not from hardcoded
    defaults. Idempotent — safe to re-run.

    Returns the number of shots updated.
    """
    from pipeline_worker import _db
    with _db() as conn:
        (styled_timeline, characters, locations,
         shot_location_ids, world_assumptions,
         shot_character_ids) = _load_project_data(project_id, conn)

    if not styled_timeline:
        logger.warning("wardrobe_engine: no styled_timeline for project=%s", project_id)
        return 0

    # Only operate on shots with a character linked (face/body shots)
    character_shots = [
        s for s in styled_timeline
        if (s.get("expression_mode") or "environment").lower() in {"face", "body"}
    ]
    if not character_shots:
        logger.info("wardrobe_engine: no character shots for project=%s", project_id)
        return 0

    if not characters:
        logger.warning("wardrobe_engine: no character found for project=%s", project_id)
        return 0

    if world_assumptions:
        logger.info("wardrobe_engine: cultural anchor — geography=%s era=%s social=%s",
                    world_assumptions.get("geography"),
                    world_assumptions.get("era"),
                    world_assumptions.get("social_context"))
    else:
        logger.warning("wardrobe_engine: no world_assumptions found for project=%s "
                       "— LLM will derive from character/location data only", project_id)

    # ── Per-character look planning ───────────────────────────────────────────
    # Build the set of characters that actually have face/body shots assigned
    # (via shot_assets.character_id, set by _link_shots_to_entities).  When no
    # per-shot character assignments exist (old projects or pre-materializer
    # path), fall back to processing all character shots under the main char.
    char_by_id: dict[int, dict] = {c["id"]: c for c in characters}
    # Set of shot indices that are face/body shots (efficient O(n) lookup below)
    character_shot_index_set: set[int] = {
        (s.get("shot_index") or s.get("timeline_index"))
        for s in character_shots
        if (s.get("shot_index") or s.get("timeline_index")) is not None
    }
    chars_with_shots: set[int] = {
        cid for idx, cid in shot_character_ids.items()
        if idx in character_shot_index_set
    }
    if not chars_with_shots:
        # Fallback: assign all character shots to the most-detailed character
        main_char = _pick_main_character(characters)
        if main_char:
            chars_with_shots = {main_char["id"]}
            # Treat every face/body shot as belonging to main_char
            for s in character_shots:
                idx = s.get("shot_index") or s.get("timeline_index")
                if idx is not None:
                    shot_character_ids[idx] = main_char["id"]
        else:
            return 0

    # ── Cap constant ─────────────────────────────────────────────────────────
    # Real filmmaking: 2-3 distinct wardrobe changes per video.  More than 3
    # creates continuity confusion.  Fewer is fine when narrative doesn't
    # justify variety.
    MAX_LOOKS: int = 3

    total_shot_updates: int = 0
    shot_wardrobe: dict[int, str] = {}
    shot_clusters: dict[int, str] = {}

    for char_id in sorted(chars_with_shots):
        character = char_by_id.get(char_id)
        if not character:
            continue

        # Shots that belong to this character
        char_shot_indices = {
            idx for idx, cid in shot_character_ids.items() if cid == char_id
        }
        char_shots = [
            s for s in character_shots
            if (s.get("shot_index") or s.get("timeline_index")) in char_shot_indices
        ]
        if not char_shots:
            continue

        # Build scene clusters for this character's shots
        clusters = _build_cluster_descriptions(char_shots, shot_location_ids, locations)
        logger.info("wardrobe_engine: char=%s(%d) — %d scene clusters for project=%s: %s",
                    character.get("name"), char_id, len(clusters),
                    project_id, list(clusters.keys()))

        # Cap to MAX_LOOKS — keep top N by shot count, remap dropped clusters
        cluster_remap: dict[str, str] = {}
        if len(clusters) > MAX_LOOKS:
            sorted_keys = sorted(clusters, key=lambda k: clusters[k]["n_shots"], reverse=True)
            kept_keys    = sorted_keys[:MAX_LOOKS]
            dropped_keys = sorted_keys[MAX_LOOKS:]
            fallback_key = kept_keys[0]
            cluster_remap = {k: fallback_key for k in dropped_keys}
            clusters = {k: clusters[k] for k in kept_keys}
            logger.info("wardrobe_engine: char=%s — capped %d clusters → %d looks "
                        "(dropped: %s)", character.get("name"),
                        len(sorted_keys), MAX_LOOKS, list(cluster_remap.keys()))

        cluster_wardrobes = _call_llm(character, clusters, world_assumptions=world_assumptions)
        logger.info("wardrobe_engine: char=%s — LLM returned %d cluster wardrobes",
                    character.get("name"), len(cluster_wardrobes))

        if not cluster_wardrobes:
            logger.warning("wardrobe_engine: LLM returned empty result for char=%s project=%s",
                           character.get("name"), project_id)
            continue

        # Map each of this character's shots to a cluster wardrobe
        for shot in char_shots:
            idx = shot.get("shot_index") or shot.get("timeline_index")
            if idx is None:
                continue
            ckey = _cluster_key(shot, shot_location_ids)
            # Apply remap for dropped clusters
            ckey = cluster_remap.get(ckey, ckey)
            wardrobe = cluster_wardrobes.get(ckey)
            if wardrobe:
                shot_wardrobe[idx] = wardrobe
                shot_clusters[idx] = ckey
                total_shot_updates += 1
            else:
                logger.debug("wardrobe_engine: no wardrobe for char=%s cluster=%s shot=%s",
                             character.get("name"), ckey, idx)

        with _db() as conn:
            # Seed character_looks rows (one per kept cluster) for this character
            _seed_look_rows(project_id, conn, char_id, clusters, cluster_wardrobes)

    # Write wardrobe_context + look_cluster_id to shot_assets in one pass
    if shot_wardrobe:
        with _db() as conn:
            _save_wardrobe_contexts(project_id, conn, shot_wardrobe, shot_clusters)

    logger.info("wardrobe_engine: updated %d shots + seeded looks for %d character(s) "
                "in project=%s", total_shot_updates, len(chars_with_shots), project_id)
    return total_shot_updates
