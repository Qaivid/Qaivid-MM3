#!/bin/bash
set -e

if [ -f requirements.txt ]; then
    pip install --quiet --disable-pip-version-check -r requirements.txt
fi

mkdir -p projects static

python3 - <<'PYEOF'
import os, psycopg
conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS video_assets (
    id          SERIAL PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    shot_index  INTEGER NOT NULL,
    file_path   TEXT,
    status      TEXT NOT NULL DEFAULT 'queued',
    error       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (project_id, shot_index)
);
CREATE INDEX IF NOT EXISTS ix_video_assets_project ON video_assets(project_id);
""")
conn.commit()
conn.close()
print("schema OK")
PYEOF
