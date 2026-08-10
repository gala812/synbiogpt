from __future__ import annotations

import mimetypes
from pathlib import Path
import re
import sqlite3

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse


ASSET_KEY_RE = re.compile(r"[0-9a-f]{64}")


def create_app(manifest: Path) -> FastAPI:
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Image asset manifest not found: {manifest}")
    connection = sqlite3.connect(f"file:{manifest.as_posix()}?mode=ro", uri=True)
    metadata = dict(connection.execute("SELECT key, value FROM manifest_metadata"))
    connection.close()
    source_root = Path(metadata["source_root"]).resolve()
    app = FastAPI(title="SynBioGPT Paper Assets", version="1")

    def lookup(asset_key: str) -> sqlite3.Row:
        if not ASSET_KEY_RE.fullmatch(asset_key):
            raise HTTPException(status_code=404, detail="Asset not found")
        database = sqlite3.connect(f"file:{manifest.as_posix()}?mode=ro", uri=True)
        database.row_factory = sqlite3.Row
        try:
            row = database.execute(
                "SELECT * FROM image_assets WHERE asset_key = ?", (asset_key,)
            ).fetchone()
        finally:
            database.close()
        if row is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return row

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/assets/{asset_key}")
    @app.head("/assets/{asset_key}", include_in_schema=False)
    def get_asset(asset_key: str):
        row = lookup(asset_key)
        path = Path(row["source_path"]).resolve()
        if source_root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Asset file unavailable")
        return FileResponse(
            path,
            media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable", "ETag": asset_key},
        )

    @app.get("/assets/{asset_key}/metadata")
    def get_metadata(asset_key: str) -> dict:
        row = lookup(asset_key)
        return {key: row[key] for key in row.keys() if key not in {"source_path", "source_markdown"}}

    @app.get("/papers/{pmcid}/assets")
    def paper_assets(pmcid: str, status: str | None = Query(default=None)) -> list[dict]:
        database = sqlite3.connect(f"file:{manifest.as_posix()}?mode=ro", uri=True)
        database.row_factory = sqlite3.Row
        query = "SELECT * FROM image_assets WHERE pmcid = ?"
        params: list[str] = [pmcid.upper()]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY relative_path"
        try:
            rows = database.execute(query, params).fetchall()
        finally:
            database.close()
        return [
            {key: row[key] for key in row.keys() if key not in {"source_path", "source_markdown"}}
            for row in rows
        ]

    return app
