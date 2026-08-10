from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable

from .recovery import recover_document


SCHEMA_VERSION = "medcpt_image_assets_v1"


def _resolve_path(value: str, prefix_maps: tuple[tuple[str, str], ...]) -> Path:
    path = Path(value)
    if path.exists():
        return path
    for source, target in prefix_maps:
        if value.startswith(source):
            candidate = Path(target + value[len(source) :])
            if candidate.exists():
                return candidate
    return path


def _worker(args: tuple[dict[str, Any], tuple[tuple[str, str], ...]]) -> dict[str, Any]:
    document, prefix_maps = args
    path = _resolve_path(document["source_markdown"], prefix_maps)
    return recover_document(path, document["pmcid"])


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _open_manifest(path: Path, source_root: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS image_assets (
            asset_key TEXT PRIMARY KEY,
            pmcid TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            source_path TEXT NOT NULL,
            status TEXT NOT NULL,
            asset_id TEXT,
            asset_type TEXT,
            label TEXT NOT NULL,
            caption TEXT NOT NULL,
            confidence TEXT NOT NULL,
            reason TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            size_bytes INTEGER,
            source_markdown TEXT NOT NULL,
            UNIQUE (pmcid, relative_path)
        );
        CREATE INDEX IF NOT EXISTS image_assets_pmcid_idx ON image_assets(pmcid);
        CREATE INDEX IF NOT EXISTS image_assets_asset_id_idx ON image_assets(asset_id);
        CREATE INDEX IF NOT EXISTS image_assets_status_idx ON image_assets(status);
        CREATE TABLE IF NOT EXISTS manifest_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT OR REPLACE INTO manifest_metadata(key, value) VALUES (?, ?)",
        (("schema_version", SCHEMA_VERSION), ("source_root", str(source_root.resolve()))),
    )
    return connection


def _insert_bindings(connection: sqlite3.Connection, bindings: list[dict[str, Any]]) -> None:
    columns = (
        "asset_key", "pmcid", "relative_path", "source_path", "status", "asset_id",
        "asset_type", "label", "caption", "confidence", "reason", "width", "height",
        "size_bytes", "source_markdown",
    )
    connection.executemany(
        f"INSERT OR REPLACE INTO image_assets ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        ([binding.get(column) for column in columns] for binding in bindings),
    )


def build_image_assets(
    *,
    documents_jsonl: Path,
    output_dir: Path,
    source_root: Path,
    limit: int = 0,
    workers: int = 8,
    documents_per_shard: int = 500,
    prefix_maps: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bindings_dir = output_dir / "bindings"
    recovered_dir = output_dir / "recovered_assets"
    bindings_dir.mkdir(exist_ok=True)
    recovered_dir.mkdir(exist_ok=True)
    documents: list[dict[str, str]] = []
    with documents_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            documents.append(
                {
                    "pmcid": record["pmcid"],
                    "source_markdown": record["source_markdown"],
                }
            )
    documents.sort(key=lambda item: item["pmcid"])
    if limit > 0:
        documents = documents[:limit]

    manifest_path = output_dir / "image_assets.sqlite3"
    if manifest_path.exists():
        manifest_path.unlink()
    connection = _open_manifest(manifest_path, source_root)
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    arguments = ((document, prefix_maps) for document in documents)
    executor = ProcessPoolExecutor(max_workers=max(1, workers)) if workers > 1 else None
    results = executor.map(_worker, arguments, chunksize=8) if executor else map(_worker, arguments)
    shard_bindings: list[dict[str, Any]] = []
    shard_assets: list[dict[str, Any]] = []
    shard_index = 0
    try:
        for index, document in enumerate(documents):
            try:
                result = next(results)
            except Exception as exc:
                errors.append({"pmcid": document["pmcid"], "error": str(exc)})
                counts["failed_documents"] += 1
                continue
            bindings = result["bindings"]
            assets = result["recovered_assets"]
            _insert_bindings(connection, bindings)
            shard_bindings.extend(bindings)
            shard_assets.extend(assets)
            counts["successful_documents"] += 1
            counts["images"] += len(bindings)
            for binding in bindings:
                counts[f"status_{binding['status']}"] += 1
                counts[f"reason_{binding['reason']}"] += 1
            if (index + 1) % documents_per_shard == 0 or index + 1 == len(documents):
                name = f"part-{shard_index:05d}.jsonl"
                _write_jsonl(bindings_dir / name, shard_bindings)
                _write_jsonl(recovered_dir / name, shard_assets)
                connection.commit()
                shard_bindings = []
                shard_assets = []
                shard_index += 1
    finally:
        if executor:
            executor.shutdown()
        connection.commit()
        connection.close()

    _write_jsonl(output_dir / "errors.jsonl", errors)
    statistics = {
        "schema_version": SCHEMA_VERSION,
        "selected_documents": len(documents),
        "shard_count": shard_index,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        **dict(sorted(counts.items())),
    }
    temporary = output_dir / "statistics.json.partial"
    temporary.write_text(json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "statistics.json")
    return statistics
