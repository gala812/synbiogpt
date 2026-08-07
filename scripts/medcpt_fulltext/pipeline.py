from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import statistics
import tempfile
import time
import traceback
from typing import Any, Iterable

from .chunking import create_chunks
from .markdown_parser import IMAGE_RE, HEADING_RE, parse_document
from .models import ChunkingConfig, DocumentCandidate
from .tokenization import MEDCPT_TOKENIZER, TokenCounter, resolve_tokenizer


SCHEMA_VERSION = "medcpt_markdown_chunk_v1"
PIPELINE_REVISION = "production_rules_v1"
INVENTORY_SCHEMA_VERSION = 1
PMCID_RE = re.compile(r"^PMC\d+$", re.I)
JSON_ID_RE = re.compile(rb'"(?:id|doc_id|pmcid)"\s*:\s*"(PMC\d+)"', re.I)

_WORKER_TOKENIZER: TokenCounter | None = None
_WORKER_CONFIG: ChunkingConfig | None = None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            for row in rows:
                handle.write(_json_bytes(row))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class _LiveJsonlFile:
    """Write visible JSONL rows while tracking manifest metadata."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("wb")
        self.content_sha256 = hashlib.sha256()
        self.row_count = 0

    def write_many(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            data = _json_bytes(row) + b"\n"
            self.handle.write(data)
            self.content_sha256.update(data)
            self.row_count += 1
        self.handle.flush()

    def close(self) -> dict[str, Any]:
        if not self.handle.closed:
            self.handle.close()
        return {
            "path": self.path.as_posix(),
            "row_count": self.row_count,
            "content_sha256": self.content_sha256.hexdigest(),
            "size_bytes": self.path.stat().st_size,
        }


class _StreamingOutputWriter:
    """Write each completed paper directly into its production shard."""

    _DIRECTORIES = {
        "chunks": "chunks",
        "parents": "parents",
        "assets": "figures_tables",
    }

    def __init__(
        self,
        output_dir: Path,
        candidates: list[DocumentCandidate],
        documents_per_shard: int,
        inspection_seed: int,
    ) -> None:
        self.output_dir = output_dir
        self.candidates = candidates
        self.documents_per_shard = documents_per_shard
        self.locations = {
            candidate.pmcid: index for index, candidate in enumerate(candidates)
        }
        rng = random.Random(inspection_seed)
        sample_count = min(20, len(candidates))
        self.sampled_ids = set(rng.sample([item.pmcid for item in candidates], sample_count))
        self.current_shard = -1
        self.shard_writers: dict[str, _LiveJsonlFile] = {}
        self.shard_records: list[dict[str, Any]] = []
        self.shard_successes = 0
        self.shard_failures = 0
        self._reset_outputs()
        self.documents = _LiveJsonlFile(output_dir / "documents.jsonl")
        self.errors = _LiveJsonlFile(output_dir / "errors.jsonl")
        self.inspection = _LiveJsonlFile(output_dir / "inspection_samples.jsonl")

    def _reset_outputs(self) -> None:
        legacy_spool = self.output_dir / ".spool"
        if legacy_spool.is_dir():
            shutil.rmtree(legacy_spool)
        for directory in self._DIRECTORIES.values():
            shard_dir = self.output_dir / directory
            shard_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ("part-*.jsonl", "part-*.jsonl.gz"):
                for path in shard_dir.glob(pattern):
                    path.unlink()
        for filename in (
            "chunks.jsonl",
            "parents.jsonl",
            "figures_tables.jsonl",
            "documents.jsonl",
            "errors.jsonl",
            "statistics.json",
            "manifest.json",
            "inspection_samples.jsonl",
        ):
            path = self.output_dir / filename
            if path.is_file():
                path.unlink()

    def _open_shard(self, shard_index: int) -> None:
        if self.shard_writers:
            self._close_shard()
        part_name = f"part-{shard_index:05d}.jsonl"
        self.shard_writers = {
            name: _LiveJsonlFile(self.output_dir / directory / part_name)
            for name, directory in self._DIRECTORIES.items()
        }
        self.current_shard = shard_index
        self.shard_successes = 0
        self.shard_failures = 0

    def _close_shard(self) -> None:
        if not self.shard_writers:
            return
        start = self.current_shard * self.documents_per_shard
        selected = self.candidates[start : start + self.documents_per_shard]
        files: dict[str, dict[str, Any]] = {}
        keys = {"chunks": "chunks", "parents": "parents", "assets": "figures_tables"}
        for name, writer in self.shard_writers.items():
            details = writer.close()
            details["path"] = writer.path.relative_to(self.output_dir).as_posix()
            files[keys[name]] = details
        self.shard_records.append(
            {
                "shard_index": self.current_shard,
                "part_name": f"part-{self.current_shard:05d}.jsonl",
                "pmcid_start": selected[0].pmcid,
                "pmcid_end": selected[-1].pmcid,
                "selected_document_count": len(selected),
                "successful_document_count": self.shard_successes,
                "failed_document_count": self.shard_failures,
                "files": files,
            }
        )
        self.shard_writers = {}

    def record(self, outcome: dict[str, Any]) -> None:
        candidate_index = self.locations[outcome["pmcid"]]
        shard_index = candidate_index // self.documents_per_shard
        if shard_index != self.current_shard:
            self._open_shard(shard_index)
        if outcome["status"] != "success":
            self.errors.write_many([outcome])
            self.shard_failures += 1
            return

        bundle = outcome.pop("bundle")
        document = bundle["document"]
        chunks = sorted(bundle["chunks"], key=lambda row: row["chunk_index"])
        parents = sorted(bundle["parents"], key=lambda row: row["parent_chunk_id"])
        assets = sorted(
            bundle["assets"], key=lambda row: (row["char_start"], row.get("label") or "")
        )
        self.documents.write_many([document])
        self.shard_writers["chunks"].write_many(chunks)
        self.shard_writers["parents"].write_many(parents)
        self.shard_writers["assets"].write_many(assets)
        self.shard_successes += 1
        if outcome["pmcid"] in self.sampled_ids:
            self.inspection.write_many([_inspection_record(document, chunks)])

    def close(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        self._close_shard()
        global_files = {
            "documents": self.documents.close(),
            "errors": self.errors.close(),
            "inspection_samples": self.inspection.close(),
        }
        for details in global_files.values():
            details["path"] = Path(details["path"]).relative_to(self.output_dir).as_posix()
        return self.shard_records, global_files


def _has_any_image(markdown_path: Path) -> bool:
    image_dir = markdown_path.parent / "images"
    try:
        return any(item.is_file() for item in image_dir.iterdir())
    except OSError:
        return False


def _candidate_score(path: Path) -> tuple[int, int, int, int]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (0, 0, 0, 0)
    refs = [match.group("path").strip().strip('"\'') for match in IMAGE_RE.finditer(raw)]
    existing = sum((path.parent / ref).is_file() for ref in refs)
    headings = sum(bool(HEADING_RE.match(line.strip())) for line in raw.splitlines())
    return existing, len(refs), headings, len(raw)


def _select_best_candidate(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    scored = [(path, _candidate_score(path)) for path in paths]
    return sorted(
        scored,
        key=lambda item: tuple(-value for value in item[1]) + (item[0].as_posix(),),
    )[0][0]


def _build_inventory(input_dir: Path, inventory_db: Path) -> float:
    """Build a deterministic Markdown inventory once, then atomically publish it."""

    started = time.perf_counter()
    grouped: dict[str, list[tuple[Path, bool]]] = {}
    for path in input_dir.rglob("*.md"):
        pmcid = path.stem.upper()
        if PMCID_RE.fullmatch(pmcid):
            grouped.setdefault(pmcid, []).append((path, _has_any_image(path)))

    inventory_db.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{inventory_db.name}.", suffix=".partial", dir=inventory_db.parent
    )
    os.close(fd)
    try:
        connection = sqlite3.connect(temp_name)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE documents (
                    pmcid TEXT PRIMARY KEY,
                    best_path_any TEXT NOT NULL,
                    duplicate_paths_any TEXT NOT NULL,
                    best_path_with_images TEXT,
                    duplicate_paths_with_images TEXT,
                    candidate_count INTEGER NOT NULL,
                    image_candidate_count INTEGER NOT NULL,
                    source_size INTEGER NOT NULL,
                    source_mtime_ns INTEGER NOT NULL
                );
                CREATE INDEX documents_has_images
                    ON documents(best_path_with_images, pmcid);
                """
            )
            metadata_rows = {
                "schema_version": str(INVENTORY_SCHEMA_VERSION),
                "input_dir": str(input_dir.resolve()),
                "document_count": str(len(grouped)),
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata_rows.items()
            )
            rows: list[tuple[Any, ...]] = []
            for pmcid in sorted(grouped):
                all_paths = sorted((item[0] for item in grouped[pmcid]), key=lambda p: p.as_posix())
                image_paths = sorted(
                    (path for path, has_images in grouped[pmcid] if has_images),
                    key=lambda p: p.as_posix(),
                )
                best_any = _select_best_candidate(all_paths)
                best_images = _select_best_candidate(image_paths) if image_paths else None
                selected = best_images or best_any
                stat = selected.stat()
                rows.append(
                    (
                        pmcid,
                        str(best_any),
                        json.dumps([str(path) for path in all_paths if path != best_any]),
                        str(best_images) if best_images else None,
                        json.dumps(
                            [str(path) for path in image_paths if path != best_images]
                            if best_images
                            else []
                        ),
                        len(all_paths),
                        len(image_paths),
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
                if len(rows) >= 1000:
                    connection.executemany(
                        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
                    )
                    rows.clear()
            if rows:
                connection.executemany(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
                )
            connection.commit()
        finally:
            connection.close()
        os.replace(temp_name, inventory_db)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return time.perf_counter() - started


def _read_inventory(
    input_dir: Path,
    inventory_db: Path,
    *,
    limit: int,
    require_images: bool,
) -> list[DocumentCandidate]:
    connection = sqlite3.connect(f"file:{inventory_db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema_version") != str(INVENTORY_SCHEMA_VERSION):
            raise RuntimeError("Inventory schema changed; rerun with --refresh-inventory")
        if metadata.get("input_dir") != str(input_dir.resolve()):
            raise RuntimeError(
                "Inventory belongs to another input directory; use a different database "
                "or rerun with --refresh-inventory"
            )
        path_column = "best_path_with_images" if require_images else "best_path_any"
        duplicates_column = (
            "duplicate_paths_with_images" if require_images else "duplicate_paths_any"
        )
        where = "WHERE best_path_with_images IS NOT NULL" if require_images else ""
        limit_clause = "LIMIT ?" if limit > 0 else ""
        parameters: tuple[int, ...] = (limit,) if limit > 0 else ()
        rows = connection.execute(
            f"SELECT pmcid, {path_column}, {duplicates_column} "
            f"FROM documents {where} ORDER BY pmcid {limit_clause}",
            parameters,
        )
        return [
            DocumentCandidate(
                pmcid=pmcid,
                markdown_path=Path(markdown_path),
                duplicate_paths=[Path(path) for path in json.loads(duplicate_paths)],
            )
            for pmcid, markdown_path, duplicate_paths in rows
        ]
    finally:
        connection.close()


def _discover_documents_with_metrics(
    input_dir: Path,
    *,
    limit: int,
    require_images: bool,
    inventory_db: Path | None,
    refresh_inventory: bool,
) -> tuple[list[DocumentCandidate], bool, float]:
    inventory_reused = False
    inventory_build_seconds = 0.0
    if inventory_db is not None:
        if refresh_inventory or not inventory_db.is_file():
            inventory_build_seconds = _build_inventory(input_dir, inventory_db)
        else:
            inventory_reused = True
        candidates = _read_inventory(
            input_dir, inventory_db, limit=limit, require_images=require_images
        )
        return candidates, inventory_reused, inventory_build_seconds
    return (
        discover_documents(input_dir, limit=limit, require_images=require_images),
        False,
        0.0,
    )


def discover_documents(
    input_dir: Path,
    *,
    limit: int,
    require_images: bool = True,
) -> list[DocumentCandidate]:
    grouped: dict[str, list[Path]] = {}
    for path in input_dir.rglob("*.md"):
        pmcid = path.stem.upper()
        if not PMCID_RE.fullmatch(pmcid):
            continue
        if require_images and not _has_any_image(path):
            continue
        grouped.setdefault(pmcid, []).append(path)

    selected: list[DocumentCandidate] = []
    for pmcid in sorted(grouped):
        paths = sorted(grouped[pmcid], key=lambda item: item.as_posix())
        best = _select_best_candidate(paths)
        selected.append(
            DocumentCandidate(
                pmcid=pmcid,
                markdown_path=best,
                duplicate_paths=[path for path in paths if path != best],
            )
        )
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def _metadata_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"))
    return [path]


def load_metadata(path: Path | None, target_pmcids: set[str]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    remaining = set(target_pmcids)
    found: dict[str, dict[str, Any]] = {}
    for jsonl_path in _metadata_files(path):
        if not remaining:
            break
        with jsonl_path.open("rb") as handle:
            for raw_line in handle:
                id_match = JSON_ID_RE.search(raw_line[:4096])
                if not id_match:
                    continue
                pmcid = id_match.group(1).decode("ascii").upper()
                if pmcid not in remaining:
                    continue
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                nested = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                found[pmcid] = {
                    "doc_id": row.get("doc_id") or row.get("id") or nested.get("doc_id") or pmcid,
                    "title": row.get("title") or nested.get("title") or "",
                    "source_file": row.get("source_file") or nested.get("source_file") or "",
                    "metadata": nested,
                    "metadata_source": str(jsonl_path),
                }
                remaining.remove(pmcid)
                if not remaining:
                    break
    return found


def _init_worker(tokenizer_name: str, config_values: dict[str, int]) -> None:
    global _WORKER_TOKENIZER, _WORKER_CONFIG
    _WORKER_TOKENIZER = resolve_tokenizer(
        tokenizer_name,
        allow_fallback=False,
        local_files_only=not tokenizer_name.startswith(("generic:", "tiktoken:")),
    )
    _WORKER_CONFIG = ChunkingConfig(**config_values)


def _set_worker_state(tokenizer: TokenCounter, config: ChunkingConfig) -> None:
    global _WORKER_TOKENIZER, _WORKER_CONFIG
    _WORKER_TOKENIZER = tokenizer
    _WORKER_CONFIG = config


def _document_record(
    document: Any,
    candidate: DocumentCandidate,
    tokenizer_name: str,
    chunks: list[dict[str, Any]],
    parents: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_image_paths = [
        block.image_path
        for block in document.blocks
        if block.kind == "image" and block.image_path
    ]
    bound_image_paths = {
        path for asset in document.assets for path in asset.image_paths
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_revision": PIPELINE_REVISION,
        "doc_id": document.pmcid,
        "pmcid": document.pmcid,
        "paper_title": document.paper_title,
        "title_source": document.title_source,
        "title_anomaly": any(
            warning in {"metadata_title_anomaly", "missing_reliable_title"}
            for warning in document.parse_warnings
        ),
        "source_markdown": document.source_markdown,
        "source_file": document.source_file,
        "selected_from_duplicate_count": len(candidate.duplicate_paths) + 1,
        "duplicate_markdown_paths": [str(path) for path in candidate.duplicate_paths],
        "tokenizer_name": tokenizer_name,
        "section_tree": [
            {"section": section, "subsections": subsections}
            for section, subsections in document.section_tree.items()
        ],
        "unknown_headings": document.unknown_headings,
        "parse_warnings": document.parse_warnings,
        "excluded_counts": document.excluded_counts,
        "chunk_count": len(chunks),
        "parent_count": len(parents),
        "figure_count": sum(asset.asset_type == "figure" for asset in document.assets),
        "table_count": sum(asset.asset_type == "table" for asset in document.assets),
        "raw_image_reference_count": len(raw_image_paths),
        "bound_image_reference_count": sum(path in bound_image_paths for path in raw_image_paths),
        "unbound_image_reference_count": sum(path not in bound_image_paths for path in raw_image_paths),
        "metadata": document.metadata,
    }


def _asset_record(asset: Any, document: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_revision": PIPELINE_REVISION,
        "doc_id": document.pmcid,
        "pmcid": document.pmcid,
        "paper_title": document.paper_title,
        "asset_type": asset.asset_type,
        "figure_id": asset.asset_id if asset.asset_type == "figure" else None,
        "table_id": asset.asset_id if asset.asset_type == "table" else None,
        "label": asset.label,
        "caption": asset.caption,
        "image_path": asset.image_paths[0] if len(asset.image_paths) == 1 else None,
        "image_paths": asset.image_paths,
        "section": asset.section,
        "subsection": asset.subsection,
        "section_path": [asset.section] + ([asset.subsection] if asset.subsection else []),
        "context_before": asset.context_before,
        "context_after": asset.context_after,
        "notes": asset.notes,
        "table_text_missing": asset.table_text_missing,
        "mapping_confidence": asset.mapping_confidence,
        "char_start": asset.char_start,
        "char_end": asset.char_end,
        "source_file": document.source_file,
        "parse_warnings": asset.parse_warnings,
    }


def _inspection_record(
    document: dict[str, Any], chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "pmcid": document["pmcid"],
        "paper_title": document["paper_title"],
        "section_tree": document["section_tree"],
        "parse_warnings": document["parse_warnings"],
        "chunk_summaries": [
            {
                "chunk_id": chunk["chunk_id"],
                "section_path": chunk["section_path"],
                "chunk_type": chunk["chunk_type"],
                "word_count": chunk["word_count"],
                "token_count": chunk["token_count"],
                "text_preview": chunk["text"][:500],
                "figure_ids": chunk["figure_ids"],
                "table_ids": chunk["table_ids"],
                "parse_warnings": chunk["parse_warnings"],
            }
            for chunk in chunks
        ],
    }


def _process_one(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    candidate = DocumentCandidate(
        pmcid=payload["pmcid"],
        markdown_path=Path(payload["markdown_path"]),
        duplicate_paths=[Path(path) for path in payload["duplicate_paths"]],
    )
    try:
        assert _WORKER_TOKENIZER is not None and _WORKER_CONFIG is not None
        document = parse_document(candidate.markdown_path, candidate.pmcid, payload["metadata"])
        chunks, parents = create_chunks(document, _WORKER_TOKENIZER, _WORKER_CONFIG)
        if not chunks:
            raise ValueError("No searchable body chunks were produced")
        for chunk in chunks:
            if chunk["word_count"] > _WORKER_CONFIG.hard_max_words:
                raise ValueError(f"chunk {chunk['chunk_id']} exceeds hard word limit")
            if chunk["token_count"] > _WORKER_CONFIG.hard_max_tokens:
                raise ValueError(f"chunk {chunk['chunk_id']} exceeds hard token limit")
        bundle = {
            "document": _document_record(document, candidate, _WORKER_TOKENIZER.name, chunks, parents),
            "chunks": chunks,
            "parents": parents,
            "assets": [_asset_record(asset, document) for asset in document.assets],
        }
        return {
            "pmcid": candidate.pmcid,
            "status": "success",
            "elapsed_seconds": time.perf_counter() - started,
            "chunk_count": len(chunks),
            "bundle": bundle,
        }
    except Exception as exc:
        return {
            "pmcid": candidate.pmcid,
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "source_markdown": str(candidate.markdown_path),
            "traceback": traceback.format_exc(limit=8),
        }


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _finalize_streaming_outputs(
    output_dir: Path,
    candidates: list[DocumentCandidate],
    shard_records: list[dict[str, Any]],
    global_files: dict[str, dict[str, Any]],
    *,
    processing_times: list[float],
    resolved_tokenizer_name: str,
    end_to_end_started: float,
    discovery_seconds: float,
    inventory_reused: bool,
    inventory_build_seconds: float,
    tokenizer_load_seconds: float,
    document_processing_wall_seconds: float,
    effective_workers: int,
    worker_start_method: str,
    documents_per_shard: int,
    input_dir: Path,
    metadata_jsonl: Path | None,
    require_images: bool,
    config_hash: str,
    config_values: dict[str, int],
) -> dict[str, Any]:
    finalize_started = time.perf_counter()
    chunk_words: list[int] = []
    chunks_per_document: list[int] = []
    section_chunk_counts: Counter[str] = Counter()
    subsection_labels: set[str] = set()
    excluded_totals: Counter[str] = Counter()
    tokenizer_names: set[str] = set()
    unknown_heading_count = 0
    title_anomaly_count = 0
    possible_merge_warning_count = 0
    raw_image_reference_count = 0
    bound_image_reference_count = 0
    unbound_image_reference_count = 0
    missing_table_text_count = 0
    image_asset_count = 0
    figure_block_count = 0
    table_block_count = 0
    chunks_above_448_tokens = 0

    documents_path = output_dir / global_files["documents"]["path"]
    for document in _iter_jsonl(documents_path):
        chunks_per_document.append(int(document["chunk_count"]))
        unknown_heading_count += len(document["unknown_headings"])
        title_anomaly_count += bool(document["title_anomaly"])
        possible_merge_warning_count += "possible_heading_body_merge" in document.get(
            "parse_warnings", []
        )
        raw_image_reference_count += int(document.get("raw_image_reference_count", 0))
        bound_image_reference_count += int(document.get("bound_image_reference_count", 0))
        unbound_image_reference_count += int(document.get("unbound_image_reference_count", 0))
        excluded_totals.update(document["excluded_counts"])

    for shard in shard_records:
        for chunk in _iter_jsonl(output_dir / shard["files"]["chunks"]["path"]):
            word_count = int(chunk["word_count"])
            chunk_words.append(word_count)
            section_chunk_counts[chunk["section"]] += 1
            if chunk.get("subsection"):
                subsection_labels.add(chunk["subsection"])
            tokenizer_names.add(chunk["tokenizer_name"])
            chunks_above_448_tokens += int(chunk["token_count"] > 448)
        for asset in _iter_jsonl(output_dir / shard["files"]["figures_tables"]["path"]):
            missing_table_text_count += bool(asset["table_text_missing"])
            image_asset_count += bool(asset["image_paths"])
            figure_block_count += asset["asset_type"] == "figure"
            table_block_count += asset["asset_type"] == "table"

    successful_documents = int(global_files["documents"]["row_count"])
    failed_documents = int(global_files["errors"]["row_count"])
    elapsed_seconds = time.perf_counter() - end_to_end_started
    statistics_row = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_revision": PIPELINE_REVISION,
        "successful_documents": successful_documents,
        "failed_documents": failed_documents,
        "skipped_documents": 0,
        "selected_documents": len(candidates),
        "newly_processed_documents": successful_documents,
        "total_processing_time_seconds": round(elapsed_seconds, 6),
        "average_seconds_per_processed_document": round(statistics.mean(processing_times), 6) if processing_times else 0.0,
        "documents_per_second": round(len(candidates) / elapsed_seconds, 6) if elapsed_seconds else 0.0,
        "end_to_end_documents_per_second": round(len(candidates) / elapsed_seconds, 6) if elapsed_seconds else 0.0,
        "processing_documents_per_second": round(len(candidates) / document_processing_wall_seconds, 6) if document_processing_wall_seconds else 0.0,
        "inventory_reused": inventory_reused,
        "inventory_build_seconds": round(inventory_build_seconds, 6),
        "discovery_seconds": round(discovery_seconds, 6),
        "tokenizer_load_seconds": round(tokenizer_load_seconds, 6),
        "document_processing_wall_seconds": round(document_processing_wall_seconds, 6),
        "merge_seconds": round(time.perf_counter() - finalize_started, 6),
        "effective_workers": effective_workers,
        "worker_start_method": worker_start_method,
        "documents_per_shard": documents_per_shard,
        "shard_count": len(shard_records),
        "total_chunks": len(chunk_words),
        "chunks_per_document_mean": round(statistics.mean(chunks_per_document), 4) if chunks_per_document else 0.0,
        "chunks_per_document_median": statistics.median(chunks_per_document) if chunks_per_document else 0.0,
        "chunk_words_mean": round(statistics.mean(chunk_words), 4) if chunk_words else 0.0,
        "chunk_words_median": statistics.median(chunk_words) if chunk_words else 0.0,
        "chunk_words_min": min(chunk_words, default=0),
        "chunk_words_max": max(chunk_words, default=0),
        "chunk_words_p95": round(_percentile(chunk_words, 0.95), 4),
        "chunks_below_80_words": sum(value < 80 for value in chunk_words),
        "chunks_above_320_words": sum(value > 320 for value in chunk_words),
        "chunks_above_448_tokens": chunks_above_448_tokens,
        "recognized_section_label_count": len(section_chunk_counts),
        "recognized_subsection_label_count": len(subsection_labels),
        "section_chunk_counts": dict(sorted(section_chunk_counts.items())),
        "unknown_heading_count": unknown_heading_count,
        "title_anomaly_count": title_anomaly_count,
        "possible_heading_body_merge_warning_count": possible_merge_warning_count,
        "missing_table_text_count": missing_table_text_count,
        "image_asset_count": image_asset_count,
        "raw_image_reference_count": raw_image_reference_count,
        "bound_image_reference_count": bound_image_reference_count,
        "unbound_image_reference_count": unbound_image_reference_count,
        "figure_block_count": figure_block_count,
        "table_block_count": table_block_count,
        "excluded_references_blocks": excluded_totals.get("references", 0),
        "excluded_advertisement_blocks": excluded_totals.get("advertisement", 0),
        "excluded_non_body_blocks": sum(excluded_totals.values()),
        "excluded_counts": dict(sorted(excluded_totals.items())),
        "tokenizer_names": sorted(tokenizer_names or {resolved_tokenizer_name}),
        "resolved_tokenizer_name": resolved_tokenizer_name,
    }
    _atomic_write_bytes(
        output_dir / "statistics.json",
        json.dumps(statistics_row, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )

    manifest = {
        "schema_version": "medcpt_chunk_manifest_v1",
        "pipeline_revision": PIPELINE_REVISION,
        "output_layout": "streaming_jsonl_shards_no_resume_v1",
        "documents_per_shard": documents_per_shard,
        "selected_documents": len(candidates),
        "successful_documents": successful_documents,
        "failed_documents": failed_documents,
        "input": {
            "input_dir": str(input_dir),
            "metadata_jsonl": str(metadata_jsonl) if metadata_jsonl else None,
            "require_images": require_images,
        },
        "tokenizer_name": resolved_tokenizer_name,
        "config_hash": config_hash,
        "chunking_config": config_values,
        "global_files": {
            **global_files,
            "statistics": {"path": "statistics.json"},
        },
        "shards": shard_records,
    }
    _atomic_write_bytes(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    return statistics_row


def run_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    metadata_jsonl: Path | None = None,
    limit: int = 0,
    workers: int = 1,
    tokenizer_name: str = MEDCPT_TOKENIZER,
    allow_tokenizer_fallback: bool = True,
    local_files_only: bool = False,
    require_images: bool = True,
    inventory_db: Path | None = None,
    refresh_inventory: bool = False,
    inspection_seed: int = 20260806,
    documents_per_shard: int = 500,
    config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    metadata_jsonl = metadata_jsonl.resolve() if metadata_jsonl else None
    inventory_db = inventory_db.resolve() if inventory_db else None
    config = config or ChunkingConfig()
    if documents_per_shard <= 0:
        raise ValueError("documents_per_shard must be greater than zero")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery_started = time.perf_counter()
    candidates, inventory_reused, inventory_build_seconds = _discover_documents_with_metrics(
        input_dir,
        limit=limit,
        require_images=require_images,
        inventory_db=inventory_db,
        refresh_inventory=refresh_inventory,
    )
    discovery_seconds = time.perf_counter() - discovery_started
    if not candidates:
        raise RuntimeError("No eligible PMCID Markdown documents were found")
    metadata = load_metadata(metadata_jsonl, {candidate.pmcid for candidate in candidates})

    # Resolve once in the parent. Hugging Face then exists in the local cache for workers.
    tokenizer_started = time.perf_counter()
    tokenizer = resolve_tokenizer(
        tokenizer_name,
        allow_fallback=allow_tokenizer_fallback,
        local_files_only=local_files_only,
    )
    tokenizer_load_seconds = time.perf_counter() - tokenizer_started
    resolved_name = tokenizer.name
    config_payload = config.as_dict()
    config_hash = hashlib.sha256(
        _json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "pipeline_revision": PIPELINE_REVISION,
                "chunking": config_payload,
                "tokenizer_name": resolved_name,
                "require_images": require_images,
            }
        )
    ).hexdigest()

    tasks: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        doc_metadata = metadata.get(candidate.pmcid, {})
        duplicate_rows.append(
            {
                "pmcid": candidate.pmcid,
                "selected_markdown": str(candidate.markdown_path),
                "discarded_markdown": [str(path) for path in candidate.duplicate_paths],
                "candidate_count": len(candidate.duplicate_paths) + 1,
            }
        )
        tasks.append(
            {
                "pmcid": candidate.pmcid,
                "markdown_path": str(candidate.markdown_path),
                "duplicate_paths": [str(path) for path in candidate.duplicate_paths],
                "metadata": doc_metadata,
            }
        )
    _atomic_write_jsonl(output_dir / "duplicate_resolution.jsonl", duplicate_rows)
    output_writer = _StreamingOutputWriter(
        output_dir,
        candidates,
        documents_per_shard,
        inspection_seed,
    )

    outcomes: list[dict[str, Any]] = []
    effective_workers = min(max(1, workers), len(tasks))
    worker_start_method = "single"
    processing_started = time.perf_counter()
    try:
        if effective_workers <= 1:
            _set_worker_state(tokenizer, config)
            for index, task in enumerate(tasks, 1):
                outcome = _process_one(task)
                output_writer.record(outcome)
                outcomes.append(outcome)
                if index % 25 == 0 or index == len(tasks):
                    print(f"processed {index}/{len(tasks)}", flush=True)
        else:
            can_fork = os.name != "nt" and "fork" in multiprocessing.get_all_start_methods()
            worker_start_method = "fork" if can_fork else "spawn"
            context = multiprocessing.get_context(worker_start_method)
            executor_options: dict[str, Any] = {
                "max_workers": effective_workers,
                "mp_context": context,
            }
            if can_fork:
                _set_worker_state(tokenizer, config)
            else:
                executor_options.update(
                    initializer=_init_worker,
                    initargs=(resolved_name, config_payload),
                )
            with ProcessPoolExecutor(**executor_options) as executor:
                task_iterator = iter(tasks)
                pending: deque[tuple[str, Any]] = deque()
                for _ in range(min(len(tasks), effective_workers * 4)):
                    task = next(task_iterator)
                    pending.append((task["pmcid"], executor.submit(_process_one, task)))
                processed = 0
                while pending:
                    pmcid, future = pending.popleft()
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = {
                            "pmcid": pmcid,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                            "elapsed_seconds": 0.0,
                        }
                    output_writer.record(outcome)
                    outcomes.append(outcome)
                    processed += 1
                    next_task = next(task_iterator, None)
                    if next_task is not None:
                        pending.append(
                            (next_task["pmcid"], executor.submit(_process_one, next_task))
                        )
                    if processed % 25 == 0 or processed == len(tasks):
                        print(f"processed {processed}/{len(tasks)}", flush=True)
    finally:
        shard_records, global_files = output_writer.close()
    document_processing_wall_seconds = time.perf_counter() - processing_started

    processing_times = [outcome["elapsed_seconds"] for outcome in outcomes]
    stats = _finalize_streaming_outputs(
        output_dir,
        candidates,
        shard_records,
        global_files,
        processing_times=processing_times,
        resolved_tokenizer_name=resolved_name,
        end_to_end_started=started,
        discovery_seconds=discovery_seconds,
        inventory_reused=inventory_reused,
        inventory_build_seconds=inventory_build_seconds,
        tokenizer_load_seconds=tokenizer_load_seconds,
        document_processing_wall_seconds=document_processing_wall_seconds,
        effective_workers=effective_workers,
        worker_start_method=worker_start_method,
        documents_per_shard=documents_per_shard,
        input_dir=input_dir,
        metadata_jsonl=metadata_jsonl,
        require_images=require_images,
        config_hash=config_hash,
        config_values=config_payload,
    )
    return stats
