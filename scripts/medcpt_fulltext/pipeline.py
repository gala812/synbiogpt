from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
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
PIPELINE_REVISION = "pilot_rules_r4"
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


def _atomic_write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        # Spool files are temporary merge inputs; level 3 balances NFS traffic and CPU time.
        with gzip.open(
            temp_name, "wt", encoding="utf-8", newline="\n", compresslevel=3
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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


def _spool_paths(output_dir: Path, pmcid: str) -> tuple[Path, Path]:
    shard = pmcid[-3:]
    base = output_dir / ".spool" / shard
    return base / f"{pmcid}.json.gz", base / f"{pmcid}.done.json"


def _source_signature(candidate: DocumentCandidate, metadata: dict[str, Any], config_hash: str) -> dict[str, Any]:
    stat = candidate.markdown_path.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_revision": PIPELINE_REVISION,
        "config_hash": config_hash,
        "source_path": str(candidate.markdown_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "metadata_sha256": hashlib.sha256(_json_bytes(metadata)).hexdigest(),
    }


def _is_committed(marker_path: Path, signature: dict[str, Any]) -> bool:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker == signature


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


def _process_one(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    candidate = DocumentCandidate(
        pmcid=payload["pmcid"],
        markdown_path=Path(payload["markdown_path"]),
        duplicate_paths=[Path(path) for path in payload["duplicate_paths"]],
    )
    bundle_path = Path(payload["bundle_path"])
    marker_path = Path(payload["marker_path"])
    try:
        assert _WORKER_TOKENIZER is not None and _WORKER_CONFIG is not None
        document = parse_document(candidate.markdown_path, candidate.pmcid, payload["metadata"])
        chunks, parents = create_chunks(document, _WORKER_TOKENIZER, _WORKER_CONFIG)
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
        _atomic_write_gzip_json(bundle_path, bundle)
        _atomic_write_bytes(marker_path, _json_bytes(payload["signature"]) + b"\n")
        return {
            "pmcid": candidate.pmcid,
            "status": "success",
            "elapsed_seconds": time.perf_counter() - started,
            "chunk_count": len(chunks),
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


def _read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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


def _write_final_outputs(
    output_dir: Path,
    candidates: list[DocumentCandidate],
    failed: list[dict[str, Any]],
    *,
    skipped_count: int,
    processing_times: list[float],
    inspection_seed: int,
    selected_count: int,
    newly_processed_count: int,
    resolved_tokenizer_name: str,
    end_to_end_started: float,
    discovery_seconds: float,
    inventory_reused: bool,
    inventory_build_seconds: float,
    tokenizer_load_seconds: float,
    document_processing_wall_seconds: float,
    processing_task_count: int,
    effective_workers: int,
    worker_start_method: str,
) -> dict[str, Any]:
    merge_started = time.perf_counter()
    chunk_words: list[int] = []
    chunks_per_document: list[int] = []
    section_chunk_counts: dict[str, int] = {}
    subsection_labels: set[str] = set()
    excluded_totals: dict[str, int] = {}
    tokenizer_names: set[str] = set()
    successful_documents = 0
    unknown_heading_count = 0
    title_anomaly_count = 0
    possible_merge_warning_count = 0
    missing_table_text_count = 0
    image_asset_count = 0
    raw_image_reference_count = 0
    bound_image_reference_count = 0
    unbound_image_reference_count = 0
    figure_block_count = 0
    table_block_count = 0
    chunks_above_448_tokens = 0

    committed_ids = []
    for candidate in candidates:
        bundle_path, marker_path = _spool_paths(output_dir, candidate.pmcid)
        if marker_path.is_file() and bundle_path.is_file():
            committed_ids.append(candidate.pmcid)
    sample_count = min(20, len(committed_ids))
    rng = random.Random(inspection_seed)
    sampled_ids = set(rng.sample(committed_ids, sample_count)) if sample_count else set()

    output_names = {
        "chunks": "chunks.jsonl",
        "parents": "parents.jsonl",
        "assets": "figures_tables.jsonl",
        "documents": "documents.jsonl",
        "errors": "errors.jsonl",
        "inspection": "inspection_samples.jsonl",
    }
    temp_paths: dict[str, str] = {}
    handles: dict[str, Any] = {}

    def write_row(name: str, row: dict[str, Any]) -> None:
        handles[name].write(_json_bytes(row))
        handles[name].write(b"\n")

    merge_failures = list(failed)
    try:
        for name, filename in output_names.items():
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".partial", dir=output_dir
            )
            temp_paths[name] = temp_name
            handles[name] = os.fdopen(fd, "wb")

        for candidate in candidates:
            bundle_path, marker_path = _spool_paths(output_dir, candidate.pmcid)
            if not marker_path.is_file() or not bundle_path.is_file():
                continue
            try:
                bundle = _read_bundle(bundle_path)
                document = bundle["document"]
                chunks = sorted(bundle["chunks"], key=lambda row: row["chunk_index"])
                parents = sorted(bundle["parents"], key=lambda row: row["parent_chunk_id"])
                assets = sorted(
                    bundle["assets"], key=lambda row: (row["char_start"], row.get("label") or "")
                )
            except (OSError, EOFError, KeyError, TypeError, json.JSONDecodeError) as exc:
                merge_failures.append(
                    {
                        "pmcid": candidate.pmcid,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": f"Cannot merge committed spool bundle: {exc}",
                        "source_markdown": str(candidate.markdown_path),
                    }
                )
                continue

            write_row("documents", document)
            for chunk in chunks:
                write_row("chunks", chunk)
                word_count = int(chunk["word_count"])
                chunk_words.append(word_count)
                section = chunk["section"]
                section_chunk_counts[section] = section_chunk_counts.get(section, 0) + 1
                if chunk.get("subsection"):
                    subsection_labels.add(chunk["subsection"])
                tokenizer_names.add(chunk["tokenizer_name"])
                chunks_above_448_tokens += int(chunk["token_count"] > 448)
            for parent in parents:
                write_row("parents", parent)
            for asset in assets:
                write_row("assets", asset)
                missing_table_text_count += bool(asset["table_text_missing"])
                image_asset_count += bool(asset["image_paths"])
                figure_block_count += asset["asset_type"] == "figure"
                table_block_count += asset["asset_type"] == "table"

            successful_documents += 1
            chunks_per_document.append(int(document["chunk_count"]))
            unknown_heading_count += len(document["unknown_headings"])
            title_anomaly_count += bool(document["title_anomaly"])
            possible_merge_warning_count += "possible_heading_body_merge" in document.get(
                "parse_warnings", []
            )
            raw_image_reference_count += int(document.get("raw_image_reference_count", 0))
            bound_image_reference_count += int(document.get("bound_image_reference_count", 0))
            unbound_image_reference_count += int(document.get("unbound_image_reference_count", 0))
            for key, value in document["excluded_counts"].items():
                excluded_totals[key] = excluded_totals.get(key, 0) + value

            if candidate.pmcid in sampled_ids:
                write_row(
                    "inspection",
                    {
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
                    },
                )

        for row in sorted(merge_failures, key=lambda item: item["pmcid"]):
            write_row("errors", row)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for name, filename in output_names.items():
            os.replace(temp_paths[name], output_dir / filename)
    except Exception:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for temp_name in temp_paths.values():
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise

    merge_seconds = time.perf_counter() - merge_started
    elapsed_seconds = time.perf_counter() - end_to_end_started
    statistics_row = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_revision": PIPELINE_REVISION,
        "successful_documents": successful_documents,
        "failed_documents": len(merge_failures),
        "skipped_documents": skipped_count,
        "selected_documents": selected_count,
        "newly_processed_documents": newly_processed_count,
        "total_processing_time_seconds": round(elapsed_seconds, 6),
        "average_seconds_per_processed_document": round(statistics.mean(processing_times), 6) if processing_times else 0.0,
        "documents_per_second": round(selected_count / elapsed_seconds, 6) if elapsed_seconds else 0.0,
        "end_to_end_documents_per_second": round(selected_count / elapsed_seconds, 6) if elapsed_seconds else 0.0,
        "processing_documents_per_second": round(processing_task_count / document_processing_wall_seconds, 6) if document_processing_wall_seconds else 0.0,
        "inventory_reused": inventory_reused,
        "inventory_build_seconds": round(inventory_build_seconds, 6),
        "discovery_seconds": round(discovery_seconds, 6),
        "tokenizer_load_seconds": round(tokenizer_load_seconds, 6),
        "document_processing_wall_seconds": round(document_processing_wall_seconds, 6),
        "merge_seconds": round(merge_seconds, 6),
        "effective_workers": effective_workers,
        "worker_start_method": worker_start_method,
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

    return statistics_row


def run_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    metadata_jsonl: Path | None = None,
    limit: int = 500,
    workers: int = 1,
    tokenizer_name: str = MEDCPT_TOKENIZER,
    allow_tokenizer_fallback: bool = True,
    local_files_only: bool = False,
    require_images: bool = True,
    force: bool = False,
    inventory_db: Path | None = None,
    refresh_inventory: bool = False,
    inspection_seed: int = 20260806,
    config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = config or ChunkingConfig()
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
    skipped = 0
    duplicate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        doc_metadata = metadata.get(candidate.pmcid, {})
        signature = _source_signature(candidate, doc_metadata, config_hash)
        bundle_path, marker_path = _spool_paths(output_dir, candidate.pmcid)
        duplicate_rows.append(
            {
                "pmcid": candidate.pmcid,
                "selected_markdown": str(candidate.markdown_path),
                "discarded_markdown": [str(path) for path in candidate.duplicate_paths],
                "candidate_count": len(candidate.duplicate_paths) + 1,
            }
        )
        if not force and bundle_path.is_file() and _is_committed(marker_path, signature):
            skipped += 1
            continue
        # A stale marker must never make an old bundle look successful if retry fails.
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        tasks.append(
            {
                "pmcid": candidate.pmcid,
                "markdown_path": str(candidate.markdown_path),
                "duplicate_paths": [str(path) for path in candidate.duplicate_paths],
                "metadata": doc_metadata,
                "bundle_path": str(bundle_path),
                "marker_path": str(marker_path),
                "signature": signature,
            }
        )
    _atomic_write_jsonl(output_dir / "duplicate_resolution.jsonl", duplicate_rows)

    outcomes: list[dict[str, Any]] = []
    effective_workers = min(max(1, workers), len(tasks)) if tasks else 0
    worker_start_method = "none"
    processing_started = time.perf_counter()
    if tasks and effective_workers <= 1:
        worker_start_method = "single"
        _set_worker_state(tokenizer, config)
        for index, task in enumerate(tasks, 1):
            outcomes.append(_process_one(task))
            if index % 25 == 0 or index == len(tasks):
                print(f"processed {index}/{len(tasks)}", flush=True)
    elif tasks:
        can_fork = os.name != "nt" and "fork" in multiprocessing.get_all_start_methods()
        worker_start_method = "fork" if can_fork else "spawn"
        context = multiprocessing.get_context(worker_start_method)
        executor_options: dict[str, Any] = {
            "max_workers": effective_workers,
            "mp_context": context,
        }
        if can_fork:
            # The read-only tokenizer is loaded once and inherited by Linux workers.
            _set_worker_state(tokenizer, config)
        else:
            executor_options.update(
                initializer=_init_worker,
                initargs=(resolved_name, config_payload),
            )
        with ProcessPoolExecutor(**executor_options) as executor:
            futures = {executor.submit(_process_one, task): task["pmcid"] for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    outcomes.append(future.result())
                except Exception as exc:
                    outcomes.append(
                        {
                            "pmcid": futures[future],
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                            "elapsed_seconds": 0.0,
                        }
                    )
                if index % 25 == 0 or index == len(futures):
                    print(f"processed {index}/{len(futures)}", flush=True)
    document_processing_wall_seconds = time.perf_counter() - processing_started

    failed = [outcome for outcome in outcomes if outcome["status"] == "failed"]
    processing_times = [outcome["elapsed_seconds"] for outcome in outcomes]
    stats = _write_final_outputs(
        output_dir,
        candidates,
        failed,
        skipped_count=skipped,
        processing_times=processing_times,
        inspection_seed=inspection_seed,
        selected_count=len(candidates),
        newly_processed_count=len(tasks) - len(failed),
        resolved_tokenizer_name=resolved_name,
        end_to_end_started=started,
        discovery_seconds=discovery_seconds,
        inventory_reused=inventory_reused,
        inventory_build_seconds=inventory_build_seconds,
        tokenizer_load_seconds=tokenizer_load_seconds,
        document_processing_wall_seconds=document_processing_wall_seconds,
        processing_task_count=len(tasks),
        effective_workers=effective_workers,
        worker_start_method=worker_start_method,
    )
    return stats
