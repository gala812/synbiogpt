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
        with gzip.open(temp_name, "wt", encoding="utf-8", newline="\n") as handle:
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
        if len(paths) == 1:
            best = paths[0]
        else:
            scored = [(path, _candidate_score(path)) for path in paths]
            best = sorted(scored, key=lambda item: tuple([-value for value in item[1]]) + (item[0].as_posix(),))[0][0]
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
    elapsed_seconds: float,
    processing_times: list[float],
    inspection_seed: int,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for candidate in candidates:
        bundle_path, marker_path = _spool_paths(output_dir, candidate.pmcid)
        if not marker_path.is_file() or not bundle_path.is_file():
            continue
        try:
            bundle = _read_bundle(bundle_path)
        except (OSError, EOFError, json.JSONDecodeError):
            continue
        documents.append(bundle["document"])
        chunks.extend(bundle["chunks"])
        parents.extend(bundle["parents"])
        assets.extend(bundle["assets"])

    documents.sort(key=lambda row: row["pmcid"])
    chunks.sort(key=lambda row: (row["pmcid"], row["chunk_index"]))
    parents.sort(key=lambda row: (row["pmcid"], row["parent_chunk_id"]))
    assets.sort(key=lambda row: (row["pmcid"], row["char_start"], row.get("label") or ""))
    failed.sort(key=lambda row: row["pmcid"])

    _atomic_write_jsonl(output_dir / "chunks.jsonl", chunks)
    _atomic_write_jsonl(output_dir / "parents.jsonl", parents)
    _atomic_write_jsonl(output_dir / "figures_tables.jsonl", assets)
    _atomic_write_jsonl(output_dir / "documents.jsonl", documents)
    _atomic_write_jsonl(output_dir / "errors.jsonl", failed)

    chunk_words = [row["word_count"] for row in chunks]
    chunks_by_doc: dict[str, int] = {}
    for document in documents:
        chunks_by_doc[document["pmcid"]] = document["chunk_count"]
    section_labels = {row["section"] for row in chunks}
    subsection_labels = {row["subsection"] for row in chunks if row["subsection"]}
    excluded_totals: dict[str, int] = {}
    for document in documents:
        for key, value in document["excluded_counts"].items():
            excluded_totals[key] = excluded_totals.get(key, 0) + value
    warning_total = lambda name: sum(
        name in document.get("parse_warnings", []) for document in documents
    )
    statistics_row = {
        "schema_version": SCHEMA_VERSION,
        "successful_documents": len(documents),
        "failed_documents": len(failed),
        "skipped_documents": skipped_count,
        "total_processing_time_seconds": round(elapsed_seconds, 6),
        "average_seconds_per_processed_document": round(statistics.mean(processing_times), 6) if processing_times else 0.0,
        "documents_per_second": round(len(candidates) / elapsed_seconds, 6) if elapsed_seconds else 0.0,
        "total_chunks": len(chunks),
        "chunks_per_document_mean": round(statistics.mean(chunks_by_doc.values()), 4) if chunks_by_doc else 0.0,
        "chunks_per_document_median": statistics.median(chunks_by_doc.values()) if chunks_by_doc else 0.0,
        "chunk_words_mean": round(statistics.mean(chunk_words), 4) if chunk_words else 0.0,
        "chunk_words_median": statistics.median(chunk_words) if chunk_words else 0.0,
        "chunk_words_min": min(chunk_words, default=0),
        "chunk_words_max": max(chunk_words, default=0),
        "chunk_words_p95": round(_percentile(chunk_words, 0.95), 4),
        "chunks_below_80_words": sum(value < 80 for value in chunk_words),
        "chunks_above_320_words": sum(value > 320 for value in chunk_words),
        "chunks_above_448_tokens": sum(row["token_count"] > 448 for row in chunks),
        "recognized_section_label_count": len(section_labels),
        "recognized_subsection_label_count": len(subsection_labels),
        "section_chunk_counts": dict(sorted((label, sum(row["section"] == label for row in chunks)) for label in section_labels)),
        "unknown_heading_count": sum(len(document["unknown_headings"]) for document in documents),
        "title_anomaly_count": sum(bool(document["title_anomaly"]) for document in documents),
        "possible_heading_body_merge_warning_count": warning_total("possible_heading_body_merge"),
        "missing_table_text_count": sum(bool(row["table_text_missing"]) for row in assets),
        "image_asset_count": sum(bool(row["image_paths"]) for row in assets),
        "raw_image_reference_count": sum(document.get("raw_image_reference_count", 0) for document in documents),
        "bound_image_reference_count": sum(document.get("bound_image_reference_count", 0) for document in documents),
        "unbound_image_reference_count": sum(document.get("unbound_image_reference_count", 0) for document in documents),
        "figure_block_count": sum(row["asset_type"] == "figure" for row in assets),
        "table_block_count": sum(row["asset_type"] == "table" for row in assets),
        "excluded_references_blocks": excluded_totals.get("references", 0),
        "excluded_advertisement_blocks": excluded_totals.get("advertisement", 0),
        "excluded_non_body_blocks": sum(excluded_totals.values()),
        "excluded_counts": dict(sorted(excluded_totals.items())),
        "tokenizer_names": sorted({row["tokenizer_name"] for row in chunks}),
    }
    _atomic_write_bytes(
        output_dir / "statistics.json",
        json.dumps(statistics_row, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )

    sample_count = min(20, len(documents))
    rng = random.Random(inspection_seed)
    sampled_ids = set(rng.sample([row["pmcid"] for row in documents], sample_count)) if sample_count else set()
    chunks_by_pmcid: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        if chunk["pmcid"] in sampled_ids:
            chunks_by_pmcid.setdefault(chunk["pmcid"], []).append(chunk)
    inspection_rows = []
    for document in documents:
        if document["pmcid"] not in sampled_ids:
            continue
        inspection_rows.append(
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
                    for chunk in chunks_by_pmcid.get(document["pmcid"], [])
                ],
            }
        )
    _atomic_write_jsonl(output_dir / "inspection_samples.jsonl", inspection_rows)
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
    inspection_seed: int = 20260806,
    config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = config or ChunkingConfig()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_documents(input_dir, limit=limit, require_images=require_images)
    if not candidates:
        raise RuntimeError("No eligible PMCID Markdown documents were found")
    metadata = load_metadata(metadata_jsonl, {candidate.pmcid for candidate in candidates})

    # Resolve once in the parent. Hugging Face then exists in the local cache for workers.
    tokenizer = resolve_tokenizer(
        tokenizer_name,
        allow_fallback=allow_tokenizer_fallback,
        local_files_only=local_files_only,
    )
    resolved_name = tokenizer.name
    config_payload = config.as_dict()
    config_hash = hashlib.sha256(
        _json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
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
    if tasks and workers <= 1:
        _init_worker(resolved_name, config_payload)
        for index, task in enumerate(tasks, 1):
            outcomes.append(_process_one(task))
            if index % 25 == 0 or index == len(tasks):
                print(f"processed {index}/{len(tasks)}", flush=True)
    elif tasks:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(resolved_name, config_payload),
        ) as executor:
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

    failed = [outcome for outcome in outcomes if outcome["status"] == "failed"]
    processing_times = [outcome["elapsed_seconds"] for outcome in outcomes]
    elapsed = time.perf_counter() - started
    stats = _write_final_outputs(
        output_dir,
        candidates,
        failed,
        skipped_count=skipped,
        elapsed_seconds=elapsed,
        processing_times=processing_times,
        inspection_seed=inspection_seed,
    )
    stats["selected_documents"] = len(candidates)
    stats["newly_processed_documents"] = len(tasks) - len(failed)
    stats["resolved_tokenizer_name"] = resolved_name
    # Persist the augmented form returned to CLI as well.
    _atomic_write_bytes(
        output_dir / "statistics.json",
        json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    return stats
