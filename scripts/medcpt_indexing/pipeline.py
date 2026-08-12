from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Encoder, IndexDocument, IndexingConfig, KeywordSink, VectorSink
from .schema import load_pmid_mapping, make_index_document

log = logging.getLogger("medcpt_fulltext_indexer")
MANIFEST_SCHEMA = "medcpt_fulltext_index_manifest_v1"


def discover_shards(chunks_dirs: Sequence[Path], limit: int = 0) -> list[Path]:
    shards = [
        shard
        for chunks_dir in chunks_dirs
        for shard in sorted(chunks_dir.glob("part-*.jsonl"))
    ]
    if not shards:
        raise FileNotFoundError(
            f"No part-*.jsonl files found in: {', '.join(map(str, chunks_dirs))}"
        )
    duplicate_names = [
        name for name, count in Counter(path.name for path in shards).items() if count > 1
    ]
    if duplicate_names:
        raise ValueError(f"Chunk shard names must be unique: {duplicate_names}")
    return shards[:limit] if limit > 0 else shards


def _batches(
    items: Iterable[IndexDocument], size: int
) -> Iterator[list[IndexDocument]]:
    batch: list[IndexDocument] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_documents(
    shard: Path,
    pmid_by_pmcid: dict[str, str],
    max_tokens: int,
) -> Iterator[IndexDocument]:
    with shard.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    raise TypeError("line is not a JSON object")
                yield make_index_document(
                    chunk,
                    pmid_by_pmcid=pmid_by_pmcid,
                    source_shard=shard.name,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                raise ValueError(f"{shard}:{line_number}: {exc}") from exc


def _file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _config_signature(config: IndexingConfig, encoder: Encoder) -> dict[str, Any]:
    return {
        "chunks_dirs": [
            str(path.resolve())
            for path in (config.chunks_dir, *config.additional_chunks_dirs)
        ],
        "mapping_db": str(config.mapping_db.resolve()),
        "mapping_db_fingerprint": _file_fingerprint(config.mapping_db),
        "collection_name": config.collection_name,
        "bm25_index_name": config.bm25_index_name,
        "distance": "dot",
        "vector_only": config.vector_only,
        "model_name": encoder.model_name,
        "dimension": encoder.dimension,
        "max_tokens": config.max_tokens,
    }


def _new_manifest(signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration": signature,
        "completed_shards": {},
        "totals": {"chunks": 0, "elapsed_seconds": 0.0},
    }


def _load_manifest(path: Path, signature: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return _new_manifest(signature)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError(f"Unsupported index manifest: {path}")
    if manifest.get("configuration") != signature:
        raise RuntimeError(
            "Index manifest configuration differs from this run. Use new collection, "
            "BM25 index, and state-file names for a new index build."
        )
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _already_complete(manifest: dict[str, Any], shard: Path) -> bool:
    record = manifest["completed_shards"].get(shard.name)
    return bool(record and record.get("input") == _file_fingerprint(shard))


def _flush(
    documents: list[IndexDocument],
    vectors: list[list[float]],
    vector_sink: VectorSink,
    keyword_sink: KeywordSink | None,
) -> int:
    if not documents:
        return 0
    vector_count = vector_sink.write(documents, vectors)
    keyword_count = keyword_sink.write(documents) if keyword_sink else len(documents)
    if vector_count != len(documents) or keyword_count != len(documents):
        raise RuntimeError(
            f"Sink count mismatch: input={len(documents)}, vector={vector_count}, "
            f"bm25={keyword_count}"
        )
    documents.clear()
    vectors.clear()
    return vector_count


def _process_shard(
    shard: Path,
    *,
    pmid_by_pmcid: dict[str, str],
    encoder: Encoder,
    vector_sink: VectorSink,
    keyword_sink: KeywordSink | None,
    config: IndexingConfig,
) -> tuple[int, float]:
    started = time.perf_counter()
    indexed = 0
    pending_documents: list[IndexDocument] = []
    pending_vectors: list[list[float]] = []

    documents = _iter_documents(shard, pmid_by_pmcid, config.max_tokens)
    for encode_batch in _batches(documents, config.encode_batch_size):
        vectors = encoder.encode([document.embedding_text for document in encode_batch])
        if len(vectors) != len(encode_batch):
            raise RuntimeError("Encoder returned a different number of vectors")
        if any(len(vector) != encoder.dimension for vector in vectors):
            raise RuntimeError("Encoder returned a vector with the wrong dimension")
        pending_documents.extend(encode_batch)
        pending_vectors.extend(vectors)
        if len(pending_documents) >= config.upload_batch_size:
            indexed += _flush(
                pending_documents, pending_vectors, vector_sink, keyword_sink
            )
            if (
                config.log_every
                and indexed % config.log_every < config.upload_batch_size
            ):
                log.info("shard=%s indexed=%d", shard.name, indexed)

    indexed += _flush(pending_documents, pending_vectors, vector_sink, keyword_sink)
    vector_count = vector_sink.count_shard(shard.name)
    keyword_count = keyword_sink.count_shard(shard.name) if keyword_sink else indexed
    if vector_count != indexed or keyword_count != indexed:
        raise RuntimeError(
            f"Remote verification failed for {shard.name}: input={indexed}, "
            f"Qdrant={vector_count}, OpenSearch={keyword_count}"
        )
    return indexed, time.perf_counter() - started


def run_indexing(
    config: IndexingConfig,
    encoder: Encoder,
    vector_sink: VectorSink,
    keyword_sink: KeywordSink | None,
) -> dict[str, Any]:
    if config.encode_batch_size <= 0 or config.upload_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if config.upload_batch_size < config.encode_batch_size:
        raise ValueError("upload_batch_size must be at least encode_batch_size")

    chunks_dirs = (config.chunks_dir, *config.additional_chunks_dirs)
    shards = discover_shards(chunks_dirs, config.limit_shards)
    pmid_by_pmcid = load_pmid_mapping(config.mapping_db)
    signature = _config_signature(config, encoder)
    manifest = _load_manifest(config.state_file, signature)
    vector_sink.ensure_ready(encoder.dimension)
    if keyword_sink:
        keyword_sink.ensure_ready()

    for shard in shards:
        if _already_complete(manifest, shard):
            expected = manifest["completed_shards"][shard.name]["chunks"]
            vector_count = vector_sink.count_shard(shard.name)
            keyword_count = keyword_sink.count_shard(shard.name) if keyword_sink else expected
            if vector_count == expected and keyword_count == expected:
                log.info("skip completed shard=%s", shard.name)
                continue
            log.warning(
                "reindex incomplete remote shard=%s expected=%d Qdrant=%d OpenSearch=%d",
                shard.name,
                expected,
                vector_count,
                keyword_count,
            )
        count, elapsed = _process_shard(
            shard,
            pmid_by_pmcid=pmid_by_pmcid,
            encoder=encoder,
            vector_sink=vector_sink,
            keyword_sink=keyword_sink,
            config=config,
        )
        manifest["completed_shards"][shard.name] = {
            "input": _file_fingerprint(shard),
            "chunks": count,
            "elapsed_seconds": round(elapsed, 6),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        totals = manifest["totals"]
        totals["chunks"] = sum(
            record["chunks"] for record in manifest["completed_shards"].values()
        )
        totals["elapsed_seconds"] = round(
            sum(
                record["elapsed_seconds"]
                for record in manifest["completed_shards"].values()
            ),
            6,
        )
        _write_manifest(config.state_file, manifest)
        log.info(
            "completed shard=%s chunks=%d elapsed=%.2fs",
            shard.name,
            count,
            elapsed,
        )
    return manifest


def validate_inputs(config: IndexingConfig) -> dict[str, Any]:
    chunks_dirs = (config.chunks_dir, *config.additional_chunks_dirs)
    shards = discover_shards(chunks_dirs, config.limit_shards)
    pmid_by_pmcid = load_pmid_mapping(config.mapping_db)
    counts: Counter[str] = Counter()
    pmcids: set[str] = set()
    digest = hashlib.sha256()
    started = time.perf_counter()
    for shard in shards:
        shard_count = 0
        for document in _iter_documents(shard, pmid_by_pmcid, config.max_tokens):
            shard_count += 1
            pmcids.add(document.metadata["pmcid"])
            counts[str(document.metadata.get("chunk_type", "paragraph"))] += 1
            digest.update(document.chunk_id.encode("utf-8"))
            digest.update(b"\n")
        counts["chunks"] += shard_count
    return {
        "shards": len(shards),
        "chunks": counts.pop("chunks", 0),
        "documents": len(pmcids),
        "chunk_types": dict(sorted(counts.items())),
        "mapping_rows": len(pmid_by_pmcid),
        "chunk_id_sha256": digest.hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
