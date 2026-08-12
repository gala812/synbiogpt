from __future__ import annotations

import json
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from medcpt_fulltext.chunking import build_embedding_text, chunk_units, word_count
    from medcpt_fulltext.models import ChunkingConfig, TextUnit
    from medcpt_fulltext.tokenization import resolve_tokenizer
    from medcpt_indexing.schema import POINT_NAMESPACE
except ModuleNotFoundError:  # Imported as scripts.medcpt_images from the repo root.
    from scripts.medcpt_fulltext.chunking import (
        build_embedding_text,
        chunk_units,
        word_count,
    )
    from scripts.medcpt_fulltext.models import ChunkingConfig, TextUnit
    from scripts.medcpt_fulltext.tokenization import resolve_tokenizer
    from scripts.medcpt_indexing.schema import POINT_NAMESPACE

from .recovery import asset_key


def _retry(operation, attempts: int = 6):
    import time

    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 30))


def _rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in sorted(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _asset_chunks(asset: dict[str, Any], token_counter, config: ChunkingConfig) -> list[dict[str, Any]]:
    asset_type = asset["asset_type"]
    kind = {
        "figure": "figure_caption",
        "table": "table_caption",
        "image": "image_caption",
    }.get(asset_type, "image_caption")
    unit = TextUnit(
        kind=kind,
        text=asset["caption"],
        section=asset.get("section") or "Unassigned",
        subsection="",
        char_start=0,
        char_end=len(asset["caption"]),
        order=0.0,
        warnings=["recovered_image_asset"],
        image_paths=list(asset["image_paths"]),
        figure_ids=[asset["asset_id"]] if asset_type == "figure" else [],
        table_ids=[asset["asset_id"]] if asset_type == "table" else [],
        source_spans=[[0, len(asset["caption"])]],
    )
    units = chunk_units([unit], token_counter, config, asset["paper_title"])
    records: list[dict[str, Any]] = []
    for index, part in enumerate(units, 1):
        chunk_id = f"{asset['asset_id']}_chunk_{index:04d}"
        embedding = build_embedding_text(
            asset["paper_title"], part.section, part.subsection, part.text
        )
        records.append(
            {
                "chunk_id": chunk_id,
                "doc_id": asset["pmcid"],
                "pmcid": asset["pmcid"],
                "paper_title": asset["paper_title"],
                "section": part.section,
                "subsection": part.subsection,
                "section_path": [part.section],
                "chunk_type": kind,
                "chunk_index": index,
                "parent_chunk_id": "",
                "text": part.text,
                "word_count": word_count(part.text),
                "text_token_count": token_counter.count(part.text),
                "token_count": token_counter.count(embedding),
                "tokenizer_name": token_counter.name,
                "char_start": 0,
                "char_end": len(part.text),
                "source_spans": [[0, len(part.text)]],
                "previous_chunk_id": None,
                "next_chunk_id": None,
                "image_paths": list(asset["image_paths"]),
                "asset_keys": list(asset["asset_keys"]),
                "image_asset_ids": [asset["asset_id"]],
                "figure_ids": part.figure_ids,
                "table_ids": part.table_ids,
                "source_file": f"{asset['pmcid']}.md",
                "parse_warnings": sorted(set(part.warnings)),
                "recovery_confidence": asset["confidence"],
                "recovery_reason": asset["reason"],
            }
        )
    for index, record in enumerate(records):
        if index:
            record["previous_chunk_id"] = records[index - 1]["chunk_id"]
        if index + 1 < len(records):
            record["next_chunk_id"] = records[index + 1]["chunk_id"]
    return records


def prepare_image_index(
    *,
    image_access_dir: Path,
    chunks_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    local_files_only: bool = True,
    records_per_shard: int = 10_000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recovered_chunks_dir = output_dir / "recovered_chunks"
    patches_dir = output_dir / "payload_patches"
    recovered_chunks_dir.mkdir(exist_ok=True)
    patches_dir.mkdir(exist_ok=True)

    additions: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"image_paths": set(), "asset_keys": set()}
    )
    new_assets: dict[str, dict[str, Any]] = {}
    for asset in _rows((image_access_dir / "recovered_assets").glob("part-*.jsonl")):
        if asset["is_new_asset"]:
            new_assets.setdefault(asset["asset_id"], asset)
        else:
            additions[asset["asset_id"]]["image_paths"].update(asset["image_paths"])
            additions[asset["asset_id"]]["asset_keys"].update(asset["asset_keys"])

    tokenizer = resolve_tokenizer(
        tokenizer_name,
        allow_fallback=False,
        local_files_only=local_files_only,
    )
    config = ChunkingConfig()
    chunk_buffer: list[dict[str, Any]] = []
    chunk_shards = new_chunk_count = 0
    for asset in sorted(new_assets.values(), key=lambda item: item["asset_id"]):
        for chunk in _asset_chunks(asset, tokenizer, config):
            if chunk["token_count"] > config.hard_max_tokens:
                raise RuntimeError(f"Recovered chunk exceeds token limit: {chunk['chunk_id']}")
            chunk_buffer.append(chunk)
            new_chunk_count += 1
            if len(chunk_buffer) >= records_per_shard:
                _atomic_jsonl(
                    recovered_chunks_dir / f"part-recovered-{chunk_shards:05d}.jsonl",
                    chunk_buffer,
                )
                chunk_buffer = []
                chunk_shards += 1
    if chunk_buffer:
        _atomic_jsonl(
            recovered_chunks_dir / f"part-recovered-{chunk_shards:05d}.jsonl",
            chunk_buffer,
        )
        chunk_shards += 1

    patch_buffer: list[dict[str, Any]] = []
    patch_shards = patch_count = 0
    for shard in sorted(chunks_dir.glob("part-*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if '"image_paths"' not in line:
                    continue
                chunk = json.loads(line)
                paths = list(chunk.get("image_paths") or [])
                if not paths:
                    continue
                identifiers = [
                    *(chunk.get("figure_ids") or []),
                    *(chunk.get("table_ids") or []),
                ]
                for identifier in identifiers:
                    for path in additions.get(identifier, {}).get("image_paths", ()):
                        if path not in paths:
                            paths.append(path)
                paths = sorted(set(paths))
                keys = [asset_key(chunk["pmcid"], path) for path in paths]
                patch_buffer.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "point_id": str(uuid.uuid5(POINT_NAMESPACE, chunk["chunk_id"])),
                        "image_paths": paths,
                        "asset_keys": keys,
                    }
                )
                patch_count += 1
                if len(patch_buffer) >= records_per_shard:
                    _atomic_jsonl(
                        patches_dir / f"part-{patch_shards:05d}.jsonl", patch_buffer
                    )
                    patch_buffer = []
                    patch_shards += 1
    if patch_buffer:
        _atomic_jsonl(patches_dir / f"part-{patch_shards:05d}.jsonl", patch_buffer)
        patch_shards += 1

    statistics = {
        "schema_version": "medcpt_image_index_integration_v1",
        "new_assets": len(new_assets),
        "existing_assets_extended": len(additions),
        "new_chunks": new_chunk_count,
        "new_chunk_shards": chunk_shards,
        "payload_patches": patch_count,
        "payload_patch_shards": patch_shards,
        "tokenizer_name": tokenizer.name,
    }
    temporary = output_dir / "statistics.json.partial"
    temporary.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir / "statistics.json")
    return statistics


def apply_payload_patches(
    *,
    patches_dir: Path,
    state_file: Path,
    qdrant_url: str,
    opensearch_url: str | None,
    collection_name: str = "fulltext_medcpt_ip_v1",
    index_name: str = "fulltext_bm25_v1",
    batch_size: int = 128,
    vector_only: bool = False,
) -> dict[str, Any]:
    from qdrant_client import QdrantClient, models

    if not vector_only and not opensearch_url:
        raise ValueError("opensearch_url is required unless vector_only is enabled")
    opensearch = None
    bulk = None
    if not vector_only:
        from opensearchpy import OpenSearch
        from opensearchpy.helpers import bulk as opensearch_bulk

        opensearch = OpenSearch(hosts=[opensearch_url], timeout=120)
        bulk = opensearch_bulk

    state = {"schema_version": "medcpt_image_payload_patch_v1", "completed": {}}
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    qdrant = QdrantClient(url=qdrant_url, timeout=120)
    patched = 0

    def flush(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        operations = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(
                    payload={
                        "image_paths": row["image_paths"],
                        "asset_keys": row["asset_keys"],
                    },
                    points=[row["point_id"]],
                    key="metadata",
                )
            )
            for row in rows
        ]
        _retry(
            lambda: qdrant.batch_update_points(
                collection_name=collection_name,
                update_operations=operations,
                wait=True,
            )
        )
        if not opensearch or not bulk:
            return
        actions = [
            {
                "_op_type": "update",
                "_index": index_name,
                "_id": f"{collection_name}:{row['chunk_id']}",
                "doc": {
                    "metadata": {
                        "image_paths": row["image_paths"],
                        "asset_keys": row["asset_keys"],
                    }
                },
            }
            for row in rows
        ]
        success, failures = _retry(
            lambda: bulk(
                opensearch,
                actions,
                refresh=False,
                raise_on_error=False,
                request_timeout=120,
            )
        )
        if failures or success != len(rows):
            raise RuntimeError(
                f"OpenSearch payload patch failed: success={success}, failures={failures[:3]}"
            )

    for path in sorted(patches_dir.glob("part-*.jsonl")):
        fingerprint = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        if state["completed"].get(path.name, {}).get("input") == fingerprint:
            patched += state["completed"][path.name]["records"]
            continue
        count = 0
        batch: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                batch.append(json.loads(line))
                if len(batch) >= batch_size:
                    flush(batch)
                    count += len(batch)
                    batch = []
        flush(batch)
        count += len(batch)
        patched += count
        state["completed"][path.name] = {"input": fingerprint, "records": count}
        state["total_records"] = sum(
            item["records"] for item in state["completed"].values()
        )
        state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_file.with_suffix(state_file.suffix + ".partial")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(state_file)
    if opensearch:
        opensearch.indices.refresh(index=index_name)
    return {"patched_records": patched, "completed_shards": len(state["completed"])}
