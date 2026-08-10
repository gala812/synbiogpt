from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .models import IndexDocument

POINT_NAMESPACE = uuid.UUID("e541b0a3-100d-5e15-9c79-b27c9c7288ad")
PMCID_RE = re.compile(r"PMC[0-9]+")
PMID_RE = re.compile(r"[0-9]+")

PAYLOAD_FIELDS = (
    "pmid",
    "pmcid",
    "paper_title",
    "section",
    "subsection",
    "section_path",
    "chunk_type",
    "chunk_index",
    "parent_chunk_id",
    "previous_chunk_id",
    "next_chunk_id",
    "image_paths",
    "asset_keys",
    "image_asset_ids",
    "figure_ids",
    "table_ids",
    "source_file",
    "source_spans",
    "word_count",
    "text_token_count",
    "token_count",
    "tokenizer_name",
    "parse_warnings",
    "recovery_confidence",
    "recovery_reason",
)


def load_pmid_mapping(database: Path) -> dict[str, str]:
    if not database.is_file():
        raise FileNotFoundError(f"PMID mapping database not found: {database}")

    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"paper_id_mapping", "index_metadata"}.issubset(tables):
            raise ValueError(
                "Official mapping database must contain paper_id_mapping and "
                "index_metadata tables"
            )

        metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
        snapshot = metadata.get("ncbi_snapshot_sha256", "")
        if metadata.get("mapping_source") != "NCBI PMC-ids.csv.gz":
            raise ValueError("PMID mapping source is not NCBI PMC-ids.csv.gz")
        if not re.fullmatch(r"[0-9a-f]{64}", snapshot):
            raise ValueError("PMID mapping has no valid NCBI snapshot SHA-256")
        if metadata.get("rows_without_official_pmid") != "0":
            raise ValueError("PMID mapping metadata reports missing official PMIDs")

        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(paper_id_mapping)")
        }
        required = {"pmcid", "pmid", "mapping_source", "ncbi_snapshot_sha256"}
        if not required.issubset(columns):
            raise ValueError(
                f"paper_id_mapping lacks provenance columns: {sorted(required - columns)}"
            )
        provenance = connection.execute(
            "SELECT DISTINCT mapping_source, ncbi_snapshot_sha256 "
            "FROM paper_id_mapping"
        ).fetchall()
        if provenance != [("NCBI PMC-ids.csv.gz", snapshot)]:
            raise ValueError("PMID mapping rows do not match audited NCBI provenance")

        result: dict[str, str] = {}
        seen_pmids: dict[str, str] = {}
        for raw_pmcid, raw_pmid in connection.execute(
            "SELECT pmcid, pmid FROM paper_id_mapping"
        ):
            pmcid = str(raw_pmcid or "").strip().upper()
            pmid = str(raw_pmid or "").strip()
            if not PMCID_RE.fullmatch(pmcid) or not PMID_RE.fullmatch(pmid):
                raise ValueError(f"Invalid official mapping row: {pmcid!r}, {pmid!r}")
            previous = result.setdefault(pmcid, pmid)
            if previous != pmid:
                raise ValueError(f"Conflicting PMIDs for {pmcid}: {previous}, {pmid}")
            prior_pmcid = seen_pmids.setdefault(pmid, pmcid)
            if prior_pmcid != pmcid:
                raise ValueError(f"PMID {pmid} maps to multiple PMCIDs")
        if not result:
            raise ValueError(f"No official PMID mappings found in {database}")
        return result
    finally:
        connection.close()


def embedding_text(chunk: dict[str, Any]) -> str:
    title = str(chunk.get("paper_title") or "").strip()
    section = str(chunk.get("section") or "Unassigned").strip()
    subsection = str(chunk.get("subsection") or "").strip()
    section_path = f"{section} > {subsection}" if subsection else section
    text = str(chunk.get("text") or "").strip()
    return f"Title: {title}\nSection: {section_path}\nText: {text}"


def make_index_document(
    chunk: dict[str, Any],
    *,
    pmid_by_pmcid: dict[str, str],
    source_shard: str,
    max_tokens: int,
) -> IndexDocument:
    chunk_id = str(chunk.get("chunk_id") or "").strip()
    pmcid = str(chunk.get("pmcid") or chunk.get("doc_id") or "").strip().upper()
    text = str(chunk.get("text") or "").strip()
    if not chunk_id:
        raise ValueError("chunk is missing chunk_id")
    if not PMCID_RE.fullmatch(pmcid):
        raise ValueError(f"chunk {chunk_id} has invalid PMCID: {pmcid!r}")
    if not text:
        raise ValueError(f"chunk {chunk_id} has empty text")

    mapped_pmid = pmid_by_pmcid.get(pmcid)
    if not mapped_pmid:
        raise ValueError(f"chunk {chunk_id} has no official PMID mapping for {pmcid}")
    embedded_pmid = str(chunk.get("pmid") or "").strip()
    if embedded_pmid and embedded_pmid != mapped_pmid:
        raise ValueError(
            f"chunk {chunk_id} PMID {embedded_pmid} conflicts with official {mapped_pmid}"
        )

    token_count = chunk.get("token_count")
    if token_count is None:
        raise ValueError(f"chunk {chunk_id} is missing token_count")
    if int(token_count) > max_tokens:
        raise ValueError(
            f"chunk {chunk_id} has {token_count} embedding tokens; maximum is {max_tokens}"
        )

    metadata = {key: chunk.get(key) for key in PAYLOAD_FIELDS if key in chunk}
    metadata.update(
        {
            "chunk_id": chunk_id,
            "doc_id": pmcid,
            "pmid": mapped_pmid,
            "pmcid": pmcid,
            "source_shard": source_shard,
        }
    )
    metadata = {key: value for key, value in metadata.items() if value is not None}
    point_id = str(uuid.uuid5(POINT_NAMESPACE, chunk_id))
    return IndexDocument(
        point_id=point_id,
        chunk_id=chunk_id,
        text=text,
        embedding_text=embedding_text(chunk),
        metadata=metadata,
    )
