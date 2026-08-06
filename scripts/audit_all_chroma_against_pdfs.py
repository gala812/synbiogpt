#!/usr/bin/env python3
"""Audit every Chroma collection against an NCBI-verified PDF PMID inventory.

The Chroma database is opened read-only. Matching is exact and considers the
identifier metadata keys ``doc_id``, ``pmid``, ``PMID`` and numeric ``title``.
No title-text or DOI inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, columns: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writerow(columns)
        writer.writerows(rows)


def load_pdf_mapping(path: Path) -> tuple[list[tuple[str, str, str]], int]:
    exact: list[tuple[str, str, str]] = []
    missing = 0
    seen_pmcids: set[str] = set()
    seen_pmids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t", escapechar="\\"):
            pmcid = row.get("pmcid", "")
            pmid = row.get("pmid", "")
            if not pmid:
                missing += 1
                continue
            if pmcid in seen_pmcids:
                raise SystemExit(f"Duplicate PMCID in PDF mapping: {pmcid}")
            if pmid in seen_pmids:
                raise SystemExit(f"Duplicate PMID in PDF mapping: {pmid}")
            seen_pmcids.add(pmcid)
            seen_pmids.add(pmid)
            exact.append((pmid, pmcid, row.get("source_path", "")))
    return exact, missing


def folder_size(path: Path) -> tuple[int, int]:
    size = files = 0
    for directory, _, filenames in os.walk(path):
        for filename in filenames:
            files += 1
            try:
                size += Path(directory, filename).stat().st_size
            except FileNotFoundError:
                pass
    return size, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--vector-dir", required=True)
    parser.add_argument("--pdf-mapping", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    database = Path(args.database).resolve()
    vector_dir = Path(args.vector_dir).resolve()
    mapping_path = Path(args.pdf_mapping).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    targets, mappings_without_pmid = load_pdf_mapping(mapping_path)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TEMP TABLE target_pmids (pmid TEXT PRIMARY KEY, pmcid TEXT NOT NULL, pdf_path TEXT NOT NULL)")
    connection.executemany("INSERT INTO target_pmids VALUES (?, ?, ?)", targets)

    # Materialize exact matches once. DISTINCT/GROUP BY collapses duplicate ID
    # metadata (for example, a PMID repeated in both doc_id and title).
    connection.execute(
        "CREATE TEMP TABLE matched_embeddings AS "
        "SELECT e.id AS embedding_row_id, e.embedding_id, c.id AS collection_id, "
        "c.name AS collection_name, t.pmid, t.pmcid, "
        "group_concat(DISTINCT identifier.key) AS identifier_keys, "
        "max(config.string_value) AS embedding_config "
        "FROM embedding_metadata identifier "
        "JOIN target_pmids t ON t.pmid = identifier.string_value "
        "JOIN embeddings e ON e.id = identifier.id "
        "JOIN segments s ON s.id = e.segment_id "
        "JOIN collections c ON c.id = s.collection "
        "LEFT JOIN embedding_metadata config ON config.id = e.id AND config.key = 'embedding_config' "
        "WHERE identifier.key IN ('doc_id', 'pmid', 'PMID', 'title') "
        "GROUP BY e.id, e.embedding_id, c.id, c.name, t.pmid, t.pmcid"
    )
    connection.execute("CREATE INDEX temp.matched_pmid_idx ON matched_embeddings(pmid)")
    connection.execute("CREATE INDEX temp.matched_collection_idx ON matched_embeddings(collection_id)")

    match_rows = list(
        connection.execute(
            "SELECT pmcid, pmid, collection_name, collection_id, count(*) AS vector_chunks, "
            "group_concat(DISTINCT identifier_keys) AS identifier_keys, "
            "group_concat(DISTINCT coalesce(embedding_config, '')) AS embedding_configs "
            "FROM matched_embeddings GROUP BY pmcid, pmid, collection_id, collection_name "
            "ORDER BY CAST(substr(pmcid, 4) AS INTEGER), collection_name"
        )
    )
    write_tsv(
        outdir / "pdf_collection_matches.tsv",
        ["pmcid", "pmid", "collection_name", "collection_id", "vector_chunks", "identifier_keys", "embedding_configs"],
        ([row[column] for column in row.keys()] for row in match_rows),
    )

    coverage_rows = list(
        connection.execute(
            "SELECT t.pmcid, t.pmid, t.pdf_path, count(DISTINCT m.collection_id) AS collection_count, "
            "count(m.embedding_row_id) AS vector_chunks, "
            "group_concat(DISTINCT m.collection_name) AS collections "
            "FROM target_pmids t LEFT JOIN matched_embeddings m ON m.pmid = t.pmid "
            "GROUP BY t.pmcid, t.pmid, t.pdf_path ORDER BY CAST(substr(t.pmcid, 4) AS INTEGER)"
        )
    )
    write_tsv(
        outdir / "pdf_coverage.tsv",
        ["pmcid", "pmid", "pdf_path", "collection_count", "vector_chunks", "collections"],
        ([row[column] for column in row.keys()] for row in coverage_rows),
    )

    collections = list(connection.execute("SELECT id, name, dimension, config_json_str FROM collections ORDER BY name"))
    vector_segments: dict[str, list[str]] = defaultdict(list)
    metadata_segments: dict[str, list[str]] = defaultdict(list)
    for row in connection.execute("SELECT id, collection, scope FROM segments"):
        if row["scope"] == "VECTOR":
            vector_segments[row["collection"]].append(row["id"])
        elif row["scope"] == "METADATA":
            metadata_segments[row["collection"]].append(row["id"])

    total_vectors = {
        row["collection_id"]: row["total_vectors"]
        for row in connection.execute(
            "SELECT s.collection AS collection_id, count(e.id) AS total_vectors "
            "FROM segments s LEFT JOIN embeddings e ON e.segment_id = s.id "
            "WHERE s.scope = 'METADATA' GROUP BY s.collection"
        )
    }
    matched_stats = {
        row["collection_id"]: row
        for row in connection.execute(
            "SELECT collection_id, count(DISTINCT pmid) AS matched_pdfs, count(*) AS matched_vector_chunks, "
            "group_concat(DISTINCT coalesce(embedding_config, '')) AS embedding_configs "
            "FROM matched_embeddings GROUP BY collection_id"
        )
    }
    collection_rows = []
    for collection in collections:
        stats = matched_stats.get(collection["id"])
        collection_rows.append(
            (
                collection["name"],
                collection["id"],
                collection["dimension"],
                total_vectors.get(collection["id"], 0),
                stats["matched_pdfs"] if stats else 0,
                stats["matched_vector_chunks"] if stats else 0,
                stats["embedding_configs"] if stats else "",
                ";".join(vector_segments.get(collection["id"], [])),
                ";".join(metadata_segments.get(collection["id"], [])),
                collection["config_json_str"] or "",
            )
        )
    write_tsv(
        outdir / "collection_coverage.tsv",
        [
            "collection_name", "collection_id", "dimension", "total_vectors", "matched_pdfs",
            "matched_vector_chunks", "matched_embedding_configs", "vector_segments", "metadata_segments",
            "collection_config",
        ],
        collection_rows,
    )

    segment_to_collection: dict[str, tuple[str, str]] = {}
    for collection in collections:
        for segment in vector_segments.get(collection["id"], []):
            segment_to_collection[segment] = (collection["name"], collection["id"])
    physical_rows = []
    physical_dirs = []
    for entry in os.scandir(vector_dir):
        if not entry.is_dir(follow_symlinks=False) or not UUID_RE.fullmatch(entry.name):
            continue
        physical_dirs.append(entry.name)
        size, file_count = folder_size(Path(entry.path))
        registered = segment_to_collection.get(entry.name)
        physical_rows.append((entry.name, bool(registered), registered[0] if registered else "", registered[1] if registered else "", size, file_count))
    physical_rows.sort()
    write_tsv(
        outdir / "physical_vector_folders.tsv",
        ["folder_uuid", "registered_vector_segment", "collection_name", "collection_id", "size_bytes", "file_count"],
        physical_rows,
    )

    matched_pdf_count = sum(row["collection_count"] > 0 for row in coverage_rows)
    multi_collection_pdfs = sum(row["collection_count"] > 1 for row in coverage_rows)
    matched_config_counts = Counter()
    for row in connection.execute(
        "SELECT coalesce(embedding_config, '') AS config, count(*) AS count FROM matched_embeddings GROUP BY embedding_config"
    ):
        matched_config_counts[row["config"]] = row["count"]
    matched_embedding_rows = connection.execute("SELECT count(*) FROM matched_embeddings").fetchone()[0]
    registered_physical = sum(bool(row[1]) for row in physical_rows)
    audit = {
        "policy": "Exact NCBI-verified PDF PMID matched against all Chroma collections; no fuzzy inference.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(database),
        "database_size_bytes": database.stat().st_size,
        "database_mtime_utc": datetime.fromtimestamp(database.stat().st_mtime, timezone.utc).isoformat(),
        "pdf_mapping": str(mapping_path),
        "pdf_mapping_sha256": sha256_file(mapping_path),
        "pdfs_with_official_pmid": len(targets),
        "pdfs_without_official_pmid": mappings_without_pmid,
        "pdfs_matched_in_any_collection": matched_pdf_count,
        "pdfs_not_matched_in_any_collection": len(targets) - matched_pdf_count,
        "pdfs_matched_in_multiple_collections": multi_collection_pdfs,
        "matched_embedding_rows": matched_embedding_rows,
        "total_collections": len(collections),
        "collections_with_vectors": sum(total_vectors.get(row["id"], 0) > 0 for row in collections),
        "collections_matching_any_pdf": len(matched_stats),
        "matched_embedding_configs": dict(matched_config_counts),
        "physical_uuid_folders": len(physical_dirs),
        "registered_physical_vector_folders": registered_physical,
        "orphan_physical_vector_folders": len(physical_dirs) - registered_physical,
        "registered_vector_segments_without_folder": len(set(segment_to_collection) - set(physical_dirs)),
    }
    (outdir / "all_collections_pdf_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
