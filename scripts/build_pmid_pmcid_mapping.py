#!/usr/bin/env python3
"""Build an audited PMID <-> PMCID crosswalk for the SynBioGPT corpora.

Only mappings explicitly present in NCBI's PMC-ids.csv.gz are accepted.  The
script intentionally does not infer mappings from titles, DOIs, or document
text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PMCID_RE = re.compile(r"^PMC[0-9]+$")
PMID_RE = re.compile(r"^[0-9]+$")


def clean_field(value: object) -> str:
    return str(value or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def normalize_pmcid(value: object) -> str | None:
    text = clean_field(value).upper()
    return text if PMCID_RE.fullmatch(text) else None


def normalize_pmid(value: object) -> str | None:
    text = clean_field(value)
    return text if PMID_RE.fullmatch(text) else None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_ncbi(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(
        args.url,
        headers={"User-Agent": "SynBioGPT-ID-Audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
        metadata = {
            "url": args.url,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_modified": response.headers.get("Last-Modified"),
            "content_length": response.headers.get("Content-Length"),
        }
    partial.replace(output)
    metadata["size_bytes"] = output.stat().st_size
    metadata["sha256"] = sha256_file(output)
    write_json(output.with_suffix(output.suffix + ".source.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def extract_mineru(args: argparse.Namespace) -> None:
    inputs = sorted(Path().glob(args.glob)) if not Path(args.glob).is_absolute() else sorted(Path(args.glob).parent.glob(Path(args.glob).name))
    if not inputs:
        raise SystemExit(f"No MinerU JSONL files matched: {args.glob}")

    records: dict[str, tuple[str, int, str]] = {}
    occurrences: Counter[str] = Counter()
    total = invalid_json = invalid_pmcid = 0
    for path in inputs:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                pmcid = normalize_pmcid(item.get("id") or (item.get("metadata") or {}).get("doc_id"))
                if not pmcid:
                    invalid_pmcid += 1
                    continue
                occurrences[pmcid] += 1
                records.setdefault(pmcid, (path.name, line_number, clean_field(item.get("title"))))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writerow(["pmcid", "fulltext_shard", "line_number", "title", "occurrences"])
        for pmcid in sorted(records, key=lambda value: int(value[3:])):
            shard, line_number, title = records[pmcid]
            writer.writerow([pmcid, shard, line_number, title, occurrences[pmcid]])

    report = {
        "input_files": len(inputs),
        "total_nonempty_records": total,
        "unique_valid_pmcids": len(records),
        "duplicate_records": sum(count - 1 for count in occurrences.values()),
        "duplicate_pmcids": sum(count > 1 for count in occurrences.values()),
        "invalid_json_records": invalid_json,
        "invalid_pmcid_records": invalid_pmcid,
        "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".audit.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def extract_pdfs(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"PDF root is not a directory: {root}")
    records: dict[str, str] = {}
    occurrences: Counter[str] = Counter()
    total_pdfs = invalid_names = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith(".pdf"):
                continue
            total_pdfs += 1
            pmcid = normalize_pmcid(Path(filename).stem)
            if not pmcid:
                invalid_names += 1
                continue
            path = Path(directory, filename)
            relative_path = path.relative_to(root).as_posix()
            occurrences[pmcid] += 1
            records.setdefault(pmcid, relative_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writerow(["pmcid", "fulltext_shard", "line_number", "title", "occurrences"])
        for pmcid in sorted(records, key=lambda value: int(value[3:])):
            writer.writerow([pmcid, records[pmcid], 0, "", occurrences[pmcid]])
    report = {
        "root": str(root),
        "total_pdf_files": total_pdfs,
        "unique_valid_pmcids": len(records),
        "duplicate_pdf_files": sum(count - 1 for count in occurrences.values()),
        "duplicate_pmcids": sum(count > 1 for count in occurrences.values()),
        "invalid_pdf_filenames": invalid_names,
        "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".audit.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_collection_spec(value: str) -> tuple[str, Path]:
    try:
        collection, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected COLLECTION=JSONL_PATH") from error
    return collection, Path(path)


def extract_specter(args: argparse.Namespace) -> None:
    records: list[tuple[str, str, str, int]] = []
    occurrences: Counter[str] = Counter()
    invalid_json = invalid_pmid = total = 0
    for collection, path in args.collection:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                pmid = normalize_pmid(item.get("id"))
                if not pmid:
                    invalid_pmid += 1
                    continue
                occurrences[pmid] += 1
                records.append((pmid, collection, path.name, line_number))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writerow(["pmid", "specter_collection", "source_file", "line_number", "occurrences"])
        for pmid, collection, source_file, line_number in sorted(records, key=lambda row: (int(row[0]), row[1])):
            writer.writerow([pmid, collection, source_file, line_number, occurrences[pmid]])

    report = {
        "collections": len(args.collection),
        "total_nonempty_records": total,
        "valid_records": len(records),
        "unique_valid_pmids": len(occurrences),
        "duplicate_records": sum(count - 1 for count in occurrences.values()),
        "duplicate_pmids": sum(count > 1 for count in occurrences.values()),
        "invalid_json_records": invalid_json,
        "invalid_pmid_records": invalid_pmid,
        "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".audit.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def extract_chroma(args: argparse.Namespace) -> None:
    database = Path(args.database)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows: list[dict[str, object]] = []
    invalid_pmids = 0
    collection_stats: dict[str, dict[str, int]] = {}
    for name in args.collection:
        collection = connection.execute("SELECT id FROM collections WHERE name = ?", (name,)).fetchone()
        if not collection:
            raise SystemExit(f"Chroma collection not found: {name}")
        segment = connection.execute(
            "SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA'",
            (collection["id"],),
        ).fetchone()
        if not segment:
            raise SystemExit(f"Metadata segment not found for Chroma collection: {name}")
        embedding_count = connection.execute(
            "SELECT count(*) FROM embeddings WHERE segment_id = ?", (segment["id"],)
        ).fetchone()[0]
        grouped = connection.execute(
            "SELECT m.string_value AS pmid, count(*) AS vector_chunks "
            "FROM embeddings e JOIN embedding_metadata m ON m.id = e.id "
            "WHERE e.segment_id = ? AND m.key = 'doc_id' GROUP BY m.string_value",
            (segment["id"],),
        )
        valid_unique = 0
        for item in grouped:
            pmid = normalize_pmid(item["pmid"])
            if not pmid:
                invalid_pmids += 1
                continue
            valid_unique += 1
            rows.append({"pmid": pmid, "specter_collection": name, "vector_chunks": item["vector_chunks"]})
        collection_stats[name] = {
            "vector_chunks": embedding_count,
            "unique_valid_pmids": valid_unique,
        }
    connection.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        columns = ["pmid", "specter_collection", "vector_chunks"]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (int(str(row["pmid"])), str(row["specter_collection"]))))
    pair_counts = Counter(str(row["pmid"]) for row in rows)
    report = {
        "database": str(database),
        "collections": collection_stats,
        "total_vector_chunks": sum(item["vector_chunks"] for item in collection_stats.values()),
        "unique_collection_pmid_pairs": len(rows),
        "unique_pmids": len(pair_counts),
        "pmids_in_multiple_collections": sum(count > 1 for count in pair_counts.values()),
        "invalid_doc_id_pmids": invalid_pmids,
        "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".audit.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", escapechar="\\"))


def map_official(args: argparse.Namespace) -> None:
    inventory_rows = load_tsv(Path(args.inventory))
    inventory_by_pmcid = {row["pmcid"]: row for row in inventory_rows}
    target_pmcids = set(inventory_by_pmcid)
    official_by_pmcid: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    rows_scanned = target_rows = 0
    ncbi_path = Path(args.ncbi)
    with gzip.open(ncbi_path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "PMCID" not in reader.fieldnames or "PMID" not in reader.fieldnames:
            raise SystemExit(f"NCBI file lacks PMCID/PMID columns: {reader.fieldnames}")
        for row in reader:
            rows_scanned += 1
            pmcid = normalize_pmcid(row.get("PMCID"))
            if pmcid not in target_pmcids:
                continue
            target_rows += 1
            pmid = normalize_pmid(row.get("PMID"))
            if not pmid:
                continue
            previous = official_by_pmcid.setdefault(pmcid, pmid)
            if previous != pmid:
                conflicts.append((pmcid, previous, pmid))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        columns = ["pmcid", "pmid", "mapping_status", "source_path", "occurrences"]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writeheader()
        for pmcid in sorted(target_pmcids, key=lambda value: int(value[3:])):
            source = inventory_by_pmcid[pmcid]
            pmid = official_by_pmcid.get(pmcid, "")
            writer.writerow(
                {
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "mapping_status": "exact_ncbi_match" if pmid else "pmcid_without_official_pmid",
                    "source_path": source.get("fulltext_shard", ""),
                    "occurrences": source.get("occurrences", "1"),
                }
            )
    report = {
        "policy": "Only explicit PMCID/PMID pairs from NCBI PMC-ids.csv.gz; no title or DOI inference.",
        "inventory_unique_pmcids": len(target_pmcids),
        "ncbi_rows_scanned": rows_scanned,
        "ncbi_rows_for_inventory": target_rows,
        "exact_ncbi_matches": len(official_by_pmcid),
        "without_official_pmid": len(target_pmcids - set(official_by_pmcid)),
        "official_mapping_conflicts": len(conflicts),
        "ncbi_sha256": sha256_file(ncbi_path),
        "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".audit.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_lookup_index(args: argparse.Namespace) -> None:
    mapping_path = Path(args.mapping).resolve()
    audit_path = Path(args.audit).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing index: {output}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("official_mapping_conflicts") != 0:
        raise SystemExit("Mapping audit contains official PMCID/PMID conflicts")
    ncbi_sha256 = clean_field(audit.get("ncbi_sha256"))
    if not re.fullmatch(r"[0-9a-f]{64}", ncbi_sha256):
        raise SystemExit("Mapping audit does not contain a valid NCBI SHA-256")

    rows = load_tsv(mapping_path)
    accepted: list[tuple[str, str, str, int, str, str, str]] = []
    missing_pmids = 0
    pmids: set[str] = set()
    pmcids: set[str] = set()
    for row in rows:
        pmcid = normalize_pmcid(row.get("pmcid"))
        pmid = normalize_pmid(row.get("pmid"))
        if not pmcid:
            raise SystemExit(f"Invalid PMCID in mapping artifact: {row.get('pmcid')!r}")
        if not pmid:
            missing_pmids += 1
            continue
        if pmid in pmids:
            raise SystemExit(f"Duplicate PMID in official mapping artifact: {pmid}")
        if pmcid in pmcids:
            raise SystemExit(f"Duplicate PMCID in official mapping artifact: {pmcid}")
        pmids.add(pmid)
        pmcids.add(pmcid)
        accepted.append(
            (
                pmid,
                pmcid,
                clean_field(row.get("fulltext_shard")),
                int(row.get("line_number") or 0),
                clean_field(row.get("title")),
                "NCBI PMC-ids.csv.gz",
                ncbi_sha256,
            )
        )

    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE paper_id_mapping ("
        "pmid TEXT NOT NULL PRIMARY KEY CHECK(pmid GLOB '[0-9]*'), "
        "pmcid TEXT NOT NULL UNIQUE CHECK(pmcid GLOB 'PMC[0-9]*'), "
        "fulltext_shard TEXT NOT NULL, line_number INTEGER NOT NULL, title TEXT NOT NULL, "
        "mapping_source TEXT NOT NULL, ncbi_snapshot_sha256 TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute("CREATE INDEX paper_id_mapping_pmcid_idx ON paper_id_mapping(pmcid)")
    connection.execute(
        "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.executemany("INSERT INTO paper_id_mapping VALUES (?, ?, ?, ?, ?, ?, ?)", accepted)
    metadata = {
        "schema_version": "1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_source": "NCBI PMC-ids.csv.gz",
        "ncbi_snapshot_sha256": ncbi_sha256,
        "source_mapping_path": str(mapping_path),
        "source_mapping_sha256": sha256_file(mapping_path),
        "input_rows": str(len(rows)),
        "indexed_rows": str(len(accepted)),
        "rows_without_official_pmid": str(missing_pmids),
    }
    connection.executemany("INSERT INTO index_metadata VALUES (?, ?)", metadata.items())
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    indexed = connection.execute("SELECT count(*) FROM paper_id_mapping").fetchone()[0]
    unique_pmids = connection.execute("SELECT count(DISTINCT pmid) FROM paper_id_mapping").fetchone()[0]
    unique_pmcids = connection.execute("SELECT count(DISTINCT pmcid) FROM paper_id_mapping").fetchone()[0]
    connection.close()
    if integrity != "ok" or indexed != len(accepted) or unique_pmids != indexed or unique_pmcids != indexed:
        raise SystemExit(
            f"Index verification failed: integrity={integrity}, rows={indexed}, "
            f"unique_pmids={unique_pmids}, unique_pmcids={unique_pmcids}"
        )
    temporary.replace(output)
    report = {
        "index": str(output),
        "size_bytes": output.stat().st_size,
        "index_sha256": sha256_file(output),
        "integrity_check": integrity,
        "input_rows": len(rows),
        "indexed_rows": indexed,
        "unique_pmids": unique_pmids,
        "unique_pmcids": unique_pmcids,
        "rows_without_official_pmid": missing_pmids,
        "mapping_source": "NCBI PMC-ids.csv.gz",
        "ncbi_snapshot_sha256": ncbi_sha256,
    }
    write_json(output.with_suffix(output.suffix + ".audit.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def lookup_index(args: argparse.Namespace) -> None:
    connection = sqlite3.connect(f"file:{Path(args.index).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    results = []
    for pmid in args.pmid or []:
        normalized = normalize_pmid(pmid)
        if not normalized:
            raise SystemExit(f"Invalid PMID: {pmid!r}")
        row = connection.execute("SELECT * FROM paper_id_mapping WHERE pmid = ?", (normalized,)).fetchone()
        results.append({"query_type": "pmid", "query": normalized, "match": dict(row) if row else None})
    for pmcid in args.pmcid or []:
        normalized = normalize_pmcid(pmcid)
        if not normalized:
            raise SystemExit(f"Invalid PMCID: {pmcid!r}")
        row = connection.execute("SELECT * FROM paper_id_mapping WHERE pmcid = ?", (normalized,)).fetchone()
        results.append({"query_type": "pmcid", "query": normalized, "match": dict(row) if row else None})
    connection.close()
    print(json.dumps(results, ensure_ascii=False, indent=2))


def build_mapping(args: argparse.Namespace) -> None:
    fulltext_rows = load_tsv(Path(args.fulltext))
    specter_rows = load_tsv(Path(args.specter))
    chroma_validation: dict[str, object] | None = None
    if args.chroma:
        chroma_rows = load_tsv(Path(args.chroma))
        source_pairs = {(row["pmid"], row["specter_collection"]) for row in specter_rows}
        chroma_pairs = {(row["pmid"], row["specter_collection"]) for row in chroma_rows}
        source_only = source_pairs - chroma_pairs
        chroma_only = chroma_pairs - source_pairs
        chroma_validation = {
            "source_collection_pmid_pairs": len(source_pairs),
            "chroma_collection_pmid_pairs": len(chroma_pairs),
            "source_pairs_missing_from_chroma": len(source_only),
            "chroma_pairs_missing_from_source": len(chroma_only),
            "exact_membership_match": not source_only and not chroma_only,
            "total_chroma_vector_chunks": sum(int(row["vector_chunks"]) for row in chroma_rows),
        }
        if source_only or chroma_only:
            raise SystemExit(f"Chroma/source PMID membership mismatch: {chroma_validation}")
    target_pmcids = {row["pmcid"] for row in fulltext_rows}
    specter_by_pmid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in specter_rows:
        specter_by_pmid[row["pmid"]].append(row)

    official_by_pmcid: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    official_rows_scanned = official_target_rows = 0
    ncbi_path = Path(args.ncbi)
    with gzip.open(ncbi_path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "PMCID" not in reader.fieldnames or "PMID" not in reader.fieldnames:
            raise SystemExit(f"NCBI file lacks PMCID/PMID columns: {reader.fieldnames}")
        for row in reader:
            official_rows_scanned += 1
            pmcid = normalize_pmcid(row.get("PMCID"))
            if pmcid not in target_pmcids:
                continue
            official_target_rows += 1
            pmid = normalize_pmid(row.get("PMID"))
            if not pmid:
                continue
            previous = official_by_pmcid.setdefault(pmcid, pmid)
            if previous != pmid:
                conflicts.append((pmcid, previous, pmid))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fulltext_by_pmcid = {row["pmcid"]: row for row in fulltext_rows}
    exact_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    matched_specter_pmids: set[str] = set()

    for pmcid in sorted(target_pmcids, key=lambda value: int(value[3:])):
        pmid = official_by_pmcid.get(pmcid, "")
        matches = specter_by_pmid.get(pmid, []) if pmid else []
        if not pmid:
            status = "pmcid_without_official_pmid"
        elif not matches:
            status = "official_pmid_not_in_target_specter"
        else:
            status = "exact_ncbi_match"
            matched_specter_pmids.add(pmid)
        base = fulltext_by_pmcid[pmcid]
        row = {
            "pmcid": pmcid,
            "pmid": pmid,
            "mapping_status": status,
            "specter_collections": ";".join(sorted({item["specter_collection"] for item in matches})),
            "fulltext_shard": base["fulltext_shard"],
            "line_number": base["line_number"],
            "title": base["title"],
        }
        all_rows.append(row)
        if status == "exact_ncbi_match":
            exact_rows.append(row)

    columns = ["pmcid", "pmid", "mapping_status", "specter_collections", "fulltext_shard", "line_number", "title"]
    for filename, rows in (("pmid_pmcid_all.tsv", all_rows), ("pmid_pmcid_exact.tsv", exact_rows)):
        with (outdir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", escapechar="\\")
            writer.writeheader()
            writer.writerows(rows)

    unmatched_specter = [row for row in specter_rows if row["pmid"] not in matched_specter_pmids]
    with (outdir / "specter_without_exact_fulltext.tsv").open("w", encoding="utf-8", newline="") as handle:
        columns_unmatched = ["pmid", "specter_collection", "source_file", "line_number", "occurrences"]
        writer = csv.DictWriter(handle, fieldnames=columns_unmatched, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writeheader()
        writer.writerows(unmatched_specter)

    with (outdir / "official_mapping_conflicts.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writerow(["pmcid", "first_pmid", "conflicting_pmid"])
        writer.writerows(conflicts)

    database = outdir / "pmid_pmcid_exact.sqlite3"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE mapping (pmid TEXT NOT NULL, pmcid TEXT NOT NULL, "
        "specter_collections TEXT NOT NULL, fulltext_shard TEXT NOT NULL, "
        "line_number INTEGER NOT NULL, title TEXT NOT NULL, "
        "PRIMARY KEY (pmid, pmcid))"
    )
    connection.executemany(
        "INSERT INTO mapping VALUES (?, ?, ?, ?, ?, ?)",
        [
            (row["pmid"], row["pmcid"], row["specter_collections"], row["fulltext_shard"], int(row["line_number"]), row["title"])
            for row in exact_rows
        ],
    )
    connection.execute("CREATE INDEX mapping_pmcid_idx ON mapping(pmcid)")
    connection.commit()
    connection.close()

    status_counts = Counter(row["mapping_status"] for row in all_rows)
    unique_specter_pmids = set(specter_by_pmid)
    audit = {
        "policy": "Only exact PMCID/PMID pairs from NCBI PMC-ids.csv.gz are accepted; no inferred mapping.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ncbi_file": str(ncbi_path),
        "ncbi_sha256": sha256_file(ncbi_path),
        "ncbi_rows_scanned": official_rows_scanned,
        "ncbi_rows_for_target_pmcids": official_target_rows,
        "fulltext_unique_pmcids": len(target_pmcids),
        "specter_records": len(specter_rows),
        "specter_unique_pmids": len(unique_specter_pmids),
        "exact_ncbi_matches": len(exact_rows),
        "unique_exact_pmids": len({row["pmid"] for row in exact_rows}),
        "unique_exact_pmcids": len({row["pmcid"] for row in exact_rows}),
        "specter_unique_pmids_without_exact_fulltext": len(unique_specter_pmids - matched_specter_pmids),
        "official_mapping_conflicts": len(conflicts),
        "chroma_source_validation": chroma_validation,
        "fulltext_status_counts": dict(sorted(status_counts.items())),
        "outputs": {
            "exact_tsv": str(outdir / "pmid_pmcid_exact.tsv"),
            "all_tsv": str(outdir / "pmid_pmcid_all.tsv"),
            "sqlite": str(database),
        },
    }
    write_json(outdir / "mapping_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-ncbi", help="Download and fingerprint NCBI PMC-ids.csv.gz")
    download.add_argument("--url", default="https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz")
    download.add_argument("--output", required=True)
    download.set_defaults(func=download_ncbi)

    mineru = subparsers.add_parser("extract-mineru", help="Extract unique PMCIDs from MinerU JSONL shards")
    mineru.add_argument("--glob", required=True)
    mineru.add_argument("--output", required=True)
    mineru.set_defaults(func=extract_mineru)

    pdfs = subparsers.add_parser("extract-pdfs", help="Extract unique PMCIDs from actual PDF filenames")
    pdfs.add_argument("--root", required=True)
    pdfs.add_argument("--output", required=True)
    pdfs.set_defaults(func=extract_pdfs)

    specter = subparsers.add_parser("extract-specter", help="Extract PMIDs from target SPECTER JSONL files")
    specter.add_argument("--collection", action="append", required=True, type=parse_collection_spec)
    specter.add_argument("--output", required=True)
    specter.set_defaults(func=extract_specter)

    chroma = subparsers.add_parser("extract-chroma", help="Extract PMID membership and chunk counts from Chroma SQLite")
    chroma.add_argument("--database", required=True)
    chroma.add_argument("--collection", action="append", required=True)
    chroma.add_argument("--output", required=True)
    chroma.set_defaults(func=extract_chroma)

    official = subparsers.add_parser("map-official", help="Map an inventory of PMCIDs to PMIDs using only NCBI")
    official.add_argument("--inventory", required=True)
    official.add_argument("--ncbi", required=True)
    official.add_argument("--output", required=True)
    official.set_defaults(func=map_official)

    index = subparsers.add_parser("build-index", help="Build a verified bidirectional PMID/PMCID SQLite index")
    index.add_argument("--mapping", required=True)
    index.add_argument("--audit", required=True)
    index.add_argument("--output", required=True)
    index.set_defaults(func=build_lookup_index)

    lookup = subparsers.add_parser("lookup", help="Query a PMID/PMCID SQLite index")
    lookup.add_argument("--index", required=True)
    lookup.add_argument("--pmid", action="append")
    lookup.add_argument("--pmcid", action="append")
    lookup.set_defaults(func=lookup_index)

    mapping = subparsers.add_parser("build", help="Build exact NCBI mapping and audit artifacts")
    mapping.add_argument("--fulltext", required=True)
    mapping.add_argument("--specter", required=True)
    mapping.add_argument("--chroma", help="Optional Chroma membership TSV for strict source validation")
    mapping.add_argument("--ncbi", required=True)
    mapping.add_argument("--outdir", required=True)
    mapping.set_defaults(func=build_mapping)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
