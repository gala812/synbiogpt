#!/usr/bin/env python3
"""Audit a deterministic sample of MinerU full texts against all Chroma collections."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import sqlite3
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", escapechar="\\"))


def write_tsv(path: Path, columns: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", escapechar="\\")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    inventory = {row["pmcid"]: row for row in read_tsv(Path(args.inventory))}
    mapping = {
        row["pmcid"]: row["pmid"]
        for row in read_tsv(Path(args.mapping))
        # Every non-empty PMID in the mapping artifact came directly from the
        # NCBI snapshot. Do not prefilter on SPECTER/Chroma availability, or
        # the validation sample would be biased toward known matches.
        if row.get("pmid")
    }
    candidates = sorted(set(inventory) & set(mapping), key=lambda value: int(value[3:]))
    if len(candidates) < args.sample_size:
        raise SystemExit(f"Only {len(candidates)} exact candidates for sample size {args.sample_size}")
    selected = sorted(random.Random(args.seed).sample(candidates, args.sample_size), key=lambda value: int(value[3:]))
    rows = []
    for pmcid in selected:
        item = inventory[pmcid]
        rows.append(
            {
                "pmcid": pmcid,
                "pmid": mapping[pmcid],
                "fulltext_shard": item["fulltext_shard"],
                "line_number": item["line_number"],
                "mineru_title": item["title"],
            }
        )
    write_tsv(Path(args.output), list(rows[0]), rows)
    print(json.dumps({"sample_size": len(rows), "seed": args.seed, "output": args.output}, ensure_ascii=False, indent=2))


def extract_fulltext(args: argparse.Namespace) -> None:
    targets = read_tsv(Path(args.targets))
    by_shard: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in targets:
        by_shard[row["fulltext_shard"]][int(row["line_number"])] = row
    found: dict[str, dict] = {}
    root = Path(args.jsonl_root)
    for shard, line_targets in by_shard.items():
        with (root / shard).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                target = line_targets.get(line_number)
                if not target:
                    continue
                item = json.loads(line)
                if item.get("id") != target["pmcid"]:
                    raise SystemExit(
                        f"Inventory location mismatch at {shard}:{line_number}: {item.get('id')} != {target['pmcid']}"
                    )
                item["pmid"] = target["pmid"]
                item["fulltext_shard"] = shard
                item["line_number"] = line_number
                found[target["pmcid"]] = item
                if len(found) == len(targets):
                    break
    missing = sorted(set(row["pmcid"] for row in targets) - set(found))
    if missing:
        raise SystemExit(f"Failed to retrieve sampled full texts: {missing}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in targets:
            handle.write(json.dumps(found[row["pmcid"]], ensure_ascii=False) + "\n")
    print(json.dumps({"retrieved": len(found), "output": str(output)}, ensure_ascii=False, indent=2))


def query_chroma(args: argparse.Namespace) -> None:
    targets = read_tsv(Path(args.targets))
    connection = sqlite3.connect(f"file:{Path(args.database).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TEMP TABLE target_pmids (pmid TEXT PRIMARY KEY, pmcid TEXT NOT NULL)")
    connection.executemany("INSERT INTO target_pmids VALUES (?, ?)", [(row["pmid"], row["pmcid"]) for row in targets])
    connection.execute(
        "CREATE TEMP TABLE sample_matches AS "
        "SELECT e.id AS row_id, e.embedding_id, c.name AS collection_name, c.id AS collection_id, "
        "t.pmid, t.pmcid, group_concat(DISTINCT identifier.key) AS identifier_keys "
        "FROM embedding_metadata identifier "
        "JOIN target_pmids t ON t.pmid = identifier.string_value "
        "JOIN embeddings e ON e.id = identifier.id "
        "JOIN segments s ON s.id = e.segment_id "
        "JOIN collections c ON c.id = s.collection "
        "WHERE identifier.key IN ('doc_id', 'pmid', 'PMID', 'title') "
        "GROUP BY e.id, e.embedding_id, c.name, c.id, t.pmid, t.pmcid"
    )
    query = connection.execute(
        "SELECT sm.*, "
        "max(CASE WHEN m.key = 'chroma:document' THEN m.string_value END) AS document, "
        "max(CASE WHEN m.key = 'embedding_config' THEN m.string_value END) AS embedding_config, "
        "max(CASE WHEN m.key = 'start_index' THEN coalesce(m.int_value, m.string_value) END) AS start_index "
        "FROM sample_matches sm LEFT JOIN embedding_metadata m ON m.id = sm.row_id "
        "GROUP BY sm.row_id, sm.embedding_id, sm.collection_name, sm.collection_id, sm.pmid, sm.pmcid, sm.identifier_keys "
        "ORDER BY CAST(sm.pmid AS INTEGER), sm.collection_name, sm.row_id"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in query:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    connection.close()
    print(json.dumps({"sample_pmids": len(targets), "matched_vector_chunks": count, "output": str(output)}, ensure_ascii=False, indent=2))


def query_uploads(args: argparse.Namespace) -> None:
    targets = {row["pmid"]: row["pmcid"] for row in read_tsv(Path(args.targets))}
    inputs = sorted(glob.glob(args.glob))
    matches = []
    availability = {pmid: {"abstract": False, "title": False} for pmid in targets}
    invalid_json = total_records = 0
    for filename in inputs:
        path = Path(filename)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                total_records += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                metadata = item.get("metadata") or {}
                candidates = {
                    str(item.get("id") or ""),
                    str(metadata.get("pmid") or ""),
                    str(metadata.get("doc_id") or ""),
                }
                for pmid in candidates & targets.keys():
                    abstract = str(item.get("text") or "").strip()
                    title = str(item.get("title") or metadata.get("title") or "").strip()
                    availability[pmid]["abstract"] |= bool(abstract)
                    availability[pmid]["title"] |= bool(title)
                    matches.append(
                        {
                            "pmcid": targets[pmid],
                            "pmid": pmid,
                            "source_file": path.name,
                            "line_number": line_number,
                            "text_length": len(abstract),
                            "title": title,
                        }
                    )
    output = Path(args.output)
    columns = ["pmcid", "pmid", "source_file", "line_number", "text_length", "title"]
    write_tsv(output, columns, matches)
    found = {row["pmid"] for row in matches}
    report = {
        "input_jsonl_files": len(inputs),
        "total_records_scanned": total_records,
        "invalid_json_records": invalid_json,
        "sample_pmids": len(targets),
        "sample_pmids_found_in_uploads": len(found),
        "found_pmids_with_abstract": sum(availability[pmid]["abstract"] for pmid in found),
        "found_pmids_with_title": sum(availability[pmid]["title"] for pmid in found),
        "found_pmids_with_both_title_and_abstract": sum(
            availability[pmid]["abstract"] and availability[pmid]["title"] for pmid in found
        ),
        "sample_pmids_missing_from_uploads": len(set(targets) - found),
        "missing": [
            {"pmcid": targets[pmid], "pmid": pmid}
            for pmid in sorted(set(targets) - found, key=int)
        ],
        "match_rows": len(matches),
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def xml_text(element: ET.Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def fetch_pubmed(pmids: list[str]) -> dict[str, dict[str, str]]:
    params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": "SynBioGPTAudit"})
    request = urllib.request.Request(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + params,
        headers={"User-Agent": "SynBioGPT-ID-Audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        root = ET.fromstring(response.read())
    records: dict[str, dict[str, str]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = xml_text(article.find(".//MedlineCitation/PMID"))
        title = xml_text(article.find(".//Article/ArticleTitle"))
        abstract = " ".join(xml_text(item) for item in article.findall(".//Article/Abstract/AbstractText"))
        records[pmid] = {"title": title, "abstract": abstract}
    return records


def normalized_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


def title_similarity(left: str, right: str) -> float:
    a = " ".join(normalized_words(left))
    b = " ".join(normalized_words(right))
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def overlap_coefficient(left: str, right: str) -> float:
    a, b = set(normalized_words(left)), set(normalized_words(right))
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def analyze(args: argparse.Namespace) -> None:
    targets = read_tsv(Path(args.targets))
    fulltexts = {}
    with Path(args.fulltext).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            fulltexts[item["pmid"]] = item
    chroma: dict[str, list[dict]] = defaultdict(list)
    with Path(args.chroma).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            chroma[item["pmid"]].append(item)
    official = fetch_pubmed([row["pmid"] for row in targets])

    results = []
    for target in targets:
        pmid = target["pmid"]
        fulltext = fulltexts[pmid]
        pubmed = official.get(pmid, {"title": "", "abstract": ""})
        matches = chroma.get(pmid, [])
        title_score = title_similarity(fulltext.get("title", ""), pubmed["title"])
        abstract_scores = [overlap_coefficient(item.get("document") or "", pubmed["abstract"]) for item in matches]
        best_abstract_score = max(abstract_scores, default=0.0)
        configs = sorted({item.get("embedding_config") or "" for item in matches})
        collections = sorted({item["collection_name"] for item in matches})
        results.append(
            {
                "pmcid": target["pmcid"],
                "pmid": pmid,
                "ncbi_pubmed_record_found": bool(pubmed["title"]),
                "mineru_title_vs_pubmed_score": f"{title_score:.4f}",
                "title_consistent": title_score >= 0.80,
                "chroma_found": bool(matches),
                "chroma_vector_chunks": len(matches),
                "chroma_collections": ";".join(collections),
                "best_chroma_abstract_overlap": f"{best_abstract_score:.4f}",
                "abstract_consistent": bool(pubmed["abstract"]) and best_abstract_score >= 0.80,
                "embedding_configs": ";".join(configs),
                "mineru_title": fulltext.get("title", ""),
                "pubmed_title": pubmed["title"],
            }
        )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_tsv(outdir / "sample_100_results.tsv", list(results[0]), results)
    summary = {
        "sample_size": len(results),
        "official_pubmed_records_found": sum(row["ncbi_pubmed_record_found"] for row in results),
        "mineru_titles_consistent_with_pubmed": sum(row["title_consistent"] for row in results),
        "pmids_found_in_any_chroma_collection": sum(row["chroma_found"] for row in results),
        "pmids_missing_from_all_chroma_collections": sum(not row["chroma_found"] for row in results),
        "chroma_abstracts_consistent_with_pubmed": sum(row["abstract_consistent"] for row in results),
        "chroma_matches_without_comparable_pubmed_abstract": sum(row["chroma_found"] and not row["abstract_consistent"] for row in results),
        "title_consistency_threshold": 0.80,
        "abstract_token_overlap_threshold": 0.80,
        "result_tsv": str(outdir / "sample_100_results.tsv"),
    }
    (outdir / "sample_100_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--inventory", required=True)
    command.add_argument("--mapping", required=True)
    command.add_argument("--sample-size", type=int, default=100)
    command.add_argument("--seed", type=int, default=20260805)
    command.add_argument("--output", required=True)
    command.set_defaults(func=prepare)
    command = commands.add_parser("extract-fulltext")
    command.add_argument("--targets", required=True)
    command.add_argument("--jsonl-root", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=extract_fulltext)
    command = commands.add_parser("query-chroma")
    command.add_argument("--targets", required=True)
    command.add_argument("--database", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=query_chroma)
    command = commands.add_parser("query-uploads")
    command.add_argument("--targets", required=True)
    command.add_argument("--glob", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=query_uploads)
    command = commands.add_parser("analyze")
    command.add_argument("--targets", required=True)
    command.add_argument("--fulltext", required=True)
    command.add_argument("--chroma", required=True)
    command.add_argument("--outdir", required=True)
    command.set_defaults(func=analyze)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
