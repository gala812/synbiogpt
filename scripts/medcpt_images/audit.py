from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from typing import Any

from .recovery import asset_key


VALID_STATUSES = {"existing_bound", "recovered", "excluded", "review"}


def audit_image_assets(
    image_access_dir: Path,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Audit the stable manifest and all binding shards in one streaming pass."""

    manifest = image_access_dir / "image_assets.sqlite3"
    statistics_path = image_access_dir / "statistics.json"
    if not manifest.is_file() or not statistics_path.is_file():
        raise FileNotFoundError("The image-access build is not complete")

    expected = json.loads(statistics_path.read_text(encoding="utf-8"))
    database = sqlite3.connect(f"file:{manifest.resolve().as_posix()}?mode=ro", uri=True)
    try:
        metadata = dict(database.execute("SELECT key, value FROM manifest_metadata"))
        sqlite_rows = int(database.execute("SELECT COUNT(*) FROM image_assets").fetchone()[0])
        sqlite_statuses = dict(
            database.execute(
                "SELECT status, COUNT(*) FROM image_assets GROUP BY status ORDER BY status"
            )
        )
        integrity = database.execute("PRAGMA quick_check").fetchone()[0]
        source_paths = (
            row[0]
            for row in database.execute("SELECT source_path FROM image_assets")
        )
        source_root = Path(metadata["source_root"]).resolve()
        paths_outside_root = missing_files = 0
        for raw_path in source_paths:
            path = Path(raw_path).resolve()
            if source_root != path and source_root not in path.parents:
                paths_outside_root += 1
            if verify_files and not path.is_file():
                missing_files += 1
    finally:
        database.close()

    counts: Counter[str] = Counter()
    document_statuses: dict[str, set[str]] = defaultdict(set)
    bad_examples: list[dict[str, Any]] = []
    shards = sorted((image_access_dir / "bindings").glob("part-*.jsonl"))
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                counts["binding_rows"] += 1
                status = row.get("status", "")
                counts[f"status_{status}"] += 1
                document_statuses[str(row.get("pmcid", ""))].add(status)
                error = ""
                if status not in VALID_STATUSES:
                    error = "invalid_status"
                elif row.get("asset_key") != asset_key(row["pmcid"], row["relative_path"]):
                    error = "asset_key_mismatch"
                elif status in {"existing_bound", "recovered"} and not row.get("asset_id"):
                    error = "usable_asset_without_id"
                if error:
                    counts[error] += 1
                    if len(bad_examples) < 20:
                        bad_examples.append(
                            {"shard": shard.name, "line": line_number, "error": error}
                        )

    zero_original_binding = sum(
        "existing_bound" not in statuses for statuses in document_statuses.values()
    )
    zero_usable_after_recovery = sum(
        not ({"existing_bound", "recovered"} & statuses)
        for statuses in document_statuses.values()
    )
    status_total = sum(counts[f"status_{status}"] for status in VALID_STATUSES)
    checks = {
        "sqlite_quick_check": integrity == "ok",
        "all_rows_classified": status_total == counts["binding_rows"],
        "statistics_row_count_matches": expected.get("images") == counts["binding_rows"],
        "shard_count_matches": expected.get("shard_count") == len(shards),
        "documents_with_images_not_above_success_count": (
            len(document_statuses) <= expected.get("successful_documents", 0)
        ),
        "asset_keys_valid": counts["asset_key_mismatch"] == 0,
        "usable_assets_have_ids": counts["usable_asset_without_id"] == 0,
        "all_paths_within_source_root": paths_outside_root == 0,
        "all_files_exist": None if not verify_files else missing_files == 0,
    }
    report = {
        "schema_version": "medcpt_image_assets_audit_v1",
        "passed": all(value is not False for value in checks.values()),
        "checks": checks,
        "binding_shards": len(shards),
        "binding_rows": counts["binding_rows"],
        "unique_image_assets": sqlite_rows,
        "binding_statuses": {
            status: counts[f"status_{status}"] for status in sorted(VALID_STATUSES)
        },
        "unique_asset_statuses": sqlite_statuses,
        "documents_with_images": len(document_statuses),
        "documents_with_zero_original_binding": zero_original_binding,
        "documents_with_zero_usable_assets_after_recovery": zero_usable_after_recovery,
        "paths_outside_root": paths_outside_root,
        "missing_files": missing_files if verify_files else None,
        "bad_examples": bad_examples,
    }
    return report
