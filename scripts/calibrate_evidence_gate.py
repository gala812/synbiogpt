#!/usr/bin/env python3
"""Export Evidence Gate labels and calibrate a raw-logit cutoff offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LABEL_COLUMNS = (
    "query_id",
    "recorded_at",
    "collection_name",
    "cross_encoder_model",
    "cross_encoder_max_tokens",
    "original_query",
    "semantic_query",
    "lexical_query",
    "chunk_id",
    "title",
    "section",
    "pmid",
    "pmcid",
    "doi",
    "rerank_rank",
    "rrf_rank",
    "retrieval_source",
    "document_text",
    "relevance_label",
    "labeler_id",
    "label_notes",
    "raw_logit",
)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(row)
    return records


def export_label_csv(input_path: Path, output_path: Path) -> int:
    """Export collected JSONL rows into an Excel-friendly labeling sheet."""

    records = _read_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(LABEL_COLUMNS),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in LABEL_COLUMNS})
    return len(records)


def _label(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip()
    if normalized not in {"0", "1"}:
        raise ValueError(f"relevance_label must be 0, 1, or blank; got {value!r}")
    return int(normalized)


def load_labeled_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records = []
    unlabeled_count = 0
    for index, row in enumerate(_read_records(path), 1):
        label = _label(row.get("relevance_label"))
        if label is None:
            unlabeled_count += 1
            continue
        try:
            raw_logit = float(row.get("raw_logit"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid raw_logit in labeled row {index}") from exc
        if not math.isfinite(raw_logit):
            raise ValueError(f"Non-finite raw_logit in labeled row {index}")

        query_id = str(row.get("query_id") or "").strip()
        semantic_query = str(row.get("semantic_query") or "").strip()
        original_query = str(row.get("original_query") or "").strip()
        if not query_id:
            query_id = hashlib.sha256(
                (semantic_query or original_query).encode("utf-8")
            ).hexdigest()[:24]
        split_group = (
            semantic_query.casefold() or original_query.casefold() or query_id
        )
        records.append(
            {
                **row,
                "query_id": query_id,
                "_split_group": split_group,
                "raw_logit": raw_logit,
                "relevance_label": label,
            }
        )
    return records, unlabeled_count


def split_by_query(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split on normalized queries so candidates from one query never leak."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    groups = sorted({row["_split_group"] for row in records})
    if validation_fraction == 0 or len(groups) < 2:
        return list(records), []

    random.Random(seed).shuffle(groups)
    validation_count = max(1, round(len(groups) * validation_fraction))
    validation_count = min(validation_count, len(groups) - 1)
    validation_groups = set(groups[:validation_count])
    calibration = [
        row for row in records if row["_split_group"] not in validation_groups
    ]
    validation = [
        row for row in records if row["_split_group"] in validation_groups
    ]
    return calibration, validation


def evaluate_threshold(
    records: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    queries: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for row in records:
        label = int(row["relevance_label"])
        accepted = float(row["raw_logit"]) >= threshold
        queries[row["query_id"]].append((label, accepted))
        if label and accepted:
            tp += 1
        elif not label and accepted:
            fp += 1
        elif not label and not accepted:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )

    supported_queries = 0
    supported_queries_with_accepted_positive = 0
    unsupported_queries = 0
    unsupported_queries_with_any_accepted = 0
    queries_with_any_accepted = 0
    queries_with_accepted_positive = 0
    for pairs in queries.values():
        has_positive = any(label for label, _ in pairs)
        has_any_accepted = any(accepted for _, accepted in pairs)
        has_accepted_positive = any(label and accepted for label, accepted in pairs)
        queries_with_any_accepted += int(has_any_accepted)
        queries_with_accepted_positive += int(has_accepted_positive)
        if has_positive:
            supported_queries += 1
            supported_queries_with_accepted_positive += int(has_accepted_positive)
        else:
            unsupported_queries += 1
            unsupported_queries_with_any_accepted += int(has_any_accepted)

    return {
        "threshold": threshold,
        "pairs": total,
        "accepted_pairs": tp + fp,
        "acceptance_rate": (tp + fp) / total if total else 0.0,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "queries": len(queries),
        "queries_with_any_evidence_rate": (
            queries_with_any_accepted / len(queries) if queries else 0.0
        ),
        "supported_query_recall": (
            supported_queries_with_accepted_positive / supported_queries
            if supported_queries
            else None
        ),
        "unsupported_query_false_evidence_rate": (
            unsupported_queries_with_any_accepted / unsupported_queries
            if unsupported_queries
            else None
        ),
        "accepted_query_support_rate": (
            queries_with_accepted_positive / queries_with_any_accepted
            if queries_with_any_accepted
            else None
        ),
    }


def _threshold_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    scores = sorted({float(row["raw_logit"]) for row in records})
    thresholds = [*scores, math.nextafter(scores[-1], math.inf)]
    return [evaluate_threshold(records, threshold) for threshold in thresholds]


def _select_threshold(
    table: list[dict[str, Any]], target_precision: float
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in table
        if row["precision"] is not None
        and row["precision"] >= target_precision
        and row["true_positive"] > 0
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row["recall"] if row["recall"] is not None else -1,
            row["precision"],
            row["f1"],
            -row["threshold"],
        ),
    )


def _score_summary(records: list[dict[str, Any]], label: int) -> dict[str, Any]:
    scores = sorted(
        float(row["raw_logit"])
        for row in records
        if int(row["relevance_label"]) == label
    )
    if not scores:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        position = (len(scores) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return scores[lower]
        weight = position - lower
        return scores[lower] * (1 - weight) + scores[upper] * weight

    return {
        "count": len(scores),
        "min": scores[0],
        "p25": percentile(0.25),
        "median": statistics.median(scores),
        "p75": percentile(0.75),
        "max": scores[-1],
    }


def build_calibration_report(
    records: list[dict[str, Any]],
    *,
    unlabeled_count: int,
    target_precision: float,
    validation_fraction: float,
    seed: int,
    min_labeled_pairs: int,
    min_positive_pairs: int,
    min_negative_pairs: int,
) -> dict[str, Any]:
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    if not records:
        raise ValueError("No labeled rows found")

    calibration, validation = split_by_query(
        records,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    calibration_table = _threshold_table(calibration)
    selected = _select_threshold(calibration_table, target_precision)
    validation_metrics = (
        evaluate_threshold(validation, selected["threshold"])
        if selected is not None and validation
        else None
    )

    positive_count = sum(int(row["relevance_label"]) for row in records)
    negative_count = len(records) - positive_count
    calibration_positive_count = sum(
        int(row["relevance_label"]) for row in calibration
    )
    validation_positive_count = sum(
        int(row["relevance_label"]) for row in validation
    )
    collections = sorted(
        {
            str(row.get("collection_name") or "").strip()
            for row in records
            if str(row.get("collection_name") or "").strip()
        }
    )
    model_signatures = sorted(
        {
            (
                str(row.get("cross_encoder_model") or "").strip(),
                str(row.get("cross_encoder_max_tokens") or "").strip(),
            )
            for row in records
            if str(row.get("cross_encoder_model") or "").strip()
        }
    )
    warnings = []
    if len(records) < min_labeled_pairs:
        warnings.append(
            f"labeled pair count {len(records)} is below {min_labeled_pairs}"
        )
    if positive_count < min_positive_pairs:
        warnings.append(
            f"positive pair count {positive_count} is below {min_positive_pairs}"
        )
    if negative_count < min_negative_pairs:
        warnings.append(
            f"negative pair count {negative_count} is below {min_negative_pairs}"
        )
    if len(collections) != 1:
        warnings.append("calibration requires exactly one collection")
    if len(model_signatures) != 1:
        warnings.append("calibration requires exactly one Cross Encoder signature")
    if selected is None:
        warnings.append("no threshold meets the requested calibration precision")
    if calibration_positive_count == 0 or calibration_positive_count == len(
        calibration
    ):
        warnings.append("calibration split does not contain both label classes")
    if validation_fraction > 0 and not validation:
        warnings.append("validation split is empty")
    if validation and (
        validation_positive_count == 0
        or validation_positive_count == len(validation)
    ):
        warnings.append("validation split does not contain both label classes")
    if validation_metrics is not None and (
        validation_metrics["precision"] is None
        or validation_metrics["precision"] < target_precision
    ):
        warnings.append("candidate threshold misses target precision on validation")

    recommendation_ready = selected is not None and not warnings
    report = {
        "schema_version": 1,
        "policy": {
            "target_precision": target_precision,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "minimum_counts": {
                "labeled_pairs": min_labeled_pairs,
                "positive_pairs": min_positive_pairs,
                "negative_pairs": min_negative_pairs,
            },
        },
        "dataset": {
            "labeled_pairs": len(records),
            "unlabeled_rows_ignored": unlabeled_count,
            "positive_pairs": positive_count,
            "negative_pairs": negative_count,
            "queries": len({row["query_id"] for row in records}),
            "collections": collections,
            "cross_encoder_signatures": [
                {"model": model, "max_tokens": max_tokens or None}
                for model, max_tokens in model_signatures
            ],
            "calibration_pairs": len(calibration),
            "validation_pairs": len(validation),
            "score_summary": {
                "relevant": _score_summary(records, 1),
                "irrelevant": _score_summary(records, 0),
            },
        },
        "candidate_threshold": selected["threshold"] if selected else None,
        "calibration_metrics": selected,
        "validation_metrics": validation_metrics,
        "recommendation_ready": recommendation_ready,
        "suggested_environment": (
            f"MEDCPT_EVIDENCE_GATE_MIN_SCORE={selected['threshold']:.17g}"
            if recommendation_ready and selected is not None
            else None
        ),
        "warnings": warnings,
        "calibration_threshold_table": calibration_table,
    }
    return report


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="export collected JSONL to CSV")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    calibrate = commands.add_parser(
        "calibrate", help="scan thresholds using completed binary labels"
    )
    calibrate.add_argument("--input", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--target-precision", type=float, required=True)
    calibrate.add_argument("--collection")
    calibrate.add_argument("--cross-encoder-model")
    calibrate.add_argument("--validation-fraction", type=float, default=0.2)
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--min-labeled-pairs", type=int, default=100)
    calibrate.add_argument("--min-positive-pairs", type=int, default=20)
    calibrate.add_argument("--min-negative-pairs", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export":
            count = export_label_csv(args.input, args.output)
            result = {"exported_rows": count, "output": str(args.output)}
        else:
            records, unlabeled_count = load_labeled_records(args.input)
            if args.collection:
                records = [
                    row
                    for row in records
                    if str(row.get("collection_name") or "") == args.collection
                ]
            if args.cross_encoder_model:
                records = [
                    row
                    for row in records
                    if str(row.get("cross_encoder_model") or "")
                    == args.cross_encoder_model
                ]
            report = build_calibration_report(
                records,
                unlabeled_count=unlabeled_count,
                target_precision=args.target_precision,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                min_labeled_pairs=args.min_labeled_pairs,
                min_positive_pairs=args.min_positive_pairs,
                min_negative_pairs=args.min_negative_pairs,
            )
            _write_json(args.output, report)
            result = {
                "labeled_pairs": len(records),
                "candidate_threshold": report["candidate_threshold"],
                "recommendation_ready": report["recommendation_ready"],
                "output": str(args.output),
                "warnings": report["warnings"],
            }
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
