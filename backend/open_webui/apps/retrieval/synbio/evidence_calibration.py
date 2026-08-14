"""Optional raw-logit collection for offline Evidence Gate calibration."""

from __future__ import annotations

import json
import math
import random
import threading
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_WRITE_LOCK = threading.Lock()
_METADATA_FIELDS = (
    "chunk_id",
    "title",
    "section",
    "pmid",
    "pmcid",
    "doi",
    "doc_id",
    "rerank_rank",
    "rrf_rank",
    "retrieval_source",
)


def collect_calibration_examples(
    *,
    path: str,
    sample_rate: float,
    max_text_chars: int,
    query: Any,
    documents: Sequence[Any],
    collection_name: str,
    cross_encoder_model: str,
    cross_encoder_max_tokens: int | None,
) -> int:
    """Append one JSONL row per reranked pair when optional sampling is enabled."""

    if not path or sample_rate <= 0 or not documents:
        return 0
    if sample_rate < 1 and random.random() >= sample_rate:
        return 0

    query_id = uuid.uuid4().hex
    recorded_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        try:
            raw_logit = float(metadata.get("cross_encoder_score"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(raw_logit):
            continue

        full_text = str(getattr(document, "page_content", "") or "")
        document_text = full_text[:max_text_chars]
        row = {
            "schema_version": 1,
            "query_id": query_id,
            "recorded_at": recorded_at,
            "collection_name": collection_name,
            "cross_encoder_model": cross_encoder_model,
            "cross_encoder_max_tokens": cross_encoder_max_tokens,
            "original_query": str(getattr(query, "original_query", "") or ""),
            "semantic_query": str(getattr(query, "semantic_query", "") or ""),
            "lexical_query": str(getattr(query, "lexical_query", "") or ""),
            "exact_terms": list(getattr(query, "exact_terms", ()) or ()),
            "raw_logit": raw_logit,
            "document_text": document_text,
            "document_text_truncated": len(full_text) > len(document_text),
            "relevance_label": None,
            "label_notes": "",
        }
        for field in _METADATA_FIELDS:
            value = metadata.get(field)
            if value is not None:
                row[field] = value
        rows.append(row)

    if not rows:
        return 0

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    with _WRITE_LOCK:
        with destination.open("a", encoding="utf-8", newline="") as handle:
            handle.write(payload)
    return len(rows)
