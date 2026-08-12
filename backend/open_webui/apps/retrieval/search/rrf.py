"""Deterministic weighted Reciprocal Rank Fusion for retrieval candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float | None = None


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    rank: int
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    rrf_score: float
    source_ranks: dict[str, int]
    source_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_chunk_id(metadata: Mapping[str, Any], fallback: str = "") -> str:
    """Resolve chunk identity without collapsing all chunks from one paper."""

    return str(
        metadata.get("chunk_id")
        or metadata.get("doc_id")
        or metadata.get("id")
        or fallback
    ).strip()


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RankedCandidate]],
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    limit: int = 150,
) -> list[FusedCandidate]:
    """Fuse ranked lists, counting each chunk at most once per source."""

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    if limit < 1:
        raise ValueError("limit must be positive")

    source_weights = dict(weights or {})
    if any(weight < 0 for weight in source_weights.values()):
        raise ValueError("RRF weights cannot be negative")

    fused: dict[str, dict[str, Any]] = {}
    for source, candidates in rankings.items():
        weight = float(source_weights.get(source, 1.0))
        if weight == 0:
            continue
        seen_in_source: set[str] = set()
        for rank, candidate in enumerate(candidates, 1):
            chunk_id = candidate.chunk_id.strip()
            if not chunk_id:
                raise ValueError(f"{source} candidate at rank {rank} has no chunk_id")
            if chunk_id in seen_in_source:
                continue
            seen_in_source.add(chunk_id)

            item = fused.setdefault(
                chunk_id,
                {
                    "text": candidate.text,
                    "metadata": dict(candidate.metadata),
                    "rrf_score": 0.0,
                    "source_ranks": {},
                    "source_scores": {},
                },
            )
            item["rrf_score"] += weight / (rrf_k + rank)
            item["source_ranks"][source] = rank
            if candidate.score is not None:
                item["source_scores"][source] = float(candidate.score)
            if not item["text"] and candidate.text:
                item["text"] = candidate.text
            for key, value in candidate.metadata.items():
                item["metadata"].setdefault(key, value)

    ordered = sorted(
        fused.items(),
        key=lambda pair: (
            -pair[1]["rrf_score"],
            -len(pair[1]["source_ranks"]),
            min(pair[1]["source_ranks"].values()),
            pair[0],
        ),
    )[:limit]

    return [
        FusedCandidate(
            rank=rank,
            chunk_id=chunk_id,
            text=item["text"],
            metadata=item["metadata"],
            rrf_score=item["rrf_score"],
            source_ranks=item["source_ranks"],
            source_scores=item["source_scores"],
        )
        for rank, (chunk_id, item) in enumerate(ordered, 1)
    ]
