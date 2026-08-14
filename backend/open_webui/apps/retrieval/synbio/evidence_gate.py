"""Post-rerank evidence filtering without assuming a calibrated score cutoff."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceGateResult:
    documents: list[Any]
    rejected_count: int


def has_evidence_documents(context: Mapping[str, Any] | None) -> bool:
    """Distinguish a real evidence payload from a nested empty result."""

    if not context:
        return False
    return any(bool(group) for group in (context.get("documents") or []))


def apply_evidence_gate(
    documents: Sequence[Any],
    *,
    min_cross_encoder_score: float | None,
) -> EvidenceGateResult:
    """Filter reranked evidence when a calibrated raw-logit cutoff is configured.

    The default cutoff is ``None``, so current raw MedCPT logits are not assigned
    an arbitrary meaning. Once configured from calibration data, the gate may
    legitimately reject every candidate.
    """

    candidates = list(documents)
    if min_cross_encoder_score is None:
        return EvidenceGateResult(candidates, 0)

    accepted = []
    for document in candidates:
        metadata = getattr(document, "metadata", {}) or {}
        score = metadata.get("cross_encoder_score")
        try:
            is_reliable = float(score) >= min_cross_encoder_score
        except (TypeError, ValueError):
            is_reliable = False
        if is_reliable:
            accepted.append(document)

    return EvidenceGateResult(accepted, len(candidates) - len(accepted))
