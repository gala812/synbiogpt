"""Protect exact scientific identifiers after reranking."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExactTermGateResult:
    documents: list[Any]
    rejected_count: int
    missing_terms: tuple[str, ...]


def gate_rejection_reason(diagnostics: Mapping[str, Any] | None) -> str | None:
    """Classify an all-candidate gate rejection for observability."""

    diagnostics = diagnostics or {}
    input_count = int(diagnostics.get("input_count") or 0)
    output_count = int(diagnostics.get("output_count") or 0)
    exact_rejected = int(diagnostics.get("exact_term_rejected_count") or 0)
    if input_count <= 0 or output_count > 0 or exact_rejected <= 0:
        return None
    return "exact_term"


def has_evidence_documents(context: Mapping[str, Any] | None) -> bool:
    """Distinguish a real evidence payload from a nested empty result."""

    if not context:
        return False
    return any(bool(group) for group in (context.get("documents") or []))


def _document_text(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    values = [
        getattr(document, "page_content", ""),
        metadata.get("paper_title"),
        metadata.get("title"),
        metadata.get("section"),
        metadata.get("subsection"),
    ]
    return "\n".join(str(value) for value in values if value)


def _term_patterns(term: str) -> tuple[re.Pattern[str], ...]:
    normalized = str(term or "").strip()
    aliases = [normalized]
    if normalized.casefold().replace(" ", "") in {"e.coli", "ecoli"}:
        aliases.append("Escherichia coli")

    patterns = []
    for alias in aliases:
        pieces = re.findall(r"[A-Za-z0-9]+", alias)
        if not pieces:
            continue
        flexible = r"[\W_]*".join(re.escape(piece) for piece in pieces)
        patterns.append(
            re.compile(rf"(?<![A-Za-z0-9]){flexible}(?![A-Za-z0-9])", re.I)
        )
    return tuple(patterns)


def apply_exact_term_gate(
    documents: Sequence[Any], *, required_terms: Sequence[str]
) -> ExactTermGateResult:
    """Reject near-name evidence when user-supplied identifiers are uncovered.

    Coverage is evaluated across the selected evidence set, so different papers
    may support different identifiers. Documents unrelated to every required
    identifier are removed. If any identifier is absent from the whole set, no
    evidence is returned rather than substituting a similarly spelled entity.
    """

    candidates = list(documents)
    terms = tuple(
        dict.fromkeys(
            str(term).strip() for term in required_terms if str(term).strip()
        )
    )
    if not candidates or not terms:
        return ExactTermGateResult(candidates, 0, ())

    patterns = {term: _term_patterns(term) for term in terms}
    matched_by_document: list[set[str]] = []
    covered: set[str] = set()
    for document in candidates:
        text = _document_text(document)
        matched = {
            term
            for term, term_patterns in patterns.items()
            if any(pattern.search(text) for pattern in term_patterns)
        }
        matched_by_document.append(matched)
        covered.update(matched)

    missing = tuple(term for term in terms if term not in covered)
    if missing:
        return ExactTermGateResult([], len(candidates), missing)

    accepted = [
        document
        for document, matched in zip(candidates, matched_by_document, strict=True)
        if matched
    ]
    return ExactTermGateResult(accepted, len(candidates) - len(accepted), ())
