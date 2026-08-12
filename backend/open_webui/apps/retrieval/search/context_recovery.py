"""Recover parent and adjacent chunks for reranked full-text anchors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

ChunkFetcher = Callable[[Sequence[str]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ContextDocument:
    page_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ContextRecoveryResult:
    documents: list[ContextDocument]
    anchor_count: int
    expanded_count: int
    token_count: int
    duplicate_count: int
    budget_skipped_count: int
    missing_count: int
    cross_document_rejection_count: int


@dataclass(slots=True)
class _WalkState:
    anchor_id: str
    anchor_rank: int
    direction: str
    chunk_id: str
    distance: int = 1
    active: bool = True
    visited_ids: set[str] = field(default_factory=set)


def _chunk_id(document: Any) -> str:
    metadata = document.metadata or {}
    return str(
        metadata.get("chunk_id") or metadata.get("id") or metadata.get("doc_id") or ""
    ).strip()


def _document_key(metadata: Mapping[str, Any]) -> tuple[str, str] | None:
    for key in ("pmcid", "pmid"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return key, value
    return None


def _token_count(document: Any) -> int:
    metadata = document.metadata or {}
    for key in ("text_token_count", "token_count"):
        value = metadata.get(key)
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return max(1, round(len(document.page_content.split()) * 1.4))


def _with_metadata(document: Any, **values: Any) -> ContextDocument:
    metadata = dict(document.metadata or {})
    metadata.update(values)
    return ContextDocument(page_content=document.page_content, metadata=metadata)


def recover_context_documents(
    anchors: Sequence[Any],
    fetch_chunks: ChunkFetcher,
    *,
    token_budget: int = 12_000,
    max_parent_chunks: int = 12,
) -> ContextRecoveryResult:
    """Expand ranked anchors while preserving their order and document boundaries.

    All anchors are retained, even when they consume the configured budget. Additional
    chunks are selected in anchor-rank and proximity order and appear after anchors.
    """

    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if max_parent_chunks < 1:
        raise ValueError("max_parent_chunks must be positive")

    unique_anchors: list[Any] = []
    anchors_by_id: dict[str, Any] = {}
    duplicate_count = 0
    for document in anchors:
        chunk_id = _chunk_id(document)
        if not chunk_id:
            continue
        if chunk_id in anchors_by_id:
            duplicate_count += 1
            continue
        anchors_by_id[chunk_id] = document
        unique_anchors.append(document)

    anchor_ids = set(anchors_by_id)
    known = dict(anchors_by_id)
    associations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parent_counts = {chunk_id: 1 for chunk_id in anchor_ids}
    states: list[_WalkState] = []
    for rank, anchor in enumerate(unique_anchors, 1):
        metadata = anchor.metadata or {}
        anchor_id = _chunk_id(anchor)
        for direction, link_key in (
            ("previous", "previous_chunk_id"),
            ("next", "next_chunk_id"),
        ):
            linked_id = str(metadata.get(link_key) or "").strip()
            if linked_id and linked_id != anchor_id:
                states.append(_WalkState(anchor_id, rank, direction, linked_id))

    missing_ids: set[str] = set()
    cross_document_rejections = 0
    while any(state.active for state in states):
        requested = sorted(
            {
                state.chunk_id
                for state in states
                if state.active
                and state.chunk_id not in known
                and state.chunk_id not in missing_ids
            }
        )
        if requested:
            fetched = fetch_chunks(requested)
            known.update(
                (chunk_id, document)
                for chunk_id, document in fetched.items()
                if chunk_id in requested
            )
            missing_ids.update(
                chunk_id for chunk_id in requested if chunk_id not in fetched
            )

        progressed = False
        for state in states:
            if not state.active:
                continue
            document = known.get(state.chunk_id)
            if document is None:
                state.active = False
                continue
            if state.chunk_id in state.visited_ids:
                state.active = False
                continue
            state.visited_ids.add(state.chunk_id)

            anchor = anchors_by_id[state.anchor_id]
            anchor_key = _document_key(anchor.metadata or {})
            context_key = _document_key(document.metadata or {})
            if anchor_key is None or context_key != anchor_key:
                cross_document_rejections += 1
                state.active = False
                continue

            anchor_parent = str(
                (anchor.metadata or {}).get("parent_chunk_id") or ""
            ).strip()
            context_parent = str(
                (document.metadata or {}).get("parent_chunk_id") or ""
            ).strip()
            same_parent = bool(anchor_parent and context_parent == anchor_parent)
            if same_parent and parent_counts[state.anchor_id] >= max_parent_chunks:
                state.active = False
                continue
            relationship = "parent" if same_parent else state.direction
            associations[state.chunk_id].append(
                {
                    "anchor_id": state.anchor_id,
                    "anchor_rank": state.anchor_rank,
                    "relationship": relationship,
                    "direction": state.direction,
                    "distance": state.distance,
                }
            )
            progressed = True

            if not same_parent:
                state.active = False
                continue
            parent_counts[state.anchor_id] += 1
            link_key = (
                "previous_chunk_id"
                if state.direction == "previous"
                else "next_chunk_id"
            )
            linked_id = str((document.metadata or {}).get(link_key) or "").strip()
            if not linked_id or linked_id == state.chunk_id:
                state.active = False
                continue
            state.chunk_id = linked_id
            state.distance += 1

        if not requested and not progressed:
            break

    output: list[ContextDocument] = []
    selected_ids: set[str] = set()
    total_tokens = 0
    for rank, anchor in enumerate(unique_anchors, 1):
        anchor_id = _chunk_id(anchor)
        count = _token_count(anchor)
        output.append(
            _with_metadata(
                anchor,
                chunk_id=anchor_id,
                context_role="anchor",
                context_anchor_chunk_id=anchor_id,
                context_anchor_rank=rank,
                context_token_count=count,
            )
        )
        selected_ids.add(anchor_id)
        total_tokens += count

    candidates = sorted(
        (
            (
                min(items, key=lambda item: (item["anchor_rank"], item["distance"])),
                chunk_id,
            )
            for chunk_id, items in associations.items()
            if chunk_id not in anchor_ids
        ),
        key=lambda item: (
            item[0]["anchor_rank"],
            0 if item[0]["relationship"] == "parent" else 1,
            item[0]["distance"],
            item[1],
        ),
    )
    budget_skipped = 0
    for primary, chunk_id in candidates:
        if chunk_id in selected_ids:
            duplicate_count += 1
            continue
        document = known[chunk_id]
        count = _token_count(document)
        if total_tokens + count > token_budget:
            budget_skipped += 1
            continue
        links = sorted(
            associations[chunk_id],
            key=lambda item: (item["anchor_rank"], item["distance"], item["direction"]),
        )
        anchor = anchors_by_id[primary["anchor_id"]]
        anchor_metadata = anchor.metadata or {}
        output.append(
            _with_metadata(
                document,
                chunk_id=chunk_id,
                context_role=primary["relationship"],
                context_relationships=links,
                context_anchor_chunk_id=primary["anchor_id"],
                context_anchor_rank=primary["anchor_rank"],
                context_anchor_chunk_ids=list(
                    dict.fromkeys(item["anchor_id"] for item in links)
                ),
                context_token_count=count,
                anchor_cross_encoder_score=anchor_metadata.get("cross_encoder_score"),
                anchor_rrf_score=anchor_metadata.get("rrf_score"),
                score=anchor_metadata.get("score"),
            )
        )
        selected_ids.add(chunk_id)
        total_tokens += count

    return ContextRecoveryResult(
        documents=output,
        anchor_count=len(unique_anchors),
        expanded_count=len(output) - len(unique_anchors),
        token_count=total_tokens,
        duplicate_count=duplicate_count,
        budget_skipped_count=budget_skipped,
        missing_count=len(missing_ids),
        cross_document_rejection_count=cross_document_rejections,
    )
