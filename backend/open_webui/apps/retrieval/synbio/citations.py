"""Normalize retrieved evidence into stable paper-level citations."""

from __future__ import annotations

from typing import Any


def _text(value: Any) -> str:
    return str(value or "").strip()


def _paper_identifier(metadata: dict[str, Any]) -> str:
    for field in ("pmcid", "pmid", "doc_id"):
        if value := _text(metadata.get(field)):
            return value
    return ""


def resolve_citation_title(metadata: dict, source_item: dict) -> str:
    """Return a paper title or stable identifier, never a collection label."""

    source = source_item.get("source") or {}
    collection_label = _text(source.get("name")) if source.get("type") == "collection" else ""
    for field in ("paper_title", "title", "document_title"):
        if title := _text(metadata.get(field)):
            if not collection_label or title.casefold() != collection_label.casefold():
                return title

    if identifier := _paper_identifier(metadata):
        return identifier

    if source.get("type") != "collection":
        for value in (metadata.get("name"), source.get("name"), metadata.get("source")):
            if title := _text(value):
                return title
    return "Unknown document"


def resolve_citation_key(metadata: dict, source_item: dict) -> str:
    """Identify one paper across its paragraph, context and visual chunks."""

    for field in ("pmcid", "pmid", "doc_id"):
        if value := _text(metadata.get(field)):
            return f"{field}:{value.casefold()}"
    return f"title:{resolve_citation_title(metadata, source_item).casefold()}"


def build_citation_sources(results: list[dict]) -> list[dict]:
    """Build one citation per paper while preserving retrieval order."""

    citations: list[dict] = []
    citation_by_key: dict[str, dict] = {}

    for item in results or []:
        documents = item.get("document") or []
        metadata_list = item.get("metadata") or []
        distances = item.get("distances") or []
        if not isinstance(documents, list):
            continue

        for index, _ in enumerate(documents):
            metadata = (
                metadata_list[index]
                if isinstance(metadata_list, list)
                and index < len(metadata_list)
                and isinstance(metadata_list[index], dict)
                else {}
            )
            key = resolve_citation_key(metadata, item)
            title = resolve_citation_title(metadata, item)
            existing = citation_by_key.get(key)
            if existing:
                if existing["title"] in {"Unknown document", _paper_identifier(existing["metadata"])}:
                    existing["title"] = title
                continue

            score = metadata.get("score")
            if score is None and isinstance(distances, list) and index < len(distances):
                score = distances[index]
            identifier = _paper_identifier(metadata)
            citation = {
                "citation_index": len(citations) + 1,
                "citation_key": key,
                "title": title,
                "source": identifier or title,
                "score": score,
                "metadata": metadata,
            }
            citations.append(citation)
            citation_by_key[key] = citation

    return citations
