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
    collection_label = (
        _text(source.get("name")) if source.get("type") == "collection" else ""
    )
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


def _merge_visual_evidence(
    citation: dict[str, Any], metadata: dict[str, Any], document_text: Any
) -> None:
    """Attach deduplicated retrieved images to their paper citation."""

    image_urls = metadata.get("image_urls") or []
    if not isinstance(image_urls, list):
        return

    image_url_list = citation["metadata"].setdefault("image_urls", [])
    assets = citation["metadata"].setdefault("visual_assets", [])
    asset_urls = {
        _text(asset.get("url")) for asset in assets if isinstance(asset, dict)
    }
    asset_ids = metadata.get("asset_ids") or []
    label = ", ".join(str(value).strip() for value in asset_ids if str(value).strip())
    caption = _text(document_text)

    for value in image_urls:
        url = _text(value)
        if not url or url in asset_urls:
            continue
        if url not in image_url_list:
            image_url_list.append(url)
        assets.append(
            {
                "url": url,
                "label": label,
                "caption": caption,
                "asset_type": _text(metadata.get("asset_type")) or "image",
            }
        )
        asset_urls.add(url)


def _merge_bibliographic_metadata(
    citation: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """Keep display metadata found by either dense or lexical retrieval."""

    aliases = {
        "journal": ("journal", "journal_title"),
        "publication_date": (
            "publication_date",
            "published_at",
            "date",
            "publication_year",
            "year",
        ),
    }
    for field, candidates in aliases.items():
        value = next(
            (
                _text(metadata.get(key))
                for key in candidates
                if _text(metadata.get(key))
            ),
            "",
        )
        if not value:
            continue
        citation["metadata"].setdefault(field, value)
        citation.setdefault(field, value)


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

        for index, document_text in enumerate(documents):
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
                if existing["title"] in {
                    "Unknown document",
                    _paper_identifier(existing["metadata"]),
                }:
                    existing["title"] = title
                _merge_visual_evidence(existing, metadata, document_text)
                _merge_bibliographic_metadata(existing, metadata)
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
                "metadata": dict(metadata),
            }
            _merge_visual_evidence(citation, metadata, document_text)
            _merge_bibliographic_metadata(citation, metadata)
            citations.append(citation)
            citation_by_key[key] = citation

    return citations
