"""Prepare retrieved text and image evidence for a multimodal chat model."""

from __future__ import annotations

from typing import Any


MAX_USER_IMAGES = 4
IMAGE_ONLY_QUERY_TEXT = "Analyze the current user image."


def current_user_image_items(
    messages: list[dict[str, Any]], *, max_images: int = MAX_USER_IMAGES
) -> list[dict[str, Any]]:
    """Return only image inputs attached to the latest user turn."""

    if max_images < 1:
        return []
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            return []
        return [
            item
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "image_url"
            and _image_url(item)
        ][:max_images]
    return []


def retain_current_user_images(
    messages: list[dict[str, Any]], *, max_images: int = MAX_USER_IMAGES
) -> tuple[list[dict[str, Any]], int]:
    """Drop historical images and cap images on the latest user turn."""

    latest_user_index = next(
        (
            index
            for index in range(len(messages or []) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )
    kept = 0
    for index, message in enumerate(messages or []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        filtered = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                filtered.append(item)
                continue
            if index == latest_user_index and kept < max_images and _image_url(item):
                filtered.append(item)
                kept += 1
        message["content"] = filtered
    return messages, kept


def build_query_generation_content(
    prompt: str,
    messages: list[dict[str, Any]],
    *,
    max_images: int = MAX_USER_IMAGES,
) -> str | list[dict[str, Any]]:
    """Attach current-turn images while preserving the pure-text payload shape."""

    images = current_user_image_items(messages, max_images=max_images)
    if not images:
        return prompt
    return [{"type": "text", "text": prompt}, *images]


def collect_retrieval_image_urls(
    sources: list[dict[str, Any]], *, max_images: int = 16
) -> list[str]:
    """Collect stable, deduplicated image URLs in retrieval evidence order."""

    if max_images < 1:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for source in sources or []:
        metadata_items = source.get("metadata") or []
        if not isinstance(metadata_items, list):
            continue
        for metadata in metadata_items:
            if not isinstance(metadata, dict):
                continue
            image_urls = metadata.get("image_urls") or []
            if not isinstance(image_urls, list):
                continue
            for url in image_urls:
                normalized = str(url).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                urls.append(normalized)
                if len(urls) >= max_images:
                    return urls
    return urls


def build_retrieval_image_files(
    image_urls: list[str], citation_sources: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Build image records, including the citation that controls inline placement."""

    evidence_by_url: dict[str, dict[str, Any]] = {}
    for citation in citation_sources or []:
        citation_index = citation.get("citation_index")
        metadata = citation.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        for asset in metadata.get("visual_assets") or []:
            if not isinstance(asset, dict):
                continue
            url = str(asset.get("url") or "").strip()
            if url:
                evidence_by_url[url] = {
                    "citation_index": citation_index,
                    "caption": str(asset.get("caption") or "").strip(),
                    "label": str(asset.get("label") or "").strip(),
                    "asset_type": str(asset.get("asset_type") or "image").strip(),
                }

    files = []
    for raw_url in image_urls:
        url = str(raw_url).strip()
        if not url:
            continue
        file = {"type": "image", "url": url}
        if evidence := evidence_by_url.get(url):
            file.update(evidence)
        files.append(file)
    return files


def _image_url(item: dict[str, Any]) -> str:
    value = item.get("image_url")
    if isinstance(value, dict):
        return str(value.get("url") or "").strip()
    return str(value or "").strip()


def inject_images_into_last_user_message(
    messages: list[dict[str, Any]],
    image_urls: list[str],
    *,
    max_images: int = 16,
) -> int:
    """Append retrieved images in the OpenAI multimodal message format."""

    if max_images < 1 or not image_urls:
        return 0

    user_message = next(
        (message for message in reversed(messages) if message.get("role") == "user"),
        None,
    )
    if user_message is None:
        return 0

    content = user_message.get("content", "")
    if isinstance(content, str):
        content_items: list[dict[str, Any]] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content_items = content
    else:
        return 0

    existing_urls = {
        url
        for item in content_items
        if isinstance(item, dict) and item.get("type") == "image_url"
        if (url := _image_url(item))
    }
    remaining = max(0, max_images - len(existing_urls))
    injected = 0
    for url in image_urls:
        normalized = str(url).strip()
        if not normalized or normalized in existing_urls or injected >= remaining:
            continue
        content_items.append(
            {"type": "image_url", "image_url": {"url": normalized}}
        )
        existing_urls.add(normalized)
        injected += 1

    if injected:
        user_message["content"] = content_items
    return injected
