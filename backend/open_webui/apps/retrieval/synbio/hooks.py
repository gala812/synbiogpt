"""Thin Open WebUI hooks for SynBioGPT query and answer preparation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from open_webui.apps.retrieval.multimodal_answer import (
    CHINESE_EVIDENCE_SYSTEM_PROMPT,
    collect_retrieval_image_urls,
    inject_images_into_last_user_message,
)

from .pipeline import RetrievalPipeline


def protect_query_messages(
    messages: list[dict[str, Any]],
    original_query: str,
    pipeline: RetrievalPipeline,
) -> list[dict[str, Any]]:
    """Copy messages and protect exact entities in the latest user turn."""

    protected = pipeline.prepare_query(original_query).protected_query
    prepared = copy.deepcopy(messages)
    for message in reversed(prepared):
        if message.get("role") == "user":
            message["content"] = protected
            break
    return prepared


def prepare_multimodal_messages(
    messages: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    max_images: int,
    add_system_message: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Add the established Chinese-answer prompt and retrieved image evidence."""

    prepared = add_system_message(CHINESE_EVIDENCE_SYSTEM_PROMPT, messages)
    image_urls = collect_retrieval_image_urls(sources, max_images=max_images)
    injected = inject_images_into_last_user_message(
        prepared, image_urls, max_images=max_images
    )
    return prepared, len(image_urls), injected
