"""Thin Open WebUI hooks for SynBioGPT query and answer preparation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from open_webui.apps.retrieval.multimodal_answer import (
    MAX_USER_IMAGES,
    collect_retrieval_image_urls,
    current_user_image_items,
    inject_images_into_last_user_message,
)
from open_webui.apps.retrieval.prompts import MULTIMODAL_EVIDENCE_SYSTEM_PROMPT

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
            content = message.get("content")
            if isinstance(content, list):
                replaced = False
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = protected
                        replaced = True
                        break
                if not replaced:
                    content.insert(0, {"type": "text", "text": protected})
            else:
                message["content"] = protected
            break
    return prepared


def prepare_multimodal_messages(
    messages: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    max_images: int,
    add_system_message: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Add the established Chinese-answer prompt and retrieved image evidence."""

    prepared = add_system_message(MULTIMODAL_EVIDENCE_SYSTEM_PROMPT, messages)
    image_urls = collect_retrieval_image_urls(sources, max_images=max_images)
    user_image_count = len(
        current_user_image_items(prepared, max_images=MAX_USER_IMAGES)
    )
    injected = inject_images_into_last_user_message(
        prepared,
        image_urls,
        max_images=max_images + user_image_count,
    )
    return prepared, image_urls, injected
