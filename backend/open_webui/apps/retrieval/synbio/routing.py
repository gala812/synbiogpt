"""Small routing helpers for application-level knowledge retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_TRAILING_PUNCTUATION = re.compile(r"[.!?。！？]+$")
_PLAIN_CHAT_MESSAGES = frozenset(
    {
        "你好",
        "您好",
        "你好呀",
        "您好呀",
        "哈喽",
        "哈啰",
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "谢谢",
        "谢谢啦",
        "多谢",
        "好的",
        "嗯嗯",
        "ok",
        "okay",
        "在吗",
        "在不在",
        "早上好",
        "晚上好",
        "再见",
        "拜拜",
        "你是谁",
        "你能做什么",
        "你能帮我做什么",
    }
)


def is_explicit_plain_chat(message: Any) -> bool:
    """Return whether the whole message is an explicit plain-chat phrase."""

    if not isinstance(message, str):
        return False

    normalized = unicodedata.normalize("NFKC", message).casefold().strip()
    normalized = _TRAILING_PUNCTUATION.sub("", normalized).strip()
    return normalized in _PLAIN_CHAT_MESSAGES


def route_flags(message: Any) -> dict[str, bool]:
    """Return request-scoped routing flags for an original user message."""

    return {"plain_chat": True} if is_explicit_plain_chat(message) else {}


def add_default_knowledge(
    files: list[dict[str, Any]] | None,
    metadata: dict[str, Any] | None,
    *,
    enabled: bool,
    collection: str,
) -> list[dict[str, Any]]:
    """Attach the production collection to real chats without replacing uploads."""

    selected = list(files or [])
    if not enabled or not collection or (metadata or {}).get("task"):
        return selected

    if any(
        item.get("id") == collection or item.get("collection_name") == collection
        for item in selected
    ):
        return selected

    return [
        {
            "id": collection,
            "name": "SynBioGPT Full-text Literature",
            "type": "collection",
            "legacy": False,
            "default": True,
        },
        *selected,
    ]
