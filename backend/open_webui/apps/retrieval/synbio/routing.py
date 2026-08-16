"""Small routing helpers for application-level knowledge retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_TRAILING_PUNCTUATION = re.compile(r"[.!?。！？]+$")
_VISUAL_EVIDENCE_RE = re.compile(
    r"(?:图片|图像|图表|插图|示意图|流程图|结构图|显微图|表格|图中|表中|"
    r"结合(?:图|表)|展示(?:图|表)|"
    r"\b(?:image|figure|fig\.?|table|diagram|chart|plot|visual)\b)",
    re.IGNORECASE,
)
_PRODUCT_CAPABILITY_EXCLUSION_RE = re.compile(
    r"(?:知识库|文献库|数据库)(?:里|中|内).*(?:有|包含|收录)|"
    r"(?:查找|查询|检索|推荐|帮我找|找).*(?:论文|文献)|"
    r"(?:关于|有关|哪些|什么).*(?:论文|文献|研究内容)",
    re.IGNORECASE,
)
_PRODUCT_CAPABILITY_PATTERNS = (
    re.compile(
        r"^(?:(?:当前|现在|这个|该)(?:系统)?|synbiogpt|你|你们)?"
        r"(?:是不是|是否|有没有|有无)?(?:已经|默认)?"
        r"(?:内置|接入|连接|配置|使用|采用|基于|拥有|具备|有)(?:了)?"
        r"(?:自己的|一个)?(?:全文(?:文献)?|文献)?(?:知识库|数据库|文献库)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:知识库|全文文献知识库|文献库)(?:是不是|是否)?(?:已经)?"
        r"(?:接入|内置|连接|启用|配置)(?:了)?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:你|synbiogpt|系统))?(?:回答问题|回答|问答)(?:时)?"
        r"(?:是否|是不是|会不会|会|能否|能不能)?"
        r"(?:使用|查询|检索|基于)(?:内置的)?(?:全文文献)?(?:知识库|文献库)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:当前|现在|这个|该)(?:系统)?|synbiogpt|你|你们)"
        r"(?:是否|是不是|会不会|会|能否|能不能)?"
        r"(?:使用|查询|检索|基于)(?:内置的)?(?:全文文献)?(?:知识库|文献库)"
        r"(?:来)?(?:回答问题|回答|进行问答)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:does|is|can)\s+(?:synbiogpt|the\s+system|it|you)\b.*\b"
        r"(?:use|have|connect(?:ed)?\s+to|include)\b.*\bknowledge\s+base$",
        re.IGNORECASE,
    ),
)
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


def _normalize_route_message(message: Any) -> str:
    if not isinstance(message, str):
        return ""

    normalized = unicodedata.normalize("NFKC", message).casefold().strip()
    return _TRAILING_PUNCTUATION.sub("", normalized).strip()


def is_explicit_plain_chat(message: Any) -> bool:
    """Return whether the whole message is an explicit plain-chat phrase."""

    normalized = _normalize_route_message(message)
    return normalized in _PLAIN_CHAT_MESSAGES


def is_product_capability_question(message: Any) -> bool:
    """Identify high-confidence questions about SynBioGPT's knowledge base."""

    normalized = _normalize_route_message(message)
    if not normalized or _PRODUCT_CAPABILITY_EXCLUSION_RE.search(normalized):
        return False
    candidate = re.sub(r"(?:吗|么|呢|嘛)$", "", normalized).strip()
    return any(pattern.fullmatch(candidate) for pattern in _PRODUCT_CAPABILITY_PATTERNS)


def route_flags(message: Any) -> dict[str, bool]:
    """Return request-scoped routing flags for an original user message."""

    if is_product_capability_question(message):
        return {"product_capability": True}
    return {"plain_chat": True} if is_explicit_plain_chat(message) else {}


def requests_visual_evidence(message: Any) -> bool:
    """Return whether the user explicitly asks for figure or table evidence."""

    return isinstance(message, str) and bool(_VISUAL_EVIDENCE_RE.search(message))


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
