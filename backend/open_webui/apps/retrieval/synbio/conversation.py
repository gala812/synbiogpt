"""Lightweight, request-scoped conversation context management."""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from open_webui.apps.retrieval.query_processor import extract_exact_terms


QUERY_HISTORY_TOKEN_BUDGET = 3_000
ANSWER_INPUT_TOKEN_BUDGET = 24_000
MAX_ACTIVE_ENTITIES = 12

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._+()/-][A-Za-z0-9+()-]*)*")
_FOLLOW_UP_RE = re.compile(
    r"(?:同时|这种|这些|那些|上述|前述|其中|如果再|再使用|那么|那|该|它|其|"
    r"为什么|怎么样|怎么做|如何控制|多少时|呢[？?]?$|"
    r"\b(?:why|what about|how about|that|those|these|this|it|also|then)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MessageWindow:
    messages: list[dict[str, Any]]
    estimated_tokens: int
    omitted_messages: int


class RecentQueryCache:
    """Small best-effort cache for follow-up fallback, never a source of evidence."""

    def __init__(self, *, max_entries: int = 512, ttl_seconds: int = 3_600):
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = max(1, ttl_seconds)
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        item = self._items.pop(key, None)
        if item is None:
            return None
        created_at, value = item
        if time.monotonic() - created_at > self.ttl_seconds:
            return None
        self._items[key] = item
        return value

    def put(self, key: str, value: Any) -> None:
        if not key:
            return
        self._items.pop(key, None)
        self._items[key] = (time.monotonic(), value)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


def message_text(message: dict[str, Any]) -> str:
    """Return textual content from plain or OpenAI multimodal messages."""

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )


def estimate_tokens(message: dict[str, Any]) -> int:
    """Conservatively estimate text and image tokens without loading a model."""

    text = message_text(message)
    cjk = len(_CJK_RE.findall(text))
    non_cjk = _CJK_RE.sub("", text)
    text_tokens = cjk + max(len(non_cjk) // 4, len(_WORD_RE.findall(non_cjk)))
    content = message.get("content")
    image_count = (
        sum(
            isinstance(item, dict) and item.get("type") == "image_url"
            for item in content
        )
        if isinstance(content, list)
        else 0
    )
    return max(1, text_tokens + image_count * 768 + 4)


def select_message_window(
    messages: list[dict[str, Any]], *, token_budget: int
) -> MessageWindow:
    """Keep system context and the newest complete messages within a token budget."""

    values = [message for message in messages or [] if isinstance(message, dict)]
    system = [message for message in values if message.get("role") == "system"]
    conversation = [message for message in values if message.get("role") != "system"]
    selected = list(system)
    used = sum(estimate_tokens(message) for message in system)

    recent: list[dict[str, Any]] = []
    for message in reversed(conversation):
        cost = estimate_tokens(message)
        if recent and used + cost > token_budget:
            break
        recent.append(message)
        used += cost
    recent.reverse()
    selected.extend(recent)
    return MessageWindow(selected, used, len(values) - len(selected))


def query_message_window(messages: list[dict[str, Any]]) -> MessageWindow:
    return select_message_window(messages, token_budget=QUERY_HISTORY_TOKEN_BUDGET)


def answer_message_window(messages: list[dict[str, Any]]) -> MessageWindow:
    return select_message_window(messages, token_budget=ANSWER_INPUT_TOKEN_BUDGET)


def is_contextual_follow_up(query: Any) -> bool:
    return isinstance(query, str) and bool(_FOLLOW_UP_RE.search(query.strip()))


def inherited_exact_terms(
    messages: list[dict[str, Any]], latest_query: str
) -> tuple[str, ...]:
    """Carry user-supplied scientific identifiers only for explicit follow-ups."""

    if not is_contextual_follow_up(latest_query):
        return ()

    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        len(messages),
    )
    inherited: list[str] = []
    seen = {term.casefold() for term in extract_exact_terms(latest_query)}
    for message in reversed(messages[:latest_user_index]):
        if message.get("role") != "user":
            continue
        for term in extract_exact_terms(message_text(message)):
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            inherited.append(term)
            if len(inherited) >= MAX_ACTIVE_ENTITIES:
                return tuple(inherited)
    return tuple(inherited)


def fallback_semantic_query(
    original_query: str, inherited_terms: tuple[str, ...]
) -> str | None:
    """Build a conservative English retry query from user-provided identifiers."""

    terms = list(
        dict.fromkeys([*extract_exact_terms(original_query), *inherited_terms])
    )
    if not terms:
        return None
    return f"Scientific literature about {' '.join(terms)}"


def conversation_key(user_id: Any, metadata: dict[str, Any] | None) -> str:
    metadata = metadata or {}
    conversation_id = metadata.get("chat_id") or metadata.get("session_id")
    return f"{user_id}:{conversation_id}" if user_id and conversation_id else ""
