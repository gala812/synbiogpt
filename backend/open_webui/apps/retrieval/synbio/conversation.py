"""Lightweight, request-scoped conversation context management."""

from __future__ import annotations

import re
import time
import unicodedata
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


@dataclass(frozen=True, slots=True)
class PaperReference:
    citation_index: int
    pmid: str
    pmcid: str
    title: str


@dataclass(frozen=True, slots=True)
class PaperFollowUp:
    status: str
    papers: tuple[PaperReference, ...]


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


_PAPER_REFERENCE_RE = re.compile(
    r"(?:这|该|上述|上面|前述)(?:一|些)?篇?(?:论文|文献|文章)|"
    r"\b(?:this|the|these|those|above|previous)\s+(?:paper|article|papers|articles)\b",
    re.IGNORECASE,
)
_PLURAL_PAPER_REFERENCE_RE = re.compile(
    r"(?:这些|上述|上面|前述)(?:论文|文献|文章)|"
    r"\b(?:these|those|above|previous)\s+(?:papers|articles)\b",
    re.IGNORECASE,
)
_PAPER_INDEX_PATTERNS = (
    re.compile(r"\[(\d{1,2})\]"),
    re.compile(r"第\s*(\d{1,2})\s*篇"),
)
_CHINESE_PAPER_INDEX_RE = re.compile(r"第\s*([一二三四五六七八九十])\s*篇")
_CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def is_paper_contextual_follow_up(query: Any) -> bool:
    """Return whether a turn explicitly refers to papers from an earlier turn."""

    if not isinstance(query, str):
        return False
    text = query.strip()
    return bool(
        _PAPER_REFERENCE_RE.search(text)
        or _CHINESE_PAPER_INDEX_RE.search(text)
        or any(pattern.search(text) for pattern in _PAPER_INDEX_PATTERNS)
    )


def _branch_messages(chat: Any) -> list[dict[str, Any]]:
    """Return persisted messages from newest to oldest on the active branch."""

    if not isinstance(chat, dict):
        return []
    history = chat.get("history")
    if not isinstance(history, dict):
        return []
    messages = history.get("messages")
    if isinstance(messages, list):
        return [item for item in reversed(messages) if isinstance(item, dict)]
    if not isinstance(messages, dict):
        return []

    current_id = history.get("currentId")
    current = messages.get(current_id)
    if not isinstance(current, dict):
        return sorted(
            (item for item in messages.values() if isinstance(item, dict)),
            key=lambda item: int(item.get("timestamp") or 0),
            reverse=True,
        )

    branch = []
    seen = set()
    while isinstance(current, dict):
        message_id = str(current.get("id") or "")
        if message_id and message_id in seen:
            break
        if message_id:
            seen.add(message_id)
        branch.append(current)
        current = messages.get(current.get("parentId"))
    return branch


def recent_specter2_papers(chat: Any) -> tuple[PaperReference, ...]:
    """Read the most recent SPECTER2 citation set from persisted chat history."""

    for message in _branch_messages(chat):
        if message.get("role") != "assistant":
            continue
        sources = message.get("sources")
        if not isinstance(sources, list):
            continue

        papers = []
        seen_pmids = set()
        for fallback_index, source in enumerate(sources, 1):
            if not isinstance(source, dict):
                continue
            metadata_values = source.get("metadata")
            metadata_list = (
                metadata_values
                if isinstance(metadata_values, list)
                else [metadata_values]
            )
            for metadata in metadata_list:
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("retrieval_source") != "specter2_paper":
                    continue
                pmid = str(metadata.get("pmid") or source.get("pmid") or "").strip()
                if not pmid or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                citation_index = source.get("citation_index") or fallback_index
                try:
                    citation_index = int(citation_index)
                except (TypeError, ValueError):
                    citation_index = fallback_index
                papers.append(
                    PaperReference(
                        citation_index=citation_index,
                        pmid=pmid,
                        pmcid=str(metadata.get("pmcid") or "").strip(),
                        title=str(
                            source.get("title")
                            or metadata.get("paper_title")
                            or metadata.get("title")
                            or ""
                        ).strip(),
                    )
                )
        if papers:
            return tuple(papers)
    return ()


def _normalized_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def resolve_paper_follow_up(
    query: Any, papers: tuple[PaperReference, ...]
) -> PaperFollowUp:
    """Resolve explicit singular, plural, numbered, or titled paper references."""

    if not papers or not is_paper_contextual_follow_up(query):
        return PaperFollowUp("none", ())
    text = str(query).strip()

    requested_indices = []
    for pattern in _PAPER_INDEX_PATTERNS:
        requested_indices.extend(int(value) for value in pattern.findall(text))
    requested_indices.extend(
        _CHINESE_NUMBERS[value] for value in _CHINESE_PAPER_INDEX_RE.findall(text)
    )
    if requested_indices:
        requested = set(requested_indices)
        selected = tuple(
            paper for paper in papers if paper.citation_index in requested
        )
        return PaperFollowUp("resolved", selected) if selected else PaperFollowUp(
            "ambiguous", papers
        )

    normalized_query = _normalized_title(text)
    title_matches = []
    for paper in papers:
        normalized = _normalized_title(paper.title)
        if len(normalized) >= 12 and normalized in normalized_query:
            title_matches.append(paper)
    if title_matches:
        matches = tuple(title_matches)
        return PaperFollowUp(
            "resolved" if len(matches) == 1 else "ambiguous", matches
        )
    if _PLURAL_PAPER_REFERENCE_RE.search(text):
        return PaperFollowUp("resolved", papers)
    if len(papers) == 1:
        return PaperFollowUp("resolved", papers)
    return PaperFollowUp("ambiguous", papers)
