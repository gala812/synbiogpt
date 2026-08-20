"""Deterministic routing for explicit paper search and recommendation requests."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_PMID_PATTERNS = (
    re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})(?:/|\b)", re.I),
    re.compile(r"\bPMID\s*[:：#]?\s*(\d{5,9})\b", re.I),
)
_PAPER = re.compile(r"论文|文献|文章|papers?|articles?|literature", re.I)
_RELATED = re.compile(
    r"相关(?:的)?(?:论文|文献)|相似(?:的)?(?:论文|文献)|类似(?:的)?(?:论文|文献)|"
    r"推荐.*(?:论文|文献)|related papers?|similar papers?|recommend.*papers?",
    re.I,
)
_SUMMARY = re.compile(r"总结|概括|解读|summari[sz]e|overview", re.I)
_SEARCH = re.compile(r"找|查找|搜索|检索|有哪些|find|search|look for", re.I)
_EXPLICIT_TITLE_PATTERNS = (
    re.compile(r"《([^》]{6,500})》"),
    re.compile(r"标题\s*(?:为|是|[:：])\s*(.+?)(?:[。！？?]|$)", re.I),
    re.compile(r"(?:paper|article)\s+(?:titled|named)\s+(.+?)(?:[?]|$)", re.I),
)
_QUOTED_TITLE_PATTERNS = (
    re.compile(r"“([^”]{2,500})”"),
    re.compile(r'"([^"\n]{2,500})"'),
)
_RELATED_TITLE_PATTERN = re.compile(
    r"(?:推荐|查找|寻找).*?与\s*(.+?)\s*相关(?:的)?(?:论文|文献)",
    re.I,
)
_SEARCHED_TITLE_PATTERN = re.compile(
    r"(?:查找|寻找|搜索|find|search(?:\s+for)?)\s*"
    r"(?:论文|文献|文章|papers?|articles?)\s*[:：]?\s*(.+)$",
    re.I,
)


@dataclass(frozen=True, slots=True)
class PaperRequest:
    intent: str
    identifier_type: str
    identifier_value: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _extract_pmid(message: str) -> str:
    for pattern in _PMID_PATTERNS:
        if match := pattern.search(message):
            return match.group(1)
    if _RELATED.search(message):
        if match := re.search(r"(?<!\d)(\d{6,9})(?!\d)", message):
            return match.group(1)
    return ""


def _clean_title(value: str) -> str:
    return value.strip().strip("“”\"").rstrip("。！？?").strip()


def _looks_like_title(value: str) -> bool:
    """Reject short quoted topics while accepting plausible supplied titles."""

    value = _clean_title(value)
    latin_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+_.()-]*", value)
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", value))
    return len(latin_words) >= 4 or cjk_count >= 8


def _extract_title(message: str) -> str:
    for pattern in _EXPLICIT_TITLE_PATTERNS:
        if match := pattern.search(message):
            return _clean_title(match.group(1))

    if not (_PAPER.search(message) or _RELATED.search(message) or _SUMMARY.search(message)):
        return ""
    for pattern in (*_QUOTED_TITLE_PATTERNS, _RELATED_TITLE_PATTERN, _SEARCHED_TITLE_PATTERN):
        if match := pattern.search(message):
            candidate = _clean_title(match.group(1))
            if _looks_like_title(candidate):
                return candidate
    return ""


def parse_paper_request(message: object) -> PaperRequest | None:
    """Recognize only explicit paper operations; ordinary research QA returns None."""

    if not isinstance(message, str) or not (text := message.strip()):
        return None
    pmid = _extract_pmid(text)
    title = _extract_title(text)
    related = bool(_RELATED.search(text))
    summary = bool(_SUMMARY.search(text))
    paper_operation = related or bool(_PAPER.search(text) and _SEARCH.search(text))
    if not (pmid or title or paper_operation):
        return None

    intent = "related_papers" if related and (pmid or title) else (
        "paper_summary" if summary and (pmid or title) else "paper_search"
    )
    if pmid:
        return PaperRequest(intent, "pmid", pmid)
    if title:
        return PaperRequest(intent, "title", title)
    return PaperRequest("paper_search", "topic", text)
