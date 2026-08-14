"""Validate and normalize model-generated scientific retrieval queries."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._+()/-][A-Za-z0-9+()-]*)*")
_STRONG_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"BBa_[A-Za-z0-9_]+|"
    r"p[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9+().-]+)+|"
    r"(?:CRISPRi|dCas9|IPTG|OD\d+)|"
    r"[A-Z]{2,}\d*|"
    r"[a-z]{2,}\d*[A-Z]\d*"
    r")(?![A-Za-z0-9_])"
)
_KNOWN_LOWERCASE_ENTITIES = {"ppc"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "which",
    "with",
}


class QueryProcessingError(ValueError):
    """Raised when a usable English retrieval query cannot be produced."""


@dataclass(frozen=True)
class QueryPreparation:
    original_query: str
    protected_query: str
    exact_terms: tuple[str, ...]


@dataclass(frozen=True)
class ProcessedQuery:
    original_query: str
    semantic_query: str
    lexical_query: str
    exact_terms: tuple[str, ...]

    @property
    def bm25_query(self) -> str:
        """Return the lexical expression with every exact entity included."""

        return _append_missing(self.lexical_query, list(self.exact_terms))

    def to_dict(self) -> dict:
        result = asdict(self)
        result["exact_terms"] = list(self.exact_terms)
        return result


class QueryProcessor:
    """Keep model query generation deterministic at the retrieval boundary."""

    def prepare(self, query: str) -> QueryPreparation:
        original = _normalize_query(query)
        if not original:
            raise QueryProcessingError("Query must not be empty")
        exact_terms = extract_exact_terms(original)
        protected, _ = _protect_terms(original, exact_terms)
        return QueryPreparation(original, protected, tuple(exact_terms))

    def process(self, query: str) -> ProcessedQuery:
        """Normalize an English query without invoking a model."""

        preparation = self.prepare(query)
        if _CJK_RE.search(preparation.original_query):
            raise QueryProcessingError(
                "Chinese queries must pass through retrieval query generation first"
            )
        return self._build(preparation, preparation.original_query, "")

    def process_model_output(
        self, query: str, model_output: str | dict
    ) -> ProcessedQuery:
        """Validate the first base-model call and restore protected entities."""

        preparation = self.prepare(query)
        data = _parse_model_output(model_output)
        semantic = data.get("semantic_query")
        if not semantic and isinstance(data.get("queries"), list):
            semantic = next((item for item in data["queries"] if item), "")

        _, placeholders = _protect_terms(
            preparation.original_query, list(preparation.exact_terms)
        )
        semantic = _restore_terms(_normalize_query(semantic), placeholders)
        if not semantic or _CJK_RE.search(semantic):
            raise QueryProcessingError(
                "The query-generation model did not return an English semantic_query"
            )

        lexical = _restore_terms(
            _normalize_query(data.get("lexical_query")), placeholders
        )
        if _CJK_RE.search(lexical):
            lexical = ""
        return self._build(preparation, semantic, lexical)

    @staticmethod
    def _build(
        preparation: QueryPreparation, semantic: str, lexical: str
    ) -> ProcessedQuery:
        lexical = lexical or _lexical_from_semantic(semantic)
        lexical = _expand_high_confidence_terms(
            preparation.original_query, semantic, lexical
        )
        return ProcessedQuery(
            original_query=preparation.original_query,
            semantic_query=semantic,
            lexical_query=lexical,
            exact_terms=preparation.exact_terms,
        )


def extract_exact_terms(query: str) -> list[str]:
    """Extract high-confidence identifiers while avoiding ordinary English words."""

    candidates: list[tuple[int, str]] = [
        (match.start(), match.group(0)) for match in _STRONG_ENTITY_RE.finditer(query)
    ]
    if _CJK_RE.search(query):
        candidates.extend(
            (match.start(), match.group(0)) for match in _WORD_RE.finditer(query)
        )
    else:
        candidates.extend(
            (match.start(), match.group(0))
            for match in _WORD_RE.finditer(query)
            if match.group(0).lower() in _KNOWN_LOWERCASE_ENTITIES
        )

    ordered = sorted(candidates, key=lambda item: item[0])
    return _unique([term for _, term in ordered])


def _parse_model_output(output: str | dict) -> dict:
    if isinstance(output, dict):
        if "semantic_query" in output or "queries" in output:
            return output
        raise QueryProcessingError(
            "Query-generation output is not a valid retrieval query object"
        )
    text = str(output or "").strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and (
            "semantic_query" in data or "queries" in data
        ):
            return data
    raise QueryProcessingError("Query-generation output is not a valid JSON object")


def _protect_terms(query: str, terms: list[str]) -> tuple[str, dict[str, str]]:
    protected = query
    placeholders: dict[str, str] = {}
    for index, term in enumerate(sorted(terms, key=len, reverse=True)):
        placeholder = f"ZXQENTITY{index}QXZ"
        protected = protected.replace(term, placeholder)
        placeholders[placeholder] = term
    return protected, placeholders


def _restore_terms(query: str, placeholders: dict[str, str]) -> str:
    restored = query
    for placeholder, term in placeholders.items():
        restored = re.sub(
            re.escape(placeholder), term, restored, flags=re.IGNORECASE
        )
        restored = _append_missing(restored, [term])
    return _normalize_query(restored)


def _lexical_from_semantic(semantic: str) -> str:
    return " ".join(
        word
        for word in _WORD_RE.findall(semantic)
        if word.lower() not in _STOPWORDS
    )


def _expand_high_confidence_terms(
    original: str, semantic: str, lexical: str
) -> str:
    combined = f"{original} {semantic}".lower()
    expansions: list[str] = []
    if "大肠杆菌" in original or re.search(
        r"\b(?:escherichia\s+coli|e\.?\s*coli)\b", combined
    ):
        expansions.extend(["Escherichia coli", "E. coli"])
    if "丁二酸" in original or re.search(
        r"\b(?:succinate|succinic\s+acid)\b", combined
    ):
        expansions.extend(["succinate", "succinic acid"])
    if "crispr干扰" in combined or re.search(
        r"\b(?:crispr\s+interference|crispri)\b", combined
    ):
        expansions.extend(["CRISPR interference", "CRISPRi"])
    if "敲除" in original or re.search(r"\b(?:delet(?:e|ion)|knockout)\b", combined):
        expansions.extend(["deletion", "knockout"])
    return _append_missing(lexical, expansions)


def _normalize_query(query: object) -> str:
    return _SPACE_RE.sub(" ", str(query or "")).strip()


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = value.strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _append_missing(text: str, values: list[str]) -> str:
    result = text.strip()
    for value in _unique(values):
        pattern = re.escape(value).replace(r"\ ", r"\s+")
        if not re.search(rf"(?<!\w){pattern}(?!\w)", result, re.IGNORECASE):
            result = f"{result} {value}".strip()
    return result
