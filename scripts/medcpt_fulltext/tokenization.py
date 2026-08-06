from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


MEDCPT_TOKENIZER = "ncbi/MedCPT-Article-Encoder"


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


@dataclass
class RegexTokenCounter:
    """Dependency-free deterministic fallback used only when no real tokenizer exists."""

    name: str = "generic:unicode_word_punctuation_v1"

    def count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)) + 2


@dataclass
class TiktokenCounter:
    encoding: object
    name: str = "tiktoken:cl100k_base"

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))


@dataclass
class HuggingFaceTokenCounter:
    tokenizer: object
    name: str

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))


def resolve_tokenizer(
    preferred: str = MEDCPT_TOKENIZER,
    *,
    allow_fallback: bool = True,
    local_files_only: bool = False,
) -> TokenCounter:
    """Load MedCPT's tokenizer, recording an explicit fallback when unavailable."""

    if preferred.startswith("generic:"):
        return RegexTokenCounter(name=preferred)
    if preferred.startswith("tiktoken:"):
        try:
            import tiktoken

            encoding_name = preferred.split(":", 1)[1]
            return TiktokenCounter(tiktoken.get_encoding(encoding_name), preferred)
        except Exception:
            if not allow_fallback:
                raise
            return RegexTokenCounter()

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            preferred,
            local_files_only=local_files_only,
            use_fast=True,
        )
        return HuggingFaceTokenCounter(tokenizer, preferred)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(f"Unable to load required tokenizer {preferred!r}") from exc

    try:
        import tiktoken

        return TiktokenCounter(tiktoken.get_encoding("cl100k_base"))
    except Exception:
        return RegexTokenCounter()

