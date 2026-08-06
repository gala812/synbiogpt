from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceBlock:
    kind: str
    text: str
    char_start: int
    char_end: int
    order: int
    heading_level: int | None = None
    image_path: str | None = None
    section: str = "Front Matter"
    subsection: str = ""
    excluded_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    asset_id: str | None = None


@dataclass
class TextUnit:
    kind: str
    text: str
    section: str
    subsection: str
    char_start: int
    char_end: int
    order: float
    warnings: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)
    source_spans: list[list[int]] = field(default_factory=list)
    is_long_split: bool = False


@dataclass
class Asset:
    asset_type: str
    asset_id: str
    label: str
    caption: str
    image_paths: list[str]
    section: str
    subsection: str
    char_start: int
    char_end: int
    order: float
    context_before: str = ""
    context_after: str = ""
    notes: str = ""
    table_text_missing: bool = False
    mapping_confidence: str = "low"
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentCandidate:
    pmcid: str
    markdown_path: Path
    duplicate_paths: list[Path] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pmcid: str
    paper_title: str
    title_source: str
    source_markdown: str
    source_file: str
    blocks: list[SourceBlock]
    assets: list[Asset]
    parse_warnings: list[str]
    section_tree: dict[str, list[str]]
    excluded_counts: dict[str, int]
    unknown_headings: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkingConfig:
    target_min_words: int = 180
    target_max_words: int = 260
    short_paragraph_words: int = 60
    min_chunk_words: int = 80
    soft_max_words: int = 280
    hard_max_words: int = 320
    long_overlap_words: int = 40
    hard_max_tokens: int = 448
    parent_min_words: int = 500
    parent_max_words: int = 900

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
