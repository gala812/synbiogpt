from __future__ import annotations

from collections import Counter, defaultdict
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .models import Asset, ParsedDocument, SourceBlock


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<path>[^)]+)\)")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
EQUATION_START_RE = re.compile(r"^\s*(?:\$\$|\\\[|\\begin\{(?:equation|align|gather|multline)\*?\})")
EQUATION_NUMBER_ONLY_RE = re.compile(r"^\s*\(\s*\d+[A-Za-z]?\s*\)\s*$")
TABLE_CAPTION_RE = re.compile(r"^\s*(?P<label>Table\s+(?:S?\d+[A-Za-z]?|[IVXLC]+))\s*[.:\-]?\s*(?P<body>.*)$", re.I | re.S)
FIGURE_CAPTION_RE = re.compile(r"^\s*(?P<label>(?:Figure|Fig\.)\s+(?:S?\d+[A-Za-z]?|[IVXLC]+))\s*[.:\-]?\s*(?P<body>.*)$", re.I | re.S)
EMBEDDED_TABLE_CAPTION_RE = re.compile(r"(?<!\w)(?P<label>Table\s+(?:S?\d+[A-Za-z]?|[IVXLC]+))\s*[.:]\s*(?P<body>.*)$", re.I | re.S)
EMBEDDED_FIGURE_CAPTION_RE = re.compile(r"(?<!\w)(?P<label>(?:Figure|Fig\.)\s+(?:S?\d+[A-Za-z]?|[IVXLC]+))\s*[.:]\s*(?P<body>.*)$", re.I | re.S)
MANGLED_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?P<label>(?:igure|gure|ure|re|e)\s+(?:S?\d+[A-Za-z]?|[IVXLC]+))\s*[.:]\s*(?P<body>.*)$",
    re.I | re.S,
)
MANGLED_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?P<label>(?:able|ble)\s+(?:S?\d+[A-Za-z]?|[IVXLC]+))\s*[.:]\s*(?P<body>.*)$",
    re.I | re.S,
)

AD_PHRASES = (
    "submit your next manuscript",
    "convenient online submission",
    "thorough peer review",
    "no space constraints",
)

INLINE_NON_BODY_RE = re.compile(
    r"^\s*(?:"
    r"Author Contributions?|Authors['’]? Contributions?|"
    r"Acknowledgements?|Acknowledgments?|"
    r"Funding(?: Information)?|"
    r"Data Availability(?: Statement)?|Availability of Data and Materials|"
    r"Conflicts? of Interest|Competing Interests?|"
    r"Institutional Review Board Statement|Informed Consent Statement|"
    r"Consent for Publication|Ethics Approval and Consent to Participate|"
    r"Publisher['’]?s Note|Declarations?"
    r")\s*:",
    re.I,
)

NON_BODY_FIXED_PHRASES = (
    "refer to web version on pubmed central for supplementary material",
    "the following supporting information can be downloaded",
)

PANEL_COMPARISON_RE = re.compile(
    r"^(?:[A-Z]+\d*|R\d+|S\d+)\s*[-:]?\s*vs\.?\s*[-:]?\s*(?:[A-Z]+\d*|R\d+|S\d+)$",
    re.I,
)
PANEL_LABEL_RE = re.compile(
    r"^\s*(?:\(?[A-Ha-h]\)?[.)]?|\(?continued\)?|merged)\s*$",
    re.I,
)

PUBLISHER_HEADING_KEYS = {
    "article",
    "article info",
    "articleinfo",
    "citation",
    "copyright",
    "correspondence",
    "graphical abstract",
    "highlights",
    "open access",
    "specialty section",
}
PUBLISHER_HEADING_PREFIXES = (
    "academic editor",
    "check for update",
    "citation ",
    "copyright ",
    "open access ",
    "ready to submit your research",
)

EXCLUDED_SECTION_KEYS = {
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "authors contributions",
    "author contributions",
    "author contribution",
    "funding",
    "funding information",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "declarations",
    "authors details",
    "author details",
    "keywords",
    "keyword",
    "data availability",
    "data availability statement",
    "declaration of competing interest",
}


def normalize_heading_key(text: str) -> str:
    value = re.sub(r"[*_`#]", "", text).strip()
    value = re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.?)\s+", "", value, flags=re.I)
    letters = value.split()
    if len(letters) >= 4 and all(len(letter) == 1 and letter.isalpha() for letter in letters):
        value = "".join(letters)
    value = value.replace("&", " and ").replace("–", "-").replace("—", "-")
    value = re.sub(r"[-_/]+", " ", value)
    value = re.sub(r"[^A-Za-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def canonical_main_section(text: str) -> str | None:
    key = normalize_heading_key(text)
    exact = {
        "abstract": "Abstract",
        "summary": "Abstract",
        "introduction": "Introduction",
        "background": "Background",
        "materials and methods": "Methods",
        "material and methods": "Methods",
        "materials and method": "Methods",
        "methods": "Methods",
        "method": "Methods",
        "methodology": "Methods",
        "experimental procedures": "Methods",
        "experimental procedure": "Methods",
        "results": "Results",
        "result": "Results",
        "results and discussion": "Results and Discussion",
        "result and discussion": "Results and Discussion",
        "discussion": "Discussion",
        "discussions": "Discussion",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "concluding remarks": "Conclusion",
        "supplementary information": "Supplementary Information",
        "supporting information": "Supplementary Information",
        "references": "References",
        "bibliography": "References",
    }
    return exact.get(key)


def _is_excluded_heading(text: str) -> bool:
    return normalize_heading_key(text) in EXCLUDED_SECTION_KEYS


def _is_publisher_heading(text: str) -> bool:
    key = normalize_heading_key(text)
    return key in PUBLISHER_HEADING_KEYS or key.startswith(PUBLISHER_HEADING_PREFIXES)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _is_page_marker(text: str) -> bool:
    return bool(
        re.fullmatch(r"\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*", text, re.I)
        or re.fullmatch(r"\s*[-–—]\s*\d+\s*[-–—]\s*", text)
    )


def _is_advertisement(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in AD_PHRASES)


def _is_inline_non_body(text: str) -> bool:
    lower = text.lower()
    return bool(INLINE_NON_BODY_RE.match(text)) or any(
        phrase in lower for phrase in NON_BODY_FIXED_PHRASES
    )


def _is_figure_panel_artifact(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if (
        len(text.split()) <= 25
        and not FIGURE_CAPTION_RE.match(text)
        and re.search(r"\bFigure\s+S?\d+[A-Za-z]?\.\s*Cont\.?", text, re.I)
    ):
        return True
    return bool(PANEL_LABEL_RE.fullmatch(text)) or all(
        PANEL_COMPARISON_RE.fullmatch(line)
        or PANEL_LABEL_RE.fullmatch(line)
        for line in lines
    )


def _is_isolated_text_fragment(text: str) -> bool:
    """Drop tiny OCR/table fragments that cannot form a useful retrieval unit."""

    words = text.split()
    return (
        0 < len(words) <= 3
        and len(text) <= 40
        and not re.search(r"[.!?][\"')\]]?\s*$", text)
    )


SECTION_NUMBER_RE = re.compile(
    r"(?P<number>\d+(?:\.\d+)+)(?:\.)?\s+(?=[A-Z])"
)


def _section_number_candidates(text: str) -> list[tuple[tuple[int, ...], int]]:
    return [
        (tuple(int(value) for value in match.group("number").split(".")), match.start())
        for match in SECTION_NUMBER_RE.finditer(text)
    ]


def _numbers_are_adjacent(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return (
        len(left) == len(right)
        and len(left) >= 2
        and left[:-1] == right[:-1]
        and right[-1] == left[-1] + 1
    )


def _repair_corrupted_heading(
    text: str,
    previous_numbers: list[tuple[int, ...]],
    next_numbers: list[tuple[int, ...]],
) -> tuple[str, list[str]]:
    """Repair page-merged headings only when adjacent numbering supports the split."""

    value = text.strip()
    warnings: list[str] = []
    midpoint = len(value) // 2
    if len(value) >= 20 and len(value) % 2 == 0:
        left, right = value[:midpoint].strip(), value[midpoint:].strip()
        if left and left.casefold() == right.casefold():
            return left, ["repaired_duplicate_heading"]

    candidates = _section_number_candidates(value)
    for number, start in reversed(candidates):
        if start <= 0:
            continue
        suffix = value[start:].strip()
        suffix_body = SECTION_NUMBER_RE.match(suffix)
        heading_words = suffix_body.group(0) if suffix_body else ""
        remainder = suffix[len(heading_words) :].strip()
        plausible_suffix = 1 <= len(remainder.split()) <= 20 and not re.search(
            r"[.!?]\s+\S", remainder
        )
        supported = any(_numbers_are_adjacent(previous, number) for previous in previous_numbers)
        supported = supported or any(_numbers_are_adjacent(number, following) for following in next_numbers)
        if plausible_suffix and supported:
            return suffix, ["repaired_page_merged_heading_sequence"]
        if plausible_suffix:
            warnings.append("possible_page_merged_heading_unverified")
    return value, warnings


def _repair_corrupted_headings(blocks: list[SourceBlock]) -> None:
    heading_indexes = [index for index, block in enumerate(blocks) if block.kind == "heading"]
    numbers_by_index = {
        index: [number for number, _ in _section_number_candidates(blocks[index].text)]
        for index in heading_indexes
    }
    for position, index in enumerate(heading_indexes):
        previous = numbers_by_index.get(heading_indexes[position - 1], []) if position else []
        following = (
            numbers_by_index.get(heading_indexes[position + 1], [])
            if position + 1 < len(heading_indexes)
            else []
        )
        blocks[index].text, repairs = _repair_corrupted_heading(
            blocks[index].text, previous, following
        )
        blocks[index].warnings.extend(repairs)


def _valid_title(text: str) -> bool:
    cleaned = _clean_text(text)
    words = cleaned.split()
    if not 3 <= len(words) <= 60:
        return False
    if canonical_main_section(cleaned) or _is_excluded_heading(cleaned):
        return False
    key = normalize_heading_key(cleaned)
    if key in {
        "article", "research article", "original article", "review", "viewpoint"
    } or _is_publisher_heading(cleaned):
        return False
    return len(re.sub(r"[^A-Za-z0-9]", "", cleaned)) >= 12


def parse_markdown_blocks(raw: str) -> list[SourceBlock]:
    """Parse block-level Markdown while retaining original character offsets."""

    lines = raw.splitlines(keepends=True)
    blocks: list[SourceBlock] = []
    pending: list[tuple[str, int, int]] = []
    pending_kind = "paragraph"
    order = 0

    def flush() -> None:
        nonlocal pending, pending_kind, order
        if not pending:
            return
        text = _clean_text("".join(item[0] for item in pending))
        if text:
            kind = pending_kind
            warnings: list[str] = []
            if EQUATION_NUMBER_ONLY_RE.fullmatch(text):
                kind = "equation"
                warnings.append("equation_number_without_body")
            blocks.append(
                SourceBlock(
                    kind=kind,
                    text=text,
                    char_start=pending[0][1],
                    char_end=pending[-1][2],
                    order=order,
                    warnings=warnings,
                )
            )
            order += 1
        pending = []
        pending_kind = "paragraph"

    offset = 0
    in_equation = False
    for line in lines:
        start, end = offset, offset + len(line)
        offset = end
        stripped = line.strip()
        if in_equation:
            pending.append((line, start, end))
            if stripped.endswith("$$") or stripped == r"\]" or re.search(r"\\end\{(?:equation|align|gather|multline)\*?\}", stripped):
                flush()
                in_equation = False
            continue
        if not stripped:
            flush()
            continue
        heading = HEADING_RE.match(stripped)
        if heading:
            flush()
            blocks.append(
                SourceBlock(
                    kind="heading",
                    text=_clean_text(heading.group(2)),
                    char_start=start,
                    char_end=end,
                    order=order,
                    heading_level=len(heading.group(1)),
                )
            )
            order += 1
            continue
        image_matches = list(IMAGE_RE.finditer(stripped))
        if image_matches and re.sub(IMAGE_RE, "", stripped).strip() == "":
            flush()
            for match in image_matches:
                image_path = PurePosixPath(match.group("path").strip().strip('"\'')).as_posix()
                blocks.append(
                    SourceBlock(
                        kind="image",
                        text=match.group(0),
                        char_start=start + match.start(),
                        char_end=start + match.end(),
                        order=order,
                        image_path=image_path,
                    )
                )
                order += 1
            continue
        if EQUATION_START_RE.match(stripped):
            flush()
            pending_kind = "equation"
            pending.append((line, start, end))
            if (stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4) or stripped == r"\]":
                flush()
            else:
                in_equation = True
            continue
        line_kind = "list" if LIST_RE.match(stripped) else "paragraph"
        if pending and pending_kind != line_kind:
            flush()
        pending_kind = line_kind
        pending.append((line, start, end))
    flush()
    return _repair_confident_heading_merges(blocks)


def _repair_confident_heading_merges(blocks: list[SourceBlock]) -> list[SourceBlock]:
    repaired: list[SourceBlock] = []
    pattern = re.compile(
        r"^(?P<head>\d+(?:\.\d+)+\.?\s+[A-Z][^.!?\n]{2,90})\.\s+(?P<body>\S.{30,})$",
        re.S,
    )
    for block in blocks:
        if block.kind != "paragraph":
            repaired.append(block)
            continue
        match = pattern.match(block.text)
        if not match or len(match.group("head").split()) > 14:
            # Flag likely merges without changing uncertain source text.
            key = normalize_heading_key(block.text[:120])
            if any(key.startswith(x + " ") for x in ("introduction", "methods", "results", "discussion", "conclusion")):
                block.warnings.append("possible_heading_body_merge")
            repaired.append(block)
            continue
        head_text = match.group("head")
        body_text = match.group("body")
        split_at = block.text.find(body_text)
        repaired.append(
            SourceBlock(
                kind="heading",
                text=head_text,
                char_start=block.char_start,
                char_end=block.char_start + split_at,
                order=block.order,
                heading_level=2,
                warnings=["repaired_heading_body_merge"],
            )
        )
        repaired.append(
            SourceBlock(
                kind="paragraph",
                text=body_text,
                char_start=block.char_start + split_at,
                char_end=block.char_end,
                order=block.order + 0.1,
                warnings=["repaired_heading_body_merge"],
            )
        )
    for index, block in enumerate(repaired):
        block.order = index
    return repaired


def _assign_structure_and_exclusions(
    blocks: list[SourceBlock], paper_title: str
) -> tuple[dict[str, list[str]], list[str], Counter[str], list[str]]:
    section = "Front Matter"
    subsection = ""
    excluded_section = False
    references_seen = False
    unknown: list[str] = []
    warnings: list[str] = []
    excluded: Counter[str] = Counter()
    section_tree: dict[str, list[str]] = defaultdict(list)
    seen_text: Counter[str] = Counter()
    normalized_title = normalize_heading_key(paper_title)
    seen_main_section = False

    _repair_corrupted_headings(blocks)
    for block in blocks:
        if block.kind == "heading":
            main = canonical_main_section(block.text)
            if main:
                section, subsection = main, ""
                seen_main_section = True
                excluded_section = main == "References"
                if main == "References":
                    references_seen = True
                section_tree.setdefault(section, [])
            elif normalize_heading_key(block.text) == normalized_title:
                block.excluded_reason = "paper_title"
                excluded["paper_title"] += 1
            elif _is_publisher_heading(block.text):
                block.excluded_reason = "publisher_heading"
                excluded["publisher_heading"] += 1
            elif _is_excluded_heading(block.text):
                subsection = block.text
                excluded_section = True
                block.excluded_reason = "excluded_section_heading"
                excluded["non_body_section"] += 1
            else:
                subsection = block.text
                if subsection not in section_tree[section]:
                    section_tree[section].append(subsection)
                heading_body = re.sub(
                    r"^\s*\d+(?:\.\d+)*\.?\s+", "", subsection
                )
                confident_subsection = (
                    section != "Front Matter"
                    and len(subsection.split()) <= 20
                    and not re.search(r"[.!?]\s+\S", heading_body)
                )
                if not confident_subsection:
                    unknown.append(block.text)
                    block.warnings.append("unknown_heading_as_subsection")
            block.section, block.subsection = section, subsection
            if references_seen and section == "References":
                block.excluded_reason = block.excluded_reason or "references"
                excluded["references"] += 1
            continue

        block.section, block.subsection = section, subsection
        inline_abstract = re.match(r"^(Abstract|Summary|Simple Summary)\s*:\s*(.+)$", block.text, re.I | re.S)
        if not seen_main_section and inline_abstract and block.kind == "paragraph":
            section = "Abstract"
            subsection = "Simple Summary" if inline_abstract.group(1).lower().startswith("simple") else ""
            block.section, block.subsection = section, subsection
            block.text = inline_abstract.group(2).strip()
            section_tree.setdefault(section, [])
            if subsection and subsection not in section_tree[section]:
                section_tree[section].append(subsection)
            seen_main_section = True
        if _is_page_marker(block.text):
            block.excluded_reason = "page_header_footer"
            excluded["page_header_footer"] += 1
            continue
        if _is_advertisement(block.text):
            block.excluded_reason = "publisher_advertisement"
            excluded["advertisement"] += 1
            continue
        if _is_inline_non_body(block.text):
            block.excluded_reason = "inline_non_body"
            excluded["inline_non_body"] += 1
            continue
        if _is_figure_panel_artifact(block.text):
            block.excluded_reason = "figure_panel_artifact"
            excluded["figure_panel_artifact"] += 1
            continue
        if block.kind == "paragraph" and _is_isolated_text_fragment(block.text):
            block.excluded_reason = "isolated_text_fragment"
            excluded["isolated_text_fragment"] += 1
            continue
        if section == "Front Matter" and block.kind != "image":
            block.excluded_reason = "front_matter"
            excluded["front_matter"] += 1
            continue
        if re.match(r"^Keywords?\s*:", block.text, re.I):
            block.excluded_reason = "keywords"
            excluded["keywords"] += 1
            continue
        if references_seen:
            block.excluded_reason = "references"
            excluded["references"] += 1
            continue
        if excluded_section:
            block.excluded_reason = "non_body_section"
            excluded["non_body_section"] += 1
            continue
        if block.kind != "image" and normalize_heading_key(block.text) == normalized_title:
            block.excluded_reason = "duplicate_title"
            excluded["duplicate_title"] += 1
            continue
        normalized = re.sub(r"\s+", " ", block.text).strip().lower()
        if normalized and seen_text[normalized] and (
            _is_advertisement(block.text) or len(normalized.split()) <= 20
        ):
            block.excluded_reason = "duplicate_template_text"
            excluded["duplicate_template"] += 1
            continue
        seen_text[normalized] += 1

    for block in blocks:
        warnings.extend(block.warnings)
    return dict(section_tree), unknown, excluded, warnings


def _caption_match(block: SourceBlock) -> tuple[str, re.Match[str]] | None:
    if block.kind not in {"paragraph", "list"}:
        return None
    table = TABLE_CAPTION_RE.match(block.text)
    if table:
        return "table", table
    table = MANGLED_TABLE_CAPTION_RE.match(block.text)
    if table:
        return "table", table
    figure = FIGURE_CAPTION_RE.match(block.text)
    if figure:
        return "figure", figure
    figure = MANGLED_FIGURE_CAPTION_RE.match(block.text)
    if figure:
        return "figure", figure
    table = EMBEDDED_TABLE_CAPTION_RE.search(block.text)
    if table and _embedded_caption_is_repeated_or_mangled(block.text, table, "table"):
        return "table", table
    figure = EMBEDDED_FIGURE_CAPTION_RE.search(block.text)
    if figure and _embedded_caption_is_repeated_or_mangled(block.text, figure, "figure"):
        return "figure", figure
    return None


def _embedded_caption_is_repeated_or_mangled(
    text: str,
    match: re.Match[str],
    asset_type: str,
) -> bool:
    label = match.group("label")
    if len(re.findall(re.escape(label), text, re.I)) >= 2:
        return True
    number = re.search(r"\b(S?\d+[A-Za-z]?|[IVXLC]+)\b", label, re.I)
    if not number:
        return False
    prefix = text[: match.start()]
    stem = r"(?:e|re|ure|gure|igure)" if asset_type == "figure" else r"(?:ble|able)"
    return bool(re.search(rf"(?:^|\s){stem}\s+{re.escape(number.group(1))}\b", prefix, re.I))


def _nearest_images(blocks: list[SourceBlock], index: int, asset_type: str) -> list[SourceBlock]:
    directions = (1, -1) if asset_type == "table" else (-1, 1)
    caption_block = blocks[index]
    post_references = caption_block.section == "References"
    for direction in directions:
        found: list[SourceBlock] = []
        cursor = index + direction
        steps = 0
        max_steps = 50 if post_references else 16
        while 0 <= cursor < len(blocks) and steps < max_steps:
            candidate = blocks[cursor]
            if candidate.kind == "image":
                found.append(candidate)
                cursor += direction
                steps += 1
                continue
            if candidate.kind == "heading" or _caption_match(candidate):
                break
            if candidate.excluded_reason and candidate.excluded_reason != "references":
                cursor += direction
                steps += 1
                continue
            if candidate.kind == "paragraph" and re.fullmatch(r"[A-Ha-h]", candidate.text.strip()):
                cursor += direction
                steps += 1
                continue
            if post_references:
                cursor += direction
                steps += 1
                continue
            if found and (
                len(candidate.text.split()) <= 45
                or re.match(r"^(?:[A-Ha-h]\b|[A-Z]\d?\s*[-:]?\s*vs\b)", candidate.text, re.I)
            ):
                cursor += direction
                steps += 1
                continue
            break
        if found:
            return sorted(found, key=lambda item: item.order)
    return []


def _reference_pattern(asset_type: str, label: str) -> re.Pattern[str]:
    number_match = re.search(r"\b(S?\d+)([A-Za-z]?)\b", label, re.I)
    if number_match:
        base = re.escape(number_match.group(1))
        suffix = re.escape(number_match.group(2)) if number_match.group(2) else r"[A-Za-z]?"
        number = base + suffix
    else:
        roman = re.search(r"\b([IVXLC]+)\b", label, re.I)
        number = re.escape(roman.group(1)) if roman else r"\d+"
    prefix = r"(?:Fig(?:ure)?\.?|Figures?)" if asset_type == "figure" else r"Tables?"
    return re.compile(rf"\b{prefix}\s*{number}\b", re.I)


def _find_reference_context(
    blocks: list[SourceBlock], index: int, asset_type: str, label: str
) -> tuple[SourceBlock | None, SourceBlock | None]:
    pattern = _reference_pattern(asset_type, label)
    contexts: list[SourceBlock | None] = []
    for direction in (-1, 1):
        matched_block: SourceBlock | None = None
        cursor = index + direction
        while 0 <= cursor < len(blocks):
            block = blocks[cursor]
            cursor += direction
            if block.excluded_reason or block.kind in {"heading", "image"}:
                continue
            if block.kind in {"paragraph", "list"} and pattern.search(block.text):
                matched_block = block
                break
        contexts.append(matched_block)
    return contexts[0], contexts[1]


def bind_assets(blocks: list[SourceBlock], pmcid: str) -> list[Asset]:
    assets: list[Asset] = []
    counters: Counter[str] = Counter()
    for index, block in enumerate(blocks):
        if block.excluded_reason and block.excluded_reason != "references":
            continue
        matched = _caption_match(block)
        if not matched:
            continue
        asset_type, match = matched
        caption_text = block.text[match.start() :].strip()
        raw_label = re.sub(r"\s+", " ", match.group("label")).strip()
        label_number = re.search(r"\b(S?\d+[A-Za-z]?|[IVXLC]+)\b", raw_label, re.I)
        if label_number:
            canonical_prefix = "Figure" if asset_type == "figure" else "Table"
            label = f"{canonical_prefix} {label_number.group(1)}"
            source_prefix = (
                r"(?:Figure|Fig\.|igure|gure|ure|re|e)"
                if asset_type == "figure"
                else r"(?:Table|able|ble)"
            )
            caption_text = re.sub(
                rf"^\s*{source_prefix}\s+" + re.escape(label_number.group(1)),
                label,
                caption_text,
                count=1,
                flags=re.I,
            )
        else:
            label = re.sub(r"^Fig\.", "Figure", raw_label, flags=re.I)
        images = _nearest_images(blocks, index, asset_type)
        if not images:
            block.warnings.append("caption_like_text_without_adjacent_image")
            continue
        counters[asset_type] += 1
        asset_id = f"{pmcid}_{asset_type}_{counters[asset_type]:04d}"
        for image in images:
            image.asset_id = asset_id
        block.asset_id = asset_id
        if images:
            low_order = min([block.order] + [image.order for image in images])
            high_order = max([block.order] + [image.order for image in images])
            for possible_panel in blocks:
                if (
                    low_order <= possible_panel.order <= high_order
                    and _is_figure_panel_artifact(possible_panel.text)
                ):
                    possible_panel.asset_id = asset_id
        before_block, after_block = _find_reference_context(blocks, index, asset_type, label)
        asset_section = block.section
        asset_subsection = block.subsection
        warnings: list[str] = []
        if asset_section == "References":
            reference_block = before_block or after_block
            if reference_block is not None:
                asset_section = reference_block.section
                asset_subsection = reference_block.subsection
            else:
                asset_section = "Unassigned"
                asset_subsection = ""
                warnings.append("post_references_asset_section_unresolved")
        notes = ""
        if asset_type == "table" and images:
            last_image_index = max(blocks.index(image) for image in images)
            if last_image_index + 1 < len(blocks):
                next_block = blocks[last_image_index + 1]
                if (
                    next_block.kind == "paragraph"
                    and (not next_block.excluded_reason or next_block.excluded_reason == "references")
                    and re.match(r"^(?:\d+\s|Note\b|Abbreviations?\b)", next_block.text, re.I)
                ):
                    notes = next_block.text
                    next_block.asset_id = asset_id
        assets.append(
            Asset(
                asset_type=asset_type,
                asset_id=asset_id,
                label=label,
                caption=block.text,
                image_paths=[image.image_path for image in images if image.image_path],
                section=asset_section,
                subsection=asset_subsection,
                char_start=min([block.char_start] + [image.char_start for image in images]),
                char_end=max([block.char_end] + [image.char_end for image in images]),
                order=min([float(block.order)] + [float(image.order) for image in images]),
                context_before=before_block.text if before_block is not None else "",
                context_after=after_block.text if after_block is not None else "",
                notes=notes,
                table_text_missing=asset_type == "table",
                mapping_confidence="high" if images else "low",
                parse_warnings=warnings,
            )
        )
        assets[-1].caption = caption_text
    return _merge_continued_assets(assets)


def _asset_label_key(asset: Asset) -> tuple[str, str]:
    number = re.search(r"\b(S?\d+|[IVXLC]+)\b", asset.label, re.I)
    return asset.asset_type, number.group(1).upper() if number else asset.label.upper()


def _merge_continued_assets(assets: list[Asset]) -> list[Asset]:
    merged: list[Asset] = []
    by_label: dict[tuple[str, str], Asset] = {}
    for asset in assets:
        key = _asset_label_key(asset)
        primary = by_label.get(key)
        if primary is None:
            by_label[key] = asset
            merged.append(asset)
            continue
        for path in asset.image_paths:
            if path not in primary.image_paths:
                primary.image_paths.append(path)
        primary.char_start = min(primary.char_start, asset.char_start)
        primary.char_end = max(primary.char_end, asset.char_end)
        primary.order = min(primary.order, asset.order)
        if len(asset.caption) > len(primary.caption) and "cont." not in asset.caption[:40].lower():
            primary.caption = asset.caption
        if not primary.context_before:
            primary.context_before = asset.context_before
        if not primary.context_after:
            primary.context_after = asset.context_after
        if primary.section == "Unassigned" and asset.section != "Unassigned":
            primary.section, primary.subsection = asset.section, asset.subsection
        primary.mapping_confidence = "high" if primary.image_paths else primary.mapping_confidence
        primary.parse_warnings = sorted(
            set(primary.parse_warnings + asset.parse_warnings + ["continued_asset_merged"])
        )
    return merged


def parse_document(
    markdown_path: Path,
    pmcid: str,
    metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    metadata = metadata or {}
    raw = markdown_path.read_text(encoding="utf-8", errors="replace")
    blocks = parse_markdown_blocks(raw)
    h1_candidates = [
        block.text
        for block in blocks
        if block.kind == "heading" and block.heading_level == 1
    ]
    markdown_title = next((text for text in h1_candidates if _valid_title(text)), "")
    opening_h2: list[str] = []
    for block in blocks:
        if block.kind in {"paragraph", "list"} and len(block.text.split()) >= 40:
            break
        if block.kind == "heading" and block.heading_level == 2:
            opening_h2.append(block.text)
    markdown_h2_title = next((text for text in opening_h2 if _valid_title(text)), "")
    metadata_title = str(metadata.get("title") or metadata.get("metadata", {}).get("title") or "").strip()
    parse_warnings: list[str] = []
    if metadata_title and not _valid_title(metadata_title):
        parse_warnings.append("metadata_title_anomaly")
    if markdown_title:
        paper_title, title_source = markdown_title, "markdown_h1"
    elif markdown_h2_title:
        paper_title, title_source = markdown_h2_title, "markdown_h2_fallback"
        parse_warnings.append("title_recovered_from_markdown_h2")
    elif _valid_title(metadata_title):
        paper_title, title_source = metadata_title, "metadata_jsonl"
    else:
        paper_title, title_source = "", "missing"
        parse_warnings.append("missing_reliable_title")

    section_tree, unknown, excluded, structural_warnings = _assign_structure_and_exclusions(
        blocks, paper_title
    )
    parse_warnings.extend(structural_warnings)
    assets = bind_assets(blocks, pmcid)
    parse_warnings.extend(warning for block in blocks for warning in block.warnings)
    parse_warnings.extend(warning for asset in assets for warning in asset.parse_warnings)
    source_file = str(
        metadata.get("source_file")
        or metadata.get("metadata", {}).get("source_file")
        or markdown_path.name
    )
    return ParsedDocument(
        pmcid=pmcid,
        paper_title=paper_title,
        title_source=title_source,
        source_markdown=str(markdown_path),
        source_file=source_file,
        blocks=blocks,
        assets=assets,
        parse_warnings=sorted(set(parse_warnings)),
        section_tree=section_tree,
        excluded_counts=dict(excluded),
        unknown_headings=unknown,
        metadata=metadata,
    )
