from __future__ import annotations

from collections import defaultdict
import hashlib
import re
from typing import Any, Iterable

from .models import Asset, ChunkingConfig, ParsedDocument, SourceBlock, TextUnit
from .tokenization import TokenCounter


WORD_RE = re.compile(r"\S+")
SENTENCE_END_RE = re.compile(r"[.!?](?:[\"'’”\]\)])?\s+")
ABBREVIATIONS = {
    "al.", "approx.", "ca.", "cf.", "dr.", "e.g.", "eq.", "eqs.", "et al.",
    "fig.", "figs.", "i.e.", "inc.", "no.", "nos.", "prof.", "ref.", "refs.",
    "sp.", "spp.", "st.", "table.", "vs.",
}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug[:48] or fallback).strip("_")


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Conservative scientific-English sentence segmentation with source offsets."""

    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.start() + len(match.group(0).rstrip())
        candidate = text[start:end].strip()
        lower = candidate.lower()
        last_two = " ".join(lower.split()[-2:])
        last_one = lower.split()[-1] if lower.split() else ""
        punctuation_pos = match.start()
        if last_one in ABBREVIATIONS or last_two in ABBREVIATIONS:
            continue
        if punctuation_pos > 0 and punctuation_pos + 1 < len(text):
            if text[punctuation_pos - 1].isdigit() and text[punctuation_pos + 1].isdigit():
                continue
        if re.search(r"\b[A-Z]\.$", candidate):
            continue
        if candidate:
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((left, end, candidate))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        left = start + len(text[start:]) - len(text[start:].lstrip())
        spans.append((left, len(text), tail))
    return spans or [(0, len(text), text.strip())]


def _force_split_sentence(
    text: str,
    token_counter: TokenCounter,
    config: ChunkingConfig,
) -> list[tuple[int, int, str, list[str]]]:
    """Last-resort split at clause/word boundaries when one sentence exceeds BERT."""

    clause_spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[;:]\s+", text):
        clause_spans.append((start, match.end()))
        start = match.end()
    if start < len(text):
        clause_spans.append((start, len(text)))
    if len(clause_spans) > 1 and all(
        token_counter.count(text[a:b]) <= config.hard_max_tokens
        and word_count(text[a:b]) <= config.hard_max_words
        for a, b in clause_spans
    ):
        return [
            (a, b, text[a:b].strip(), ["long_sentence_split_at_clause_boundary"])
            for a, b in clause_spans
            if text[a:b].strip()
        ]

    words = list(WORD_RE.finditer(text))
    parts: list[tuple[int, int, str, list[str]]] = []
    cursor = 0
    while cursor < len(words):
        end_index = min(cursor + config.target_max_words, len(words))
        while end_index > cursor + 1:
            start_char = words[cursor].start()
            end_char = words[end_index - 1].end()
            candidate = text[start_char:end_char]
            if token_counter.count(candidate) <= config.hard_max_tokens:
                break
            end_index -= 1
        if end_index <= cursor:
            end_index = cursor + 1
        start_char = words[cursor].start()
        end_char = words[end_index - 1].end()
        candidate = text[start_char:end_char]
        if end_index == cursor + 1 and token_counter.count(candidate) > config.hard_max_tokens:
            char_cursor = start_char
            while char_cursor < end_char:
                low, high = char_cursor + 1, end_char
                best = low
                while low <= high:
                    middle = (low + high) // 2
                    if token_counter.count(text[char_cursor:middle]) <= config.hard_max_tokens:
                        best = middle
                        low = middle + 1
                    else:
                        high = middle - 1
                parts.append(
                    (
                        char_cursor,
                        best,
                        text[char_cursor:best],
                        ["forced_oversize_token_sequence_split"],
                    )
                )
                char_cursor = best
            cursor = end_index
            continue
        parts.append(
            (
                start_char,
                end_char,
                candidate,
                ["forced_long_sentence_word_boundary_split"],
            )
        )
        if end_index >= len(words):
            cursor = end_index
        else:
            cursor = max(cursor + 1, end_index - config.long_overlap_words)
    return parts


def _split_long_unit(
    unit: TextUnit,
    token_counter: TokenCounter,
    config: ChunkingConfig,
) -> list[TextUnit]:
    sentence_parts: list[tuple[int, int, str, list[str]]] = []
    for start, end, sentence in _sentence_spans(unit.text):
        if (
            word_count(sentence) > config.hard_max_words
            or token_counter.count(sentence) > config.hard_max_tokens
        ):
            for a, b, text, warnings in _force_split_sentence(sentence, token_counter, config):
                sentence_parts.append((start + a, start + b, text, warnings))
        else:
            sentence_parts.append((start, end, sentence, []))

    base_groups: list[list[tuple[int, int, str, list[str]]]] = []
    current: list[tuple[int, int, str, list[str]]] = []
    base_limit = max(config.target_min_words, config.target_max_words - config.long_overlap_words)
    for part in sentence_parts:
        proposed = " ".join([item[2] for item in current] + [part[2]])
        if current and (
            word_count(proposed) > base_limit
            or token_counter.count(proposed) > config.hard_max_tokens
        ):
            base_groups.append(current)
            current = [part]
        else:
            current.append(part)
    if current:
        base_groups.append(current)

    # Rebalance a very small tail before adding overlap.
    if len(base_groups) > 1:
        tail_text = " ".join(item[2] for item in base_groups[-1])
        previous_text = " ".join(item[2] for item in base_groups[-2])
        combined = f"{previous_text} {tail_text}".strip()
        if (
            word_count(tail_text) < config.min_chunk_words
            and word_count(combined) <= config.hard_max_words
            and token_counter.count(combined) <= config.hard_max_tokens
        ):
            base_groups[-2].extend(base_groups.pop())

    groups: list[list[tuple[int, int, str, list[str]]]] = []
    for index, group in enumerate(base_groups):
        if index == 0:
            groups.append(group)
            continue
        overlap: list[tuple[int, int, str, list[str]]] = []
        overlap_words = 0
        for prior in reversed(base_groups[index - 1]):
            count = word_count(prior[2])
            if overlap and overlap_words + count > config.long_overlap_words * 1.5:
                break
            overlap.insert(0, prior)
            overlap_words += count
            if overlap_words >= config.long_overlap_words:
                break
        candidate = overlap + group
        candidate_text = " ".join(item[2] for item in candidate)
        while overlap and (
            word_count(candidate_text) > config.target_max_words
            or token_counter.count(candidate_text) > config.hard_max_tokens
        ):
            overlap.pop(0)
            candidate = overlap + group
            candidate_text = " ".join(item[2] for item in candidate)
        groups.append(candidate)

    split_units: list[TextUnit] = []
    for index, group in enumerate(groups):
        text = " ".join(item[2] for item in group).strip()
        starts = [item[0] for item in group]
        ends = [item[1] for item in group]
        warnings = list(unit.warnings)
        warnings.extend(warning for item in group for warning in item[3])
        warnings.append("long_paragraph_sentence_split")
        # Offsets are conservative because Markdown whitespace was normalized.
        split_units.append(
            TextUnit(
                kind=unit.kind,
                text=text,
                section=unit.section,
                subsection=unit.subsection,
                char_start=unit.char_start + min(starts),
                char_end=min(unit.char_end, unit.char_start + max(ends)),
                order=unit.order + index / 1000.0,
                warnings=sorted(set(warnings)),
                image_paths=list(unit.image_paths),
                figure_ids=list(unit.figure_ids),
                table_ids=list(unit.table_ids),
                source_spans=[[unit.char_start + min(starts), min(unit.char_end, unit.char_start + max(ends))]],
                is_long_split=True,
            )
        )
    return split_units


def _asset_reference_ids(text: str, assets: list[Asset]) -> tuple[list[str], list[str], list[str]]:
    figures: list[str] = []
    tables: list[str] = []
    paths: list[str] = []
    for asset in assets:
        number = re.search(r"\b(S?\d+[A-Za-z]?|[IVXLC]+)\b", asset.label, re.I)
        if not number:
            continue
        prefix = r"(?:Fig(?:ure)?\.?)" if asset.asset_type == "figure" else r"Table"
        if re.search(rf"\b{prefix}s?\s*{re.escape(number.group(1))}\b", text, re.I):
            (figures if asset.asset_type == "figure" else tables).append(asset.asset_id)
            paths.extend(asset.image_paths)
    return sorted(set(figures)), sorted(set(tables)), sorted(set(paths))


def build_text_units(document: ParsedDocument) -> list[TextUnit]:
    units: list[TextUnit] = []
    for block in document.blocks:
        if block.excluded_reason or block.asset_id:
            continue
        if block.kind not in {"paragraph", "list", "equation"}:
            continue
        figures, tables, paths = _asset_reference_ids(block.text, document.assets)
        units.append(
            TextUnit(
                kind=block.kind,
                text=block.text,
                section=block.section,
                subsection=block.subsection,
                char_start=block.char_start,
                char_end=block.char_end,
                order=float(block.order),
                warnings=list(block.warnings),
                image_paths=paths,
                figure_ids=figures,
                table_ids=tables,
                source_spans=[[block.char_start, block.char_end]],
            )
        )
    return units


def _asset_units(asset: Asset) -> list[TextUnit]:
    parts = [asset.caption]
    if asset.notes:
        parts.append(f"Table notes: {asset.notes}")
    if asset.context_before:
        parts.append(f"Context before: {asset.context_before}")
    if asset.context_after and asset.context_after != asset.context_before:
        parts.append(f"Context after: {asset.context_after}")
    return [
        TextUnit(
            kind=f"{asset.asset_type}_caption",
            text="\n\n".join(parts),
            section=asset.section,
            subsection=asset.subsection,
            char_start=asset.char_start,
            char_end=asset.char_end,
            order=asset.order + 0.0001,
            warnings=list(asset.parse_warnings),
            image_paths=list(asset.image_paths),
            figure_ids=[asset.asset_id] if asset.asset_type == "figure" else [],
            table_ids=[asset.asset_id] if asset.asset_type == "table" else [],
            source_spans=[[asset.char_start, asset.char_end]],
        )
    ]


def _combine_units(units: list[TextUnit]) -> TextUnit:
    kinds = {unit.kind for unit in units}
    kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    return TextUnit(
        kind=kind,
        text="\n\n".join(unit.text for unit in units),
        section=units[0].section,
        subsection=units[0].subsection,
        char_start=min(unit.char_start for unit in units),
        char_end=max(unit.char_end for unit in units),
        order=min(unit.order for unit in units),
        warnings=sorted({warning for unit in units for warning in unit.warnings}),
        image_paths=sorted({path for unit in units for path in unit.image_paths}),
        figure_ids=sorted({value for unit in units for value in unit.figure_ids}),
        table_ids=sorted({value for unit in units for value in unit.table_ids}),
        source_spans=[span for unit in units for span in unit.source_spans],
        is_long_split=any(unit.is_long_split for unit in units),
    )


def _within_hard_limits(text: str, token_counter: TokenCounter, config: ChunkingConfig) -> bool:
    return (
        word_count(text) <= config.hard_max_words
        and token_counter.count(text) <= config.hard_max_tokens
    )


def chunk_units(
    units: list[TextUnit], token_counter: TokenCounter, config: ChunkingConfig
) -> list[TextUnit]:
    expanded: list[TextUnit] = []
    for unit in units:
        if not _within_hard_limits(unit.text, token_counter, config):
            expanded.extend(_split_long_unit(unit, token_counter, config))
        else:
            expanded.append(unit)

    output: list[TextUnit] = []
    current: list[TextUnit] = []

    def flush() -> None:
        nonlocal current
        if current:
            output.append(_combine_units(current))
            current = []

    previous_path: tuple[str, str] | None = None
    for unit in expanded:
        path = (unit.section, unit.subsection)
        if previous_path is not None and path != previous_path:
            flush()
        previous_path = path
        if unit.kind in {"figure_caption", "table_caption"}:
            flush()
            output.append(unit)
            continue
        if unit.is_long_split:
            flush()
            output.append(unit)
            continue
        if not current:
            current = [unit]
            continue
        current_text = _combine_units(current).text
        proposed_text = f"{current_text}\n\n{unit.text}"
        current_words = word_count(current_text)
        proposed_words = word_count(proposed_text)
        should_add = (
            proposed_words <= config.soft_max_words
            and token_counter.count(proposed_text) <= config.hard_max_tokens
            and (current_words < config.target_min_words or proposed_words <= config.target_max_words)
        )
        if should_add:
            current.append(unit)
        else:
            flush()
            current = [unit]
    flush()

    # Prefer merging an undersized tail backward within the same path.
    index = len(output) - 1
    while index > 0:
        current_unit = output[index]
        previous = output[index - 1]
        same_path = (current_unit.section, current_unit.subsection) == (
            previous.section,
            previous.subsection,
        )
        mergeable_type = current_unit.kind not in {"figure_caption", "table_caption"}
        if same_path and mergeable_type and word_count(current_unit.text) < config.min_chunk_words:
            combined = _combine_units([previous, current_unit])
            if (
                word_count(combined.text) <= config.soft_max_words
                and token_counter.count(combined.text) <= config.hard_max_tokens
                and previous.kind not in {"figure_caption", "table_caption"}
            ):
                output[index - 1] = combined
                output.pop(index)
        index -= 1
    return sorted(output, key=lambda unit: unit.order)


def _make_parents(
    chunks: list[dict[str, Any]], document: ParsedDocument, config: ChunkingConfig
) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_path: tuple[str, str] | None = None
    for chunk in chunks:
        path = (chunk["section"], chunk["subsection"])
        if current and path != previous_path:
            groups.append(current)
            current = []
        proposed_words = sum(item["word_count"] for item in current) + chunk["word_count"]
        if current and proposed_words > config.parent_max_words:
            groups.append(current)
            current = []
        current.append(chunk)
        previous_path = path
    if current:
        groups.append(current)

    # Merge a short tail group backward only when it remains inside one subsection.
    merged_groups: list[list[dict[str, Any]]] = []
    for group in groups:
        if merged_groups:
            prior = merged_groups[-1]
            same_path = (prior[0]["section"], prior[0]["subsection"]) == (
                group[0]["section"], group[0]["subsection"]
            )
            combined_words = sum(item["word_count"] for item in prior + group)
            if same_path and sum(item["word_count"] for item in group) < config.parent_min_words and combined_words <= config.parent_max_words:
                prior.extend(group)
                continue
        merged_groups.append(group)

    for index, group in enumerate(merged_groups, 1):
        parent_id = f"{document.pmcid}_parent_{index:04d}"
        for chunk in group:
            chunk["parent_chunk_id"] = parent_id
        text = "\n\n".join(chunk["text"] for chunk in group)
        warnings: list[str] = []
        total_words = word_count(text)
        if total_words < config.parent_min_words:
            warnings.append("short_parent_unavoidable")
        if total_words > config.parent_max_words:
            warnings.append("long_parent_unavoidable")
        parents.append(
            {
                "parent_chunk_id": parent_id,
                "doc_id": document.pmcid,
                "pmcid": document.pmcid,
                "paper_title": document.paper_title,
                "section": group[0]["section"],
                "subsection": group[0]["subsection"],
                "section_path": group[0]["section_path"],
                "child_chunk_ids": [chunk["chunk_id"] for chunk in group],
                "text": text,
                "word_count": total_words,
                "char_start": min(chunk["char_start"] for chunk in group),
                "char_end": max(chunk["char_end"] for chunk in group),
                "source_file": document.source_file,
                "parse_warnings": warnings,
            }
        )
    return parents


def create_chunks(
    document: ParsedDocument,
    token_counter: TokenCounter,
    config: ChunkingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units = build_text_units(document)
    for asset in document.assets:
        units.extend(_asset_units(asset))
    units.sort(key=lambda unit: unit.order)
    child_units = chunk_units(units, token_counter, config)

    chunks: list[dict[str, Any]] = []
    path_counters: defaultdict[tuple[str, str], int] = defaultdict(int)
    used_chunk_ids: set[str] = set()
    for index, unit in enumerate(child_units, 1):
        path = (unit.section, unit.subsection)
        path_counters[path] += 1
        section_slug = _slug(unit.section, "section")
        subsection_slug = _slug(unit.subsection, "main")
        chunk_id = f"{document.pmcid}_{section_slug}_{subsection_slug}_{path_counters[path]:04d}"
        if chunk_id in used_chunk_ids:
            path_hash = hashlib.sha1(f"{unit.section}\n{unit.subsection}".encode("utf-8")).hexdigest()[:8]
            chunk_id = f"{document.pmcid}_{section_slug}_{subsection_slug}_{path_hash}_{path_counters[path]:04d}"
        used_chunk_ids.add(chunk_id)
        section_path = [unit.section] + ([unit.subsection] if unit.subsection else [])
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": document.pmcid,
                "pmcid": document.pmcid,
                "paper_title": document.paper_title,
                "section": unit.section,
                "subsection": unit.subsection,
                "section_path": section_path,
                "chunk_type": unit.kind,
                "chunk_index": index,
                "parent_chunk_id": "",
                "text": unit.text,
                "word_count": word_count(unit.text),
                "token_count": token_counter.count(unit.text),
                "tokenizer_name": token_counter.name,
                "char_start": unit.char_start,
                "char_end": unit.char_end,
                "source_spans": unit.source_spans,
                "previous_chunk_id": None,
                "next_chunk_id": None,
                "image_paths": unit.image_paths,
                "figure_ids": unit.figure_ids,
                "table_ids": unit.table_ids,
                "source_file": document.source_file,
                "parse_warnings": sorted(set(unit.warnings)),
            }
        )
    for index, chunk in enumerate(chunks):
        if index:
            chunk["previous_chunk_id"] = chunks[index - 1]["chunk_id"]
        if index + 1 < len(chunks):
            chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"]
    parents = _make_parents(chunks, document, config)
    return chunks, parents
