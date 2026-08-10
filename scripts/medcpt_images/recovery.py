from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any

try:
    from medcpt_fulltext.markdown_parser import _caption_match, parse_document
except ModuleNotFoundError:  # Imported as scripts.medcpt_images from the repo root.
    from scripts.medcpt_fulltext.markdown_parser import _caption_match, parse_document


RECOVERY_CAPTION_RE = re.compile(
    r"^\s*(?P<prefix>fig(?:ure)?\.?|igure|gure|ure|re|e|table|able|ble)"
    r"\s*[-:]?\s*(?P<number>S?\d+[A-Za-z]?|[IVXLC]+)\s*[.:-]?\s*(?P<body>.*)$",
    re.I | re.S,
)
REFERENCE_HEADING_RE = re.compile(r"^(?:references|bibliography)$", re.I)


@dataclass(frozen=True)
class ImageInfo:
    width: int | None
    height: int | None
    size_bytes: int | None
    quality: str


def asset_key(pmcid: str, relative_path: str) -> str:
    value = f"{pmcid.upper()}\0{relative_path}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe image path: {value!r}")
    return path.as_posix()


def _image_info(path: Path) -> ImageInfo:
    try:
        from PIL import Image

        size_bytes = path.stat().st_size
        with Image.open(path) as image:
            width, height = image.size
        aspect = max(width / max(height, 1), height / max(width, 1))
        if width < 80 or height < 80 or width * height < 20_000:
            quality = "tiny"
        elif aspect >= 12:
            quality = "extreme_aspect"
        elif width >= 300 and height >= 200:
            quality = "normal"
        else:
            quality = "fragment"
        return ImageInfo(width, height, size_bytes, quality)
    except Exception:
        return ImageInfo(None, None, None, "unreadable")


def _caption(block: Any) -> tuple[str, str, str, bool] | None:
    matched = _caption_match(block)
    if matched:
        asset_type, match = matched
        number = re.search(r"\b(S?\d+[A-Za-z]?|[IVXLC]+)\b", match.group("label"), re.I)
        if number:
            label = f"{'Figure' if asset_type == 'figure' else 'Table'} {number.group(1)}"
            return asset_type, label, block.text.strip(), True
    loose = RECOVERY_CAPTION_RE.match(block.text) if block.kind in {"paragraph", "list"} else None
    if not loose:
        return None
    prefix = loose.group("prefix").lower()
    asset_type = "table" if prefix in {"table", "able", "ble"} else "figure"
    label = f"{'Table' if asset_type == 'table' else 'Figure'} {loose.group('number')}"
    return asset_type, label, block.text.strip(), False


def _crosses_boundary(blocks: list[Any], start: int, end: int, anchor: int) -> bool:
    for index in range(min(start, end) + 1, max(start, end)):
        block = blocks[index]
        if block.kind == "heading":
            return True
        if index != anchor and _caption(block):
            return True
    return False


def _prose_between(blocks: list[Any], start: int, end: int) -> tuple[int, int]:
    count = words = 0
    for index in range(min(start, end) + 1, max(start, end)):
        block = blocks[index]
        if block.excluded_reason or block.kind not in {"paragraph", "list"}:
            continue
        text = block.text.strip()
        if re.fullmatch(r"\(?[A-Ha-h]\)?[.)]?", text):
            continue
        count += 1
        words += len(text.split())
    return count, words


def _context(blocks: list[Any], index: int, direction: int) -> str:
    recoverable_exclusions = {None, "front_matter", "isolated_text_fragment"}
    cursor = index + direction
    while 0 <= cursor < len(blocks):
        block = blocks[cursor]
        if block.kind == "heading":
            return ""
        if (
            block.kind in {"paragraph", "list"}
            and block.excluded_reason in recoverable_exclusions
        ):
            return block.text.strip()
        cursor += direction
    return ""


def _new_asset_id(pmcid: str, asset_type: str, label: str, position: int) -> str:
    digest = hashlib.sha1(f"{label}\0{position}".encode("utf-8")).hexdigest()[:12]
    return f"{pmcid}_{asset_type}_recovered_{digest}"


def recover_document(
    markdown_path: Path,
    pmcid: str,
    *,
    radius: int = 48,
    max_images_per_asset: int = 16,
) -> dict[str, Any]:
    """Classify every Markdown image and recover high-confidence associations."""

    document = parse_document(markdown_path, pmcid)
    blocks = document.blocks
    image_indices = [index for index, block in enumerate(blocks) if block.kind == "image"]
    info_by_index: dict[int, ImageInfo] = {}
    for index in image_indices:
        relative = _safe_relative_path(blocks[index].image_path or "")
        info_by_index[index] = _image_info(markdown_path.parent / relative)

    assets = {asset.asset_id: asset for asset in document.assets}
    asset_by_path = {
        _safe_relative_path(relative): asset
        for asset in document.assets
        for relative in asset.image_paths
    }
    assignments: dict[int, dict[str, Any]] = {}
    for index in image_indices:
        block = blocks[index]
        relative = _safe_relative_path(block.image_path or "")
        asset = asset_by_path.get(relative)
        if asset is not None:
            assignments[index] = {
                "asset_id": asset.asset_id,
                "asset_type": asset.asset_type,
                "label": asset.label,
                "caption": asset.caption,
                "confidence": "high",
                "reason": "existing_binding",
                "is_new_asset": False,
            }

    existing_by_path = {
        _safe_relative_path(blocks[index].image_path or ""): assignment
        for index, assignment in assignments.items()
    }
    for index in image_indices:
        relative = _safe_relative_path(blocks[index].image_path or "")
        if index not in assignments and relative in existing_by_path:
            assignments[index] = dict(existing_by_path[relative])

    proposals: list[tuple[int, int, int, dict[str, Any]]] = []
    for caption_index, block in enumerate(blocks):
        matched = _caption(block)
        if not matched:
            continue
        asset_type, label, caption_text, strict = matched
        existing_id = block.asset_id if block.asset_id in assets else None
        candidate_id = existing_id or _new_asset_id(pmcid, asset_type, label, block.char_start)
        current_count = sum(item.get("asset_id") == candidate_id for item in assignments.values())
        preferred_direction = 1 if asset_type == "table" else -1
        for image_index in image_indices:
            if image_index in assignments:
                continue
            distance = abs(image_index - caption_index)
            if distance > radius or _crosses_boundary(blocks, caption_index, image_index, caption_index):
                continue
            info = info_by_index[image_index]
            if info.quality in {"unreadable", "extreme_aspect"}:
                continue
            if info.quality == "tiny" and (not strict or distance > 10):
                continue
            direction = 1 if image_index > caption_index else -1
            prose_blocks, prose_words = _prose_between(blocks, caption_index, image_index)
            if direction != preferred_direction and prose_blocks:
                continue
            if direction == preferred_direction and (prose_blocks > 2 or prose_words > 160):
                continue
            score = 100 - distance + (12 if direction == preferred_direction else 0) + (12 if strict else 0)
            proposal = {
                "asset_id": candidate_id,
                "asset_type": asset_type,
                "label": label,
                "caption": caption_text,
                "confidence": "high" if strict and distance <= 20 else "medium",
                "reason": "caption_recovery" if strict else "mangled_caption_recovery",
                "is_new_asset": existing_id is None,
            }
            proposals.append((-score, image_index, current_count, proposal))

    assigned_per_asset: dict[str, int] = {}
    for _, image_index, existing_count, proposal in sorted(
        proposals, key=lambda item: (item[0], item[1], item[3]["asset_id"])
    ):
        if image_index in assignments:
            continue
        count = assigned_per_asset.get(proposal["asset_id"], existing_count)
        if count >= max_images_per_asset:
            continue
        assignments[image_index] = proposal
        assigned_per_asset[proposal["asset_id"]] = count + 1

    remaining = [index for index in image_indices if index not in assignments]
    groups: list[list[int]] = []
    for index in remaining:
        if (
            groups
            and index - groups[-1][-1] <= 6
            and blocks[index].section == blocks[groups[-1][-1]].section
            and not _crosses_boundary(blocks, groups[-1][-1], index, groups[-1][-1])
        ):
            groups[-1].append(index)
        else:
            groups.append([index])

    for group in groups:
        useful = [
            index
            for index in group
            if info_by_index[index].quality in {"normal", "fragment"}
        ]
        if not useful:
            continue
        first = useful[0]
        if blocks[first].section == "References":
            continue
        context_before = _context(blocks, first, -1)
        context_after = _context(blocks, useful[-1], 1)
        context_text = "\n\n".join(value for value in (context_before, context_after) if value)
        if not context_text:
            context_text = (
                f"Image from paper: {document.paper_title}. "
                f"Section: {blocks[first].section or 'Unassigned'}."
            )
        identifier = _new_asset_id(pmcid, "image", "context-only", blocks[first].char_start)
        for index in useful[:max_images_per_asset]:
            assignments[index] = {
                "asset_id": identifier,
                "asset_type": "image",
                "label": "Context-only image",
                "caption": context_text,
                "confidence": "low",
                "reason": "context_only_recovery",
                "is_new_asset": True,
            }

    confidence_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    best_by_path: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, assignment in assignments.items():
        relative = _safe_relative_path(blocks[index].image_path or "")
        current = best_by_path.get(relative)
        rank = confidence_rank.get(assignment["confidence"], 0)
        if assignment["reason"] == "existing_binding":
            rank += 10
        if current is None or rank > current[0]:
            best_by_path[relative] = (rank, assignment)
    for index in image_indices:
        relative = _safe_relative_path(blocks[index].image_path or "")
        if relative in best_by_path:
            assignments[index] = dict(best_by_path[relative][1])

    after_references = False
    bindings: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.kind == "heading" and REFERENCE_HEADING_RE.fullmatch(block.text.strip()):
            after_references = True
        if block.kind != "image":
            continue
        relative = _safe_relative_path(block.image_path or "")
        info = info_by_index[index]
        assignment = assignments.get(index)
        if assignment:
            status = "existing_bound" if assignment["reason"] == "existing_binding" else "recovered"
            values = assignment
        else:
            if info.quality in {"tiny", "extreme_aspect"}:
                reason = f"excluded_{info.quality}"
            elif after_references:
                reason = "unresolved_after_references"
            elif info.quality == "unreadable":
                reason = "excluded_unreadable"
            else:
                reason = "unresolved_no_caption_or_context"
            status = "excluded" if reason.startswith("excluded_") else "review"
            values = {
                "asset_id": None,
                "asset_type": None,
                "label": "",
                "caption": "",
                "confidence": "none",
                "reason": reason,
                "is_new_asset": False,
            }
        bindings.append(
            {
                "asset_key": asset_key(pmcid, relative),
                "pmcid": pmcid,
                "relative_path": relative,
                "source_path": str((markdown_path.parent / relative).resolve()),
                "status": status,
                **values,
                "width": info.width,
                "height": info.height,
                "size_bytes": info.size_bytes,
                "source_markdown": str(markdown_path),
            }
        )

    recovered_assets: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        if binding["status"] != "recovered":
            continue
        item = recovered_assets.setdefault(
            binding["asset_id"],
            {
                "asset_id": binding["asset_id"],
                "pmcid": pmcid,
                "asset_type": binding["asset_type"],
                "label": binding["label"],
                "caption": binding["caption"],
                "confidence": binding["confidence"],
                "reason": binding["reason"],
                "is_new_asset": binding["is_new_asset"],
                "image_paths": [],
                "asset_keys": [],
                "paper_title": document.paper_title,
                "section": next(
                    (block.section for block in blocks if block.image_path == binding["relative_path"]),
                    "Unassigned",
                ),
            },
        )
        if binding["relative_path"] not in item["image_paths"]:
            item["image_paths"].append(binding["relative_path"])
            item["asset_keys"].append(binding["asset_key"])

    return {
        "pmcid": pmcid,
        "paper_title": document.paper_title,
        "source_markdown": str(markdown_path),
        "bindings": bindings,
        "recovered_assets": list(recovered_assets.values()),
    }
