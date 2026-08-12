"""Select deduplicated figure and table evidence from recovered text context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ASSET_KEY_RE = re.compile(r"[0-9a-f]{64}")
CAPTION_TYPES = {"figure_caption", "table_caption", "image_caption"}
ROLE_PRIORITY = {"anchor": 0, "parent": 1, "previous": 2, "next": 2}


@dataclass(frozen=True, slots=True)
class AssetEvidenceDocument:
    page_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AssetExpansionResult:
    documents: list[AssetEvidenceDocument]
    candidate_group_count: int
    selected_group_count: int
    selected_image_count: int
    duplicate_key_count: int
    invalid_key_count: int
    group_limit_skipped_count: int
    image_limit_skipped_count: int
    unresolved_url_count: int


def _values(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key) or []
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _asset_type(metadata: dict[str, Any]) -> str:
    types = set()
    if _values(metadata, "table_ids") or metadata.get("chunk_type") == "table_caption":
        types.add("table")
    if (
        _values(metadata, "figure_ids")
        or metadata.get("chunk_type") == "figure_caption"
    ):
        types.add("figure")
    if (
        _values(metadata, "image_asset_ids")
        or metadata.get("chunk_type") == "image_caption"
    ):
        types.add("image")
    return next(iter(types)) if len(types) == 1 else "mixed"


def _logical_ids(metadata: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            _values(metadata, "figure_ids")
            + _values(metadata, "table_ids")
            + _values(metadata, "image_asset_ids")
        )
    )


def _source_priority(document: Any) -> tuple[int, int, int, str]:
    metadata = document.metadata or {}
    try:
        anchor_rank = int(metadata.get("context_anchor_rank") or 10_000)
    except (TypeError, ValueError):
        anchor_rank = 10_000
    role = str(metadata.get("context_role") or "")
    chunk_type = str(metadata.get("chunk_type") or "")
    return (
        anchor_rank,
        ROLE_PRIORITY.get(role, 3),
        0 if chunk_type in CAPTION_TYPES else 1,
        str(metadata.get("chunk_id") or ""),
    )


def _group_id(metadata: dict[str, Any], logical_ids: list[str]) -> str:
    if logical_ids:
        return "+".join(sorted(logical_ids))
    return str(metadata.get("chunk_id") or "unidentified_asset")


def _evidence_text(group: dict[str, Any]) -> str:
    source = group["best_source"]
    metadata = source.metadata or {}
    label = ", ".join(group["logical_ids"]) or group["group_id"]
    prefix = group["asset_type"].replace("mixed", "figure/table").title()
    text = source.page_content.strip()
    if metadata.get("chunk_type") in CAPTION_TYPES:
        return f"{prefix} evidence {label}: {text}"
    return f"{prefix} evidence {label}. Referenced by: {text}"


def expand_asset_evidence(
    context_documents: list[Any],
    *,
    asset_base_url: str = "",
    max_asset_groups: int = 8,
    max_images: int = 16,
) -> AssetExpansionResult:
    """Create atomic logical-asset groups without visual inference or OCR."""

    if max_asset_groups < 1:
        raise ValueError("max_asset_groups must be positive")
    if max_images < 1:
        raise ValueError("max_images must be positive")

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_keys = 0
    observed_keys = 0
    for document in context_documents:
        metadata = dict(document.metadata or {})
        raw_keys = _values(metadata, "asset_keys")
        if not raw_keys:
            continue
        valid_keys = []
        for key in raw_keys:
            observed_keys += 1
            if ASSET_KEY_RE.fullmatch(key):
                valid_keys.append(key)
            else:
                invalid_keys += 1
        if not valid_keys:
            continue

        logical_ids = _logical_ids(metadata)
        group_id = _group_id(metadata, logical_ids)
        pmcid = str(metadata.get("pmcid") or "").strip().upper()
        key = (pmcid, group_id)
        group = groups.setdefault(
            key,
            {
                "pmcid": pmcid,
                "group_id": group_id,
                "logical_ids": set(),
                "asset_types": set(),
                "asset_keys": [],
                "paths_by_key": {},
                "source_chunk_ids": [],
                "source_roles": [],
                "best_source": document,
            },
        )
        group["logical_ids"].update(logical_ids)
        group["asset_types"].add(_asset_type(metadata))
        paths = _values(metadata, "image_paths")
        for index, asset_key in enumerate(valid_keys):
            if asset_key not in group["asset_keys"]:
                group["asset_keys"].append(asset_key)
            if index < len(paths):
                group["paths_by_key"].setdefault(asset_key, paths[index])
        chunk_id = str(metadata.get("chunk_id") or "").strip()
        if chunk_id and chunk_id not in group["source_chunk_ids"]:
            group["source_chunk_ids"].append(chunk_id)
        role = str(metadata.get("context_role") or "").strip()
        if role and role not in group["source_roles"]:
            group["source_roles"].append(role)
        if _source_priority(document) < _source_priority(group["best_source"]):
            group["best_source"] = document

    ordered_groups = []
    for group in groups.values():
        group["logical_ids"] = sorted(group["logical_ids"])
        types = group.pop("asset_types")
        group["asset_type"] = next(iter(types)) if len(types) == 1 else "mixed"
        ordered_groups.append(group)
    ordered_groups.sort(
        key=lambda group: (
            _source_priority(group["best_source"]),
            group["pmcid"],
            group["group_id"],
        )
    )

    selected_keys: set[str] = set()
    selected: list[AssetEvidenceDocument] = []
    group_limit_skipped = image_limit_skipped = 0
    base_url = asset_base_url.rstrip("/")
    for group in ordered_groups:
        new_keys = [key for key in group["asset_keys"] if key not in selected_keys]
        if not new_keys:
            continue
        if len(selected) >= max_asset_groups:
            group_limit_skipped += 1
            continue
        if len(selected_keys) + len(new_keys) > max_images:
            image_limit_skipped += 1
            continue

        source_metadata = group["best_source"].metadata or {}
        image_urls = (
            [f"{base_url}/assets/{key}" for key in new_keys] if base_url else []
        )
        selected.append(
            AssetEvidenceDocument(
                page_content=_evidence_text(group),
                metadata={
                    "chunk_id": f"asset:{group['pmcid']}:{group['group_id']}",
                    "pmcid": group["pmcid"],
                    "context_role": "asset",
                    "evidence_type": "visual_asset",
                    "asset_type": group["asset_type"],
                    "asset_group_id": group["group_id"],
                    "asset_ids": group["logical_ids"],
                    "asset_keys": new_keys,
                    "image_urls": image_urls,
                    "image_paths": [
                        group["paths_by_key"].get(key, "") for key in new_keys
                    ],
                    "source_chunk_ids": group["source_chunk_ids"],
                    "source_context_roles": group["source_roles"],
                    "context_anchor_chunk_id": source_metadata.get(
                        "context_anchor_chunk_id"
                    ),
                    "context_anchor_rank": source_metadata.get("context_anchor_rank"),
                    "anchor_cross_encoder_score": source_metadata.get(
                        "anchor_cross_encoder_score",
                        source_metadata.get("cross_encoder_score"),
                    ),
                    "anchor_rrf_score": source_metadata.get(
                        "anchor_rrf_score", source_metadata.get("rrf_score")
                    ),
                    "score": source_metadata.get("score"),
                    "title": source_metadata.get("title")
                    or source_metadata.get("paper_title"),
                    "paper_title": source_metadata.get("paper_title")
                    or source_metadata.get("title"),
                    "source": source_metadata.get("source"),
                    "section": source_metadata.get("section"),
                    "subsection": source_metadata.get("subsection"),
                    "table_text_missing": group["asset_type"] in {"table", "mixed"},
                    "visual_content_analyzed": False,
                },
            )
        )
        selected_keys.update(new_keys)

    return AssetExpansionResult(
        documents=selected,
        candidate_group_count=len(ordered_groups),
        selected_group_count=len(selected),
        selected_image_count=len(selected_keys),
        duplicate_key_count=max(
            0,
            observed_keys
            - invalid_keys
            - len({key for group in groups.values() for key in group["asset_keys"]}),
        ),
        invalid_key_count=invalid_keys,
        group_limit_skipped_count=group_limit_skipped,
        image_limit_skipped_count=image_limit_skipped,
        unresolved_url_count=0 if base_url else len(selected_keys),
    )
