import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/search/asset_expansion.py"
)
SPEC = importlib.util.spec_from_file_location("asset_expansion", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
expand_asset_evidence = MODULE.expand_asset_evidence


def document(chunk_id, keys, *, pmcid="PMC1", figure_ids=None, role="parent", rank=1):
    return SimpleNamespace(
        page_content=f"Figure 1. Evidence from {chunk_id}.",
        metadata={
            "chunk_id": chunk_id,
            "pmcid": pmcid,
            "chunk_type": "figure_caption",
            "context_role": role,
            "context_anchor_rank": rank,
            "context_anchor_chunk_id": "anchor-1",
            "figure_ids": figure_ids or ["PMC1_figure_0001"],
            "table_ids": [],
            "asset_keys": keys,
            "image_paths": [f"images/{index}.jpg" for index in range(len(keys))],
            "score": 3.0,
            "paper_title": "Synthetic biology evidence",
            "source": "part-00000.jsonl",
        },
    )


def key(character):
    return character * 64


def test_groups_multipanel_figure_and_builds_stable_urls():
    result = expand_asset_evidence(
        [document("caption", [key("a"), key("b")])],
        asset_base_url="http://assets:8011/",
    )

    assert result.selected_group_count == 1
    assert result.selected_image_count == 2
    evidence = result.documents[0]
    assert evidence.metadata["asset_type"] == "figure"
    assert evidence.metadata["image_urls"] == [
        f"http://assets:8011/assets/{key('a')}",
        f"http://assets:8011/assets/{key('b')}",
    ]
    assert evidence.metadata["visual_content_analyzed"] is False
    assert evidence.metadata["title"] == "Synthetic biology evidence"
    assert evidence.metadata["source"] == "part-00000.jsonl"


def test_deduplicates_same_asset_across_context_chunks():
    documents = [
        document("paragraph", [key("a")], role="anchor", rank=2),
        document("caption", [key("a"), key("b")], rank=2),
    ]
    result = expand_asset_evidence(documents)

    assert result.selected_group_count == 1
    assert result.selected_image_count == 2
    assert result.duplicate_key_count == 1
    assert result.unresolved_url_count == 2


def test_never_partially_selects_oversized_logical_figure():
    oversized = document("large", [key(chr(ord("a") + index)) for index in range(5)])
    small = document(
        "small",
        [key("f")],
        figure_ids=["PMC1_figure_0002"],
        rank=2,
    )
    result = expand_asset_evidence(
        [oversized, small],
        max_images=3,
    )

    assert result.image_limit_skipped_count == 1
    assert result.selected_image_count == 1
    assert result.documents[0].metadata["asset_group_id"] == "PMC1_figure_0002"


def test_keeps_assets_from_different_papers_separate_and_rejects_bad_keys():
    result = expand_asset_evidence(
        [
            document("one", [key("a"), "bad"]),
            document("two", [key("b")], pmcid="PMC2"),
        ]
    )

    assert result.selected_group_count == 2
    assert result.invalid_key_count == 1
    assert {item.metadata["pmcid"] for item in result.documents} == {"PMC1", "PMC2"}
