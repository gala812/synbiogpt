import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/search/rrf.py"
)
SPEC = importlib.util.spec_from_file_location("retrieval_rrf", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RankedCandidate = MODULE.RankedCandidate
reciprocal_rank_fusion = MODULE.reciprocal_rank_fusion
resolve_chunk_id = MODULE.resolve_chunk_id


def candidate(chunk_id: str, score: float = 1.0):
    return RankedCandidate(chunk_id, f"text {chunk_id}", {"chunk_id": chunk_id}, score)


def test_rrf_deduplicates_chunk_id_and_rewards_two_source_hits():
    fused = reciprocal_rank_fusion(
        {
            "dense": [candidate("A"), candidate("B")],
            "bm25": [candidate("B"), candidate("C")],
        },
        rrf_k=60,
        limit=10,
    )

    assert [item.chunk_id for item in fused] == ["B", "A", "C"]
    assert fused[0].source_ranks == {"dense": 2, "bm25": 1}
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert len({item.chunk_id for item in fused}) == len(fused)


def test_duplicate_inside_one_source_only_contributes_once():
    fused = reciprocal_rank_fusion(
        {"dense": [candidate("A"), candidate("A"), candidate("B")]},
        rrf_k=60,
    )

    assert fused[0].source_ranks == {"dense": 1}
    assert fused[0].rrf_score == pytest.approx(1 / 61)
    assert len(fused) == 2


def test_weights_limit_and_validation_are_deterministic():
    fused = reciprocal_rank_fusion(
        {"dense": [candidate("A")], "bm25": [candidate("B")]},
        weights={"dense": 1.0, "bm25": 0.5},
        limit=1,
    )

    assert [item.chunk_id for item in fused] == ["A"]
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion({}, rrf_k=0)
    with pytest.raises(ValueError, match="negative"):
        reciprocal_rank_fusion({"dense": []}, weights={"dense": -1})


def test_chunk_id_takes_priority_over_paper_doc_id():
    metadata = {"chunk_id": "PMC1_results_0001", "doc_id": "PMC1", "id": "point"}

    assert resolve_chunk_id(metadata) == "PMC1_results_0001"
