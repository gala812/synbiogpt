import importlib.util
import sys
from pathlib import Path

from langchain_core.documents import Document

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/search/context_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("context_recovery", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
recover_context_documents = MODULE.recover_context_documents


def chunk(
    chunk_id,
    *,
    pmcid="PMC1",
    parent="PMC1_parent_0001",
    previous=None,
    next_=None,
    tokens=100,
    score=None,
):
    metadata = {
        "chunk_id": chunk_id,
        "pmcid": pmcid,
        "parent_chunk_id": parent,
        "previous_chunk_id": previous,
        "next_chunk_id": next_,
        "text_token_count": tokens,
    }
    if score is not None:
        metadata.update(
            score=score,
            cross_encoder_score=score,
            rrf_score=0.01,
        )
    return Document(page_content=f"text {chunk_id}", metadata=metadata)


def test_recovers_parent_and_boundary_neighbors_in_batches():
    store = {
        "A": chunk("A", previous="Z", next_="B"),
        "B": chunk("B", previous="A", next_="C", score=5.0),
        "C": chunk("C", previous="B", next_="D"),
        "Z": chunk("Z", parent="PMC1_parent_0000", next_="A"),
        "D": chunk("D", parent="PMC1_parent_0002", previous="C"),
    }
    calls = []

    def fetch(ids):
        calls.append(list(ids))
        return {value: store[value] for value in ids if value in store}

    result = recover_context_documents([store["B"]], fetch, token_budget=1_000)

    assert [doc.metadata["chunk_id"] for doc in result.documents] == [
        "B",
        "A",
        "C",
        "D",
        "Z",
    ]
    assert result.documents[0].metadata["context_role"] == "anchor"
    assert {doc.metadata["context_role"] for doc in result.documents[1:3]} == {"parent"}
    assert result.expanded_count == 4
    assert len(calls) == 2


def test_preserves_anchor_order_deduplicates_and_tracks_shared_context():
    store = {
        "A": chunk("A", next_="B"),
        "B": chunk("B", previous="A", next_="C", score=4.0),
        "C": chunk("C", previous="B", next_="D", score=3.0),
        "D": chunk("D", parent="PMC1_parent_0002", previous="C"),
    }

    result = recover_context_documents(
        [store["B"], store["C"]],
        lambda ids: {value: store[value] for value in ids if value in store},
        token_budget=1_000,
    )

    assert [doc.metadata["chunk_id"] for doc in result.documents[:2]] == ["B", "C"]
    assert len({doc.metadata["chunk_id"] for doc in result.documents}) == len(
        result.documents
    )
    shared = next(doc for doc in result.documents if doc.metadata["chunk_id"] == "D")
    assert shared.metadata["context_anchor_chunk_ids"] == ["B", "C"]


def test_rejects_cross_document_links_and_enforces_expansion_budget():
    anchor = chunk("A", next_="B", tokens=100, score=2.0)
    foreign = chunk("B", pmcid="PMC2", previous="A", tokens=100)
    result = recover_context_documents(
        [anchor],
        lambda ids: {"B": foreign},
        token_budget=150,
    )

    assert [doc.metadata["chunk_id"] for doc in result.documents] == ["A"]
    assert result.cross_document_rejection_count == 1

    local = chunk("B", previous="A", tokens=100)
    budgeted = recover_context_documents(
        [anchor],
        lambda ids: {"B": local},
        token_budget=150,
    )
    assert budgeted.expanded_count == 0
    assert budgeted.budget_skipped_count == 1


def test_missing_links_and_cycles_terminate_cleanly():
    anchor = chunk("A", next_="B", score=1.0)
    cycle = chunk("B", previous="A", next_="A")
    result = recover_context_documents(
        [anchor],
        lambda ids: {value: cycle for value in ids if value == "B"},
    )

    assert [doc.metadata["chunk_id"] for doc in result.documents] == ["A", "B"]
    assert result.missing_count == 0
