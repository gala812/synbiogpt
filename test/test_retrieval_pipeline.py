import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from langchain_core.documents import Document

ROOT = Path(__file__).parents[1] / "backend/open_webui/apps/retrieval"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package in (
    "open_webui",
    "open_webui.apps",
    "open_webui.apps.retrieval",
    "open_webui.apps.retrieval.models",
    "open_webui.apps.retrieval.search",
    "open_webui.apps.retrieval.synbio",
):
    sys.modules.setdefault(package, ModuleType(package))

QUERY = load("open_webui.apps.retrieval.query_processor", ROOT / "query_processor.py")
MEDCPT = load("open_webui.apps.retrieval.models.medcpt", ROOT / "models/medcpt.py")
RRF = load("open_webui.apps.retrieval.search.rrf", ROOT / "search/rrf.py")
CONTEXT = load(
    "open_webui.apps.retrieval.search.context_recovery",
    ROOT / "search/context_recovery.py",
)
ASSETS = load(
    "open_webui.apps.retrieval.search.asset_expansion",
    ROOT / "search/asset_expansion.py",
)
CONFIG = load(
    "open_webui.apps.retrieval.synbio.config", ROOT / "synbio/config.py"
)
EVIDENCE_GATE = load(
    "open_webui.apps.retrieval.synbio.evidence_gate",
    ROOT / "synbio/evidence_gate.py",
)
PIPELINE = load(
    "open_webui.apps.retrieval.synbio.pipeline", ROOT / "synbio/pipeline.py"
)
MULTIMODAL = load(
    "open_webui.apps.retrieval.multimodal_answer", ROOT / "multimodal_answer.py"
)
HOOKS = load("open_webui.apps.retrieval.synbio.hooks", ROOT / "synbio/hooks.py")

ProcessedQuery = QUERY.ProcessedQuery
RankedCandidate = RRF.RankedCandidate
RetrievalConfig = CONFIG.RetrievalConfig
RetrievalPipeline = PIPELINE.RetrievalPipeline


def candidate(chunk_id, source, score):
    return RankedCandidate(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"chunk_id": chunk_id, f"{source}_score": score},
        score=score,
    )


class Reranker:
    is_medcpt_cross_encoder = True
    uses_raw_logits = True

    def score(self, query, documents):
        assert query == "succinate production"
        return [2.0 if "B" in text else 1.0 for text in documents]


def test_exact_term_gate_env_switch_defaults_on(monkeypatch):
    monkeypatch.delenv("MEDCPT_EXACT_TERM_GATE_ENABLED", raising=False)
    assert RetrievalConfig.from_env().exact_term_gate_enabled is True

    monkeypatch.setenv("MEDCPT_EXACT_TERM_GATE_ENABLED", "false")
    assert RetrievalConfig.from_env().exact_term_gate_enabled is False


def test_gate_rejection_reason_reports_only_exact_term_rejection():
    base = {
        "input_count": 2,
        "output_count": 0,
        "exact_term_rejected_count": 0,
    }

    assert EVIDENCE_GATE.gate_rejection_reason(base) is None
    assert EVIDENCE_GATE.gate_rejection_reason(
        {**base, "exact_term_rejected_count": 2}
    ) == "exact_term"
    assert EVIDENCE_GATE.gate_rejection_reason(
        {**base, "output_count": 1, "exact_term_rejected_count": 1}
    ) is None


def test_formal_pipeline_preserves_hybrid_order_and_metadata():
    pipeline = RetrievalPipeline(
        RetrievalConfig(candidate_limit=10, cross_encoder_top_k=2)
    )
    query = ProcessedQuery(
        "succinate production",
        "succinate production",
        "succinate production",
        (),
    )
    result = pipeline.search_ranked(
        query,
        dense_search=lambda _: [candidate("A", "dense", 0.9), candidate("B", "dense", 0.8)],
        bm25_search=lambda _: [candidate("B", "bm25", 8.0), candidate("C", "bm25", 7.0)],
        embedding_function=lambda _: [],
        reranking_function=Reranker(),
        output_k=10,
        relevance_threshold=0.0,
    )

    assert [item.chunk_id for item in result.fused_candidates] == ["B", "A", "C"]
    assert [doc.metadata["chunk_id"] for doc in result.reranked_documents] == ["B", "A"]
    assert result.reranked_documents[0].metadata["retrieval_source"] == "hybrid"
    assert result.reranked_documents[0].metadata["rerank_rank"] == 1


def test_cross_encoder_raw_scores_rank_but_do_not_filter_evidence():
    pipeline = RetrievalPipeline(
        RetrievalConfig(candidate_limit=10, cross_encoder_top_k=2)
    )
    query = ProcessedQuery(
        "succinate production",
        "succinate production",
        "succinate production",
        (),
    )

    result = pipeline.search_ranked(
        query,
        dense_search=lambda _: [
            candidate("A", "dense", 0.9),
            candidate("B", "dense", 0.8),
        ],
        bm25_search=lambda _: [candidate("B", "bm25", 8.0)],
        embedding_function=lambda _: [],
        reranking_function=Reranker(),
        output_k=10,
        relevance_threshold=0.0,
    )

    assert [item.chunk_id for item in result.fused_candidates] == ["B", "A"]
    assert [doc.metadata["chunk_id"] for doc in result.reranked_documents] == [
        "B",
        "A",
    ]
    assert result.timings["rerank_seconds"] >= 0
    assert result.timings["evidence_gate_seconds"] >= 0
    assert result.gate_diagnostics["rejection_reason"] is None
    assert not EVIDENCE_GATE.has_evidence_documents({"documents": [[]]})


def test_exact_term_gate_can_be_disabled_without_changing_retrieval():
    query = ProcessedQuery(
        "pckA regulation",
        "succinate production",
        "pckA regulation",
        ("pckA",),
        ("pckA",),
    )
    search = lambda _: [candidate("pcnB", "dense", 0.9)]

    enabled = RetrievalPipeline(
        RetrievalConfig(candidate_limit=10, exact_term_gate_enabled=True)
    ).search_ranked(
        query,
        dense_search=search,
        bm25_search=lambda _: [],
        embedding_function=lambda _: [],
        reranking_function=Reranker(),
        output_k=10,
        relevance_threshold=0.0,
    )
    disabled = RetrievalPipeline(
        RetrievalConfig(candidate_limit=10, exact_term_gate_enabled=False)
    ).search_ranked(
        query,
        dense_search=search,
        bm25_search=lambda _: [],
        embedding_function=lambda _: [],
        reranking_function=Reranker(),
        output_k=10,
        relevance_threshold=0.0,
    )

    assert enabled.reranked_documents == []
    assert enabled.gate_diagnostics["rejection_reason"] == "exact_term"
    assert enabled.gate_diagnostics["exact_term_rejected_count"] == 1
    assert len(disabled.reranked_documents) == 1
    assert disabled.gate_diagnostics["rejection_reason"] is None
    assert disabled.gate_diagnostics["exact_term_rejected_count"] == 0


def test_formal_pipeline_expands_context_then_assets():
    key = "a" * 64
    anchor = Document(
        page_content="Figure 1. Result.",
        metadata={
            "chunk_id": "A",
            "pmcid": "PMC1",
            "next_chunk_id": "B",
            "text_token_count": 10,
            "figure_ids": ["PMC1_figure_0001"],
            "asset_keys": [key],
            "image_paths": ["images/a.jpg"],
            "chunk_type": "figure_caption",
        },
    )
    neighbor = SimpleNamespace(
        page_content="Neighbor.",
        metadata={
            "chunk_id": "B",
            "pmcid": "PMC1",
            "previous_chunk_id": "A",
            "text_token_count": 10,
        },
    )
    pipeline = RetrievalPipeline(
        RetrievalConfig(asset_base_url="http://assets", asset_max_images=2)
    )
    result = pipeline.expand_evidence(
        [anchor], lambda ids: {"B": neighbor} if "B" in ids else {}
    )

    assert result.context is not None
    assert result.context.expanded_count == 1
    assert result.assets is not None
    assert result.assets.selected_image_count == 1
    assert result.documents[-1].metadata["image_urls"] == [
        f"http://assets/assets/{key}"
    ]


def test_formal_pipeline_can_preserve_rrf_only_cli_behavior():
    pipeline = RetrievalPipeline(RetrievalConfig(candidate_limit=10))
    calls = []
    result = pipeline.search_ranked(
        "succinate production",
        dense_search=lambda _: calls.append("dense") or [
            candidate("A", "dense", 0.9)
        ],
        bm25_search=lambda _: calls.append("bm25") or [
            candidate("B", "bm25", 8.0)
        ],
        embedding_function=lambda _: (_ for _ in ()).throw(AssertionError()),
        reranking_function=None,
        output_k=10,
        relevance_threshold=0.0,
        rerank_enabled=False,
        dense_first=True,
    )

    assert calls == ["dense", "bm25"]
    assert [item.chunk_id for item in result.fused_candidates] == ["A", "B"]
    assert result.reranked_documents == []
    assert result.timings["rerank_seconds"] == 0.0


def test_webui_hooks_preserve_query_and_multimodal_message_shapes():
    pipeline = RetrievalPipeline()
    messages = [{"role": "user", "content": "ldhA deletion"}]
    protected = HOOKS.protect_query_messages(messages, "ldhA deletion", pipeline)
    assert messages == [{"role": "user", "content": "ldhA deletion"}]
    assert protected[0]["content"] == "ZXQENTITY0QXZ deletion"

    prepared, image_urls, injected = HOOKS.prepare_multimodal_messages(
        messages,
        [{"metadata": [{"image_urls": ["http://assets/a"]}]}],
        max_images=1,
        add_system_message=lambda prompt, values: [
            {"role": "system", "content": prompt},
            *values,
        ],
    )
    assert image_urls == ["http://assets/a"]
    assert injected == 1
    assert prepared[-1]["content"][-1] == {
        "type": "image_url",
        "image_url": {"url": "http://assets/a"},
    }


def test_exact_term_gate_rejects_similarly_spelled_substitutes():
    documents = [
        Document(
            page_content="A CRISPRi screen identified pcnB repression.",
            metadata={"paper_title": "CRISPRi screen"},
        )
    ]

    result = EVIDENCE_GATE.apply_exact_term_gate(
        documents, required_terms=("CRISPRi", "pckA")
    )

    assert result.documents == []
    assert result.rejected_count == 1
    assert result.missing_terms == ("pckA",)


def test_exact_term_gate_accepts_coverage_across_multiple_documents():
    crispri = Document(
        page_content="CRISPRi enables programmable transcriptional repression.",
        metadata={},
    )
    target = Document(
        page_content="The pckA target controls phosphoenolpyruvate metabolism.",
        metadata={},
    )
    unrelated = Document(page_content="Generic fermentation background.", metadata={})

    result = EVIDENCE_GATE.apply_exact_term_gate(
        [crispri, target, unrelated], required_terms=("CRISPRi", "pckA")
    )

    assert result.documents == [crispri, target]
    assert result.rejected_count == 1
    assert result.missing_terms == ()


def test_exact_term_gate_understands_common_entity_punctuation_variants():
    document = Document(
        page_content=(
            "Recombinant expression used Escherichia coli BL21 (DE3) and "
            "the pET28a vector."
        ),
        metadata={},
    )

    result = EVIDENCE_GATE.apply_exact_term_gate(
        [document], required_terms=("E. coli", "BL21(DE3)", "pET-28a(+)")
    )

    assert result.documents == [document]
    assert result.missing_terms == ()
