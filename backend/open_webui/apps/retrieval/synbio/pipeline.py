"""The single formal SynBioGPT full-text retrieval pipeline."""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict

from open_webui.apps.retrieval.models.medcpt import build_medcpt_rerank_text
from open_webui.apps.retrieval.query_processor import ProcessedQuery, QueryProcessor
from open_webui.apps.retrieval.search.asset_expansion import (
    AssetExpansionResult,
    expand_asset_evidence,
)
from open_webui.apps.retrieval.search.context_recovery import (
    ContextRecoveryResult,
    recover_context_documents,
)
from open_webui.apps.retrieval.search.rrf import (
    FusedCandidate,
    RankedCandidate,
    reciprocal_rank_fusion,
    resolve_chunk_id,
)

from .config import RetrievalConfig
from .evidence_gate import (
    apply_exact_term_gate,
    gate_rejection_reason,
)

log = logging.getLogger("synbiogpt.app.retrieval.pipeline")


@dataclass(frozen=True, slots=True)
class RetrievalRun:
    query: ProcessedQuery
    dense_candidates: list[RankedCandidate]
    bm25_candidates: list[RankedCandidate]
    fused_candidates: list[FusedCandidate]
    reranked_documents: list[Document]
    timings: dict[str, float]
    gate_diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvidenceExpansion:
    documents: list[Any]
    context: ContextRecoveryResult | None
    assets: AssetExpansionResult | None
    timings: dict[str, float]


def _selector(value: Any, collection_name: str) -> Any:
    select = getattr(value, "for_collection", None)
    return select(collection_name) if callable(select) else value


def _asset_urls(metadata: dict[str, Any], base_url: str) -> dict[str, Any]:
    keys = metadata.get("asset_keys") or []
    if base_url and isinstance(keys, list):
        metadata["image_urls"] = [
            f"{base_url}/assets/{key}"
            for key in keys
            if isinstance(key, str) and len(key) == 64
        ]
    return metadata


class RerankCompressor(BaseDocumentCompressor):
    """Compatibility-preserving reranker used by the formal pipeline."""

    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None
        if reranking and not getattr(self, "_reranker_device_logged", False):
            setattr(self, "_reranker_device_logged", True)
            device = None
            try:
                device = str(self.reranking_function.model.device)
            except Exception:
                pass
            log.info(
                "[DEBUG] reranker=%s device=%s",
                type(self.reranking_function),
                device,
            )
        if reranking:
            document_texts = [
                build_medcpt_rerank_text(doc.page_content, doc.metadata)
                for doc in documents
            ]
            scorer = getattr(self.reranking_function, "score", None)
            if callable(scorer):
                scores = scorer(query, document_texts)
            else:
                scores = self.reranking_function.predict(
                    list(zip([query] * len(document_texts), document_texts))
                )
        else:
            from sentence_transformers import util

            query_embedding = self.embedding_function(query)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents]
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        values = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        docs_with_scores = list(zip(documents, values))
        uses_raw_logits = getattr(self.reranking_function, "uses_raw_logits", False)
        if self.r_score and not uses_raw_logits:
            docs_with_scores = [
                (document, score)
                for document, score in docs_with_scores
                if score >= self.r_score
            ]

        ranked = sorted(
            docs_with_scores, key=operator.itemgetter(1), reverse=True
        )[: self.top_n]
        return [
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "score": score,
                    "cross_encoder_score": score,
                    "rerank_rank": rank,
                },
            )
            for rank, (document, score) in enumerate(ranked, 1)
        ]


class RetrievalPipeline:
    """Orchestrate query processing, hybrid recall, reranking and evidence expansion."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self.config = config or RetrievalConfig.from_env()
        self.query_processor = query_processor or QueryProcessor()

    def prepare_query(self, query: str):
        return self.query_processor.prepare(query)

    def process_query(self, query: str | ProcessedQuery) -> ProcessedQuery:
        if isinstance(query, ProcessedQuery):
            return query
        return self.query_processor.process(query)

    def process_generated_query(
        self,
        query: str,
        model_output: str | dict,
        inherited_exact_terms: tuple[str, ...] = (),
        *,
        allow_no_retrieval: bool = False,
    ) -> ProcessedQuery:
        return self.query_processor.process_model_output(
            query,
            model_output,
            inherited_exact_terms=inherited_exact_terms,
            allow_no_retrieval=allow_no_retrieval,
        )

    def process_fallback_query(
        self,
        query: str,
        semantic_query: str,
        inherited_exact_terms: tuple[str, ...] = (),
    ) -> ProcessedQuery:
        return self.query_processor.process_fallback(
            query,
            semantic_query,
            inherited_exact_terms=inherited_exact_terms,
        )

    @staticmethod
    def ranked_from_documents(
        source: str, documents: Sequence[Any]
    ) -> list[RankedCandidate]:
        candidates = []
        for document in documents:
            metadata = dict(document.metadata or {})
            fallback = str(metadata.get("vector_id") or "")
            candidates.append(
                RankedCandidate(
                    chunk_id=resolve_chunk_id(metadata, fallback),
                    text=document.page_content,
                    metadata=metadata,
                    score=metadata.get(f"{source}_score"),
                )
            )
        return candidates

    def fuse(
        self,
        dense: Sequence[RankedCandidate],
        bm25: Sequence[RankedCandidate],
    ) -> list[FusedCandidate]:
        return reciprocal_rank_fusion(
            {"dense": dense, "bm25": bm25},
            weights={
                "dense": self.config.dense_weight,
                "bm25": self.config.bm25_weight,
            },
            rrf_k=self.config.rrf_k,
            limit=self.config.candidate_limit,
        )

    def fused_documents(
        self, candidates: Sequence[FusedCandidate]
    ) -> list[Document]:
        documents = []
        for candidate in candidates:
            metadata = dict(candidate.metadata)
            metadata.update(
                chunk_id=candidate.chunk_id,
                rrf_rank=candidate.rank,
                rrf_score=candidate.rrf_score,
            )
            for source, rank in candidate.source_ranks.items():
                metadata[f"{source}_rank"] = rank
            for source, score in candidate.source_scores.items():
                metadata[f"{source}_score"] = score
            sources = sorted(candidate.source_ranks)
            metadata["retrieval_source"] = (
                "hybrid" if len(sources) > 1 else sources[0]
            )
            metadata["retrieval_sources"] = sources
            documents.append(
                Document(
                    page_content=candidate.text,
                    metadata=_asset_urls(metadata, self.config.asset_base_url),
                )
            )
        return documents

    def rerank(
        self,
        query: ProcessedQuery,
        candidates: Sequence[FusedCandidate],
        *,
        embedding_function: Callable,
        reranking_function: Any,
        top_n: int,
        relevance_threshold: float,
    ) -> list[Document]:
        if not candidates:
            return []
        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=top_n,
            reranking_function=reranking_function,
            r_score=relevance_threshold,
        )
        return list(
            compressor.compress_documents(
                self.fused_documents(candidates), query.semantic_query
            )
        )

    @staticmethod
    def rerank_candidates(
        query: ProcessedQuery,
        candidates: Sequence[FusedCandidate],
        cross_encoder: Any,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the established offline rerank report representation."""

        documents = [
            build_medcpt_rerank_text(candidate.text, candidate.metadata)
            for candidate in candidates
        ]
        scores = cross_encoder.score(query.semantic_query, documents)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: (-item[1], item[0].rank, item[0].chunk_id),
        )[:top_k]
        results = []
        for rerank_rank, (candidate, score) in enumerate(ranked, 1):
            item = candidate.to_dict()
            item["rrf_rank"] = item.pop("rank")
            item["rerank_rank"] = rerank_rank
            item["cross_encoder_score"] = score
            item["rank_change"] = item["rrf_rank"] - rerank_rank
            results.append(item)
        return results

    @staticmethod
    def candidate_documents(candidates: Sequence[dict[str, Any]]) -> list[Document]:
        """Adapt offline candidate dictionaries to context-recovery documents."""

        documents = []
        for item in candidates:
            metadata = dict(item["metadata"])
            for key in (
                "chunk_id",
                "rrf_rank",
                "rerank_rank",
                "cross_encoder_score",
            ):
                metadata[key] = item[key]
            documents.append(Document(page_content=item["text"], metadata=metadata))
        return documents

    def search_ranked(
        self,
        query: str | ProcessedQuery,
        *,
        dense_search: Callable[[ProcessedQuery], Sequence[RankedCandidate]],
        bm25_search: Callable[[ProcessedQuery], Sequence[RankedCandidate]],
        embedding_function: Callable,
        reranking_function: Any,
        output_k: int,
        relevance_threshold: float,
        collection_name: str = "",
        rerank_enabled: bool = True,
        tolerate_bm25_failure: bool = False,
        dense_first: bool = False,
    ) -> RetrievalRun:
        processed = self.process_query(query)
        timings: dict[str, float] = {}

        def run_bm25() -> list[RankedCandidate]:
            started = time.perf_counter()
            try:
                return list(bm25_search(processed))
            except Exception:
                if not tolerate_bm25_failure:
                    raise
                log.exception("[HYBRID] OpenSearch BM25 search failed")
                return []
            finally:
                timings["bm25_search_seconds"] = time.perf_counter() - started

        def run_dense() -> list[RankedCandidate]:
            started = time.perf_counter()
            try:
                return list(dense_search(processed))
            finally:
                timings["dense_search_seconds"] = time.perf_counter() - started

        if dense_first:
            dense = run_dense()
            bm25 = run_bm25()
        else:
            bm25 = run_bm25()
            dense = run_dense()

        started = time.perf_counter()
        fused = self.fuse(dense, bm25)
        timings["fusion_seconds"] = time.perf_counter() - started

        reranked = []
        timings["rerank_seconds"] = 0.0
        timings["evidence_gate_seconds"] = 0.0
        gate_diagnostics: dict[str, Any] = {
            "input_count": 0,
            "output_count": 0,
            "exact_term_rejected_count": 0,
            "exact_term_missing_terms": [],
            "rejection_reason": None,
        }
        if rerank_enabled:
            selected_reranker = _selector(reranking_function, collection_name)
            top_n = (
                self.config.cross_encoder_top_k
                if getattr(selected_reranker, "is_medcpt_cross_encoder", False)
                else output_k
            )
            started = time.perf_counter()
            reranked = self.rerank(
                processed,
                fused,
                embedding_function=embedding_function,
                reranking_function=selected_reranker,
                top_n=top_n,
                relevance_threshold=relevance_threshold,
            )
            timings["rerank_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            exact_term_rejected_count = 0
            exact_term_missing_terms: tuple[str, ...] = ()
            if self.config.exact_term_gate_enabled:
                exact_term_gated = apply_exact_term_gate(
                    reranked,
                    required_terms=processed.required_terms,
                )
                reranked = exact_term_gated.documents
                exact_term_rejected_count = exact_term_gated.rejected_count
                exact_term_missing_terms = exact_term_gated.missing_terms
            else:
                reranked = list(reranked)

            gate_diagnostics = {
                "input_count": len(reranked) + exact_term_rejected_count,
                "output_count": len(reranked),
                "exact_term_rejected_count": exact_term_rejected_count,
                "exact_term_missing_terms": list(exact_term_missing_terms),
            }
            gate_diagnostics["rejection_reason"] = gate_rejection_reason(
                gate_diagnostics
            )
            timings["evidence_gate_seconds"] = time.perf_counter() - started
            log.info(
                "[EVIDENCE_GATE] candidates=%d accepted=%d "
                "exact_term_rejected=%d missing_terms=%s reason=%s "
                "exact_term_gate_enabled=%s",
                gate_diagnostics["input_count"],
                len(reranked),
                exact_term_rejected_count,
                list(exact_term_missing_terms),
                gate_diagnostics["rejection_reason"],
                self.config.exact_term_gate_enabled,
            )
        return RetrievalRun(
            processed,
            dense,
            bm25,
            fused,
            reranked,
            timings,
            gate_diagnostics,
        )

    def expand_evidence(
        self,
        documents: Sequence[Any],
        fetch_chunks: Callable[[Sequence[str]], Mapping[str, Any]],
        *,
        context_enabled: bool | None = None,
        asset_enabled: bool | None = None,
        strip_context_image_urls: bool = False,
    ) -> EvidenceExpansion:
        use_context = (
            self.config.context_recovery_enabled
            if context_enabled is None
            else context_enabled
        )
        use_assets = (
            self.config.asset_expansion_enabled
            if asset_enabled is None
            else asset_enabled
        )
        if not documents or not use_context:
            return EvidenceExpansion(list(documents), None, None, {})

        timings: dict[str, float] = {}
        started = time.perf_counter()
        context = recover_context_documents(
            documents,
            fetch_chunks,
            token_budget=self.config.context_token_budget,
            max_parent_chunks=self.config.context_max_parent_chunks,
        )
        timings["context_recovery_seconds"] = time.perf_counter() - started

        assets = None
        asset_documents: list[Any] = []
        if use_assets:
            started = time.perf_counter()
            assets = self.expand_assets(context.documents)
            timings["asset_expansion_seconds"] = time.perf_counter() - started
            asset_documents = assets.documents
            if strip_context_image_urls:
                for document in context.documents:
                    document.metadata.pop("image_urls", None)

        return EvidenceExpansion(
            [*context.documents, *asset_documents], context, assets, timings
        )

    def expand_assets(self, documents: Sequence[Any]) -> AssetExpansionResult:
        return expand_asset_evidence(
            documents,
            asset_base_url=self.config.asset_base_url,
            max_asset_groups=self.config.asset_max_groups,
            max_images=self.config.asset_max_images,
        )


def select_for_collection(value: Any, collection_name: str) -> Any:
    """Public compatibility helper for collection-specific model routers."""

    return _selector(value, collection_name)


def add_asset_urls(metadata: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Public compatibility helper for legacy WebUI result conversion."""

    return _asset_urls(metadata, base_url)
