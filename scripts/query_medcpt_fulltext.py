from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODELS_ROOT = PROJECT_ROOT / "backend/open_webui/apps/retrieval/models"
SEARCH_ROOT = PROJECT_ROOT / "backend/open_webui/apps/retrieval/search"
RETRIEVAL_ROOT = PROJECT_ROOT / "backend/open_webui/apps/retrieval"
MEDCPT_MODELS = _load_module(
    "medcpt_models",
    MODELS_ROOT / "medcpt.py",
)
for package in (
    "open_webui",
    "open_webui.apps",
    "open_webui.apps.retrieval",
    "open_webui.apps.retrieval.models",
    "open_webui.apps.retrieval.search",
    "open_webui.apps.retrieval.synbio",
):
    sys.modules.setdefault(package, ModuleType(package))
sys.modules["open_webui.apps.retrieval.models.medcpt"] = MEDCPT_MODELS
MedCPTQueryEncoder = MEDCPT_MODELS.MedCPTQueryEncoder
MedCPTCrossEncoder = MEDCPT_MODELS.MedCPTCrossEncoder
MedCPTDenseRetriever = _load_module(
    "medcpt_dense_retriever",
    SEARCH_ROOT / "medcpt_dense.py",
).MedCPTDenseRetriever
QUERY_PROCESSOR = _load_module(
    "fulltext_query_processor",
    RETRIEVAL_ROOT / "query_processor.py",
)
sys.modules["open_webui.apps.retrieval.query_processor"] = QUERY_PROCESSOR
QueryProcessor = QUERY_PROCESSOR.QueryProcessor
OPENSEARCH_BM25 = _load_module(
    "fulltext_opensearch_bm25",
    SEARCH_ROOT / "opensearch_bm25.py",
)
sys.modules["open_webui.apps.retrieval.search.opensearch_bm25"] = OPENSEARCH_BM25
search_bm25 = OPENSEARCH_BM25.search_bm25
fetch_chunks_by_ids = OPENSEARCH_BM25.fetch_chunks_by_ids
RRF = _load_module(
    "fulltext_rrf",
    SEARCH_ROOT / "rrf.py",
)
sys.modules["open_webui.apps.retrieval.search.rrf"] = RRF
RankedCandidate = RRF.RankedCandidate
resolve_chunk_id = RRF.resolve_chunk_id
CONTEXT_RECOVERY = _load_module(
    "fulltext_context_recovery",
    SEARCH_ROOT / "context_recovery.py",
)
sys.modules["open_webui.apps.retrieval.search.context_recovery"] = CONTEXT_RECOVERY
ASSET_EXPANSION = _load_module(
    "fulltext_asset_expansion",
    SEARCH_ROOT / "asset_expansion.py",
)
sys.modules["open_webui.apps.retrieval.search.asset_expansion"] = ASSET_EXPANSION
SYNBIO_CONFIG = _load_module(
    "open_webui.apps.retrieval.synbio.config",
    RETRIEVAL_ROOT / "synbio/config.py",
)
_load_module(
    "open_webui.apps.retrieval.synbio.evidence_calibration",
    RETRIEVAL_ROOT / "synbio/evidence_calibration.py",
)
_load_module(
    "open_webui.apps.retrieval.synbio.evidence_gate",
    RETRIEVAL_ROOT / "synbio/evidence_gate.py",
)
SYNBIO_PIPELINE = _load_module(
    "open_webui.apps.retrieval.synbio.pipeline",
    RETRIEVAL_ROOT / "synbio/pipeline.py",
)
RetrievalConfig = SYNBIO_CONFIG.RetrievalConfig
RetrievalPipeline = SYNBIO_PIPELINE.RetrievalPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MedCPT and BM25 full-text recall against production indexes."
    )
    queries = parser.add_mutually_exclusive_group(required=True)
    queries.add_argument(
        "--query",
        action="append",
        help="English query; lexical terms are derived locally.",
    )
    queries.add_argument(
        "--query-json",
        action="append",
        help="Structured Query Processor JSON with original/semantic/lexical fields.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MEDCPT_QUERY_ENCODER_MODEL"),
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URI", "http://localhost:6333"),
    )
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection", default="fulltext_medcpt_ip_v1")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=150)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--pmid", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--cross-encoder-model",
        default=os.getenv("MEDCPT_CROSS_ENCODER_MODEL"),
    )
    parser.add_argument("--cross-device")
    parser.add_argument(
        "--cross-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--cross-max-tokens", type=int, default=512)
    parser.add_argument("--cross-batch-size", type=int, default=32)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--context-token-budget", type=int, default=12_000)
    parser.add_argument("--context-max-parent-chunks", type=int, default=12)
    parser.add_argument("--no-context-recovery", action="store_true")
    parser.add_argument(
        "--asset-base-url", default=os.getenv("PAPER_ASSET_BASE_URL", "")
    )
    parser.add_argument("--asset-max-groups", type=int, default=8)
    parser.add_argument("--asset-max-images", type=int, default=16)
    parser.add_argument("--no-asset-expansion", action="store_true")
    parser.add_argument(
        "--opensearch-url",
        default=os.getenv("OPENSEARCH_URI", "http://localhost:9200"),
    )
    parser.add_argument(
        "--bm25-index",
        default=os.getenv("MEDCPT_BM25_INDEX", "fulltext_bm25_v1"),
    )
    parser.add_argument(
        "--opensearch-username", default=os.getenv("OPENSEARCH_USERNAME")
    )
    parser.add_argument(
        "--opensearch-password", default=os.getenv("OPENSEARCH_PASSWORD")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _processed_queries(args) -> list:
    processor = QueryProcessor()
    if args.query:
        return [processor.process(query) for query in args.query]

    processed = []
    for value in args.query_json:
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --query-json: {exc}") from exc
        original = data.get("original_query")
        if not original:
            raise SystemExit("--query-json requires original_query")
        processed.append(processor.process_model_output(original, data))
    return processed


def _dense_hit_id(hit) -> str:
    return resolve_chunk_id(hit.metadata, hit.point_id)


def _pipeline(args) -> RetrievalPipeline:
    return RetrievalPipeline(
        RetrievalConfig.from_env().with_overrides(
            bm25_top_k=args.top_k,
            vector_top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            cross_encoder_top_k=args.rerank_top_k,
            context_recovery_enabled=not args.no_context_recovery,
            context_token_budget=args.context_token_budget,
            context_max_parent_chunks=args.context_max_parent_chunks,
            asset_expansion_enabled=not args.no_asset_expansion,
            asset_max_groups=args.asset_max_groups,
            asset_max_images=args.asset_max_images,
            asset_base_url=args.asset_base_url.rstrip("/"),
            fulltext_collections=frozenset({args.collection}),
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model:
        raise SystemExit("--model or MEDCPT_QUERY_ENCODER_MODEL is required")
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if args.candidate_limit < 1:
        raise SystemExit("--candidate-limit must be positive")
    if args.rerank_top_k < 1:
        raise SystemExit("--rerank-top-k must be positive")
    if args.context_token_budget < 1:
        raise SystemExit("--context-token-budget must be positive")
    if args.context_max_parent_chunks < 1:
        raise SystemExit("--context-max-parent-chunks must be positive")
    if args.asset_max_groups < 1:
        raise SystemExit("--asset-max-groups must be positive")
    if args.asset_max_images < 1:
        raise SystemExit("--asset-max-images must be positive")

    os.environ["OPENSEARCH_URI"] = args.opensearch_url
    os.environ["OPENSEARCH_SSL"] = str(
        args.opensearch_url.lower().startswith("https://")
    ).lower()
    os.environ.setdefault("OPENSEARCH_CERT_VERIFY", "false")
    if args.opensearch_username:
        os.environ["OPENSEARCH_USERNAME"] = args.opensearch_username
    if args.opensearch_password:
        os.environ["OPENSEARCH_PASSWORD"] = args.opensearch_password

    started = time.perf_counter()
    model_started = time.perf_counter()
    encoder = MedCPTQueryEncoder(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        local_files_only=args.local_files_only,
    )
    model_load_seconds = time.perf_counter() - model_started
    cross_encoder = None
    cross_model_load_seconds = 0.0
    if args.cross_encoder_model:
        cross_started = time.perf_counter()
        cross_encoder = MedCPTCrossEncoder(
            args.cross_encoder_model,
            device=args.cross_device or args.device,
            dtype=args.cross_dtype,
            max_tokens=args.cross_max_tokens,
            batch_size=args.cross_batch_size,
            local_files_only=args.local_files_only,
        )
        cross_model_load_seconds = time.perf_counter() - cross_started
    retriever = MedCPTDenseRetriever(
        url=args.qdrant_url,
        collection_name=args.collection,
        encoder=encoder,
        api_key=args.qdrant_api_key,
    )
    collection = retriever.validate_collection()
    pipeline = _pipeline(args)

    query_reports = []
    for query in _processed_queries(args):
        vector = []
        encode_seconds = 0.0
        dense_search_seconds = 0.0
        bm25_search_seconds = 0.0
        dense_hits = []
        bm25_hits = []

        def dense_search(processed_query):
            nonlocal vector, encode_seconds, dense_hits, dense_search_seconds
            encode_started = time.perf_counter()
            vector = retriever.encode_query(processed_query.semantic_query)
            encode_seconds = time.perf_counter() - encode_started
            search_started = time.perf_counter()
            dense_hits = retriever.search_vector(
                vector, limit=args.top_k, pmids=args.pmid
            )
            dense_search_seconds = time.perf_counter() - search_started
            return [
                RankedCandidate(
                    chunk_id=_dense_hit_id(hit),
                    text=hit.text,
                    metadata=hit.metadata,
                    score=hit.score,
                )
                for hit in dense_hits
            ]

        def bm25_search(processed_query):
            nonlocal bm25_hits, bm25_search_seconds
            search_started = time.perf_counter()
            bm25_hits = search_bm25(
                [args.collection],
                processed_query.bm25_query,
                top_k=args.top_k,
                index_name=args.bm25_index,
                candidate_pmids=args.pmid or None,
                exact_terms=list(processed_query.exact_terms),
            )
            bm25_search_seconds = time.perf_counter() - search_started
            return [
                RankedCandidate(
                    chunk_id=resolve_chunk_id(
                        hit.get("metadata") or {}, hit.get("doc_id", "")
                    ),
                    text=hit.get("text", ""),
                    metadata=hit.get("metadata") or {},
                    score=hit.get("score"),
                )
                for hit in bm25_hits
            ]

        run = pipeline.search_ranked(
            query,
            dense_search=dense_search,
            bm25_search=bm25_search,
            embedding_function=lambda _: [],
            reranking_function=None,
            output_k=args.candidate_limit,
            relevance_threshold=0.0,
            rerank_enabled=False,
            dense_first=True,
        )
        dense_ids = {_dense_hit_id(hit) for hit in dense_hits}
        bm25_ids = {
            resolve_chunk_id(hit.get("metadata") or {}, hit.get("doc_id", ""))
            for hit in bm25_hits
        }
        fused_candidates = run.fused_candidates
        fusion_seconds = run.timings["fusion_seconds"]
        reranked_candidates = []
        rerank_seconds = 0.0
        if cross_encoder is not None:
            rerank_started = time.perf_counter()
            reranked_candidates = pipeline.rerank_candidates(
                query, fused_candidates, cross_encoder, args.rerank_top_k
            )
            rerank_seconds = time.perf_counter() - rerank_started
        recovered_context = []
        asset_evidence = []
        context_recovery_seconds = 0.0
        asset_expansion_seconds = 0.0
        context_stats = None
        asset_stats = None
        if reranked_candidates and not args.no_context_recovery:
            expansion = pipeline.expand_evidence(
                pipeline.candidate_documents(reranked_candidates),
                lambda chunk_ids: fetch_chunks_by_ids(
                    args.collection,
                    list(chunk_ids),
                    index_name=args.bm25_index,
                ),
                asset_enabled=not args.no_asset_expansion,
            )
            context_recovery_seconds = expansion.timings[
                "context_recovery_seconds"
            ]
            context_result = expansion.context
            assert context_result is not None
            recovered_context = [
                {"text": document.page_content, "metadata": document.metadata}
                for document in context_result.documents
            ]
            context_stats = {
                key: getattr(context_result, key)
                for key in (
                    "anchor_count",
                    "expanded_count",
                    "token_count",
                    "duplicate_count",
                    "budget_skipped_count",
                    "missing_count",
                    "cross_document_rejection_count",
                )
            }
            if expansion.assets is not None:
                asset_result = expansion.assets
                asset_expansion_seconds = expansion.timings.get(
                    "asset_expansion_seconds", 0.0
                )
                asset_evidence = [
                    {"text": document.page_content, "metadata": document.metadata}
                    for document in asset_result.documents
                ]
                asset_stats = {
                    key: getattr(asset_result, key)
                    for key in (
                        "candidate_group_count",
                        "selected_group_count",
                        "selected_image_count",
                        "duplicate_key_count",
                        "invalid_key_count",
                        "group_limit_skipped_count",
                        "image_limit_skipped_count",
                        "unresolved_url_count",
                    )
                }
        unique_candidate_count = len(dense_ids | bm25_ids)
        query_reports.append(
            {
                "query": query.to_dict(),
                "bm25_query": query.bm25_query,
                "query_vector_dimension": len(vector),
                "query_vector_norm": sum(value * value for value in vector) ** 0.5,
                "encode_seconds": round(encode_seconds, 6),
                "dense_search_seconds": round(dense_search_seconds, 6),
                "bm25_search_seconds": round(bm25_search_seconds, 6),
                "dense_hit_count": len(dense_hits),
                "bm25_hit_count": len(bm25_hits),
                "overlap_count": len(dense_ids & bm25_ids),
                "overlap_chunk_ids": sorted(dense_ids & bm25_ids),
                "unique_candidate_count": unique_candidate_count,
                "duplicates_removed": len(dense_hits)
                + len(bm25_hits)
                - unique_candidate_count,
                "fusion_seconds": round(fusion_seconds, 6),
                "fused_candidate_count": len(fused_candidates),
                "fused_candidates": [
                    candidate.to_dict() for candidate in fused_candidates
                ],
                "rerank_seconds": round(rerank_seconds, 6),
                "reranked_candidate_count": len(reranked_candidates),
                "reranked_candidates": reranked_candidates,
                "context_recovery_seconds": round(context_recovery_seconds, 6),
                "context_recovery": context_stats,
                "recovered_context": recovered_context,
                "asset_expansion_seconds": round(asset_expansion_seconds, 6),
                "asset_expansion": asset_stats,
                "asset_evidence": asset_evidence,
                "dense_hits": [hit.to_dict() for hit in dense_hits],
                "bm25_hits": bm25_hits,
            }
        )

    report = {
        "schema_version": (
            "fulltext_asset_expansion_smoke_v1"
            if (
                cross_encoder is not None
                and not args.no_context_recovery
                and not args.no_asset_expansion
            )
            else (
                "fulltext_context_recovery_smoke_v1"
                if cross_encoder is not None and not args.no_context_recovery
                else (
                    "fulltext_cross_rerank_smoke_v1"
                    if cross_encoder is not None
                    else "fulltext_rrf_smoke_v1"
                )
            )
        ),
        "model_name": encoder.model_name,
        "model_load_seconds": round(model_load_seconds, 6),
        "cross_encoder_model": (
            cross_encoder.model_name if cross_encoder is not None else None
        ),
        "cross_model_load_seconds": round(cross_model_load_seconds, 6),
        "device": args.device,
        "dtype": args.dtype,
        "collection": collection,
        "bm25_index": args.bm25_index,
        "rrf": {
            "k": args.rrf_k,
            "weights": {"dense": args.dense_weight, "bm25": args.bm25_weight},
            "candidate_limit": args.candidate_limit,
        },
        "rerank": {
            "top_k": args.rerank_top_k,
            "max_tokens": args.cross_max_tokens,
            "batch_size": args.cross_batch_size,
            "raw_logits": True,
        },
        "context_recovery": {
            "enabled": cross_encoder is not None and not args.no_context_recovery,
            "token_budget": args.context_token_budget,
            "max_parent_chunks": args.context_max_parent_chunks,
            "includes_images": False,
        },
        "asset_expansion": {
            "enabled": (
                cross_encoder is not None
                and not args.no_context_recovery
                and not args.no_asset_expansion
            ),
            "asset_base_url": args.asset_base_url or None,
            "max_groups": args.asset_max_groups,
            "max_images": args.asset_max_images,
            "visual_inference": False,
            "ocr": False,
        },
        "pmid_filter": args.pmid,
        "queries": query_reports,
        "total_seconds": round(time.perf_counter() - started, 6),
    }
    if args.output:
        _write_report(args.output, report)
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
