"""Configuration for the existing SynBioGPT full-text retrieval behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _enabled(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """One immutable snapshot of the current retrieval environment."""

    default_knowledge_enabled: bool = True
    default_collection: str = "fulltext_medcpt_ip_v1"
    bm25_top_k: int = 100
    vector_top_k: int = 100
    candidate_limit: int = 150
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    cross_encoder_top_k: int = 10
    context_recovery_enabled: bool = True
    context_token_budget: int = 12_000
    context_max_parent_chunks: int = 12
    asset_expansion_enabled: bool = True
    asset_max_groups: int = 8
    asset_max_images: int = 16
    asset_base_url: str = ""
    fulltext_collections: frozenset[str] = frozenset({"fulltext_medcpt_ip_v1"})

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        collections = frozenset(
            name.strip()
            for name in os.getenv(
                "MEDCPT_QUERY_ENCODER_COLLECTIONS", "fulltext_medcpt_ip_v1"
            ).split(",")
            if name.strip()
        )
        return cls(
            default_knowledge_enabled=_enabled(
                "SYNBIO_DEFAULT_KNOWLEDGE_ENABLED", "true"
            ),
            default_collection=os.getenv(
                "SYNBIO_DEFAULT_COLLECTION", "fulltext_medcpt_ip_v1"
            ).strip(),
            bm25_top_k=max(1, int(os.getenv("RAG_HYBRID_BM25_TOP_K", "100"))),
            vector_top_k=max(1, int(os.getenv("RAG_HYBRID_VECTOR_TOP_K", "100"))),
            candidate_limit=max(
                1, int(os.getenv("RAG_HYBRID_RERANK_CANDIDATE_LIMIT", "150"))
            ),
            rrf_k=max(1, int(os.getenv("RAG_HYBRID_RRF_K", "60"))),
            dense_weight=float(os.getenv("RAG_HYBRID_DENSE_WEIGHT", "1.0")),
            bm25_weight=float(os.getenv("RAG_HYBRID_BM25_WEIGHT", "1.0")),
            cross_encoder_top_k=max(
                1, int(os.getenv("MEDCPT_CROSS_ENCODER_TOP_K", "10"))
            ),
            context_recovery_enabled=_enabled(
                "MEDCPT_CONTEXT_RECOVERY_ENABLED", "true"
            ),
            context_token_budget=max(
                1, int(os.getenv("MEDCPT_CONTEXT_TOKEN_BUDGET", "12000"))
            ),
            context_max_parent_chunks=max(
                1, int(os.getenv("MEDCPT_CONTEXT_MAX_PARENT_CHUNKS", "12"))
            ),
            asset_expansion_enabled=_enabled(
                "MEDCPT_ASSET_EXPANSION_ENABLED", "true"
            ),
            asset_max_groups=max(
                1, int(os.getenv("MEDCPT_ASSET_MAX_GROUPS", "8"))
            ),
            asset_max_images=max(
                1, int(os.getenv("MEDCPT_ASSET_MAX_IMAGES", "16"))
            ),
            asset_base_url=os.getenv("PAPER_ASSET_BASE_URL", "").rstrip("/"),
            fulltext_collections=collections,
        )

    def with_overrides(self, **values) -> "RetrievalConfig":
        return replace(self, **values)
