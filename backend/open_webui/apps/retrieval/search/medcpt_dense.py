from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DenseSearchHit:
    rank: int
    point_id: str
    score: float
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MedCPTDenseRetriever:
    """Query a 768-dimensional raw-CLS MedCPT Qdrant collection."""

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        encoder,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout: int = 120,
        client=None,
    ) -> None:
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Install qdrant-client for MedCPT retrieval") from exc
            client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )
        self.client = client
        self.collection_name = collection_name
        self.encoder = encoder

    @staticmethod
    def _distance_name(value: Any) -> str:
        return str(getattr(value, "value", value)).lower()

    def validate_collection(self) -> dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        dimension = getattr(vectors, "size", None)
        distance = self._distance_name(getattr(vectors, "distance", None))
        if dimension != self.encoder.dimension:
            raise RuntimeError(
                f"Collection dimension is {dimension}, encoder dimension is "
                f"{self.encoder.dimension}"
            )
        if distance != "dot":
            raise RuntimeError(
                f"Collection {self.collection_name!r} uses {distance}, expected dot"
            )
        return {
            "collection_name": self.collection_name,
            "status": str(getattr(info, "status", "unknown")),
            "points_count": int(getattr(info, "points_count", 0)),
            "indexed_vectors_count": int(
                getattr(info, "indexed_vectors_count", 0)
            ),
            "dimension": dimension,
            "distance": distance,
        }

    def encode_query(self, query: str) -> list[float]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        vector = self.encoder.encode(query)
        if len(vector) != self.encoder.dimension:
            raise RuntimeError(
                f"Query encoder returned {len(vector)} values, expected "
                f"{self.encoder.dimension}"
            )
        return vector

    @staticmethod
    def _pmid_filter(pmids: Sequence[str] | None):
        values = sorted({str(pmid).strip() for pmid in pmids or () if str(pmid).strip()})
        if not values:
            return None
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.pmid",
                    match=models.MatchAny(any=values),
                )
            ]
        )

    def search_vector(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        pmids: Sequence[str] | None = None,
    ) -> list[DenseSearchHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if len(vector) != self.encoder.dimension:
            raise ValueError(
                f"Query vector has {len(vector)} values, expected "
                f"{self.encoder.dimension}"
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            query_filter=self._pmid_filter(pmids),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[DenseSearchHit] = []
        for rank, point in enumerate(response.points, 1):
            payload = dict(point.payload or {})
            metadata = dict(payload.get("metadata") or {})
            hits.append(
                DenseSearchHit(
                    rank=rank,
                    point_id=str(point.id),
                    score=float(point.score),
                    text=str(payload.get("text") or ""),
                    metadata=metadata,
                )
            )
        return hits

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        pmids: Sequence[str] | None = None,
    ) -> list[DenseSearchHit]:
        return self.search_vector(
            self.encode_query(query),
            limit=limit,
            pmids=pmids,
        )
