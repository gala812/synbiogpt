import logging
from typing import Any, Optional

from qdrant_client import QdrantClient as Qclient
from qdrant_client.http.models import PointStruct
from qdrant_client import models

from open_webui.apps.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import (
    QDRANT_API_KEY,
    QDRANT_BATCH_SIZE,
    QDRANT_COLLECTION_PREFIX,
    QDRANT_SCROLL_LIMIT,
    QDRANT_URI,
)


log = logging.getLogger(__name__)


class QdrantClient:
    def __init__(self):
        if not QDRANT_URI:
            raise ValueError("QDRANT_URI is required when VECTOR_DB=qdrant")

        self.collection_prefix = (QDRANT_COLLECTION_PREFIX or "").strip()
        self.batch_size = QDRANT_BATCH_SIZE
        self.scroll_limit = QDRANT_SCROLL_LIMIT
        self.client = Qclient(url=QDRANT_URI, api_key=QDRANT_API_KEY or None)

        log.info(
            "Qdrant vector client initialized uri=%s collection_prefix=%r",
            QDRANT_URI,
            self.collection_prefix,
        )

    def _collection_name(self, collection_name: str) -> str:
        if self.collection_prefix:
            return f"{self.collection_prefix}_{collection_name}"
        return collection_name

    def _payload_text(self, payload: dict[str, Any]) -> str:
        return (
            payload.get("text")
            or payload.get("document")
            or payload.get("page_content")
            or ""
        )

    def _payload_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return dict(metadata)

        return {
            key: value
            for key, value in payload.items()
            if key not in {"text", "document", "page_content"}
        }

    def _point_vector(self, point) -> Optional[list[float | int]]:
        vector = getattr(point, "vector", None)
        if vector is None:
            return None
        if isinstance(vector, dict):
            vector = next(iter(vector.values()), None)
        if vector is None:
            return None
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return list(vector)

    def _result_to_get_result(self, points, include_vectors: bool = False) -> GetResult:
        ids = []
        documents = []
        metadatas = []
        vectors = []

        for point in points:
            payload = point.payload or {}
            ids.append(str(point.id))
            documents.append(self._payload_text(payload))
            metadatas.append(self._payload_metadata(payload))

            if include_vectors:
                vector = self._point_vector(point)
                if vector is not None:
                    vectors.append(vector)

        return GetResult(
            ids=[ids],
            documents=[documents],
            metadatas=[metadatas],
            vectors=[vectors] if include_vectors else None,
        )

    def _filter_to_qdrant_filter(self, filter: Optional[dict]) -> Optional[models.Filter]:
        if not filter:
            return None

        must = []
        for key, value in filter.items():
            field_key = key if key.startswith("metadata.") else f"metadata.{key}"

            if isinstance(value, list):
                condition = models.FieldCondition(
                    key=field_key, match=models.MatchAny(any=value)
                )
            else:
                condition = models.FieldCondition(
                    key=field_key, match=models.MatchValue(value=value)
                )
            must.append(condition)

        return models.Filter(must=must)

    def _create_collection(self, collection_name: str, dimension: int):
        collection_name_with_prefix = self._collection_name(collection_name)
        self.client.create_collection(
            collection_name=collection_name_with_prefix,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )

        log.info("Qdrant collection created collection=%s", collection_name_with_prefix)

    def _create_collection_if_not_exists(self, collection_name, dimension):
        if not self.has_collection(collection_name=collection_name):
            self._create_collection(collection_name=collection_name, dimension=dimension)

    def _create_points(self, items: list[VectorItem]):
        points = []
        for item in items:
            metadata = dict(item["metadata"] or {})
            points.append(
                PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload={
                        "text": item["text"],
                        "metadata": metadata,
                    },
                )
            )
        return points

    def _scroll(
        self,
        collection_name: str,
        filter: Optional[dict] = None,
        limit: Optional[int] = None,
        include_vectors: bool = False,
    ):
        qdrant_collection = self._collection_name(collection_name)
        scroll_filter = self._filter_to_qdrant_filter(filter)
        points = []
        offset = None
        remaining = limit

        while True:
            batch_limit = self.scroll_limit
            if remaining is not None:
                if remaining <= 0:
                    break
                batch_limit = min(batch_limit, remaining)

            batch, offset = self.client.scroll(
                collection_name=qdrant_collection,
                scroll_filter=scroll_filter,
                limit=batch_limit,
                offset=offset,
                with_payload=True,
                with_vectors=include_vectors,
            )
            points.extend(batch)

            if remaining is not None:
                remaining -= len(batch)

            if offset is None or not batch:
                break

        return points

    def has_collection(self, collection_name: str) -> bool:
        return self.client.collection_exists(self._collection_name(collection_name))

    def delete_collection(self, collection_name: str):
        qdrant_collection = self._collection_name(collection_name)
        if not self.client.collection_exists(qdrant_collection):
            return None
        return self.client.delete_collection(collection_name=qdrant_collection)

    def search(
        self,
        collection_name: str,
        vectors: list[list[float | int]],
        limit: int,
        where: Optional[dict] = None,
    ) -> Optional[SearchResult]:
        if not self.has_collection(collection_name):
            return None

        query_response = self.client.query_points(
            collection_name=self._collection_name(collection_name),
            query=vectors[0],
            limit=limit,
            query_filter=self._filter_to_qdrant_filter(where),
            with_payload=True,
            with_vectors=False,
        )
        get_result = self._result_to_get_result(query_response.points)

        # Keep Chroma-compatible distance semantics: lower is better for cosine.
        distances = [
            1 - point.score if point.score is not None else None
            for point in query_response.points
        ]

        return SearchResult(
            ids=get_result.ids,
            documents=get_result.documents,
            metadatas=get_result.metadatas,
            distances=[distances],
        )

    def query(self, collection_name: str, filter: dict, limit: Optional[int] = None):
        if not self.has_collection(collection_name):
            return None
        try:
            points = self._scroll(
                collection_name=collection_name,
                filter=filter,
                limit=limit,
                include_vectors=True,
            )
            return self._result_to_get_result(points, include_vectors=True)
        except Exception:
            log.exception("Qdrant query failed collection=%s filter=%s", collection_name, filter)
            return None

    def get(
        self, collection_name: str, where: Optional[dict] = None
    ) -> Optional[GetResult]:
        if not self.has_collection(collection_name):
            return None
        try:
            points = self._scroll(
                collection_name=collection_name,
                filter=where,
                include_vectors=False,
            )
            return self._result_to_get_result(points, include_vectors=False)
        except Exception:
            log.exception("Qdrant get failed collection=%s where=%s", collection_name, where)
            return None

    def insert(self, collection_name: str, items: list[VectorItem]):
        if not items:
            return None
        self._create_collection_if_not_exists(collection_name, len(items[0]["vector"]))
        return self.upsert(collection_name=collection_name, items=items)

    def upsert(self, collection_name: str, items: list[VectorItem]):
        if not items:
            return None
        self._create_collection_if_not_exists(collection_name, len(items[0]["vector"]))
        qdrant_collection = self._collection_name(collection_name)
        points = self._create_points(items)
        for start in range(0, len(points), self.batch_size):
            self.client.upsert(
                collection_name=qdrant_collection,
                points=points[start : start + self.batch_size],
                wait=True,
            )

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        if not self.has_collection(collection_name):
            return None

        qdrant_collection = self._collection_name(collection_name)
        if ids:
            return self.client.delete(
                collection_name=qdrant_collection,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )

        effective_filter = filter or metadata
        if effective_filter:
            qdrant_filter = self._filter_to_qdrant_filter(effective_filter)
            return self.client.delete(
                collection_name=qdrant_collection,
                points_selector=models.FilterSelector(filter=qdrant_filter),
                wait=True,
            )

        return None

    def reset(self):
        collection_names = self.client.get_collections().collections
        for collection_name in collection_names:
            if self.collection_prefix and not collection_name.name.startswith(
                f"{self.collection_prefix}_"
            ):
                continue
            self.client.delete_collection(collection_name=collection_name.name)
