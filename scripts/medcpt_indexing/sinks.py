from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from .models import IndexDocument

log = logging.getLogger("medcpt_fulltext_indexer")


def _with_retries(operation, *, label: str, attempts: int = 6):
    """Retry idempotent remote operations after transient service failures."""

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 30)
            log.warning(
                "%s failed; retrying in %ds (%d/%d)",
                label,
                delay,
                attempt,
                attempts,
                exc_info=True,
            )
            time.sleep(delay)


class QdrantVectorSink:
    UPSERT_BATCH_SIZE = 512

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout: int = 120,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to write MedCPT vectors") from exc
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )
        self.collection_name = collection_name

    def ensure_ready(self, dimension: int) -> None:
        from qdrant_client import models

        expected_distance = models.Distance.DOT

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=expected_distance,
                ),
            )
        else:
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            existing_dimension = getattr(vectors, "size", None)
            existing_distance = getattr(vectors, "distance", None)
            if existing_dimension != dimension:
                raise RuntimeError(
                    f"Qdrant collection {self.collection_name!r} has dimension "
                    f"{existing_dimension}, expected {dimension}"
                )
            if existing_distance != expected_distance:
                raise RuntimeError(
                    f"Qdrant collection {self.collection_name!r} uses distance "
                    f"{existing_distance}, expected {expected_distance}"
                )

        for field in (
            "metadata.pmid",
            "metadata.pmcid",
            "metadata.section",
            "metadata.chunk_type",
            "metadata.source_shard",
        ):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def write(
        self,
        documents: Sequence[IndexDocument],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        from qdrant_client import models

        if len(documents) != len(vectors):
            raise ValueError("Qdrant document/vector batch sizes differ")
        written = 0
        for start in range(0, len(documents), self.UPSERT_BATCH_SIZE):
            document_batch = documents[start : start + self.UPSERT_BATCH_SIZE]
            vector_batch = vectors[start : start + self.UPSERT_BATCH_SIZE]
            points = [
                models.PointStruct(
                    id=document.point_id,
                    vector=list(vector),
                    payload={"text": document.text, "metadata": document.metadata},
                )
                for document, vector in zip(document_batch, vector_batch, strict=True)
            ]
            _with_retries(
                lambda points=points: self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                ),
                label="Qdrant upsert",
            )
            written += len(points)
        return written

    def count_shard(self, source_shard: str) -> int:
        from qdrant_client import models

        result = _with_retries(
            lambda: self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source_shard",
                            match=models.MatchValue(value=source_shard),
                        )
                    ]
                ),
                exact=True,
            ),
            label="Qdrant shard count",
        )
        return int(result.count)


class OpenSearchKeywordSink:
    def __init__(
        self,
        *,
        url: str,
        index_name: str,
        collection_name: str,
        username: str | None = None,
        password: str | None = None,
        verify_certs: bool = False,
        shards: int = 1,
        replicas: int = 0,
    ) -> None:
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:
            raise RuntimeError("Install opensearch-py to build the BM25 index") from exc
        auth = (username or "", password or "") if username or password else None
        self.client = OpenSearch(
            hosts=[url],
            http_auth=auth,
            use_ssl=url.lower().startswith("https://"),
            verify_certs=verify_certs,
            ssl_show_warn=verify_certs,
            timeout=120,
        )
        self.index_name = index_name
        self.collection_name = collection_name
        self.shards = shards
        self.replicas = replicas

    def ensure_ready(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        body = {
            "settings": {
                "index": {
                    "number_of_shards": self.shards,
                    "number_of_replicas": self.replicas,
                    "refresh_interval": "30s",
                }
            },
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "collection_name": {"type": "keyword"},
                    "pmid": {"type": "keyword"},
                    "pmcid": {"type": "keyword"},
                    "source_shard": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "title": {"type": "text"},
                    "section_text": {"type": "text"},
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "enabled": False},
                    "source": {"type": "keyword"},
                },
            },
        }
        self.client.indices.create(index=self.index_name, body=body)

    def _source(self, document: IndexDocument) -> dict[str, Any]:
        metadata = document.metadata
        subsection = str(metadata.get("subsection") or "").strip()
        section = str(metadata.get("section") or "Unassigned").strip()
        return {
            "doc_id": document.chunk_id,
            "collection_name": self.collection_name,
            "pmid": metadata["pmid"],
            "pmcid": metadata["pmcid"],
            "source_shard": metadata["source_shard"],
            "chunk_type": metadata.get("chunk_type", "paragraph"),
            "section": section,
            "title": metadata.get("paper_title", ""),
            "section_text": f"{section} {subsection}".strip(),
            "text": document.text,
            "metadata": metadata,
            "source": "fulltext_bm25",
        }

    def write(self, documents: Sequence[IndexDocument]) -> int:
        if not documents:
            return 0
        from opensearchpy.helpers import bulk

        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": f"{self.collection_name}:{document.chunk_id}",
                "_source": self._source(document),
            }
            for document in documents
        ]
        success, failures = _with_retries(
            lambda: bulk(
                self.client,
                actions,
                refresh=False,
                raise_on_error=False,
                request_timeout=120,
            ),
            label="OpenSearch bulk write",
        )
        if failures:
            raise RuntimeError(f"OpenSearch bulk indexing had failures: {failures[:3]}")
        return int(success)

    def count_shard(self, source_shard: str) -> int:
        _with_retries(
            lambda: self.client.indices.refresh(index=self.index_name),
            label="OpenSearch refresh",
        )
        response = _with_retries(
            lambda: self.client.count(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"collection_name": self.collection_name}},
                                {"term": {"source_shard": source_shard}},
                            ]
                        }
                    }
                },
            ),
            label="OpenSearch shard count",
        )
        return int(response["count"])
