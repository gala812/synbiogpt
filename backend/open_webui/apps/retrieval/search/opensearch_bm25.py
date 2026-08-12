import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("synbiogpt.app.retrieval.search.opensearch_bm25")
log.setLevel(os.getenv("RAG_LOG_LEVEL", os.getenv("GLOBAL_LOG_LEVEL", "INFO")).upper())

DEFAULT_BM25_INDEX_NAME = "open_webui_bm25"
DEFAULT_MEDCPT_BM25_INDEX_NAME = "fulltext_bm25_v1"
DEFAULT_MEDCPT_COLLECTION_NAME = "fulltext_medcpt_ip_v1"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    page_content: str
    metadata: dict[str, Any]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _get_index_name(index_name: str | None = None) -> str:
    return index_name or os.getenv("OPENSEARCH_BM25_INDEX", DEFAULT_BM25_INDEX_NAME)


def _get_search_index_name(
    collection_names: list[str], index_name: str | None = None
) -> str:
    if index_name:
        return index_name
    medcpt_collections = {
        name.strip()
        for name in os.getenv(
            "MEDCPT_QUERY_ENCODER_COLLECTIONS", DEFAULT_MEDCPT_COLLECTION_NAME
        ).split(",")
        if name.strip()
    }
    if any(name in medcpt_collections for name in collection_names):
        return os.getenv("MEDCPT_BM25_INDEX", DEFAULT_MEDCPT_BM25_INDEX_NAME)
    return _get_index_name()


def _get_client():
    from opensearchpy import OpenSearch

    opensearch_uri = os.getenv("OPENSEARCH_URI", "https://localhost:9200")
    opensearch_ssl = os.getenv("OPENSEARCH_SSL", "true")
    opensearch_cert_verify = os.getenv("OPENSEARCH_CERT_VERIFY", "false")
    opensearch_username = os.getenv("OPENSEARCH_USERNAME")
    opensearch_password = os.getenv("OPENSEARCH_PASSWORD")

    http_auth = None
    if opensearch_username or opensearch_password:
        http_auth = (opensearch_username, opensearch_password)

    return OpenSearch(
        hosts=[opensearch_uri],
        use_ssl=_as_bool(opensearch_ssl),
        verify_certs=_as_bool(opensearch_cert_verify),
        http_auth=http_auth,
    )


def ensure_bm25_index(index_name: str | None = None) -> str:
    index = _get_index_name(index_name)
    client = _get_client()

    if client.indices.exists(index=index):
        return index

    body = {
        "settings": {
            "index": {
                "number_of_shards": int(os.getenv("OPENSEARCH_BM25_SHARDS", "1")),
                "number_of_replicas": int(os.getenv("OPENSEARCH_BM25_REPLICAS", "0")),
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "collection_name": {"type": "keyword"},
                "pmid": {"type": "keyword"},
                "pmcid": {"type": "keyword"},
                "source_shard": {"type": "keyword"},
                "chunk_type": {"type": "keyword"},
                "section": {"type": "keyword"},
                "file_id": {"type": "keyword"},
                "title": {"type": "text"},
                "section_text": {"type": "text"},
                "text": {"type": "text"},
                "journal": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 512},
                    },
                },
                "publication_date": {
                    "type": "date",
                    "ignore_malformed": True,
                },
                "metadata": {"type": "object", "enabled": True},
                "source": {"type": "keyword"},
            }
        },
    }
    client.indices.create(index=index, body=body)
    log.info("[BM25] created OpenSearch index=%s", index)
    return index


def _coerce_document(doc: Any) -> dict[str, Any]:
    if isinstance(doc, dict):
        text = doc.get("text") or doc.get("page_content") or ""
        metadata = dict(doc.get("metadata") or {})
        doc_id = (
            doc.get("doc_id")
            or doc.get("id")
            or metadata.get("doc_id")
            or metadata.get("id")
        )
        collection_name = doc.get("collection_name") or metadata.get("collection_name")
        title = doc.get("title") or metadata.get("title")
        file_id = doc.get("file_id") or metadata.get("file_id")
        journal = doc.get("journal") or metadata.get("journal")
        publication_date = doc.get("publication_date") or metadata.get(
            "publication_date"
        )
        source = doc.get("source") or metadata.get("source")
    else:
        text = getattr(doc, "page_content", "") or ""
        metadata = dict(getattr(doc, "metadata", {}) or {})
        doc_id = metadata.get("doc_id") or metadata.get("id")
        collection_name = metadata.get("collection_name")
        title = metadata.get("title")
        file_id = metadata.get("file_id")
        journal = metadata.get("journal")
        publication_date = metadata.get("publication_date")
        source = metadata.get("source")

    if doc_id is None:
        raise ValueError("BM25 document is missing doc_id/id")
    if not collection_name:
        raise ValueError(f"BM25 document {doc_id!r} is missing collection_name")

    doc_id = str(doc_id)
    metadata["id"] = doc_id
    metadata["doc_id"] = doc_id
    metadata["collection_name"] = collection_name

    return {
        "doc_id": doc_id,
        "collection_name": collection_name,
        "file_id": file_id,
        "title": title,
        "text": text,
        "journal": journal,
        "publication_date": publication_date,
        "metadata": metadata,
        "source": source,
    }


def _drop_none_values(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if value is not None}


def _document_storage_id(document: dict[str, Any]) -> str:
    return f"{document['collection_name']}:{document['doc_id']}"


def index_bm25_documents(
    docs: Iterable[Any],
    index_name: str | None = None,
    refresh: bool = False,
) -> int:
    from opensearchpy.helpers import bulk

    index = ensure_bm25_index(index_name)
    actions = []
    for doc in docs:
        document = _drop_none_values(_coerce_document(doc))
        actions.append(
            {
                "_op_type": "index",
                "_index": index,
                "_id": _document_storage_id(document),
                "_source": document,
            }
        )

    if not actions:
        return 0

    success_count, _ = bulk(_get_client(), actions, refresh=refresh)
    log.info("[BM25] indexed documents index=%s count=%d", index, success_count)
    return int(success_count)


def search_bm25(
    collection_names: list[str],
    query: str,
    top_k: int = 100,
    index_name: str | None = None,
    candidate_pmids: list[str] | None = None,
    exact_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not collection_names:
        return []

    index = _get_search_index_name(collection_names, index_name)
    client = _get_client()
    filters = [{"terms": {"collection_name": collection_names}}]
    if candidate_pmids:
        filters.append({"terms": {"pmid": [str(value) for value in candidate_pmids]}})

    exact_terms = list(
        dict.fromkeys(term.strip() for term in exact_terms or [] if term.strip())
    )
    bool_query: dict[str, Any] = {
        "filter": filters,
        "must": [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "section_text^1.5", "text"],
                    "minimum_should_match": "30%",
                }
            }
        ],
    }
    if exact_terms:
        bool_query["should"] = [
            {
                "multi_match": {
                    "query": term,
                    "type": "phrase",
                    "fields": ["title^6", "section_text^3", "text^4"],
                    "boost": 4,
                }
            }
            for term in exact_terms
        ]
        bool_query["minimum_should_match"] = 1

    body = {
        "size": top_k,
        "_source": [
            "doc_id",
            "text",
            "metadata",
            "title",
            "pmid",
            "pmcid",
            "chunk_type",
            "section",
            "source_shard",
            "journal",
            "publication_date",
            "collection_name",
            "file_id",
        ],
        "query": {"bool": bool_query},
    }
    result = client.search(index=index, body=body)

    hits = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        metadata = dict(source.get("metadata") or {})
        doc_id = (
            source.get("doc_id")
            or metadata.get("doc_id")
            or metadata.get("id")
            or hit.get("_id")
        )
        if doc_id is not None:
            metadata["id"] = str(doc_id)
            metadata["doc_id"] = str(doc_id)
        if source.get("collection_name"):
            metadata["collection_name"] = source.get("collection_name")
        for field in ("pmid", "pmcid", "chunk_type", "section", "source_shard"):
            if source.get(field) is not None:
                metadata.setdefault(field, source[field])

        hits.append(
            {
                "doc_id": str(doc_id) if doc_id is not None else hit.get("_id"),
                "text": source.get("text", ""),
                "metadata": metadata,
                "score": hit.get("_score", 0.0),
                "source": "bm25",
            }
        )

    return hits


def fetch_chunks_by_ids(
    collection_name: str,
    chunk_ids: list[str],
    index_name: str | None = None,
) -> dict[str, Any]:
    """Fetch chunk text and metadata in one OpenSearch request."""

    unique_ids = list(
        dict.fromkeys(str(value).strip() for value in chunk_ids if str(value).strip())
    )
    if not unique_ids:
        return {}
    index = _get_search_index_name([collection_name], index_name)
    storage_ids = [f"{collection_name}:{chunk_id}" for chunk_id in unique_ids]
    source_fields = [
        "doc_id",
        "text",
        "metadata",
        "title",
        "pmid",
        "pmcid",
        "chunk_type",
        "section",
        "source_shard",
        "collection_name",
    ]
    client = _get_client()
    response = client.mget(
        index=index,
        body={"ids": storage_ids},
        _source=source_fields,
    )
    documents: dict[str, RetrievedChunk] = {}

    def add_source(source: dict[str, Any]) -> None:
        chunk_id = str(source.get("doc_id") or "").strip()
        if not chunk_id or chunk_id not in unique_ids:
            return
        metadata = dict(source.get("metadata") or {})
        metadata["chunk_id"] = chunk_id
        metadata["collection_name"] = source.get("collection_name") or collection_name
        for field in ("pmid", "pmcid", "chunk_type", "section", "source_shard"):
            if source.get(field) is not None:
                metadata.setdefault(field, source[field])
        documents[chunk_id] = RetrievedChunk(
            page_content=str(source.get("text") or ""),
            metadata=metadata,
        )

    for hit in response.get("docs", []):
        if hit.get("found"):
            add_source(hit.get("_source") or {})

    missing_ids = [chunk_id for chunk_id in unique_ids if chunk_id not in documents]
    if missing_ids:
        fallback = client.search(
            index=index,
            body={
                "size": len(missing_ids),
                "_source": source_fields,
                "query": {"terms": {"doc_id": missing_ids}},
            },
        )
        for hit in fallback.get("hits", {}).get("hits", []):
            add_source(hit.get("_source") or {})
    return documents
