import logging
import os
from typing import Optional, Union
import time

import asyncio
import requests

from huggingface_hub import snapshot_download
from langchain_core.documents import Document

from open_webui.apps.retrieval.vector.connector import VECTOR_DB_CLIENT
from open_webui.apps.retrieval.query_processor import ProcessedQuery
from open_webui.apps.retrieval.synbio import RetrievalConfig, RetrievalPipeline
from open_webui.apps.retrieval.synbio.pipeline import (
    add_asset_urls,
)
from open_webui.utils.misc import get_last_user_message

from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger("synbiogpt.app.retrieval.utils")
log.setLevel(SRC_LOG_LEVELS["RAG"])


from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from open_webui.apps.webui.models.knowledge import Knowledges


def _embedding_function_for_collection(embedding_function, collection_name: str):
    selector = getattr(embedding_function, "for_collection", None)
    return selector(collection_name) if callable(selector) else embedding_function


def _is_base_knowledge_base(kb) -> bool:
    meta = getattr(kb, "meta", None)
    if isinstance(meta, dict) and meta.get("tag_source", False):
        return True

    data = getattr(kb, "data", None)
    if isinstance(data, dict) and data.get("tag_source", False):
        return True

    return False


def _get_base_knowledge_collection_names(exclude: Optional[set[str]] = None) -> list[str]:
    exclude_ids = exclude or set()
    collections: list[str] = []

    for kb in Knowledges.get_knowledge_bases():
        kb_id = getattr(kb, "id", None)
        if not kb_id or kb_id in exclude_ids:
            continue
        if _is_base_knowledge_base(kb):
            collections.append(kb_id)

    return collections


def _prepend_base_knowledge_collections(collection_names: list[str]) -> list[str]:
    explicit_collections = [name for name in collection_names if name]
    base_collections = _get_base_knowledge_collection_names(exclude=set(explicit_collections))

    merged: list[str] = []
    seen = set()
    for collection_name in [*base_collections, *explicit_collections]:
        if collection_name not in seen:
            seen.add(collection_name)
            merged.append(collection_name)
    return merged


def prepend_base_knowledge_collections(collection_names: list[str]) -> list[str]:
    return _prepend_base_knowledge_collections(collection_names)


class VectorSearchRetriever(BaseRetriever):
    collection_name: Any
    embedding_function: Any
    top_k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        encoder = _embedding_function_for_collection(
            self.embedding_function, self.collection_name
        )
        result = VECTOR_DB_CLIENT.search(
            collection_name=self.collection_name,
            vectors=[encoder(query)],
            limit=self.top_k,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


def query_doc(
    collection_name: str,
    query_embedding: list[float],
    k: int,
):
    try:
        result = VECTOR_DB_CLIENT.search(
            collection_name=collection_name,
            vectors=[query_embedding],
            limit=k,
        )

        log.info(f"query_doc:result {result.ids} {result.metadatas}")
        return result
    except Exception as e:
        print(e)
        raise e


_RETRIEVAL_PIPELINE = RetrievalPipeline(RetrievalConfig.from_env())


def prepare_retrieval_query(query: str):
    """Protect exact entities before the first base-model call."""

    return _RETRIEVAL_PIPELINE.prepare_query(query)


def process_generated_retrieval_query(
    query: str, model_output: str | dict
) -> ProcessedQuery:
    """Validate and normalize the structured output of query generation."""

    return _RETRIEVAL_PIPELINE.process_generated_query(query, model_output)


def process_retrieval_query(query: str | ProcessedQuery) -> ProcessedQuery:
    """Expose one shared preprocessing route for Dense, BM25, and SPECTER2."""

    return _RETRIEVAL_PIPELINE.process_query(query)


def invalidate_collection_cache(collection_name: str) -> None:
    log.info("[CACHE] invalidate skipped collection=%s", collection_name)


def _add_image_urls(metadata: dict) -> dict:
    """Expose stable image URLs without leaking storage paths."""

    return add_asset_urls(metadata, _RETRIEVAL_PIPELINE.config.asset_base_url)


def _vector_result_to_documents(result) -> list[Document]:
    ids = result.ids[0] if result and result.ids and result.ids[0] else []
    documents = result.documents[0] if result and result.documents and result.documents[0] else []
    metadatas = result.metadatas[0] if result and result.metadatas and result.metadatas[0] else []

    docs = []
    for idx, text in enumerate(documents):
        metadata = dict(metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {})
        if idx < len(ids):
            metadata.setdefault("vector_id", ids[idx])
        metadata.setdefault("retrieval_source", "vector")
        docs.append(Document(page_content=text, metadata=_add_image_urls(metadata)))
    return docs


def _bm25_results_to_documents(results: list[dict]) -> list[Document]:
    docs = []
    for result in results:
        metadata = dict(result.get("metadata") or {})
        doc_id = result.get("doc_id") or metadata.get("doc_id") or metadata.get("id")
        if doc_id is not None:
            metadata["id"] = str(doc_id)
            metadata["doc_id"] = str(doc_id)
        metadata["bm25_score"] = result.get("score")
        metadata["retrieval_source"] = "bm25"
        docs.append(
            Document(
                page_content=result.get("text", ""),
                metadata=_add_image_urls(metadata),
            )
        )
    return docs


def _query_doc_with_opensearch_hybrid(
    collection_name: str,
    query: ProcessedQuery,
    embedding_function,
    k: int,
    reranking_function,
    r: float,
    vector_call_index: int = 1,
    vector_call_total: int = 1,
) -> dict:
    def bm25_search(processed_query):
        from open_webui.apps.retrieval.search.opensearch_bm25 import search_bm25

        bm25_results = search_bm25(
            [collection_name],
            processed_query.bm25_query,
            top_k=_RETRIEVAL_PIPELINE.config.bm25_top_k,
            exact_terms=list(processed_query.exact_terms),
        )
        bm25_docs = _bm25_results_to_documents(bm25_results)
        for document in bm25_docs:
            document.metadata.setdefault("collection_name", collection_name)
        return _RETRIEVAL_PIPELINE.ranked_from_documents("bm25", bm25_docs)

    def dense_search(processed_query):
        query_encoder = _embedding_function_for_collection(
            embedding_function, collection_name
        )
        encode_started = time.perf_counter()
        query_embedding = query_encoder(processed_query.semantic_query)
        log.info(
            "[PERF] rag.query_embedding duration=%.3fs collection=%s vector_call=%d/%d",
            time.perf_counter() - encode_started,
            collection_name,
            vector_call_index,
            vector_call_total,
        )
        search_started = time.perf_counter()
        vector_result = query_doc(
            collection_name=collection_name,
            query_embedding=query_embedding,
            k=_RETRIEVAL_PIPELINE.config.vector_top_k,
        )
        vector_docs = _vector_result_to_documents(vector_result)
        for document in vector_docs:
            document.metadata.setdefault("collection_name", collection_name)
        log.info(
            "[PERF] rag.vector_db_search duration=%.3fs hits=%d top_k=%d collection=%s vector_call=%d/%d",
            time.perf_counter() - search_started,
            len(vector_docs),
            _RETRIEVAL_PIPELINE.config.vector_top_k,
            collection_name,
            vector_call_index,
            vector_call_total,
        )
        return _RETRIEVAL_PIPELINE.ranked_from_documents("dense", vector_docs)

    run = _RETRIEVAL_PIPELINE.search_ranked(
        query,
        dense_search=dense_search,
        bm25_search=bm25_search,
        embedding_function=embedding_function,
        reranking_function=reranking_function,
        output_k=k,
        relevance_threshold=r,
        collection_name=collection_name,
        tolerate_bm25_failure=True,
    )
    docs = run.reranked_documents
    log.info(
        "[PERF] rag.bm25_search duration=%.3fs hits=%d top_k=%d collection=%s",
        run.timings["bm25_search_seconds"],
        len(run.bm25_candidates),
        _RETRIEVAL_PIPELINE.config.bm25_top_k,
        collection_name,
    )
    log.info(
        "[PERF] rag.vector_search duration=%.3fs hits=%d top_k=%d collection=%s vector_call=%d/%d",
        run.timings["dense_search_seconds"],
        len(run.dense_candidates),
        _RETRIEVAL_PIPELINE.config.vector_top_k,
        collection_name,
        vector_call_index,
        vector_call_total,
    )
    log.info(
        "[PERF] rag.merge_dedupe duration=%.3fs bm25_hits=%d vector_hits=%d merged=%d limit=%d collection=%s",
        run.timings["fusion_seconds"],
        len(run.bm25_candidates),
        len(run.dense_candidates),
        len(run.fused_candidates),
        _RETRIEVAL_PIPELINE.config.candidate_limit,
        collection_name,
    )
    log.info(
        "[PERF] rag.reranker duration=%.3fs candidates=%d out=%d",
        run.timings["rerank_seconds"],
        len(run.fused_candidates),
        len(docs),
    )

    return {
        "distances": [[d.metadata.get("score") for d in docs]],
        "documents": [[d.page_content for d in docs]],
        "metadatas": [[d.metadata for d in docs]],
    }


def query_doc_with_hybrid_search(
    collection_name: str,
    query: str | ProcessedQuery,
    embedding_function,
    k: int,
    reranking_function,
    r: float,
    vector_call_index: int = 1,
    vector_call_total: int = 1,
) -> dict:
    t0 = time.perf_counter()
    try:
        processed_query = process_retrieval_query(query)
        result = _query_doc_with_opensearch_hybrid(
            collection_name=collection_name,
            embedding_function=embedding_function,
            query=processed_query,
            k=k,
            reranking_function=reranking_function,
            r=r,
            vector_call_index=vector_call_index,
            vector_call_total=vector_call_total,
        )
        log.info(
            "[PERF] rag.total_hybrid duration=%.3fs collection=%s vector_call=%d/%d",
            time.perf_counter() - t0,
            collection_name,
            vector_call_index,
            vector_call_total,
        )
        return result

    except Exception:
        log.info(
            "[PERF] rag.total_hybrid_failed duration=%.3fs",
            time.perf_counter() - t0,
        )
        raise


def merge_and_sort_query_results(
    query_results: list[dict], k: int, reverse: bool = False
) -> list[dict]:
    # Initialize lists to store combined data
    combined_distances = []
    combined_documents = []
    combined_metadatas = []

    for data in query_results:
        combined_distances.extend(data["distances"][0])
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])

    # Create a list of tuples (distance, document, metadata)
    combined = list(zip(combined_distances, combined_documents, combined_metadatas))

    # Sort the list based on distances
    combined.sort(key=lambda x: x[0], reverse=reverse)

    # We don't have anything :-(
    if not combined:
        sorted_distances = []
        sorted_documents = []
        sorted_metadatas = []
    else:
        # Unzip the sorted list
        sorted_distances, sorted_documents, sorted_metadatas = zip(*combined)

        # Slicing the lists to include only k elements
        sorted_distances = list(sorted_distances)[:k]
        sorted_documents = list(sorted_documents)[:k]
        sorted_metadatas = list(sorted_metadatas)[:k]

    # Create the output dictionary
    result = {
        "distances": [sorted_distances],
        "documents": [sorted_documents],
        "metadatas": [sorted_metadatas],
    }

    return result


def _recover_fulltext_context(result: dict, collection_names: list[str]) -> dict:
    collection_name = next(
        (
            name
            for name in collection_names
            if name in _RETRIEVAL_PIPELINE.config.fulltext_collections
        ),
        None,
    )
    if not _RETRIEVAL_PIPELINE.config.context_recovery_enabled or not collection_name:
        return result

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    if not documents or len(documents) != len(metadatas):
        return result

    from open_webui.apps.retrieval.search.opensearch_bm25 import (
        fetch_chunks_by_ids,
    )

    anchors = [
        Document(page_content=text, metadata=dict(metadata or {}))
        for text, metadata in zip(documents, metadatas, strict=True)
    ]
    started = time.perf_counter()
    try:
        expanded = _RETRIEVAL_PIPELINE.expand_evidence(
            anchors,
            lambda chunk_ids: fetch_chunks_by_ids(collection_name, list(chunk_ids)),
            asset_enabled=False,
        )
        recovered = expanded.context
        if recovered is None:
            return result
    except Exception:
        log.exception(
            "[CONTEXT] recovery failed; returning reranked anchors collection=%s",
            collection_name,
        )
        return result

    asset_documents = []
    assets = None
    if _RETRIEVAL_PIPELINE.config.asset_expansion_enabled:
        asset_started = time.perf_counter()
        assets = _RETRIEVAL_PIPELINE.expand_assets(recovered.documents)
        asset_documents = assets.documents
        for document in recovered.documents:
            document.metadata.pop("image_urls", None)
        log.info(
            "[PERF] rag.asset_expansion duration=%.3fs candidates=%d groups=%d "
            "images=%d duplicate_keys=%d image_limit_skipped=%d unresolved_urls=%d",
            time.perf_counter() - asset_started,
            assets.candidate_group_count,
            assets.selected_group_count,
            assets.selected_image_count,
            assets.duplicate_key_count,
            assets.image_limit_skipped_count,
            assets.unresolved_url_count,
        )

    log.info(
        "[PERF] rag.context_recovery duration=%.3fs anchors=%d expanded=%d "
        "tokens=%d budget_skipped=%d missing=%d cross_document_rejections=%d",
        time.perf_counter() - started,
        recovered.anchor_count,
        recovered.expanded_count,
        recovered.token_count,
        recovered.budget_skipped_count,
        recovered.missing_count,
        recovered.cross_document_rejection_count,
    )
    output_documents = [*recovered.documents, *asset_documents]
    return {
        "distances": [[doc.metadata.get("score") for doc in output_documents]],
        "documents": [[doc.page_content for doc in output_documents]],
        "metadatas": [[doc.metadata for doc in output_documents]],
    }


def query_collection(
    collection_names: list[str],
    queries: list[str | ProcessedQuery],
    embedding_function,
    k: int,
) -> dict:
    results = []
    for query in queries:
        semantic_query = process_retrieval_query(query).semantic_query
        embeddings_by_encoder = {}
        for collection_name in collection_names:
            if collection_name:
                try:
                    encoder = _embedding_function_for_collection(
                        embedding_function, collection_name
                    )
                    encoder_key = id(encoder)
                    if encoder_key not in embeddings_by_encoder:
                        embeddings_by_encoder[encoder_key] = encoder(semantic_query)
                    result = query_doc(
                        collection_name=collection_name,
                        k=k,
                        query_embedding=embeddings_by_encoder[encoder_key],
                    )
                    if result is not None:
                        results.append(result.model_dump())
                except Exception as e:
                    log.exception(f"Error when querying the collection: {e}")
            else:
                pass

    return merge_and_sort_query_results(results, k=k)


# Original query_collection_with_hybrid_search
def query_collection_with_hybrid_search(
    collection_names: list[str],
    queries: list[str | ProcessedQuery],
    embedding_function,
    k: int,
    reranking_function,
    r: float,
) -> dict:
    results = []
    error = False
    collection_vector_counts = {collection_name: 0 for collection_name in collection_names if collection_name}
    query_count = len(queries)
    collection_count = len(collection_vector_counts)
    log.info(
        "[PERF] rag.hybrid_collections collections=%d queries=%d expected_vector_searches=%d collection_names=%s",
        collection_count,
        query_count,
        collection_count * query_count,
        list(collection_vector_counts.keys()),
    )

    for collection_name in collection_names:
        try:
            for query in queries:
                collection_vector_counts[collection_name] = (
                    collection_vector_counts.get(collection_name, 0) + 1
                )
                result = query_doc_with_hybrid_search(
                    collection_name=collection_name,
                    query=query,
                    embedding_function=embedding_function,
                    k=k,
                    reranking_function=reranking_function,
                    r=r,
                    vector_call_index=collection_vector_counts[collection_name],
                    vector_call_total=query_count,
                )
                results.append(result)
        except Exception as e:
            log.exception(
                "Error when querying the collection with " f"hybrid_search: {e}"
            )
            error = True

    log.info(
        "[PERF] rag.hybrid_vector_search_counts collections=%d counts=%s total_vector_searches=%d",
        collection_count,
        collection_vector_counts,
        sum(collection_vector_counts.values()),
    )

    if error and not results:
        raise Exception(
            "Hybrid search failed for all collections. Using Non hybrid search as fallback."
        )

    merged = merge_and_sort_query_results(results, k=k, reverse=True)
    return _recover_fulltext_context(merged, collection_names)


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    embedding_batch_size,
):
    if embedding_engine == "":
        return lambda query: embedding_function.encode(query).tolist()
    elif embedding_engine in ["ollama", "openai"]:
        func = lambda query: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            url=url,
            key=key,
        )

        def generate_multiple(query, func):
            if isinstance(query, list):
                embeddings = []
                for i in range(0, len(query), embedding_batch_size):
                    embeddings.extend(func(query[i : i + embedding_batch_size]))
                return embeddings
            else:
                return func(query)

        return lambda query: generate_multiple(query, func)

def get_sources_from_files(
    files,
    queries,
    embedding_function,
    k,
    reranking_function,
    r,
    hybrid_search,
):
    # ---------------- logs ----------------
    log.debug("======================= retrieval start =======================")
    log.info("[DEBUG] queries=%r", queries)
    log.info(
        "[DEBUG] files=%s",
        [(f.get("id"), f.get("name"), f.get("type")) for f in (files or [])],
    )

    t0 = time.perf_counter()
    log.info(
        "[TIMING] get_sources_from_files start files=%d queries=%d hybrid=%s",
        len(files or []),
        len(queries) if hasattr(queries, "__len__") else -1,
        hybrid_search,
    )
    # -------------------------------------

    extracted_collections: list[str] = []
    relevant_contexts: list[dict] = []

    for idx, file in enumerate(files or [], 1):
        tf0 = time.perf_counter()

        # 1) full context returns directly without retrieval
        if file.get("context") == "full":
            context = {
                "documents": [[file.get("file", {}).get("data", {}).get("content")]],
                "metadatas": [[{"file_id": file.get("id"), "name": file.get("name")}]],
            }
            relevant_contexts.append({**context, "file": file})
            log.info(
                "[TIMING] file[%d/%d] full-context-direct %.3fs",
                idx,
                len(files or []),
                time.perf_counter() - tf0,
            )
            continue

        # 2) resolve collection_names
        collection_names: list[str] = []
        if file.get("type") == "collection":
            if file.get("legacy"):
                collection_names = file.get("collection_names", []) or []
            else:
                if file.get("id"):
                    collection_names.append(file["id"])
        elif file.get("collection_name"):
            collection_names.append(file["collection_name"])
        elif file.get("id"):
            if file.get("legacy"):
                collection_names.append(f"{file['id']}")
            else:
                collection_names.append(f"file-{file['id']}")

        # dedupe and skip already processed collections
        if file.get("type") == "collection":
            try:
                collection_names = _prepend_base_knowledge_collections(collection_names)
                log.info(
                    "[BASE_KB] merged collection_names=%s for file_id=%s",
                    collection_names,
                    file.get("id"),
                )
            except Exception:
                log.exception("[BASE_KB] failed to merge base knowledge collections")

        deduped_collection_names = []
        seen_collection_names = set()
        for collection_name in collection_names:
            if collection_name and collection_name not in seen_collection_names:
                seen_collection_names.add(collection_name)
                deduped_collection_names.append(collection_name)

        collection_names = [
            collection_name
            for collection_name in deduped_collection_names
            if collection_name not in extracted_collections
        ]
        if not collection_names:
            log.info(
                "[TIMING] file[%d/%d] skipped (already extracted) %.3fs",
                idx,
                len(files or []),
                time.perf_counter() - tf0,
            )
            continue

        # 3) text type uses content directly
        if file.get("type") == "text":
            context = file.get("content")
            if context:
                relevant_contexts.append({"documents": [[context]], "metadatas": [[{}]], "file": file})
            log.info(
                "[TIMING] file[%d/%d] text-direct %.3fs",
                idx,
                len(files or []),
                time.perf_counter() - tf0,
            )
            extracted_collections.extend(collection_names)
            continue
        # execute retrieval
        context = None

        if hybrid_search:
            th0 = time.perf_counter()
            try:
                context = query_collection_with_hybrid_search(
                    collection_names=collection_names,
                    queries=queries,
                    embedding_function=embedding_function,
                    k=k,
                    reranking_function=reranking_function,
                    r=r,
                )
                log.info(
                    "[TIMING] file[%d/%d] hybrid_search %.3fs collections=%d",
                    idx,
                    len(files or []),
                    time.perf_counter() - th0,
                    len(collection_names),
                )
            except Exception:
                log.exception("[TIMING] hybrid_search failed, fallback non-hybrid")
                context = None

        if (not hybrid_search) or (context is None):
            tn0 = time.perf_counter()
            try:
                context = query_collection(
                    collection_names=collection_names,
                    queries=queries,
                    embedding_function=embedding_function,
                    k=k,
                )
                log.info(
                    "[TIMING] file[%d/%d] non_hybrid_search %.3fs collections=%d",
                    idx,
                    len(files or []),
                    time.perf_counter() - tn0,
                    len(collection_names),
                )
            except Exception:
                log.exception("[TIMING] non_hybrid_search failed")
                context = None

        extracted_collections.extend(collection_names)

        log.info(
            "[TIMING] file[%d/%d] total %.3fs",
            idx,
            len(files or []),
            time.perf_counter() - tf0,
        )

        # 6) collect context
        if context:
            if "data" in file:
                try:
                    del file["data"]
                except Exception:
                    pass
            relevant_contexts.append({**context, "file": file})

    # 7) build sources
    ts0 = time.perf_counter()
    sources: list[dict] = []
    for ctx in relevant_contexts:
        try:
            if "documents" in ctx and "metadatas" in ctx:
                source = {
                    "source": ctx["file"],
                    "document": ctx["documents"][0],
                    "metadata": ctx["metadatas"][0],
                }
                if "distances" in ctx and ctx["distances"]:
                    source["distances"] = ctx["distances"][0]
                sources.append(source)
        except Exception:
            log.exception("[TIMING] build sources failed")

    log.info("[TIMING] build_sources %.3fs", time.perf_counter() - ts0)
    log.info(
        "[TIMING] get_sources_from_files done total %.3fs sources=%d",
        time.perf_counter() - t0,
        len(sources),
    )
    return sources


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


def generate_openai_batch_embeddings(
    model: str, texts: list[str], url: str = "https://api.openai.com/v1", key: str = ""
) -> Optional[list[list[float]]]:
    try:
        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={"input": texts, "model": model},
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        print(e)
        return None


def generate_ollama_batch_embeddings(
    model: str, texts: list[str], url: str, key: str = ""
) -> Optional[list[list[float]]]:
    try:
        r = requests.post(
            f"{url}/api/embed",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={"input": texts, "model": model},
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        print(e)
        return None


def generate_embeddings(engine: str, model: str, text: Union[str, list[str]], **kwargs):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")

    if engine == "ollama":
        if isinstance(text, list):
            embeddings = generate_ollama_batch_embeddings(
                **{"model": model, "texts": text, "url": url, "key": key}
            )
        else:
            embeddings = generate_ollama_batch_embeddings(
                **{"model": model, "texts": [text], "url": url, "key": key}
            )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "openai":
        if isinstance(text, list):
            embeddings = generate_openai_batch_embeddings(model, text, url, key)
        else:
            embeddings = generate_openai_batch_embeddings(model, [text], url, key)

        return embeddings[0] if isinstance(text, str) else embeddings
