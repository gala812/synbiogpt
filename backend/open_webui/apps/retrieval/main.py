# TODO: Merge this with the webui_app and make it a single app

import json
import logging
import mimetypes
import os
import shutil

import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tiktoken


from open_webui.storage.provider import Storage
from open_webui.apps.webui.models.knowledge import Knowledges
from open_webui.apps.webui.models.users import Users
from open_webui.apps.retrieval.vector.connector import VECTOR_DB_CLIENT
from open_webui.apps.retrieval.models.medcpt import (
    CollectionEmbeddingRouter,
    CollectionRerankerRouter,
    MedCPTCrossEncoder,
    MedCPTQueryEncoder,
)

# Document loaders
from open_webui.apps.retrieval.loaders.main import Loader
from open_webui.apps.retrieval.loaders.youtube import YoutubeLoader

# Web search engines
from open_webui.apps.retrieval.web.main import SearchResult
from open_webui.apps.retrieval.web.utils import get_web_loader
from open_webui.apps.retrieval.web.brave import search_brave
from open_webui.apps.retrieval.web.mojeek import search_mojeek
from open_webui.apps.retrieval.web.duckduckgo import search_duckduckgo
from open_webui.apps.retrieval.web.google_pse import search_google_pse
from open_webui.apps.retrieval.web.jina_search import search_jina
from open_webui.apps.retrieval.web.searchapi import search_searchapi
from open_webui.apps.retrieval.web.searxng import search_searxng
from open_webui.apps.retrieval.web.serper import search_serper
from open_webui.apps.retrieval.web.serply import search_serply
from open_webui.apps.retrieval.web.serpstack import search_serpstack
from open_webui.apps.retrieval.web.tavily import search_tavily
from open_webui.apps.retrieval.web.bing import search_bing


from open_webui.apps.retrieval.utils import (
    get_embedding_function,
    get_model_path,
    invalidate_collection_cache,
    prepend_base_knowledge_collections,
    query_collection,
    query_collection_with_hybrid_search,
)


from open_webui.apps.webui.models.files import Files
from open_webui.config import (
    BRAVE_SEARCH_API_KEY,
    MOJEEK_SEARCH_API_KEY,
    TIKTOKEN_ENCODING_NAME,
    RAG_TEXT_SPLITTER,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTENT_EXTRACTION_ENGINE,
    CORS_ALLOW_ORIGIN,
    ENABLE_RAG_HYBRID_SEARCH,
    ENABLE_RAG_LOCAL_WEB_FETCH,
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
    ENABLE_RAG_WEB_SEARCH,
    ENV,
    GOOGLE_PSE_API_KEY,
    GOOGLE_PSE_ENGINE_ID,
    PDF_EXTRACT_IMAGES,
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_FILE_MAX_COUNT,
    RAG_FILE_MAX_SIZE,
    RAG_OPENAI_API_BASE_URL,
    RAG_OPENAI_API_KEY,
    RAG_OLLAMA_BASE_URL,
    RAG_OLLAMA_API_KEY,
    RAG_RELEVANCE_THRESHOLD,
    RAG_RERANKING_MODEL,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    DEFAULT_RAG_TEMPLATE,
    RAG_TEMPLATE,
    RAG_TOP_K,
    RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
    RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
    RAG_WEB_SEARCH_ENGINE,
    RAG_WEB_SEARCH_RESULT_COUNT,
    JINA_API_KEY,
    SEARCHAPI_API_KEY,
    SEARCHAPI_ENGINE,
    SEARXNG_QUERY_URL,
    SERPER_API_KEY,
    SERPLY_API_KEY,
    SERPSTACK_API_KEY,
    SERPSTACK_HTTPS,
    TAVILY_API_KEY,
    BING_SEARCH_V7_ENDPOINT,
    BING_SEARCH_V7_SUBSCRIPTION_KEY,
    TIKA_SERVER_URL,
    UPLOAD_DIR,
    YOUTUBE_LOADER_LANGUAGE,
    YOUTUBE_LOADER_PROXY_URL,
    DEFAULT_LOCALE,
    AppConfig,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    SRC_LOG_LEVELS,
    DEVICE_TYPE,
    DOCKER,
)
from open_webui.utils.misc import (
    calculate_sha256,
    calculate_sha256_string,
    extract_folders_after_data_docs,
    sanitize_filename,
)
from open_webui.utils.utils import get_admin_user, get_verified_user

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.documents import Document


log = logging.getLogger("synbiogpt.app.retrieval.main")
log.setLevel(SRC_LOG_LEVELS["RAG"])

app = FastAPI(
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url=None,
)

app.state.config = AppConfig()

app.state.config.TOP_K = RAG_TOP_K
app.state.config.RELEVANCE_THRESHOLD = RAG_RELEVANCE_THRESHOLD
app.state.config.FILE_MAX_SIZE = RAG_FILE_MAX_SIZE
app.state.config.FILE_MAX_COUNT = RAG_FILE_MAX_COUNT

app.state.config.ENABLE_RAG_HYBRID_SEARCH = ENABLE_RAG_HYBRID_SEARCH
app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = (
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION
)

app.state.config.CONTENT_EXTRACTION_ENGINE = CONTENT_EXTRACTION_ENGINE
app.state.config.TIKA_SERVER_URL = TIKA_SERVER_URL

app.state.config.TEXT_SPLITTER = RAG_TEXT_SPLITTER
app.state.config.TIKTOKEN_ENCODING_NAME = TIKTOKEN_ENCODING_NAME

app.state.config.CHUNK_SIZE = CHUNK_SIZE
app.state.config.CHUNK_OVERLAP = CHUNK_OVERLAP

app.state.config.RAG_EMBEDDING_ENGINE = RAG_EMBEDDING_ENGINE
app.state.config.RAG_EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
app.state.config.RAG_EMBEDDING_BATCH_SIZE = RAG_EMBEDDING_BATCH_SIZE
app.state.config.RAG_RERANKING_MODEL = RAG_RERANKING_MODEL
app.state.config.RAG_TEMPLATE = RAG_TEMPLATE

app.state.config.OPENAI_API_BASE_URL = RAG_OPENAI_API_BASE_URL
app.state.config.OPENAI_API_KEY = RAG_OPENAI_API_KEY

app.state.config.OLLAMA_BASE_URL = RAG_OLLAMA_BASE_URL
app.state.config.OLLAMA_API_KEY = RAG_OLLAMA_API_KEY

app.state.config.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

app.state.config.YOUTUBE_LOADER_LANGUAGE = YOUTUBE_LOADER_LANGUAGE
app.state.config.YOUTUBE_LOADER_PROXY_URL = YOUTUBE_LOADER_PROXY_URL
app.state.YOUTUBE_LOADER_TRANSLATION = None


app.state.config.ENABLE_RAG_WEB_SEARCH = ENABLE_RAG_WEB_SEARCH
app.state.config.RAG_WEB_SEARCH_ENGINE = RAG_WEB_SEARCH_ENGINE
app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST = RAG_WEB_SEARCH_DOMAIN_FILTER_LIST

app.state.config.SEARXNG_QUERY_URL = SEARXNG_QUERY_URL
app.state.config.GOOGLE_PSE_API_KEY = GOOGLE_PSE_API_KEY
app.state.config.GOOGLE_PSE_ENGINE_ID = GOOGLE_PSE_ENGINE_ID
app.state.config.BRAVE_SEARCH_API_KEY = BRAVE_SEARCH_API_KEY
app.state.config.MOJEEK_SEARCH_API_KEY = MOJEEK_SEARCH_API_KEY
app.state.config.SERPSTACK_API_KEY = SERPSTACK_API_KEY
app.state.config.SERPSTACK_HTTPS = SERPSTACK_HTTPS
app.state.config.SERPER_API_KEY = SERPER_API_KEY
app.state.config.SERPLY_API_KEY = SERPLY_API_KEY
app.state.config.TAVILY_API_KEY = TAVILY_API_KEY
app.state.config.SEARCHAPI_API_KEY = SEARCHAPI_API_KEY
app.state.config.SEARCHAPI_ENGINE = SEARCHAPI_ENGINE
app.state.config.JINA_API_KEY = JINA_API_KEY
app.state.config.BING_SEARCH_V7_ENDPOINT = BING_SEARCH_V7_ENDPOINT
app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY = BING_SEARCH_V7_SUBSCRIPTION_KEY

app.state.config.RAG_WEB_SEARCH_RESULT_COUNT = RAG_WEB_SEARCH_RESULT_COUNT
app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS = RAG_WEB_SEARCH_CONCURRENT_REQUESTS


def update_embedding_model(
    embedding_model: str,
    auto_update: bool = False,
):
    if embedding_model and app.state.config.RAG_EMBEDDING_ENGINE == "":
        from sentence_transformers import SentenceTransformer

        try:
            app.state.sentence_transformer_ef = SentenceTransformer(
                get_model_path(embedding_model, auto_update),
                device=DEVICE_TYPE,
                trust_remote_code=RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
            )
        except Exception as e:
            log.debug(f"Error loading SentenceTransformer: {e}")
            app.state.sentence_transformer_ef = None
    else:
        app.state.sentence_transformer_ef = None


def update_reranking_model(
    reranking_model: str,
    auto_update: bool = False,
):
    if reranking_model:
        if any(model in reranking_model for model in ["jinaai/jina-colbert-v2"]):
            try:
                from open_webui.apps.retrieval.models.colbert import ColBERT

                app.state.sentence_transformer_rf = ColBERT(
                    get_model_path(reranking_model, auto_update),
                    env="docker" if DOCKER else None,
                )
            except Exception as e:
                log.error(f"ColBERT: {e}")
                app.state.sentence_transformer_rf = None
                app.state.config.ENABLE_RAG_HYBRID_SEARCH = False
        else:
            import sentence_transformers
            import torch
            import os

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            try:
                app.state.sentence_transformer_rf = sentence_transformers.CrossEncoder(
                    get_model_path(reranking_model, auto_update),
                    device=device,
                    trust_remote_code=RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
                )
                try:
                    app.state.sentence_transformer_rf.model.to(device)
                except Exception as e:
                    log.info("[DEBUG] CrossEncoder model.to(%s) failed: %s", device, e)
                p = next(app.state.sentence_transformer_rf.model.parameters())
                log.info("[DEBUG] CrossEncoder param device=%s dtype=%s", p.device, p.dtype)
            except:
                log.error("CrossEncoder error")
                app.state.sentence_transformer_rf = None
                app.state.config.ENABLE_RAG_HYBRID_SEARCH = False
    else:
        app.state.sentence_transformer_rf = None


update_embedding_model(
    app.state.config.RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
)

update_reranking_model(
    app.state.config.RAG_RERANKING_MODEL,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
)


def _medcpt_query_collections() -> set[str]:
    return {
        name.strip()
        for name in os.getenv(
            "MEDCPT_QUERY_ENCODER_COLLECTIONS", "fulltext_medcpt_ip_v1"
        ).split(",")
        if name.strip()
    }


def _load_medcpt_query_encoder():
    model_name = os.getenv("MEDCPT_QUERY_ENCODER_MODEL", "").strip()
    if not model_name:
        return None

    encoder = MedCPTQueryEncoder(
        model_name,
        device=os.getenv("MEDCPT_QUERY_ENCODER_DEVICE", DEVICE_TYPE),
        dtype=os.getenv("MEDCPT_QUERY_ENCODER_DTYPE", "auto"),
        max_tokens=int(os.getenv("MEDCPT_QUERY_ENCODER_MAX_TOKENS", "64")),
        local_files_only=os.getenv(
            "MEDCPT_QUERY_ENCODER_LOCAL_FILES_ONLY", "true"
        ).lower()
        in {"1", "true", "yes", "on"},
    )
    log.info(
        "Loaded MedCPT Query Encoder model=%s device=%s dimension=%d",
        model_name,
        os.getenv("MEDCPT_QUERY_ENCODER_DEVICE", DEVICE_TYPE),
        encoder.dimension,
    )
    return encoder


def _load_medcpt_cross_encoder():
    model_name = os.getenv("MEDCPT_CROSS_ENCODER_MODEL", "").strip()
    if not model_name:
        return None

    encoder = MedCPTCrossEncoder(
        model_name,
        device=os.getenv("MEDCPT_CROSS_ENCODER_DEVICE", DEVICE_TYPE),
        dtype=os.getenv("MEDCPT_CROSS_ENCODER_DTYPE", "auto"),
        max_tokens=int(os.getenv("MEDCPT_CROSS_ENCODER_MAX_TOKENS", "512")),
        batch_size=int(os.getenv("MEDCPT_CROSS_ENCODER_BATCH_SIZE", "32")),
        local_files_only=os.getenv(
            "MEDCPT_CROSS_ENCODER_LOCAL_FILES_ONLY", "true"
        ).lower()
        in {"1", "true", "yes", "on"},
    )
    log.info(
        "Loaded MedCPT Cross Encoder model=%s device=%s",
        model_name,
        os.getenv("MEDCPT_CROSS_ENCODER_DEVICE", DEVICE_TYPE),
    )
    return encoder


def _build_embedding_function():
    default = get_embedding_function(
        app.state.config.RAG_EMBEDDING_ENGINE,
        app.state.config.RAG_EMBEDDING_MODEL,
        app.state.sentence_transformer_ef,
        (
            app.state.config.OPENAI_API_BASE_URL
            if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
            else app.state.config.OLLAMA_BASE_URL
        ),
        (
            app.state.config.OPENAI_API_KEY
            if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
            else app.state.config.OLLAMA_API_KEY
        ),
        app.state.config.RAG_EMBEDDING_BATCH_SIZE,
    )
    encoder = app.state.medcpt_query_encoder
    if encoder is None:
        return default

    collection_names = _medcpt_query_collections()
    if not collection_names:
        raise RuntimeError("MEDCPT_QUERY_ENCODER_COLLECTIONS cannot be empty")
    return CollectionEmbeddingRouter(
        default,
        {name: encoder.encode for name in collection_names},
    )


def _build_reranking_function(default):
    if isinstance(default, CollectionRerankerRouter):
        default = default.default
    encoder = app.state.medcpt_cross_encoder
    if encoder is None:
        return default
    return CollectionRerankerRouter(
        default,
        {name: encoder for name in _medcpt_query_collections()},
    )


app.state.medcpt_query_encoder = _load_medcpt_query_encoder()
app.state.EMBEDDING_FUNCTION = _build_embedding_function()
app.state.medcpt_cross_encoder = _load_medcpt_cross_encoder()
app.state.sentence_transformer_rf = _build_reranking_function(
    app.state.sentence_transformer_rf
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_tags_from_metadatas(metadatas: list) -> list[str]:
    tags = set()
    if not metadatas:
        return []
    for metadata_list in metadatas:
        if not metadata_list:
            continue
        for metadata in metadata_list:
            if not isinstance(metadata, dict):
                continue
            value = metadata.get("tags")
            if isinstance(value, list):
                for tag in value:
                    if isinstance(tag, str) and tag:
                        tags.add(tag)
            elif isinstance(value, str) and value:
                if value.startswith("|") and value.endswith("|"):
                    parts = [p for p in value.split("|") if p]
                    for tag in parts:
                        tags.add(tag)
                else:
                    tags.add(value)
    return sorted(tags)


def _encode_tags_value(value):
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        if items:
            return "|" + "|".join(items) + "|"
        return ""
    if isinstance(value, str):
        return value
    return value


class CollectionNameForm(BaseModel):
    collection_name: Optional[str] = None


class ProcessUrlForm(CollectionNameForm):
    url: str


class SearchForm(CollectionNameForm):
    query: str


@app.get("/")
async def get_status():
    return {
        "status": True,
        "chunk_size": app.state.config.CHUNK_SIZE,
        "chunk_overlap": app.state.config.CHUNK_OVERLAP,
        "template": app.state.config.RAG_TEMPLATE,
        "embedding_engine": app.state.config.RAG_EMBEDDING_ENGINE,
        "embedding_model": app.state.config.RAG_EMBEDDING_MODEL,
        "reranking_model": app.state.config.RAG_RERANKING_MODEL,
        "embedding_batch_size": app.state.config.RAG_EMBEDDING_BATCH_SIZE,
    }


@app.get("/embedding")
async def get_embedding_config(user=Depends(get_admin_user)):
    medcpt_encoder = app.state.medcpt_query_encoder
    return {
        "status": True,
        "embedding_engine": app.state.config.RAG_EMBEDDING_ENGINE,
        "embedding_model": app.state.config.RAG_EMBEDDING_MODEL,
        "embedding_batch_size": app.state.config.RAG_EMBEDDING_BATCH_SIZE,
        "openai_config": {
            "url": app.state.config.OPENAI_API_BASE_URL,
            "key": app.state.config.OPENAI_API_KEY,
        },
        "ollama_config": {
            "url": app.state.config.OLLAMA_BASE_URL,
            "key": app.state.config.OLLAMA_API_KEY,
        },
        "medcpt_query_encoder": {
            "enabled": medcpt_encoder is not None,
            "model": medcpt_encoder.model_name if medcpt_encoder else None,
            "dimension": medcpt_encoder.dimension if medcpt_encoder else None,
            "collections": sorted(_medcpt_query_collections()),
        },
    }


@app.get("/reranking")
async def get_reraanking_config(user=Depends(get_admin_user)):
    return {
        "status": True,
        "reranking_model": app.state.config.RAG_RERANKING_MODEL,
    }


class OpenAIConfigForm(BaseModel):
    url: str
    key: str


class OllamaConfigForm(BaseModel):
    url: str
    key: str


class EmbeddingModelUpdateForm(BaseModel):
    openai_config: Optional[OpenAIConfigForm] = None
    ollama_config: Optional[OllamaConfigForm] = None
    embedding_engine: str
    embedding_model: str
    embedding_batch_size: Optional[int] = 1


@app.post("/embedding/update")
async def update_embedding_config(
    form_data: EmbeddingModelUpdateForm, user=Depends(get_admin_user)
):
    log.info(
        f"Updating embedding model: {app.state.config.RAG_EMBEDDING_MODEL} to {form_data.embedding_model}"
    )
    try:
        app.state.config.RAG_EMBEDDING_ENGINE = form_data.embedding_engine
        app.state.config.RAG_EMBEDDING_MODEL = form_data.embedding_model

        if app.state.config.RAG_EMBEDDING_ENGINE in ["ollama", "openai"]:
            if form_data.openai_config is not None:
                app.state.config.OPENAI_API_BASE_URL = form_data.openai_config.url
                app.state.config.OPENAI_API_KEY = form_data.openai_config.key

            if form_data.ollama_config is not None:
                app.state.config.OLLAMA_BASE_URL = form_data.ollama_config.url
                app.state.config.OLLAMA_API_KEY = form_data.ollama_config.key

            app.state.config.RAG_EMBEDDING_BATCH_SIZE = form_data.embedding_batch_size

        update_embedding_model(app.state.config.RAG_EMBEDDING_MODEL)

        app.state.EMBEDDING_FUNCTION = _build_embedding_function()

        return {
            "status": True,
            "embedding_engine": app.state.config.RAG_EMBEDDING_ENGINE,
            "embedding_model": app.state.config.RAG_EMBEDDING_MODEL,
            "embedding_batch_size": app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            "openai_config": {
                "url": app.state.config.OPENAI_API_BASE_URL,
                "key": app.state.config.OPENAI_API_KEY,
            },
            "ollama_config": {
                "url": app.state.config.OLLAMA_BASE_URL,
                "key": app.state.config.OLLAMA_API_KEY,
            },
        }
    except Exception as e:
        log.exception(f"Problem updating embedding model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class RerankingModelUpdateForm(BaseModel):
    reranking_model: str


@app.post("/reranking/update")
async def update_reranking_config(
    form_data: RerankingModelUpdateForm, user=Depends(get_admin_user)
):
    log.info(
        f"Updating reranking model: {app.state.config.RAG_RERANKING_MODEL} to {form_data.reranking_model}"
    )
    try:
        app.state.config.RAG_RERANKING_MODEL = form_data.reranking_model

        update_reranking_model(app.state.config.RAG_RERANKING_MODEL, True)
        app.state.sentence_transformer_rf = _build_reranking_function(
            app.state.sentence_transformer_rf
        )

        return {
            "status": True,
            "reranking_model": app.state.config.RAG_RERANKING_MODEL,
        }
    except Exception as e:
        log.exception(f"Problem updating reranking model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.get("/config")
async def get_rag_config(user=Depends(get_admin_user)):
    return {
        "status": True,
        "pdf_extract_images": app.state.config.PDF_EXTRACT_IMAGES,
        "content_extraction": {
            "engine": app.state.config.CONTENT_EXTRACTION_ENGINE,
            "tika_server_url": app.state.config.TIKA_SERVER_URL,
        },
        "chunk": {
            "text_splitter": app.state.config.TEXT_SPLITTER,
            "chunk_size": app.state.config.CHUNK_SIZE,
            "chunk_overlap": app.state.config.CHUNK_OVERLAP,
        },
        "file": {
            "max_size": app.state.config.FILE_MAX_SIZE,
            "max_count": app.state.config.FILE_MAX_COUNT,
        },
        "youtube": {
            "language": app.state.config.YOUTUBE_LOADER_LANGUAGE,
            "translation": app.state.YOUTUBE_LOADER_TRANSLATION,
            "proxy_url": app.state.config.YOUTUBE_LOADER_PROXY_URL,
        },
        "web": {
            "web_loader_ssl_verification": app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
            "search": {
                "enabled": app.state.config.ENABLE_RAG_WEB_SEARCH,
                "engine": app.state.config.RAG_WEB_SEARCH_ENGINE,
                "searxng_query_url": app.state.config.SEARXNG_QUERY_URL,
                "google_pse_api_key": app.state.config.GOOGLE_PSE_API_KEY,
                "google_pse_engine_id": app.state.config.GOOGLE_PSE_ENGINE_ID,
                "brave_search_api_key": app.state.config.BRAVE_SEARCH_API_KEY,
                "mojeek_search_api_key": app.state.config.MOJEEK_SEARCH_API_KEY,
                "serpstack_api_key": app.state.config.SERPSTACK_API_KEY,
                "serpstack_https": app.state.config.SERPSTACK_HTTPS,
                "serper_api_key": app.state.config.SERPER_API_KEY,
                "serply_api_key": app.state.config.SERPLY_API_KEY,
                "tavily_api_key": app.state.config.TAVILY_API_KEY,
                "searchapi_api_key": app.state.config.SEARCHAPI_API_KEY,
                "seaarchapi_engine": app.state.config.SEARCHAPI_ENGINE,
                "jina_api_key": app.state.config.JINA_API_KEY,
                "bing_search_v7_endpoint": app.state.config.BING_SEARCH_V7_ENDPOINT,
                "bing_search_v7_subscription_key": app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
                "result_count": app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                "concurrent_requests": app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
            },
        },
    }


class FileConfig(BaseModel):
    max_size: Optional[int] = None
    max_count: Optional[int] = None


class ContentExtractionConfig(BaseModel):
    engine: str = ""
    tika_server_url: Optional[str] = None


class ChunkParamUpdateForm(BaseModel):
    text_splitter: Optional[str] = None
    chunk_size: int
    chunk_overlap: int


class YoutubeLoaderConfig(BaseModel):
    language: list[str]
    translation: Optional[str] = None
    proxy_url: str = ""


class WebSearchConfig(BaseModel):
    enabled: bool
    engine: Optional[str] = None
    searxng_query_url: Optional[str] = None
    google_pse_api_key: Optional[str] = None
    google_pse_engine_id: Optional[str] = None
    brave_search_api_key: Optional[str] = None
    mojeek_search_api_key: Optional[str] = None
    serpstack_api_key: Optional[str] = None
    serpstack_https: Optional[bool] = None
    serper_api_key: Optional[str] = None
    serply_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    searchapi_api_key: Optional[str] = None
    searchapi_engine: Optional[str] = None
    jina_api_key: Optional[str] = None
    bing_search_v7_endpoint: Optional[str] = None
    bing_search_v7_subscription_key: Optional[str] = None
    result_count: Optional[int] = None
    concurrent_requests: Optional[int] = None


class WebConfig(BaseModel):
    search: WebSearchConfig
    web_loader_ssl_verification: Optional[bool] = None


class ConfigUpdateForm(BaseModel):
    pdf_extract_images: Optional[bool] = None
    file: Optional[FileConfig] = None
    content_extraction: Optional[ContentExtractionConfig] = None
    chunk: Optional[ChunkParamUpdateForm] = None
    youtube: Optional[YoutubeLoaderConfig] = None
    web: Optional[WebConfig] = None


@app.post("/config/update")
async def update_rag_config(form_data: ConfigUpdateForm, user=Depends(get_admin_user)):
    app.state.config.PDF_EXTRACT_IMAGES = (
        form_data.pdf_extract_images
        if form_data.pdf_extract_images is not None
        else app.state.config.PDF_EXTRACT_IMAGES
    )

    if form_data.file is not None:
        app.state.config.FILE_MAX_SIZE = form_data.file.max_size
        app.state.config.FILE_MAX_COUNT = form_data.file.max_count

    if form_data.content_extraction is not None:
        log.info(f"Updating text settings: {form_data.content_extraction}")
        app.state.config.CONTENT_EXTRACTION_ENGINE = form_data.content_extraction.engine
        app.state.config.TIKA_SERVER_URL = form_data.content_extraction.tika_server_url

    if form_data.chunk is not None:
        app.state.config.TEXT_SPLITTER = form_data.chunk.text_splitter
        app.state.config.CHUNK_SIZE = form_data.chunk.chunk_size
        app.state.config.CHUNK_OVERLAP = form_data.chunk.chunk_overlap

    log.info("[CONFIG_UPDATE] splitter=%s size=%s overlap=%s",
        app.state.config.TEXT_SPLITTER,
        app.state.config.CHUNK_SIZE,
        app.state.config.CHUNK_OVERLAP)

    if form_data.youtube is not None:
        app.state.config.YOUTUBE_LOADER_LANGUAGE = form_data.youtube.language
        app.state.config.YOUTUBE_LOADER_PROXY_URL = form_data.youtube.proxy_url
        app.state.YOUTUBE_LOADER_TRANSLATION = form_data.youtube.translation

    if form_data.web is not None:
        app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = (
            # Note: When UI "Bypass SSL verification for Websites"=True then ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION=False
            form_data.web.web_loader_ssl_verification
        )

        app.state.config.ENABLE_RAG_WEB_SEARCH = form_data.web.search.enabled
        app.state.config.RAG_WEB_SEARCH_ENGINE = form_data.web.search.engine
        app.state.config.SEARXNG_QUERY_URL = form_data.web.search.searxng_query_url
        app.state.config.GOOGLE_PSE_API_KEY = form_data.web.search.google_pse_api_key
        app.state.config.GOOGLE_PSE_ENGINE_ID = (
            form_data.web.search.google_pse_engine_id
        )
        app.state.config.BRAVE_SEARCH_API_KEY = (
            form_data.web.search.brave_search_api_key
        )
        app.state.config.MOJEEK_SEARCH_API_KEY = (
            form_data.web.search.mojeek_search_api_key
        )
        app.state.config.SERPSTACK_API_KEY = form_data.web.search.serpstack_api_key
        app.state.config.SERPSTACK_HTTPS = form_data.web.search.serpstack_https
        app.state.config.SERPER_API_KEY = form_data.web.search.serper_api_key
        app.state.config.SERPLY_API_KEY = form_data.web.search.serply_api_key
        app.state.config.TAVILY_API_KEY = form_data.web.search.tavily_api_key
        app.state.config.SEARCHAPI_API_KEY = form_data.web.search.searchapi_api_key
        app.state.config.SEARCHAPI_ENGINE = form_data.web.search.searchapi_engine

        app.state.config.JINA_API_KEY = form_data.web.search.jina_api_key
        app.state.config.BING_SEARCH_V7_ENDPOINT = (
            form_data.web.search.bing_search_v7_endpoint
        )
        app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY = (
            form_data.web.search.bing_search_v7_subscription_key
        )

        app.state.config.RAG_WEB_SEARCH_RESULT_COUNT = form_data.web.search.result_count
        app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS = (
            form_data.web.search.concurrent_requests
        )

    return {
        "status": True,
        "pdf_extract_images": app.state.config.PDF_EXTRACT_IMAGES,
        "file": {
            "max_size": app.state.config.FILE_MAX_SIZE,
            "max_count": app.state.config.FILE_MAX_COUNT,
        },
        "content_extraction": {
            "engine": app.state.config.CONTENT_EXTRACTION_ENGINE,
            "tika_server_url": app.state.config.TIKA_SERVER_URL,
        },
        "chunk": {
            "text_splitter": app.state.config.TEXT_SPLITTER,
            "chunk_size": app.state.config.CHUNK_SIZE,
            "chunk_overlap": app.state.config.CHUNK_OVERLAP,
        },
        "youtube": {
            "language": app.state.config.YOUTUBE_LOADER_LANGUAGE,
            "proxy_url": app.state.config.YOUTUBE_LOADER_PROXY_URL,
            "translation": app.state.YOUTUBE_LOADER_TRANSLATION,
        },
        "web": {
            "web_loader_ssl_verification": app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
            "search": {
                "enabled": app.state.config.ENABLE_RAG_WEB_SEARCH,
                "engine": app.state.config.RAG_WEB_SEARCH_ENGINE,
                "searxng_query_url": app.state.config.SEARXNG_QUERY_URL,
                "google_pse_api_key": app.state.config.GOOGLE_PSE_API_KEY,
                "google_pse_engine_id": app.state.config.GOOGLE_PSE_ENGINE_ID,
                "brave_search_api_key": app.state.config.BRAVE_SEARCH_API_KEY,
                "mojeek_search_api_key": app.state.config.MOJEEK_SEARCH_API_KEY,
                "serpstack_api_key": app.state.config.SERPSTACK_API_KEY,
                "serpstack_https": app.state.config.SERPSTACK_HTTPS,
                "serper_api_key": app.state.config.SERPER_API_KEY,
                "serply_api_key": app.state.config.SERPLY_API_KEY,
                "serachapi_api_key": app.state.config.SEARCHAPI_API_KEY,
                "searchapi_engine": app.state.config.SEARCHAPI_ENGINE,
                "tavily_api_key": app.state.config.TAVILY_API_KEY,
                "jina_api_key": app.state.config.JINA_API_KEY,
                "bing_search_v7_endpoint": app.state.config.BING_SEARCH_V7_ENDPOINT,
                "bing_search_v7_subscription_key": app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
                "result_count": app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                "concurrent_requests": app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
            },
        },
    }


@app.get("/template")
async def get_rag_template(user=Depends(get_verified_user)):
    return {
        "status": True,
        "template": app.state.config.RAG_TEMPLATE,
    }

def _normalize_meta(meta):
    if meta is None:
        return None
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        s = meta.strip()
        if not s or s.lower() == "null":
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None

# @app.get("/tags")
# def get_vector_tags(user=Depends(get_verified_user)):
#     try:
#         log.info("[TAGS] enter user_id=%s", getattr(user, "id", None))

#         tag_sources = []
#         for kb in Knowledges.get_knowledge_bases_by_user_id(user.id, "read"):
#             meta = _normalize_meta(getattr(kb, "meta", None))
#             if meta and meta.get("tag_source", False):
#                 tag_sources.append(kb)

#         log.info(
#             "[TAGS] tag_sources count=%d ids=%s",
#             len(tag_sources),
#             [kb.id for kb in tag_sources],
#         )

#         if not tag_sources:
#             log.info("[TAGS] no tag_sources, return empty list")
#             return []

#         all_tags = set()

#         for kb in tag_sources:
#             log.info("[TAGS] scan collection kb_id=%s", kb.id)

#             has_col = VECTOR_DB_CLIENT.has_collection(collection_name=kb.id)
#             log.info("[TAGS] has_collection kb_id=%s has=%s", kb.id, has_col)
#             if not has_col:
#                 continue

#             result = VECTOR_DB_CLIENT.get(collection_name=kb.id)
#             # 注意：result 的结构依赖你的 VectorDB client，这里只做“安全打印”
#             metadatas = (result.metadatas if result else []) or []
#             meta_count = len(metadatas)

#             # 打印少量样例：只取前 1 条 metadata 的 keys，避免泄露内容
#             meta_keys_sample = []
#             if meta_count > 0 and isinstance(metadatas[0], dict):
#                 meta_keys_sample = list(metadatas[0].keys())

#             log.info(
#                 "[TAGS] get result kb_id=%s result_none=%s metadatas_count=%d meta_keys_sample=%s",
#                 kb.id,
#                 result is None,
#                 meta_count,
#                 meta_keys_sample,
#             )

#             tags = _extract_tags_from_metadatas(metadatas)
#             log.info(
#                 "[TAGS] extracted kb_id=%s tags_count=%d tags_sample=%s",
#                 kb.id,
#                 len(tags),
#                 sorted(list(tags))[:10],  # 只打前10个样例
#             )

#             before = len(all_tags)
#             all_tags.update(tags)
#             after = len(all_tags)

#             log.info(
#                 "[TAGS] merge kb_id=%s all_tags_before=%d all_tags_after=%d delta=%d",
#                 kb.id,
#                 before,
#                 after,
#                 after - before,
#             )

#         output = [{"id": tag, "name": tag} for tag in sorted(all_tags)]
#         log.info(
#             "[TAGS] done total_unique_tags=%d output_count=%d output_sample=%s",
#             len(all_tags),
#             len(output),
#             [o["id"] for o in output[:10]],
#         )

#         return output

#     except Exception as e:
#         log.exception("[TAGS] failed error=%s", e)
#         return []
from fastapi import Query

@app.get("/tags")
def get_vector_tags(
    query: str = Query("", alias="query"),
    limit: int = Query(20, ge=1, le=200),
    user=Depends(get_verified_user),
):
    try:
        q = (query or "").strip()
        log.info("[TAGS] enter user_id=%s query=%s limit=%s", getattr(user, "id", None), q, limit)

        # 关键：空 query 不扫库，避免页面加载/聚焦就全量读取向量库
        if not q:
            log.info("[TAGS] empty query, return empty list")
            return []

        q_lower = q.lower()

        tag_sources = []
        for kb in Knowledges.get_knowledge_bases_by_user_id(user.id, "read"):
            meta = _normalize_meta(getattr(kb, "meta", None))
            if meta and meta.get("tag_source", False):
                tag_sources.append(kb)

        log.info("[TAGS] tag_sources count=%d ids=%s", len(tag_sources), [kb.id for kb in tag_sources])

        if not tag_sources:
            log.info("[TAGS] no tag_sources, return empty list")
            return []

        matched = []
        seen = set()

        def maybe_add(tag: str):
            if tag in seen:
                return False
            # 匹配策略：contains；需要更严格可换 startswith
            if q_lower in tag.lower():
                seen.add(tag)
                matched.append(tag)
                return True
            return False

        for kb in tag_sources:
            if len(matched) >= limit:
                break

            log.info("[TAGS] scan collection kb_id=%s", kb.id)

            has_col = VECTOR_DB_CLIENT.has_collection(collection_name=kb.id)
            log.info("[TAGS] has_collection kb_id=%s has=%s", kb.id, has_col)
            if not has_col:
                continue

            result = VECTOR_DB_CLIENT.get(collection_name=kb.id)
            metadatas = (result.metadatas if result else []) or []

            tags = _extract_tags_from_metadatas(metadatas)

            # 稳定遍历（可选，但推荐）
            for t in sorted(tags):
                if maybe_add(t) and len(matched) >= limit:
                    break

        output = [{"id": t, "name": t} for t in matched]
        log.info("[TAGS] done matched=%d output_sample=%s", len(output), [o["id"] for o in output[:10]])
        return output

    except Exception as e:
        log.exception("[TAGS] failed error=%s", e)
        return []

@app.get("/query/settings")
async def get_query_settings(user=Depends(get_admin_user)):
    return {
        "status": True,
        "template": app.state.config.RAG_TEMPLATE,
        "k": app.state.config.TOP_K,
        "r": app.state.config.RELEVANCE_THRESHOLD,
        "hybrid": app.state.config.ENABLE_RAG_HYBRID_SEARCH,
    }


class QuerySettingsForm(BaseModel):
    k: Optional[int] = None
    r: Optional[float] = None
    template: Optional[str] = None
    hybrid: Optional[bool] = None


@app.post("/query/settings/update")
async def update_query_settings(
    form_data: QuerySettingsForm, user=Depends(get_admin_user)
):
    app.state.config.RAG_TEMPLATE = form_data.template
    app.state.config.TOP_K = form_data.k if form_data.k else 4
    app.state.config.RELEVANCE_THRESHOLD = form_data.r if form_data.r else 0.0

    app.state.config.ENABLE_RAG_HYBRID_SEARCH = (
        form_data.hybrid if form_data.hybrid else False
    )

    return {
        "status": True,
        "template": app.state.config.RAG_TEMPLATE,
        "k": app.state.config.TOP_K,
        "r": app.state.config.RELEVANCE_THRESHOLD,
        "hybrid": app.state.config.ENABLE_RAG_HYBRID_SEARCH,
    }


####################################
#
# Document process and retrieval
#
####################################


def _get_docs_info(docs: list[Document]) -> str:
    docs_info = set()

    # Trying to select relevant metadata identifying the document.
    for doc in docs:
        metadata = getattr(doc, "metadata", {})
        doc_name = metadata.get("name", "")
        if not doc_name:
            doc_name = metadata.get("title", "")
        if not doc_name:
            doc_name = metadata.get("source", "")
        if doc_name:
            docs_info.add(doc_name)

    return ", ".join(docs_info)

import re

def _normalize_tag(t: str) -> str:
    return re.sub(r"\s+", "", str(t or "")).strip()

def _inject_tag_flags(metadata: dict) -> None:
    """
    目标：
    - 保留你原来的 metadata["tags"] = "|a|b|"（用于展示/抽取 tags）
    - 额外写入 metadata["tag__a"]=True 这种字段，供 where 过滤
    """
    raw = metadata.get("tags")
    tags: list[str] = []

    if isinstance(raw, list):
        tags = [_normalize_tag(x) for x in raw]
    elif isinstance(raw, str) and raw:
        s = raw.strip()
        if s.startswith("|") and s.endswith("|"):
            parts = [p for p in s.split("|") if p]
            tags = [_normalize_tag(x) for x in parts]
        else:
            tags = [_normalize_tag(s)]
    else:
        return

    tags = [t for t in tags if t]
    if not tags:
        return

    # 保持你原来的 pipe 格式（可选，但建议保留，方便你 /tags 抽取）
    metadata["tags"] = "|" + "|".join(tags) + "|"

    # ✅ 写入可过滤字段
    for t in tags:
        metadata[f"tag__{t}"] = True


def _normalize_retrieval_identity(metadata: dict, collection_name: str) -> None:
    doc_id = metadata.get("id") or metadata.get("doc_id")
    if doc_id is not None:
        doc_id = str(doc_id)
        metadata["id"] = doc_id
        metadata["doc_id"] = doc_id
    metadata["collection_name"] = collection_name


def _index_docs_to_bm25(texts: list[str], metadatas: list[dict]) -> None:
    bm25_docs = []
    for text, metadata in zip(texts, metadatas):
        doc_id = metadata.get("doc_id") or metadata.get("id")
        collection_name = metadata.get("collection_name")
        if not doc_id or not collection_name:
            continue

        bm25_docs.append(
            {
                "doc_id": doc_id,
                "collection_name": collection_name,
                "title": metadata.get("title", ""),
                "text": text,
                "journal": metadata.get("journal", ""),
                "publication_date": metadata.get("publication_date"),
                "metadata": metadata,
            }
        )

    if not bm25_docs:
        return

    try:
        from open_webui.apps.retrieval.search.opensearch_bm25 import (
            ensure_bm25_index,
            index_bm25_documents,
        )

        ensure_bm25_index()
        indexed_count = index_bm25_documents(bm25_docs)
        log.info("[BM25] indexed sidecar documents count=%d", indexed_count)
    except Exception as e:
        log.exception("BM25 index write failed: %s", e)


def save_docs_to_vector_db(
    docs,
    collection_name,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
    split: bool = True,
    add: bool = False,
    precomputed_embeddings: Optional[list[list[float | int]]] = None,
) -> bool:
    log.info(
        f"save_docs_to_vector_db: document {_get_docs_info(docs)} {collection_name}"
    )
    log.info(
        "[VECTORIZE] collection=%s | docs=%d | first_len=%d",
        collection_name,
        len(docs),
        sum(len(d.page_content) for d in docs),
    )
    # Check if entries with the same hash (metadata.hash) already exist
    if metadata and "hash" in metadata:
        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={"hash": metadata["hash"]},
        )

        if result is not None:
            existing_doc_ids = result.ids[0]
            if existing_doc_ids:
                log.info(f"Document with hash {metadata['hash']} already exists")
                if add:
                    return True
                raise ValueError(ERROR_MESSAGES.DUPLICATE_CONTENT)

    if split:
        # if app.state.config.TEXT_SPLITTER in ["", "character"]:
        splitter = (app.state.config.TEXT_SPLITTER or "token").strip().lower()

        if splitter == "character":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=app.state.config.CHUNK_SIZE,
                chunk_overlap=app.state.config.CHUNK_OVERLAP,
                add_start_index=True,
            )
        # elif app.state.config.TEXT_SPLITTER == "token":
        elif splitter == "token":
            log.info("Using token text splitter: %s", app.state.config.TIKTOKEN_ENCODING_NAME)

            tiktoken.get_encoding(str(app.state.config.TIKTOKEN_ENCODING_NAME))
            text_splitter = TokenTextSplitter(
                encoding_name=str(app.state.config.TIKTOKEN_ENCODING_NAME),
                chunk_size=app.state.config.CHUNK_SIZE,
                chunk_overlap=app.state.config.CHUNK_OVERLAP,
                add_start_index=True,
            )
        else:
            raise ValueError(ERROR_MESSAGES.DEFAULT("Invalid text splitter"))

        before = len(docs)
        docs = text_splitter.split_documents(docs)
        after = len(docs)
        log.info("[SPLIT] before=%d after=%d chunk_size=%d overlap=%d",
                before, after, app.state.config.CHUNK_SIZE, app.state.config.CHUNK_OVERLAP)
        log.info("[SPLIT] sample_meta=%s", docs[0].metadata if docs else None)

    if len(docs) == 0:
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

    texts = [doc.page_content for doc in docs]
    metadatas = [
        {
            **doc.metadata,
            **(metadata if metadata else {}),
            "embedding_config": json.dumps(
                {
                    "engine": app.state.config.RAG_EMBEDDING_ENGINE,
                    "model": app.state.config.RAG_EMBEDDING_MODEL,
                }
            ),
        }
        for doc in docs
    ]

    # ChromaDB does not like datetime formats
    # for meta-data so convert them to string.
    for metadata in metadatas:
        _normalize_retrieval_identity(metadata, collection_name)

        for key, value in metadata.items():
            if isinstance(value, datetime):
                metadata[key] = str(value)
            elif isinstance(value, list):
                metadata[key] = _encode_tags_value(value)

        _inject_tag_flags(metadata)

    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=collection_name):
            log.info(f"collection {collection_name} already exists")

            if overwrite:
                VECTOR_DB_CLIENT.delete_collection(collection_name=collection_name)
                log.info(f"deleting existing collection {collection_name}")
            elif add is False:
                log.info(
                    f"collection {collection_name} already exists, overwrite is False and add is False"
                )
                return True

        log.info(f"adding to collection {collection_name}")
        embedding_function = get_embedding_function(
            app.state.config.RAG_EMBEDDING_ENGINE,
            app.state.config.RAG_EMBEDDING_MODEL,
            app.state.sentence_transformer_ef,
            (
                app.state.config.OPENAI_API_BASE_URL
                if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                else app.state.config.OLLAMA_BASE_URL
            ),
            (
                app.state.config.OPENAI_API_KEY
                if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                else app.state.config.OLLAMA_API_KEY
            ),
            app.state.config.RAG_EMBEDDING_BATCH_SIZE,
        )

        if precomputed_embeddings is not None:
            if len(precomputed_embeddings) != len(texts):
                raise ValueError(
                    ERROR_MESSAGES.DEFAULT(
                        "Precomputed embeddings length mismatch with texts"
                    )
                )
            embeddings = precomputed_embeddings
            log.info("[VECTORIZE] using precomputed embeddings count=%d", len(embeddings))
        else:
            embeddings = embedding_function(
                list(map(lambda x: x.replace("\n", " "), texts))
            )

        items = [
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "vector": embeddings[idx],
                "metadata": metadatas[idx],
            }
            for idx, text in enumerate(texts)
        ]

        VECTOR_DB_CLIENT.insert(
            collection_name=collection_name,
            items=items,
        )
        _index_docs_to_bm25(texts, metadatas)
        invalidate_collection_cache(collection_name)

        return True
    except Exception as e:
        log.exception(e)
        raise e


class ProcessFileForm(BaseModel):
    file_id: str
    content: Optional[str] = None
    collection_name: Optional[str] = None


@app.post("/process/file")
def process_file(
    form_data: ProcessFileForm,
    user=Depends(get_verified_user),
):
    try:
        file = Files.get_file_by_id(form_data.file_id)

        collection_name = form_data.collection_name
        precomputed_embeddings = None
        split_for_save = True

        if collection_name is None:
            collection_name = f"file-{file.id}"

        if form_data.content:
            # Update the content in the file
            # Usage: /files/{file_id}/data/content/update

            VECTOR_DB_CLIENT.delete_collection(collection_name=f"file-{file.id}")

            docs = [
                Document(
                    page_content=form_data.content.replace("<br/>", "\n"),
                    metadata={
                        **file.meta,
                        "name": file.filename,
                        "created_by": file.user_id,
                        "file_id": file.id,
                        "source": file.filename,
                    },
                )
            ]

            text_content = form_data.content
        elif form_data.collection_name:
            # Check if the file has already been processed and save the content
            # Usage: /knowledge/{id}/file/add, /knowledge/{id}/file/update

            result = VECTOR_DB_CLIENT.query(
                collection_name=f"file-{file.id}", filter={"file_id": file.id}
            )

            if result is not None and len(result.ids[0]) > 0:
                docs = [
                    Document(
                        page_content=result.documents[0][idx],
                        metadata=result.metadatas[0][idx],
                    )
                    for idx, id in enumerate(result.ids[0])
                ]

                vectors = result.vectors[0] if result.vectors else None
                if vectors is not None and len(vectors) == len(docs):
                    precomputed_embeddings = vectors
                    split_for_save = False
                    log.info(
                        "[PROCESS_FILE] reuse embeddings from file collection count=%d",
                        len(vectors),
                    )
            else:
                docs = [
                    Document(
                        page_content=file.data.get("content", ""),
                        metadata={
                            **file.meta,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                ]

            text_content = file.data.get("content", "")
        else:
            # Process the file and save the content
            # Usage: /files/
            file_path = file.path
            if file_path:
                file_path = Storage.get_file(file_path)
                loader = Loader(
                    engine=app.state.config.CONTENT_EXTRACTION_ENGINE,
                    TIKA_SERVER_URL=app.state.config.TIKA_SERVER_URL,
                    PDF_EXTRACT_IMAGES=app.state.config.PDF_EXTRACT_IMAGES,
                )
                docs = loader.load(
                    file.filename, file.meta.get("content_type"), file_path
                )

                docs = [
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                    for doc in docs
                ]
                log.info(
                        "[PROCESS_FILE] docs=%d | first_doc_len=%d | first_doc_head=%r",
                        len(docs),
                        len(docs[0].page_content) if docs else -1,
                        docs[0].page_content[:120] if docs else None,
                    )
            else:
                docs = [
                    Document(
                        page_content=file.data.get("content", ""),
                        metadata={
                            **file.meta,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                ]
            text_content = " ".join([doc.page_content for doc in docs])

        log.debug(f"text_content: {text_content}")
        Files.update_file_data_by_id(
            file.id,
            {"content": text_content},
        )

        hash = calculate_sha256_string(text_content)
        Files.update_file_hash_by_id(file.id, hash)

        try:
            log.info(
                "[PROCESS_FILE][BEFORE_EMBED] docs=%d | total_chars=%d | sample=%r",
                len(docs),
                sum(len(d.page_content or "") for d in docs),
                docs[0].page_content[:200] if docs else None,
            )

            result = save_docs_to_vector_db(
                docs=docs,
                collection_name=collection_name,
                metadata={
                    "file_id": file.id,
                    "name": file.filename,
                    "hash": hash,
                },
                split=split_for_save,
                add=(True if form_data.collection_name else False),
                precomputed_embeddings=precomputed_embeddings,
            )

            if result:
                Files.update_file_metadata_by_id(
                    file.id,
                    {
                        "collection_name": collection_name,
                    },
                )

                return {
                    "status": True,
                    "collection_name": collection_name,
                    "filename": file.filename,
                    "content": text_content,
                }
        except Exception as e:
            raise e
    except Exception as e:
        log.exception(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )


class ProcessTextForm(BaseModel):
    name: str
    content: str
    collection_name: Optional[str] = None


@app.post("/process/text")
def process_text(
    form_data: ProcessTextForm,
    user=Depends(get_verified_user),
):
    collection_name = form_data.collection_name
    if collection_name is None:
        collection_name = calculate_sha256_string(form_data.content)

    docs = [
        Document(
            page_content=form_data.content,
            metadata={"name": form_data.name, "created_by": user.id},
        )
    ]
    text_content = form_data.content
    log.debug(f"text_content: {text_content}")

    result = save_docs_to_vector_db(docs, collection_name)

    if result:
        return {
            "status": True,
            "collection_name": collection_name,
            "content": text_content,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(),
        )


@app.post("/process/youtube")
def process_youtube_video(form_data: ProcessUrlForm, user=Depends(get_verified_user)):
    try:
        collection_name = form_data.collection_name
        if not collection_name:
            collection_name = calculate_sha256_string(form_data.url)[:63]

        loader = YoutubeLoader(
            form_data.url,
            language=app.state.config.YOUTUBE_LOADER_LANGUAGE,
            proxy_url=app.state.config.YOUTUBE_LOADER_PROXY_URL,
        )

        docs = loader.load()
        content = " ".join([doc.page_content for doc in docs])
        log.debug(f"text_content: {content}")
        save_docs_to_vector_db(docs, collection_name, overwrite=True)

        return {
            "status": True,
            "collection_name": collection_name,
            "filename": form_data.url,
            "file": {
                "data": {
                    "content": content,
                },
                "meta": {
                    "name": form_data.url,
                },
            },
        }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.post("/process/web")
def process_web(form_data: ProcessUrlForm, user=Depends(get_verified_user)):
    try:
        collection_name = form_data.collection_name
        if not collection_name:
            collection_name = calculate_sha256_string(form_data.url)[:63]

        loader = get_web_loader(
            form_data.url,
            verify_ssl=app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
            requests_per_second=app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
        )
        docs = loader.load()
        content = " ".join([doc.page_content for doc in docs])
        log.debug(f"text_content: {content}")
        save_docs_to_vector_db(docs, collection_name, overwrite=True)

        return {
            "status": True,
            "collection_name": collection_name,
            "filename": form_data.url,
            "file": {
                "data": {
                    "content": content,
                },
                "meta": {
                    "name": form_data.url,
                },
            },
        }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def search_web(engine: str, query: str) -> list[SearchResult]:
    """Search the web using a search engine and return the results as a list of SearchResult objects.
    Will look for a search engine API key in environment variables in the following order:
    - SEARXNG_QUERY_URL
    - GOOGLE_PSE_API_KEY + GOOGLE_PSE_ENGINE_ID
    - BRAVE_SEARCH_API_KEY
    - MOJEEK_SEARCH_API_KEY
    - SERPSTACK_API_KEY
    - SERPER_API_KEY
    - SERPLY_API_KEY
    - TAVILY_API_KEY
    - SEARCHAPI_API_KEY + SEARCHAPI_ENGINE (by default `google`)
    Args:
        query (str): The query to search for
    """

    # TODO: add playwright to search the web
    if engine == "searxng":
        if app.state.config.SEARXNG_QUERY_URL:
            return search_searxng(
                app.state.config.SEARXNG_QUERY_URL,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SEARXNG_QUERY_URL found in environment variables")
    elif engine == "google_pse":
        if (
            app.state.config.GOOGLE_PSE_API_KEY
            and app.state.config.GOOGLE_PSE_ENGINE_ID
        ):
            return search_google_pse(
                app.state.config.GOOGLE_PSE_API_KEY,
                app.state.config.GOOGLE_PSE_ENGINE_ID,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception(
                "No GOOGLE_PSE_API_KEY or GOOGLE_PSE_ENGINE_ID found in environment variables"
            )
    elif engine == "brave":
        if app.state.config.BRAVE_SEARCH_API_KEY:
            return search_brave(
                app.state.config.BRAVE_SEARCH_API_KEY,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No BRAVE_SEARCH_API_KEY found in environment variables")
    elif engine == "mojeek":
        if app.state.config.MOJEEK_SEARCH_API_KEY:
            return search_mojeek(
                app.state.config.MOJEEK_SEARCH_API_KEY,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No MOJEEK_SEARCH_API_KEY found in environment variables")
    elif engine == "serpstack":
        if app.state.config.SERPSTACK_API_KEY:
            return search_serpstack(
                app.state.config.SERPSTACK_API_KEY,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
                https_enabled=app.state.config.SERPSTACK_HTTPS,
            )
        else:
            raise Exception("No SERPSTACK_API_KEY found in environment variables")
    elif engine == "serper":
        if app.state.config.SERPER_API_KEY:
            return search_serper(
                app.state.config.SERPER_API_KEY,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SERPER_API_KEY found in environment variables")
    elif engine == "serply":
        if app.state.config.SERPLY_API_KEY:
            return search_serply(
                app.state.config.SERPLY_API_KEY,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SERPLY_API_KEY found in environment variables")
    elif engine == "duckduckgo":
        return search_duckduckgo(
            query,
            app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
            app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    elif engine == "tavily":
        if app.state.config.TAVILY_API_KEY:
            return search_tavily(
                app.state.config.TAVILY_API_KEY,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
            )
        else:
            raise Exception("No TAVILY_API_KEY found in environment variables")
    elif engine == "searchapi":
        if app.state.config.SEARCHAPI_API_KEY:
            return search_searchapi(
                app.state.config.SEARCHAPI_API_KEY,
                app.state.config.SEARCHAPI_ENGINE,
                query,
                app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
                app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SEARCHAPI_API_KEY found in environment variables")
    elif engine == "jina":
        return search_jina(
            app.state.config.JINA_API_KEY,
            query,
            app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
        )
    elif engine == "bing":
        return search_bing(
            app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
            app.state.config.BING_SEARCH_V7_ENDPOINT,
            str(DEFAULT_LOCALE),
            query,
            app.state.config.RAG_WEB_SEARCH_RESULT_COUNT,
            app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    else:
        raise Exception("No search engine API key found in environment variables")


@app.post("/process/web/search")
def process_web_search(form_data: SearchForm, user=Depends(get_verified_user)):
    try:
        logging.info(
            f"trying to web search with {app.state.config.RAG_WEB_SEARCH_ENGINE, form_data.query}"
        )
        web_results = search_web(
            app.state.config.RAG_WEB_SEARCH_ENGINE, form_data.query
        )
    except Exception as e:
        log.exception(e)

        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.WEB_SEARCH_ERROR(e),
        )

    try:
        collection_name = form_data.collection_name
        if collection_name == "":
            collection_name = calculate_sha256_string(form_data.query)[:63]

        urls = [result.link for result in web_results]

        loader = get_web_loader(
            urls,
            verify_ssl=app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
            requests_per_second=app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
        )
        docs = loader.aload()

        save_docs_to_vector_db(docs, collection_name, overwrite=True)

        return {
            "status": True,
            "collection_name": collection_name,
            "filenames": urls,
        }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class QueryDocForm(BaseModel):
    collection_name: str
    query: str
    k: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None
    tags: Optional[list[str]] = None 


@app.post("/query/doc")
def query_doc_handler(
    form_data: QueryDocForm,
    user=Depends(get_verified_user),
):
    log.info("===============================================================")
    log.info("[QUERY_DOC] collection=%s tags=%s", form_data.collection_name, form_data.tags)
    try:
        collection_names = prepend_base_knowledge_collections([form_data.collection_name])
        log.info("[QUERY_DOC] merged collection_names=%s", collection_names)

        if app.state.config.ENABLE_RAG_HYBRID_SEARCH:
            return query_collection_with_hybrid_search(
                collection_names=collection_names,
                queries=[form_data.query],
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
                reranking_function=app.state.sentence_transformer_rf,
                r=(
                    form_data.r if form_data.r else app.state.config.RELEVANCE_THRESHOLD
                ),
            )
        else:
            return query_collection(
                collection_names=collection_names,
                queries=[form_data.query],
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
            )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class QueryCollectionsForm(BaseModel):
    collection_names: list[str]
    query: str
    k: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


@app.post("/query/collection")
def query_collection_handler(
    form_data: QueryCollectionsForm,
    user=Depends(get_verified_user),
):
    try:
        merged_collection_names = prepend_base_knowledge_collections(form_data.collection_names)
        log.info(
            "[QUERY_COLLECTION] merged collection_names=%s from input=%s",
            merged_collection_names,
            form_data.collection_names,
        )

        if app.state.config.ENABLE_RAG_HYBRID_SEARCH:
            return query_collection_with_hybrid_search(
                collection_names=merged_collection_names,
                queries=[form_data.query],
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
                reranking_function=app.state.sentence_transformer_rf,
                r=(
                    form_data.r if form_data.r else app.state.config.RELEVANCE_THRESHOLD
                ),
            )
        else:
            return query_collection(
                collection_names=merged_collection_names,
                queries=[form_data.query],
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
            )

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


####################################
#
# Vector DB operations
#
####################################


class DeleteForm(BaseModel):
    collection_name: str
    file_id: str


@app.post("/delete")
def delete_entries_from_collection(form_data: DeleteForm, user=Depends(get_admin_user)):
    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=form_data.collection_name):
            file = Files.get_file_by_id(form_data.file_id)
            hash = file.hash

            VECTOR_DB_CLIENT.delete(
                collection_name=form_data.collection_name,
                metadata={"hash": hash},
            )
            return {"status": True}
        else:
            return {"status": False}
    except Exception as e:
        log.exception(e)
        return {"status": False}


@app.post("/reset/db")
def reset_vector_db(user=Depends(get_admin_user)):
    VECTOR_DB_CLIENT.reset()
    Knowledges.delete_all_knowledge()


@app.post("/reset/uploads")
def reset_upload_dir(user=Depends(get_admin_user)) -> bool:
    folder = f"{UPLOAD_DIR}"
    try:
        # Check if the directory exists
        if os.path.exists(folder):
            # Iterate over all the files and directories in the specified directory
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"The directory {folder} does not exist")
    except Exception as e:
        print(f"Failed to process the directory {folder}. Reason: {e}")
    return True


if ENV == "dev":

    @app.get("/ef")
    async def get_embeddings():
        return {"result": app.state.EMBEDDING_FUNCTION("hello world")}

    @app.get("/ef/{text}")
    async def get_embeddings_text(text: str):
        return {"result": app.state.EMBEDDING_FUNCTION(text)}

####################################
#
# PDF to JSONL conversion
#
####################################

class ParsePdfResult(BaseModel):
    source_pdf: str
    doc_id: str
    title: str
    jsonl_filename: str
    content: str
    status: str

from open_webui.apps.retrieval.pdf_tools import parse_pdf_to_jsonl

# @app.post("/parse-pdf")
# async def parse_pdf(
#     files: list[UploadFile] = File(...),
#     user=Depends(get_verified_user)
# ):

#     MAX_PDF_UPLOAD = 10

#     if len(files) > MAX_PDF_UPLOAD:
#         raise HTTPException(
#             status_code=400,
#             detail=f"最多上传 {MAX_PDF_UPLOAD} 个 PDF"
#         )

#     results = []

#     for file in files:

#         if not file.filename.lower().endswith(".pdf"):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"{file.filename} 不是 PDF"
#             )

#         temp_path = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")

#         with open(temp_path, "wb") as f:
#             f.write(await file.read())

#         try:

#             jsonl_content = parse_pdf_to_jsonl(temp_path)

#             results.append({
#                 "source_pdf": file.filename,
#                 "jsonl_filename": f"{temp_path.stem}.jsonl",
#                 "content": jsonl_content,
#                 "status": "success"
#             })

#         except Exception as e:

#             results.append({
#                 "source_pdf": file.filename,
#                 "status": "failed",
#                 "error": str(e)
#             })

#         finally:

#             if temp_path.exists():
#                 temp_path.unlink()

#     return {
#         "mode": "sync",
#         "results": results
#     }
@app.post("/parse-pdf")
async def parse_pdf(
    files: list[UploadFile] = File(...),
    user=Depends(get_verified_user),
):
    DEFAULT_PDF_UPLOAD_LIMIT = 20

    user_info = user.info or {}

    is_admin = getattr(user, "role", None) == "admin"

    if not is_admin:
        pdf_upload_limit = user_info.get("pdf_upload_limit", DEFAULT_PDF_UPLOAD_LIMIT)
        pdf_upload_used = user_info.get("pdf_upload_used", 0)
        remaining_pdf_upload = max(0, pdf_upload_limit - pdf_upload_used)

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未上传 PDF 文件",
        )

    if not is_admin and len(files) > remaining_pdf_upload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"当前剩余可上传 PDF 数量为 {remaining_pdf_upload}，本次最多只能上传 {remaining_pdf_upload} 个 PDF",
        )

    results = []

    for file in files:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            results.append(
                {
                    "source_pdf": filename,
                    "status": "failed",
                    "error": "仅支持 PDF 文件",
                }
            )
            continue

        temp_path = Path(UPLOAD_DIR) / f"{uuid.uuid4()}_{filename}"

        try:
            with open(temp_path, "wb") as f:
                f.write(await file.read())

            jsonl_content = parse_pdf_to_jsonl(temp_path)
            first_record = json.loads(jsonl_content)
            real_title = first_record.get("title") or temp_path.stem

            results.append(
                {
                    "source_pdf": filename,
                    "doc_id": temp_path.stem,
                    "title": real_title,
                    "jsonl_filename": f"{temp_path.stem}.jsonl",
                    "content": jsonl_content,
                    "status": "success",
                }
            )

        except Exception as e:
            log.exception(e)
            results.append(
                {
                    "source_pdf": filename,
                    "status": "failed",
                    "error": str(e),
                }
            )
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    success_count = sum(1 for item in results if item.get("status") == "success")

    if not is_admin and success_count > 0:
        updated_info = dict(user_info)
        updated_info["pdf_upload_limit"] = pdf_upload_limit
        updated_info["pdf_upload_used"] = pdf_upload_used + success_count
        Users.update_user_by_id(user.id, {"info": updated_info})

    return {
        "mode": "sync",
        "total": len(files),
        "results": results,
    }
