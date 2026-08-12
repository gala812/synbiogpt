from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .encoder import MedCPTArticleEncoder
from .models import IndexingConfig
from .pipeline import run_indexing, validate_inputs
from .sinks import OpenSearchKeywordSink, QdrantVectorSink


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed MedCPT full-text chunks into Qdrant and optionally OpenSearch."
    )
    parser.add_argument(
        "--chunks-dir",
        required=True,
        type=Path,
        action="append",
        help="Chunk shard directory; repeat to index additional chunk sources.",
    )
    parser.add_argument("--mapping-db", required=True, type=Path)
    parser.add_argument("--model", default=os.getenv("MEDCPT_ARTICLE_ENCODER"))
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Per-shard manifest; defaults beside the chunks directory.",
    )
    parser.add_argument("--collection", default="fulltext_medcpt_ip_v1")
    parser.add_argument("--bm25-index", default="fulltext_bm25_v1")
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Write only Qdrant; keep the existing BM25 index unchanged.",
    )
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--upload-batch-size", type=int, default=1024)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=10_000)

    parser.add_argument(
        "--qdrant-url", default=os.getenv("QDRANT_URI", "http://localhost:6333")
    )
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--qdrant-prefer-grpc", action="store_true")
    parser.add_argument(
        "--opensearch-url", default=os.getenv("OPENSEARCH_URI", "http://localhost:9200")
    )
    parser.add_argument(
        "--opensearch-username", default=os.getenv("OPENSEARCH_USERNAME")
    )
    parser.add_argument(
        "--opensearch-password", default=os.getenv("OPENSEARCH_PASSWORD")
    )
    parser.add_argument(
        "--opensearch-verify-certs",
        type=_bool,
        default=_bool(os.getenv("OPENSEARCH_CERT_VERIFY", "false")),
    )
    parser.add_argument(
        "--opensearch-shards",
        type=int,
        default=int(os.getenv("OPENSEARCH_BM25_SHARDS", "1")),
    )
    parser.add_argument(
        "--opensearch-replicas",
        type=int,
        default=int(os.getenv("OPENSEARCH_BM25_REPLICAS", "0")),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    state_file = (
        args.state_file or args.chunks_dir[0].parent / "medcpt_ip_index_manifest.json"
    )
    config = IndexingConfig(
        chunks_dir=args.chunks_dir[0],
        additional_chunks_dirs=tuple(args.chunks_dir[1:]),
        mapping_db=args.mapping_db,
        state_file=state_file,
        collection_name=args.collection,
        bm25_index_name=args.bm25_index,
        vector_only=args.vector_only,
        encode_batch_size=args.encode_batch_size,
        upload_batch_size=args.upload_batch_size,
        max_tokens=args.max_tokens,
        limit_shards=args.limit_shards,
        log_every=args.log_every,
    )
    if args.validate_only:
        print(json.dumps(validate_inputs(config), ensure_ascii=False, indent=2))
        return 0

    if not args.model:
        parser.error("--model or MEDCPT_ARTICLE_ENCODER is required for indexing")

    encoder = MedCPTArticleEncoder(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        local_files_only=args.local_files_only,
    )
    vector_sink = QdrantVectorSink(
        url=args.qdrant_url,
        collection_name=args.collection,
        api_key=args.qdrant_api_key,
        prefer_grpc=args.qdrant_prefer_grpc,
    )
    keyword_sink = None
    if not args.vector_only:
        keyword_sink = OpenSearchKeywordSink(
            url=args.opensearch_url,
            index_name=args.bm25_index,
            collection_name=args.collection,
            username=args.opensearch_username,
            password=args.opensearch_password,
            verify_certs=args.opensearch_verify_certs,
            shards=args.opensearch_shards,
            replicas=args.opensearch_replicas,
        )
    manifest = run_indexing(config, encoder, vector_sink, keyword_sink)
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))
    return 0
