import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


log = logging.getLogger("backfill_bm25_from_jsonl")
_BM25_MODULE = None


def _load_bm25_module():
    global _BM25_MODULE
    if _BM25_MODULE is not None:
        return _BM25_MODULE

    module_path = BACKEND_DIR / "open_webui" / "apps" / "retrieval" / "search" / "opensearch_bm25.py"
    spec = importlib.util.spec_from_file_location("opensearch_bm25_sidecar", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load BM25 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BM25_MODULE = module
    return _BM25_MODULE


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _get_index_name(index_name: str | None) -> str:
    return index_name or os.getenv("OPENSEARCH_BM25_INDEX", "open_webui_bm25")


def _get_opensearch_base_url() -> str:
    return os.getenv("OPENSEARCH_URI", "https://localhost:9200").rstrip("/") + "/"


def _get_rest_auth() -> tuple[str, str] | None:
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")
    if username or password:
        return username or "", password or ""
    return None


def _rest_request(method: str, path: str, **kwargs):
    import requests

    verify = _as_bool(os.getenv("OPENSEARCH_CERT_VERIFY", "false"))
    url = urljoin(_get_opensearch_base_url(), path.lstrip("/"))
    response = requests.request(
        method,
        url,
        auth=_get_rest_auth(),
        verify=verify,
        timeout=120,
        **kwargs,
    )
    return response


def _bm25_index_body() -> dict[str, Any]:
    return {
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
                "file_id": {"type": "keyword"},
                "title": {"type": "text"},
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


def _drop_none_values(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if value is not None}


def _document_storage_id(document: dict[str, Any]) -> str:
    return f"{document['collection_name']}:{document['doc_id']}"


def _rest_ensure_bm25_index(index_name: str | None) -> str:
    index = _get_index_name(index_name)
    exists_response = _rest_request("HEAD", index)
    if exists_response.status_code == 200:
        return index
    if exists_response.status_code != 404:
        raise RuntimeError(
            f"OpenSearch index check failed status={exists_response.status_code} "
            f"body={exists_response.text}"
        )

    create_response = _rest_request("PUT", index, json=_bm25_index_body())
    if create_response.status_code not in {200, 201}:
        raise RuntimeError(
            f"OpenSearch index create failed status={create_response.status_code} "
            f"body={create_response.text}"
        )
    log.info("[BM25_BACKFILL] created OpenSearch index=%s via REST", index)
    return index


def _rest_index_bm25_documents(
    docs: list[dict[str, Any]],
    *,
    index_name: str | None,
    refresh: bool,
) -> int:
    if not docs:
        return 0

    index = _rest_ensure_bm25_index(index_name)
    lines = []
    for doc in docs:
        document = _drop_none_values(doc)
        lines.append(json.dumps({"index": {"_id": _document_storage_id(document)}}))
        lines.append(json.dumps(document, ensure_ascii=False))

    params = "?refresh=true" if refresh else ""
    response = _rest_request(
        "POST",
        f"{index}/_bulk{params}",
        data="\n".join(lines) + "\n",
        headers={"Content-Type": "application/x-ndjson"},
    )
    if response.status_code not in {200, 201}:
        raise RuntimeError(
            f"OpenSearch bulk index failed status={response.status_code} "
            f"body={response.text[:1000]}"
        )

    result = response.json()
    if result.get("errors"):
        failures = [
            item
            for item in result.get("items", [])
            if item.get("index", {}).get("status", 200) >= 300
        ]
        raise RuntimeError(
            f"OpenSearch bulk index had {len(failures)} failures; "
            f"sample={failures[:3]}"
        )
    return len(result.get("items", []))


def _normalize_document(item: dict[str, Any], collection_name: str) -> dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    doc_id = (
        item.get("id")
        or item.get("doc_id")
        or metadata.get("id")
        or metadata.get("doc_id")
    )
    text = item.get("text")

    if doc_id is None:
        raise ValueError("missing id/doc_id/metadata.id")
    if not text:
        raise ValueError(f"document {doc_id!r} missing text")

    doc_id = str(doc_id)
    metadata["id"] = doc_id
    metadata["doc_id"] = doc_id
    metadata["collection_name"] = collection_name

    return {
        "doc_id": doc_id,
        "collection_name": collection_name,
        "file_id": item.get("file_id") or metadata.get("file_id"),
        "title": item.get("title") or metadata.get("title", ""),
        "text": text,
        "journal": item.get("journal") or metadata.get("journal", ""),
        "publication_date": item.get("publication_date")
        or metadata.get("publication_date"),
        "metadata": metadata,
        "source": item.get("source") or metadata.get("source") or "jsonl_backfill",
    }


def _iter_jsonl_documents(
    file_path: Path,
    collection_name: str,
    limit: int | None = None,
) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and line_no > limit:
                break

            line = line.strip()
            if not line:
                yield line_no, None, "empty line"
                continue

            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("line is not a JSON object")
                yield line_no, _normalize_document(item, collection_name), None
            except Exception as e:
                yield line_no, None, str(e)


def _flush_batch(
    batch: list[dict[str, Any]],
    *,
    index_name: str | None,
    refresh: bool,
    client: str,
) -> int:
    if not batch:
        return 0

    if client in {"module", "auto"}:
        try:
            bm25 = _load_bm25_module()
            return bm25.index_bm25_documents(
                batch,
                index_name=index_name,
                refresh=refresh,
            )
        except ModuleNotFoundError as e:
            if client == "module":
                raise
            log.warning(
                "[BM25_BACKFILL] module client unavailable (%s); falling back to REST",
                e,
            )

    return _rest_index_bm25_documents(batch, index_name=index_name, refresh=refresh)


def backfill_jsonl(
    file_path: Path,
    collection_name: str,
    batch_size: int,
    index_name: str | None,
    refresh: bool,
    limit: int | None,
    client: str,
) -> dict[str, Any]:
    if client in {"module", "auto"}:
        try:
            bm25 = _load_bm25_module()
            bm25.ensure_bm25_index(index_name)
        except ModuleNotFoundError as e:
            if client == "module":
                raise
            log.warning(
                "[BM25_BACKFILL] module client unavailable (%s); falling back to REST",
                e,
            )
            _rest_ensure_bm25_index(index_name)
    else:
        _rest_ensure_bm25_index(index_name)

    started_at = time.perf_counter()
    batch: list[dict[str, Any]] = []
    parsed_count = 0
    indexed_count = 0
    skipped_count = 0
    error_samples: list[str] = []

    for line_no, doc, error in _iter_jsonl_documents(
        file_path=file_path,
        collection_name=collection_name,
        limit=limit,
    ):
        if error:
            skipped_count += 1
            if len(error_samples) < 5:
                error_samples.append(f"line {line_no}: {error}")
            continue

        parsed_count += 1
        batch.append(doc)

        if len(batch) >= batch_size:
            indexed_count += _flush_batch(
                batch,
                index_name=index_name,
                refresh=refresh,
                client=client,
            )
            elapsed = time.perf_counter() - started_at
            rate = indexed_count / elapsed if elapsed else 0
            print(
                f"[BM25_BACKFILL] indexed={indexed_count} parsed={parsed_count} "
                f"skipped={skipped_count} elapsed={elapsed:.1f}s rate={rate:.1f}/s",
                flush=True,
            )
            batch.clear()

    indexed_count += _flush_batch(
        batch,
        index_name=index_name,
        refresh=refresh,
        client=client,
    )

    elapsed = time.perf_counter() - started_at
    return {
        "file": str(file_path),
        "collection_name": collection_name,
        "parsed": parsed_count,
        "indexed": indexed_count,
        "skipped": skipped_count,
        "elapsed_sec": round(elapsed, 3),
        "rate_per_sec": round(indexed_count / elapsed, 2) if elapsed else 0,
        "client": client,
        "error_samples": error_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill a JSONL file into the OpenSearch BM25 sidecar index."
    )
    parser.add_argument("--file", required=True, help="Path to source JSONL file.")
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Collection name to write into BM25 documents.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of JSONL rows to bulk index per batch.",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="OpenSearch BM25 index name. Defaults to OPENSEARCH_BM25_INDEX.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the BM25 index after each bulk write. Useful for small validation runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max line count for small validation runs.",
    )
    parser.add_argument(
        "--client",
        choices=["auto", "module", "rest"],
        default="auto",
        help="OpenSearch write client. auto uses the BM25 module, then REST fallback.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    summary = backfill_jsonl(
        file_path=file_path,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        index_name=args.index,
        refresh=args.refresh,
        limit=args.limit,
        client=args.client,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
