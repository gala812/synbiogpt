from __future__ import annotations

import argparse
import json
from pathlib import Path

from medcpt_images.integration import apply_payload_patches


def main() -> int:
    parser = argparse.ArgumentParser(description="Add stable image keys and recovered panels to existing indexes.")
    parser.add_argument("--patches-dir", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--opensearch-url", required=True)
    parser.add_argument("--collection", default="fulltext_medcpt_v1")
    parser.add_argument("--bm25-index", default="fulltext_bm25_v1")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = apply_payload_patches(
        patches_dir=args.patches_dir,
        state_file=args.state_file,
        qdrant_url=args.qdrant_url,
        opensearch_url=args.opensearch_url,
        collection_name=args.collection,
        index_name=args.bm25_index,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
