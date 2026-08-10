from __future__ import annotations

import argparse
import json
from pathlib import Path

from medcpt_images.pipeline import build_image_assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover MinerU images and build a stable access manifest.")
    parser.add_argument("--documents-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--documents-per-shard", type=int, default=500)
    parser.add_argument("--path-prefix", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args()
    prefix_maps = []
    for value in args.path_prefix:
        if "=" not in value:
            parser.error("--path-prefix must use OLD=NEW")
        prefix_maps.append(tuple(value.split("=", 1)))
    statistics = build_image_assets(
        documents_jsonl=args.documents_jsonl,
        output_dir=args.output_dir,
        source_root=args.source_root,
        limit=args.limit,
        workers=args.workers,
        documents_per_shard=args.documents_per_shard,
        prefix_maps=tuple(prefix_maps),
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not statistics.get("failed_documents") else 2


if __name__ == "__main__":
    raise SystemExit(main())
