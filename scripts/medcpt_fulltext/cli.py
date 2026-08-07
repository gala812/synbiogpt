from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .models import ChunkingConfig
from .pipeline import run_pipeline
from .tokenization import MEDCPT_TOKENIZER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically chunk MinerU Markdown papers for MedCPT/BM25 retrieval."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--inventory-db",
        type=Path,
        help=(
            "Reusable SQLite Markdown inventory. Defaults to "
            "OUTPUT_DIR/../article_inventory.sqlite3."
        ),
    )
    parser.add_argument(
        "--refresh-inventory",
        action="store_true",
        help="Rescan the input tree and atomically rebuild the reusable inventory.",
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--tokenizer", default=MEDCPT_TOKENIZER)
    parser.add_argument(
        "--require-medcpt-tokenizer",
        action="store_true",
        help="Fail instead of using tiktoken/generic fallback when MedCPT cannot load.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--include-without-images",
        action="store_true",
        help="Include Markdown papers whose sibling images directory is empty/missing.",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess successfully committed papers.")
    parser.add_argument("--inspection-seed", type=int, default=20260806)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inventory_db = args.inventory_db or args.output_dir.parent / "article_inventory.sqlite3"
    stats = run_pipeline(
        input_dir=args.input_dir,
        metadata_jsonl=args.metadata_jsonl,
        output_dir=args.output_dir,
        limit=args.limit,
        workers=max(1, args.workers),
        tokenizer_name=args.tokenizer,
        allow_tokenizer_fallback=not args.require_medcpt_tokenizer,
        local_files_only=args.local_files_only,
        require_images=not args.include_without_images,
        force=args.force,
        inventory_db=inventory_db,
        refresh_inventory=args.refresh_inventory,
        inspection_seed=args.inspection_seed,
        config=ChunkingConfig(),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if stats["failed_documents"] == 0 else 2
