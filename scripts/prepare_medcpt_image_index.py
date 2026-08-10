from __future__ import annotations

import argparse
import json
from pathlib import Path

from medcpt_images.integration import prepare_image_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare recovered image chunks and stable-key payload patches.")
    parser.add_argument("--image-access-dir", required=True, type=Path)
    parser.add_argument("--chunks-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--records-per-shard", type=int, default=10_000)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = prepare_image_index(
        image_access_dir=args.image_access_dir,
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        local_files_only=args.local_files_only,
        records_per_shard=args.records_per_shard,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
