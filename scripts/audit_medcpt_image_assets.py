from __future__ import annotations

import argparse
import json
from pathlib import Path

from medcpt_images.audit import audit_image_assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit stable paper-image bindings.")
    parser.add_argument("--image-access-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    report = audit_image_assets(
        args.image_access_dir,
        verify_files=args.verify_files,
    )
    value = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(args.output)
    print(value, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
