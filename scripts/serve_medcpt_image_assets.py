from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from medcpt_images.service import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve paper images by stable asset key.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    uvicorn.run(create_app(args.manifest), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
