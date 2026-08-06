"""Standard-library smoke test, including multiprocessing and resume behavior."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.medcpt_fulltext.pipeline import run_pipeline


def _paper(root: Path, pmcid: str, section: str, image: str) -> None:
    auto = root / pmcid / "auto"
    (auto / "images").mkdir(parents=True)
    (auto / "images" / image).write_bytes(b"jpeg")
    sentences = " ".join(
        f"Engineered pathway variant {index} increased metabolite production in controlled cultures."
        for index in range(35)
    )
    (auto / f"{pmcid}.md").write_text(
        f"# Deterministic Pilot Paper {pmcid}\n\n"
        f"## {section}\n\n{sentences}\n\n"
        f"![](images/{image})\n\nFigure 1. Engineered pathway overview.\n",
        encoding="utf-8",
    )


def main() -> int:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        input_dir = root / "input"
        output_dir = root / "output"
        _paper(input_dir, "PMC0000101", "Results", "figure1.jpg")
        _paper(input_dir, "PMC0000102", "Methods", "figure2.jpg")
        first = run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            limit=2,
            workers=2,
            tokenizer_name="generic:smoke_v1",
        )
        before = (output_dir / "chunks.jsonl").read_bytes()
        second = run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            limit=2,
            workers=2,
            tokenizer_name="generic:smoke_v1",
        )
        assert first["successful_documents"] == 2
        assert first["failed_documents"] == 0
        assert second["skipped_documents"] == 2
        assert before == (output_dir / "chunks.jsonl").read_bytes()
        stats = json.loads((output_dir / "statistics.json").read_text(encoding="utf-8"))
        assert stats["chunks_above_320_words"] == 0
        assert stats["chunks_above_448_tokens"] == 0
    print("PASS multiprocessing/resume smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
