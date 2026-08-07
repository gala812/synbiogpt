from __future__ import annotations

import json
from pathlib import Path

from scripts.medcpt_fulltext.chunking import create_chunks
from scripts.medcpt_fulltext.markdown_parser import parse_document
from scripts.medcpt_fulltext.models import ChunkingConfig
from scripts.medcpt_fulltext.pipeline import discover_documents, run_pipeline
from scripts.medcpt_fulltext.tokenization import RegexTokenCounter


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count)) + "."


def _write_paper(root: Path, pmcid: str, markdown: str, image_names: list[str]) -> Path:
    auto = root / "task" / pmcid / "auto"
    images = auto / "images"
    images.mkdir(parents=True)
    for name in image_names:
        (images / name).write_bytes(b"jpeg")
    path = auto / f"{pmcid}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def test_parser_repairs_sections_binds_assets_and_excludes_non_body(tmp_path: Path) -> None:
    markdown = f"""Published in final edited form as: Example Journal.

# Engineered Microbial Consortium Improves Bioproduct Yield

Alice Author, Bob Author

Department of Synthetic Biology, Example University

## Abstract

{_words('abstract', 90)}

## 1. Introduction

{_words('intro', 100)} Figure 1 summarizes the construct design.

![](images/figure1.jpg)

Figure 1. Genetic construct and pathway organization.

## 2. Materials and Methods

## 2.1. Bioinformatics analysis

{_words('method', 110)} Table 1 lists all strains.

Table 1. Engineered strains used in this study.
![](images/table1.jpg)
1 Values are reported for three biological replicates.

{_words('methodb', 100)}

## References

Reference that must not be indexed.

Submit your next manuscript and enjoy convenient online submission.
"""
    path = _write_paper(tmp_path, "PMC0000001", markdown, ["figure1.jpg", "table1.jpg"])
    document = parse_document(
        path,
        "PMC0000001",
        {"title": "MATERIALS AND METHODS", "source_file": "PMC0000001.pdf"},
    )

    assert document.paper_title == "Engineered Microbial Consortium Improves Bioproduct Yield"
    assert "metadata_title_anomaly" in document.parse_warnings
    assert document.excluded_counts["front_matter"] >= 2
    assert document.excluded_counts["references"] >= 1
    assert document.section_tree["Methods"] == ["2.1. Bioinformatics analysis"]
    assert len(document.assets) == 2
    figure = next(asset for asset in document.assets if asset.asset_type == "figure")
    table = next(asset for asset in document.assets if asset.asset_type == "table")
    assert figure.image_paths == ["images/figure1.jpg"]
    assert "Figure 1" in figure.label
    assert "Figure 1" in figure.context_before
    assert table.image_paths == ["images/table1.jpg"]
    assert table.table_text_missing is True
    assert table.notes.startswith("1 Values")


def test_chunking_enforces_limits_parents_and_stable_ids(tmp_path: Path) -> None:
    sentences = " ".join(
        f"Engineered strain number {index} increased pathway productivity under controlled conditions."
        for index in range(90)
    )
    markdown = f"""# Reproducible Synthetic Biology Study of Engineered Strains

## Results

{sentences}
"""
    path = _write_paper(tmp_path, "PMC0000002", markdown, ["unused.jpg"])
    document = parse_document(path, "PMC0000002")
    tokenizer = RegexTokenCounter()
    chunks1, parents1 = create_chunks(document, tokenizer, ChunkingConfig())
    chunks2, parents2 = create_chunks(document, tokenizer, ChunkingConfig())

    assert [chunk["chunk_id"] for chunk in chunks1] == [chunk["chunk_id"] for chunk in chunks2]
    assert parents1 == parents2
    assert len(chunks1) >= 3
    assert all(chunk["word_count"] <= 320 for chunk in chunks1)
    assert all(chunk["token_count"] <= 448 for chunk in chunks1)
    assert all(chunk["parent_chunk_id"] for chunk in chunks1)
    assert chunks1[0]["previous_chunk_id"] is None
    assert chunks1[-1]["next_chunk_id"] is None
    assert all(
        chunks1[index]["next_chunk_id"] == chunks1[index + 1]["chunk_id"]
        for index in range(len(chunks1) - 1)
    )


def test_post_references_figures_and_equation_warning_are_preserved(tmp_path: Path) -> None:
    markdown = f"""# Synthetic Circuit Dynamics in Engineered Cells

## Results

{_words('result', 100)} Figure 1 summarizes the circuit response.

(7)

## References

Reference entry that is excluded from text chunks.

![](images/panel_a.jpg)
A
![](images/panel_b.jpg)
Figure 1. Multi-panel response of the engineered circuit.

![](images/panel_c.jpg)
Figure 1. Cont.

Funding: This boilerplate funding statement must not be indexed.
"""
    path = _write_paper(
        tmp_path,
        "PMC0000003",
        markdown,
        ["panel_a.jpg", "panel_b.jpg", "panel_c.jpg"],
    )
    document = parse_document(path, "PMC0000003")
    assert len(document.assets) == 1
    asset = document.assets[0]
    assert asset.image_paths == ["images/panel_a.jpg", "images/panel_b.jpg", "images/panel_c.jpg"]
    assert asset.section == "Results"
    assert asset.context_before.startswith("result0")
    assert "continued_asset_merged" in asset.parse_warnings
    assert document.excluded_counts["inline_non_body"] == 1
    assert any(
        "equation_number_without_body" in block.warnings
        for block in document.blocks
        if block.kind == "equation"
    )


def test_forced_long_sentence_uses_word_overlap_without_exceeding_limits(tmp_path: Path) -> None:
    long_sentence = " ".join(f"sequenceword{index}" for index in range(620)) + "."
    path = _write_paper(
        tmp_path,
        "PMC0000004",
        f"# Long Synthetic Sequence Description in Engineered Cells\n\n## Results\n\n{long_sentence}\n",
        ["unused.jpg"],
    )
    chunks, _ = create_chunks(
        parse_document(path, "PMC0000004"),
        RegexTokenCounter(),
        ChunkingConfig(),
    )
    assert len(chunks) >= 3
    assert all(chunk["word_count"] <= 320 for chunk in chunks)
    assert all(chunk["token_count"] <= 448 for chunk in chunks)
    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    assert first_words[-40:] == second_words[:40]


def test_discovery_deduplicates_and_pipeline_resumes_deterministically(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    body = _words("result", 210)
    first = _write_paper(
        input_root / "a",
        "PMC0000010",
        f"# Reliable Paper Title for Pilot Ten\n\n## Results\n\n{body}\n\n![](images/f1.jpg)\n\nFigure 1. Result overview.\n",
        ["f1.jpg"],
    )
    # Duplicate parse with less content and no valid Markdown image reference.
    _write_paper(
        input_root / "b",
        "PMC0000010",
        "# Reliable Paper Title for Pilot Ten\n\n## Results\n\nshort text\n",
        ["unused.jpg"],
    )
    _write_paper(
        input_root / "c",
        "PMC0000020",
        f"# Reliable Paper Title for Pilot Twenty\n\n## Methods\n\n{body}\n\nTable 1. Values.\n![](images/t1.jpg)\n",
        ["t1.jpg"],
    )

    candidates = discover_documents(input_root, limit=2, require_images=True)
    assert [candidate.pmcid for candidate in candidates] == ["PMC0000010", "PMC0000020"]
    assert candidates[0].markdown_path == first
    assert len(candidates[0].duplicate_paths) == 1

    stats1 = run_pipeline(
        input_dir=input_root,
        output_dir=output_root,
        limit=2,
        workers=1,
        tokenizer_name="generic:test_v1",
    )
    chunks_before = (output_root / "chunks.jsonl").read_bytes()
    stats2 = run_pipeline(
        input_dir=input_root,
        output_dir=output_root,
        limit=2,
        workers=1,
        tokenizer_name="generic:test_v1",
    )
    chunks_after = (output_root / "chunks.jsonl").read_bytes()

    assert stats1["successful_documents"] == 2
    assert stats1["failed_documents"] == 0
    assert stats2["skipped_documents"] == 2
    assert chunks_before == chunks_after
    expected = {
        "chunks.jsonl",
        "parents.jsonl",
        "figures_tables.jsonl",
        "documents.jsonl",
        "errors.jsonl",
        "statistics.json",
        "inspection_samples.jsonl",
    }
    assert expected.issubset({path.name for path in output_root.iterdir()})
    rows = [json.loads(line) for line in (output_root / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == sorted(rows, key=lambda row: (row["pmcid"], row["chunk_index"]))
    required_chunk_fields = {
        "chunk_id", "doc_id", "pmcid", "paper_title", "section", "subsection",
        "section_path", "chunk_type", "chunk_index", "parent_chunk_id", "text",
        "word_count", "token_count", "char_start", "char_end", "previous_chunk_id",
        "next_chunk_id", "image_paths", "figure_ids", "table_ids", "source_file",
        "parse_warnings",
    }
    assert required_chunk_fields.issubset(rows[0])
    stats = json.loads((output_root / "statistics.json").read_text(encoding="utf-8"))
    assert stats["chunks_above_320_words"] == 0
    assert stats["chunks_above_448_tokens"] == 0
    assert "missing_table_text_count" in stats
    assert "excluded_references_blocks" in stats
