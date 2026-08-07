from __future__ import annotations

import json
from pathlib import Path

from scripts.medcpt_fulltext.chunking import build_embedding_text, create_chunks
from scripts.medcpt_fulltext.cli import build_parser
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


def test_cli_defaults_to_all_documents_and_500_per_shard() -> None:
    args = build_parser().parse_args(["--input-dir", "input", "--output-dir", "output"])
    assert args.limit == 0
    assert args.documents_per_shard == 500


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
    assert all(
        chunk["token_count"]
        == tokenizer.count(
            build_embedding_text(
                chunk["paper_title"],
                chunk["section"],
                chunk["subsection"],
                chunk["text"],
            )
        )
        for chunk in chunks1
    )
    assert all(chunk["text_token_count"] == tokenizer.count(chunk["text"]) for chunk in chunks1)
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


def test_repairs_page_merged_subheading_and_excludes_panel_artifacts(tmp_path: Path) -> None:
    markdown = f"""# Multiomics Analysis of an Engineered Biological System

## Results

## ITY_DN38401_c0_g1_i1_1, and TRINITY_DN3.3. Gene Ontology Analysis

{_words('ontology', 100)}

(GO:0005198)Figure 5. Cont.
R1-vs-R0
R0-vs-S0
R1-vs-S1

## of anthelmintics in susceptible strain3.4. KEGG Pathway Analysis

R1 -vs- S1

{_words('pathway', 100)}
"""
    path = _write_paper(tmp_path, "PMC0000005", markdown, ["unused.jpg"])
    document = parse_document(path, "PMC0000005")
    assert document.unknown_headings == []
    assert document.section_tree["Results"] == [
        "3.3. Gene Ontology Analysis",
        "3.4. KEGG Pathway Analysis",
    ]
    assert document.excluded_counts["figure_panel_artifact"] == 2


def test_does_not_repair_unverified_number_inside_heading(tmp_path: Path) -> None:
    markdown = f"""# General Structural Repair Safety Study

## Results

## Protein construct version3.3. Analysis workflow

{_words('result', 100)}
"""
    path = _write_paper(tmp_path, "PMC0000006", markdown, ["unused.jpg"])
    document = parse_document(path, "PMC0000006")

    assert document.section_tree["Results"] == [
        "Protein construct version3.3. Analysis workflow"
    ]
    assert "possible_page_merged_heading_unverified" in document.parse_warnings


def test_recovers_h2_title_and_excludes_publisher_headings_and_panel_labels(
    tmp_path: Path,
) -> None:
    markdown = f"""## Viewpoint

## Genetic Engineering and the Clinician

## Open access

## A B S T R A C T

{_words('abstract', 90)}

## Results

(a)
B)
(Continued)
Merged

I

50%

{_words('result', 100)}
"""
    path = _write_paper(tmp_path, "PMC0000007", markdown, ["unused.jpg"])
    document = parse_document(path, "PMC0000007")

    assert document.paper_title == "Genetic Engineering and the Clinician"
    assert document.title_source == "markdown_h2_fallback"
    assert "title_recovered_from_markdown_h2" in document.parse_warnings
    assert document.excluded_counts["publisher_heading"] == 1
    assert document.excluded_counts["figure_panel_artifact"] == 1
    assert document.excluded_counts["isolated_text_fragment"] == 2
    assert "Abstract" in document.section_tree


def test_binds_conservatively_mangled_figure_and_table_captions(tmp_path: Path) -> None:
    markdown = f"""# Reliable Study of Engineered Biological Systems

## Results

{_words('result', 100)}

![](images/figure2.jpg)

e 2. Fermentation response across engineered strains.

able 1. Engineered strains used in fermentation.

![](images/table1.jpg)
"""
    path = _write_paper(
        tmp_path,
        "PMC0000008",
        markdown,
        ["figure2.jpg", "table1.jpg"],
    )
    document = parse_document(path, "PMC0000008")

    assert [(asset.asset_type, asset.label) for asset in document.assets] == [
        ("figure", "Figure 2"),
        ("table", "Table 1"),
    ]
    assert document.assets[0].image_paths == ["images/figure2.jpg"]
    assert document.assets[1].image_paths == ["images/table1.jpg"]


def test_recovers_unstructured_body_and_removes_exact_duplicate_chunks(
    tmp_path: Path,
) -> None:
    repeated = _words("repeated", 210)
    markdown = f"""# Narrative Paper Without Standard Section Headings

Author Name

{_words('opening', 90)}

{repeated}

{repeated}

## References

Reference text that must be excluded.
"""
    path = _write_paper(tmp_path, "PMC0000009", markdown, ["unused.jpg"])
    document = parse_document(path, "PMC0000009")
    chunks, _ = create_chunks(document, RegexTokenCounter(), ChunkingConfig())

    assert "body_recovered_without_standard_sections" in document.parse_warnings
    assert chunks
    assert {chunk["section"] for chunk in chunks} == {"Unassigned"}
    assert document.excluded_counts["duplicate_chunk"] == 1
    assert sum(chunk["text"].count("repeated0") for chunk in chunks) == 1


def test_discovery_deduplicates_and_pipeline_resumes_deterministically(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    inventory_db = tmp_path / "article_inventory.sqlite3"
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
        inventory_db=inventory_db,
        documents_per_shard=1,
    )
    chunk_parts = sorted((output_root / "chunks").glob("part-*.jsonl"))
    chunks_before = [path.read_bytes() for path in chunk_parts]
    stats2 = run_pipeline(
        input_dir=input_root,
        output_dir=output_root,
        limit=2,
        workers=1,
        tokenizer_name="generic:test_v1",
        inventory_db=inventory_db,
        documents_per_shard=1,
    )
    chunks_after = [path.read_bytes() for path in chunk_parts]

    assert stats1["successful_documents"] == 2
    assert stats1["failed_documents"] == 0
    assert stats2["skipped_documents"] == 2
    assert stats1["inventory_reused"] is False
    assert stats1["inventory_build_seconds"] > 0
    assert stats2["inventory_reused"] is True
    assert stats2["inventory_build_seconds"] == 0
    assert stats2["worker_start_method"] == "none"
    assert chunks_before == chunks_after
    expected = {
        "chunks",
        "parents",
        "figures_tables",
        "documents.jsonl",
        "errors.jsonl",
        "statistics.json",
        "manifest.json",
        "inspection_samples.jsonl",
    }
    assert expected.issubset({path.name for path in output_root.iterdir()})
    assert len(chunk_parts) == 2
    rows = []
    for path in chunk_parts:
        with path.open("rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    assert rows == sorted(rows, key=lambda row: (row["pmcid"], row["chunk_index"]))
    required_chunk_fields = {
        "chunk_id", "doc_id", "pmcid", "paper_title", "section", "subsection",
        "section_path", "chunk_type", "chunk_index", "parent_chunk_id", "text",
        "word_count", "token_count", "char_start", "char_end", "previous_chunk_id",
        "text_token_count", "next_chunk_id", "image_paths", "figure_ids",
        "table_ids", "source_file",
        "parse_warnings",
    }
    assert required_chunk_fields.issubset(rows[0])
    stats = json.loads((output_root / "statistics.json").read_text(encoding="utf-8"))
    assert stats["chunks_above_320_words"] == 0
    assert stats["chunks_above_448_tokens"] == 0
    assert "missing_table_text_count" in stats
    assert "excluded_references_blocks" in stats
    assert stats["documents_per_shard"] == 1
    assert stats["shard_count"] == 2
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_layout"] == "document_count_jsonl_shards_v1"
    assert [shard["selected_document_count"] for shard in manifest["shards"]] == [1, 1]
    assert all(shard["failed_document_count"] == 0 for shard in manifest["shards"])


def test_zero_chunk_document_is_reported_as_failure(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_paper(
        input_root,
        "PMC0000030",
        "# Metadata Only Record\n\nAuthor Name\n\n## References\n\nOne reference.\n",
        ["unused.jpg"],
    )

    stats = run_pipeline(
        input_dir=input_root,
        output_dir=output_root,
        limit=1,
        workers=1,
        tokenizer_name="generic:test_v1",
    )

    assert stats["successful_documents"] == 0
    assert stats["failed_documents"] == 1
    assert (output_root / "documents.jsonl").read_text(encoding="utf-8") == ""
    errors = [
        json.loads(line)
        for line in (output_root / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert errors[0]["error_message"] == "No searchable body chunks were produced"
