from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.medcpt_images.pipeline import build_image_assets
from scripts.medcpt_images.audit import audit_image_assets
from scripts.medcpt_images.integration import prepare_image_index
from scripts.medcpt_images.recovery import asset_key, recover_document


def _image(path: Path, size: tuple[int, int] = (640, 480)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _paper(tmp_path: Path, body: str, names: tuple[str, ...]) -> Path:
    paper = tmp_path / "PMC123" / "auto"
    for name in names:
        _image(paper / "images" / name)
    markdown = paper / "PMC123.md"
    markdown.write_text(
        "# A synthetic biology paper\n\n## Results\n\n" + body,
        encoding="utf-8",
    )
    return markdown


def test_recovers_extra_panel_and_context_only_image(tmp_path):
    filler = " ".join(["measurement"] * 70)
    markdown = _paper(
        tmp_path,
        (
            "![](images/panel-a.jpg)\n\n"
            f"{filler}\n\n"
            "![](images/panel-b.jpg)\n\n"
            "Figure 1. Engineered pathway activity in two conditions.\n\n"
            "The pathway increased product yield.\n\n"
            "![](images/context.jpg)\n\n"
            "Fermentation performance remained stable across replicates.\n"
        ),
        ("panel-a.jpg", "panel-b.jpg", "context.jpg"),
    )

    result = recover_document(markdown, "PMC123")
    by_path = {item["relative_path"]: item for item in result["bindings"]}

    assert by_path["images/panel-b.jpg"]["status"] == "existing_bound"
    assert by_path["images/panel-a.jpg"]["status"] == "recovered"
    assert by_path["images/panel-a.jpg"]["asset_id"] == by_path["images/panel-b.jpg"]["asset_id"]
    assert by_path["images/context.jpg"]["status"] == "recovered"
    assert by_path["images/context.jpg"]["asset_type"] == "image"


def test_recovers_paper_with_no_original_binding_and_excludes_tiny_image(tmp_path):
    separators = "\n\n".join(chr(ord("A") + index % 8) for index in range(18))
    markdown = _paper(
        tmp_path,
        (
            "![](images/figure.jpg)\n\n"
            f"{separators}\n\n"
            "Fig. 2: A recovered figure caption with experimental evidence.\n\n"
            "## Discussion\n\n"
            "A separate discussion paragraph.\n\n"
            "![](images/tiny.jpg)\n"
        ),
        ("figure.jpg", "tiny.jpg"),
    )
    _image(markdown.parent / "images" / "tiny.jpg", (30, 20))

    result = recover_document(markdown, "PMC123")
    by_path = {item["relative_path"]: item for item in result["bindings"]}

    assert by_path["images/figure.jpg"]["status"] == "recovered"
    assert by_path["images/figure.jpg"]["is_new_asset"] is True
    assert by_path["images/tiny.jpg"]["status"] == "excluded"
    assert by_path["images/tiny.jpg"]["reason"] == "excluded_tiny"


def test_pipeline_builds_stable_sqlite_manifest(tmp_path):
    markdown = _paper(
        tmp_path,
        "![](images/result.jpg)\n\nFigure 1. Product yield.\n",
        ("result.jpg",),
    )
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        json.dumps({"pmcid": "PMC123", "source_markdown": str(markdown)}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    statistics = build_image_assets(
        documents_jsonl=documents,
        output_dir=output,
        source_root=tmp_path,
        workers=1,
    )

    assert statistics["successful_documents"] == 1
    assert statistics["status_existing_bound"] == 1
    assert (output / "image_assets.sqlite3").is_file()
    binding = json.loads((output / "bindings" / "part-00000.jsonl").read_text(encoding="utf-8"))
    assert binding["asset_key"] == asset_key("PMC123", "images/result.jpg")
    report = audit_image_assets(output, verify_files=True)
    assert report["passed"] is True
    assert report["binding_rows"] == 1


def test_prepare_image_index_creates_incremental_chunks_and_payload_patches(tmp_path):
    access = tmp_path / "access" / "recovered_assets"
    chunks = tmp_path / "chunks"
    output = tmp_path / "integration"
    access.mkdir(parents=True)
    chunks.mkdir()
    recovered = [
        {
            "asset_id": "PMC123_figure_recovered_a",
            "pmcid": "PMC123",
            "asset_type": "figure",
            "label": "Figure 9",
            "caption": "Figure 9. A newly recovered pathway figure.",
            "confidence": "high",
            "reason": "caption_recovery",
            "is_new_asset": True,
            "image_paths": ["images/new.jpg"],
            "asset_keys": [asset_key("PMC123", "images/new.jpg")],
            "paper_title": "A synthetic biology paper",
            "section": "Results",
        },
        {
            "asset_id": "PMC123_figure_0001",
            "pmcid": "PMC123",
            "asset_type": "figure",
            "label": "Figure 1",
            "caption": "Figure 1. Existing figure.",
            "confidence": "high",
            "reason": "caption_recovery",
            "is_new_asset": False,
            "image_paths": ["images/panel-b.jpg"],
            "asset_keys": [asset_key("PMC123", "images/panel-b.jpg")],
            "paper_title": "A synthetic biology paper",
            "section": "Results",
        },
    ]
    (access / "part-00000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in recovered), encoding="utf-8"
    )
    existing_chunk = {
        "chunk_id": "PMC123_results_main_0001",
        "pmcid": "PMC123",
        "image_paths": ["images/panel-a.jpg"],
        "figure_ids": ["PMC123_figure_0001"],
        "table_ids": [],
    }
    (chunks / "part-00000.jsonl").write_text(
        json.dumps(existing_chunk) + "\n", encoding="utf-8"
    )

    statistics = prepare_image_index(
        image_access_dir=tmp_path / "access",
        chunks_dir=chunks,
        output_dir=output,
        tokenizer_name="generic:test",
    )

    assert statistics["new_assets"] == 1
    assert statistics["new_chunks"] == 1
    assert statistics["payload_patches"] == 1
    patch = json.loads(
        (output / "payload_patches" / "part-00000.jsonl").read_text(encoding="utf-8")
    )
    assert patch["image_paths"] == ["images/panel-a.jpg", "images/panel-b.jpg"]
    assert len(patch["asset_keys"]) == 2
