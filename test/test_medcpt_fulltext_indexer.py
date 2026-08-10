from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.medcpt_indexing.models import IndexDocument, IndexingConfig
from scripts.medcpt_indexing.pipeline import run_indexing, validate_inputs
from scripts.medcpt_indexing.schema import (
    embedding_text,
    load_pmid_mapping,
    make_index_document,
)
from scripts.medcpt_indexing.sinks import _with_retries


def _mapping_database(path: Path, rows=(("1", "PMC1"), ("2", "PMC2"))) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE paper_id_mapping ("
        "pmid TEXT PRIMARY KEY, pmcid TEXT UNIQUE, "
        "mapping_source TEXT, ncbi_snapshot_sha256 TEXT)"
    )
    snapshot = "a" * 64
    connection.executemany(
        "INSERT INTO paper_id_mapping VALUES (?, ?, ?, ?)",
        [(*row, "NCBI PMC-ids.csv.gz", snapshot) for row in rows],
    )
    connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT)")
    connection.executemany(
        "INSERT INTO index_metadata VALUES (?, ?)",
        [
            ("mapping_source", "NCBI PMC-ids.csv.gz"),
            ("ncbi_snapshot_sha256", snapshot),
            ("rows_without_official_pmid", "0"),
        ],
    )
    connection.commit()
    connection.close()
    return path


def _chunk(chunk_id: str, pmcid: str, *, chunk_type: str = "paragraph") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": pmcid,
        "pmcid": pmcid,
        "paper_title": "A synthetic biology paper",
        "section": "Results",
        "subsection": "Pathway yield",
        "section_path": ["Results", "Pathway yield"],
        "chunk_type": chunk_type,
        "chunk_index": 1,
        "parent_chunk_id": f"{pmcid}_parent_0001",
        "text": "The engineered pathway increased product yield.",
        "word_count": 7,
        "text_token_count": 10,
        "token_count": 24,
        "tokenizer_name": "ncbi/MedCPT-Article-Encoder",
        "image_paths": ["images/result.jpg"] if chunk_type == "figure_caption" else [],
        "figure_ids": (
            [f"{pmcid}_figure_0001"] if chunk_type == "figure_caption" else []
        ),
        "table_ids": [],
    }


def _write_shard(path: Path, chunks: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks),
        encoding="utf-8",
    )


class FakeEncoder:
    dimension = 3
    model_name = "fake-medcpt"

    def encode(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class FakeVectorSink:
    def __init__(self):
        self.points: dict[str, tuple[IndexDocument, list[float]]] = {}
        self.dimension = None

    def ensure_ready(self, dimension):
        self.dimension = dimension

    def write(self, documents, vectors):
        for document, vector in zip(documents, vectors, strict=True):
            self.points[document.point_id] = (document, list(vector))
        return len(documents)

    def count_shard(self, source_shard):
        return sum(
            document.metadata["source_shard"] == source_shard
            for document, _ in self.points.values()
        )


class FakeKeywordSink:
    def __init__(self):
        self.documents: dict[str, IndexDocument] = {}
        self.ready = False

    def ensure_ready(self):
        self.ready = True

    def write(self, documents):
        for document in documents:
            self.documents[document.chunk_id] = document
        return len(documents)

    def count_shard(self, source_shard):
        return sum(
            document.metadata["source_shard"] == source_shard
            for document in self.documents.values()
        )


def test_index_document_uses_official_pmid_and_preserves_figure_metadata():
    chunk = _chunk("PMC1_results_0001", "PMC1", chunk_type="figure_caption")
    document = make_index_document(
        chunk,
        pmid_by_pmcid={"PMC1": "1"},
        source_shard="part-00000.jsonl",
        max_tokens=448,
    )

    assert document.metadata["pmid"] == "1"
    assert document.metadata["figure_ids"] == ["PMC1_figure_0001"]
    assert document.metadata["image_paths"] == ["images/result.jpg"]
    assert document.embedding_text == (
        "Title: A synthetic biology paper\n"
        "Section: Results > Pathway yield\n"
        "Text: The engineered pathway increased product yield."
    )
    assert (
        document.point_id
        == make_index_document(
            chunk,
            pmid_by_pmcid={"PMC1": "1"},
            source_shard="part-00000.jsonl",
            max_tokens=448,
        ).point_id
    )


def test_index_document_rejects_missing_mapping_and_oversized_input():
    chunk = _chunk("PMC1_results_0001", "PMC1")
    with pytest.raises(ValueError, match="no official PMID"):
        make_index_document(
            chunk,
            pmid_by_pmcid={},
            source_shard="part-00000.jsonl",
            max_tokens=448,
        )

    chunk["token_count"] = 449
    with pytest.raises(ValueError, match="maximum is 448"):
        make_index_document(
            chunk,
            pmid_by_pmcid={"PMC1": "1"},
            source_shard="part-00000.jsonl",
            max_tokens=448,
        )


def test_pipeline_dual_writes_and_continues_after_pilot(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_shard(
        chunks_dir / "part-00000.jsonl",
        [_chunk("PMC1_results_0001", "PMC1"), _chunk("PMC1_results_0002", "PMC1")],
    )
    _write_shard(
        chunks_dir / "part-00001.jsonl",
        [_chunk("PMC2_results_0001", "PMC2", chunk_type="table_caption")],
    )
    mapping_db = _mapping_database(tmp_path / "mapping.sqlite3")
    state_file = tmp_path / "index_manifest.json"
    encoder = FakeEncoder()
    vector_sink = FakeVectorSink()
    keyword_sink = FakeKeywordSink()

    pilot = IndexingConfig(
        chunks_dir=chunks_dir,
        mapping_db=mapping_db,
        state_file=state_file,
        encode_batch_size=1,
        upload_batch_size=2,
        limit_shards=1,
    )
    manifest = run_indexing(pilot, encoder, vector_sink, keyword_sink)
    assert manifest["totals"]["chunks"] == 2
    assert len(vector_sink.points) == len(keyword_sink.documents) == 2

    full = IndexingConfig(
        chunks_dir=chunks_dir,
        mapping_db=mapping_db,
        state_file=state_file,
        encode_batch_size=1,
        upload_batch_size=2,
    )
    manifest = run_indexing(full, encoder, vector_sink, keyword_sink)
    assert manifest["totals"]["chunks"] == 3
    assert len(vector_sink.points) == len(keyword_sink.documents) == 3
    assert set(manifest["completed_shards"]) == {
        "part-00000.jsonl",
        "part-00001.jsonl",
    }


def test_validate_inputs_reports_chunk_types(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_shard(
        chunks_dir / "part-00000.jsonl",
        [
            _chunk("PMC1_results_0001", "PMC1"),
            _chunk("PMC1_figure_0001", "PMC1", chunk_type="figure_caption"),
        ],
    )
    config = IndexingConfig(
        chunks_dir=chunks_dir,
        mapping_db=_mapping_database(tmp_path / "mapping.sqlite3"),
        state_file=tmp_path / "state.json",
    )

    report = validate_inputs(config)

    assert report["shards"] == 1
    assert report["chunks"] == 2
    assert report["documents"] == 1
    assert report["chunk_types"] == {"figure_caption": 1, "paragraph": 1}


def test_load_mapping_rejects_unverified_database(tmp_path):
    database = tmp_path / "unverified.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE mapping (pmid TEXT, pmcid TEXT)")
    connection.execute("INSERT INTO mapping VALUES ('123', 'PMC456')")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="Official mapping database"):
        load_pmid_mapping(database)


def test_remote_operation_retries_transient_failure(monkeypatch):
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary disconnect")
        return "ok"

    monkeypatch.setattr("scripts.medcpt_indexing.sinks.time.sleep", lambda _: None)

    assert _with_retries(operation, label="test") == "ok"
    assert attempts == 3
