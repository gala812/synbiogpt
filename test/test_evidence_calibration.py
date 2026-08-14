import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR = load(
    "evidence_calibration",
    ROOT / "backend/open_webui/apps/retrieval/synbio/evidence_calibration.py",
)
CALIBRATOR = load(
    "calibrate_evidence_gate",
    ROOT / "scripts/calibrate_evidence_gate.py",
)


def test_raw_logit_collection_is_structured_and_label_ready(tmp_path):
    destination = tmp_path / "samples.jsonl"
    query = SimpleNamespace(
        original_query="为什么？",
        semantic_query="Why does CRISPRi improve succinate production?",
        lexical_query="CRISPRi succinate production mechanism",
        exact_terms=("CRISPRi",),
    )
    documents = [
        SimpleNamespace(
            page_content="direct evidence text",
            metadata={
                "chunk_id": "A",
                "title": "Study A",
                "cross_encoder_score": 2.5,
                "rerank_rank": 1,
            },
        ),
        SimpleNamespace(
            page_content="unrelated text",
            metadata={
                "chunk_id": "B",
                "cross_encoder_score": -1.0,
                "rerank_rank": 2,
            },
        ),
    ]

    count = COLLECTOR.collect_calibration_examples(
        path=str(destination),
        sample_rate=1.0,
        max_text_chars=8,
        query=query,
        documents=documents,
        collection_name="fulltext_medcpt_ip_v1",
        cross_encoder_model="MedCPT/Cross-Encoder",
        cross_encoder_max_tokens=512,
    )

    rows = [json.loads(line) for line in destination.read_text("utf-8").splitlines()]
    assert count == 2
    assert len({row["query_id"] for row in rows}) == 1
    assert rows[0]["semantic_query"].startswith("Why does CRISPRi")
    assert rows[0]["raw_logit"] == 2.5
    assert rows[0]["document_text"] == "direct e"
    assert rows[0]["document_text_truncated"] is True
    assert rows[0]["relevance_label"] is None


def test_collection_is_disabled_without_both_path_and_sampling(tmp_path):
    document = SimpleNamespace(
        page_content="text", metadata={"cross_encoder_score": 1.0}
    )
    assert (
        COLLECTOR.collect_calibration_examples(
            path="",
            sample_rate=1.0,
            max_text_chars=100,
            query=SimpleNamespace(),
            documents=[document],
            collection_name="collection",
            cross_encoder_model="model",
            cross_encoder_max_tokens=512,
        )
        == 0
    )
    destination = tmp_path / "disabled.jsonl"
    assert (
        COLLECTOR.collect_calibration_examples(
            path=str(destination),
            sample_rate=0.0,
            max_text_chars=100,
            query=SimpleNamespace(),
            documents=[document],
            collection_name="collection",
            cross_encoder_model="model",
            cross_encoder_max_tokens=512,
        )
        == 0
    )
    assert not destination.exists()


def test_label_export_and_threshold_calibration(tmp_path):
    jsonl_path = tmp_path / "samples.jsonl"
    rows = [
        {
            "query_id": "q1",
            "semantic_query": "query one",
            "collection_name": "collection",
            "cross_encoder_model": "model",
            "cross_encoder_max_tokens": 512,
            "chunk_id": "p1",
            "raw_logit": 3.0,
            "document_text": "positive one",
            "relevance_label": 1,
        },
        {
            "query_id": "q2",
            "semantic_query": "query two",
            "collection_name": "collection",
            "cross_encoder_model": "model",
            "cross_encoder_max_tokens": 512,
            "chunk_id": "p2",
            "raw_logit": 2.0,
            "document_text": "positive two",
            "relevance_label": 1,
        },
        {
            "query_id": "q3",
            "semantic_query": "query three",
            "collection_name": "collection",
            "cross_encoder_model": "model",
            "cross_encoder_max_tokens": 512,
            "chunk_id": "n1",
            "raw_logit": 1.0,
            "document_text": "negative one",
            "relevance_label": 0,
        },
        {
            "query_id": "q4",
            "semantic_query": "query four",
            "collection_name": "collection",
            "cross_encoder_model": "model",
            "cross_encoder_max_tokens": 512,
            "chunk_id": "n2",
            "raw_logit": -1.0,
            "document_text": "negative two",
            "relevance_label": 0,
        },
    ]
    jsonl_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    csv_path = tmp_path / "labels.csv"

    assert CALIBRATOR.export_label_csv(jsonl_path, csv_path) == 4
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert exported[0]["document_text"] == "positive one"

    labeled, unlabeled_count = CALIBRATOR.load_labeled_records(csv_path)
    report = CALIBRATOR.build_calibration_report(
        labeled,
        unlabeled_count=unlabeled_count,
        target_precision=1.0,
        validation_fraction=0.0,
        seed=42,
        min_labeled_pairs=4,
        min_positive_pairs=2,
        min_negative_pairs=2,
    )

    assert report["candidate_threshold"] == 2.0
    assert report["calibration_metrics"]["precision"] == 1.0
    assert report["calibration_metrics"]["recall"] == 1.0
    assert report["recommendation_ready"] is True
    assert report["suggested_environment"] == (
        "MEDCPT_EVIDENCE_GATE_MIN_SCORE=2"
    )


def test_query_level_split_prevents_candidate_leakage():
    records = [
        {
            "query_id": f"q-{query}-{candidate}",
            "_split_group": query,
            "raw_logit": float(candidate),
            "relevance_label": candidate % 2,
        }
        for query in ("a", "b", "c", "d")
        for candidate in (1, 2)
    ]

    calibration, validation = CALIBRATOR.split_by_query(
        records, validation_fraction=0.5, seed=42
    )

    calibration_groups = {row["_split_group"] for row in calibration}
    validation_groups = {row["_split_group"] for row in validation}
    assert calibration_groups
    assert validation_groups
    assert calibration_groups.isdisjoint(validation_groups)
