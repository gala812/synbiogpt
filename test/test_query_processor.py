import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/query_processor.py"
)
SPEC = importlib.util.spec_from_file_location("query_processor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

QueryProcessor = MODULE.QueryProcessor
QueryProcessingError = MODULE.QueryProcessingError


def test_chinese_query_generation_is_validated_and_expanded():
    processor = QueryProcessor()
    original = "大肠杆菌中敲除 ldhA 如何提高丁二酸产量？"
    preparation = processor.prepare(original)

    assert "ldhA" not in preparation.protected_query
    assert "ZXQENTITY0QXZ" in preparation.protected_query

    result = processor.process_model_output(
        original,
        {
            "original_query": preparation.protected_query,
            "semantic_query": (
                "How does ZXQENTITY0QXZ deletion improve succinate production "
                "in Escherichia coli?"
            ),
            "lexical_query": (
                "Escherichia coli ZXQENTITY0QXZ succinate deletion"
            ),
            "exact_terms": ["ZXQENTITY0QXZ"],
        },
    )

    assert result.original_query == original
    assert result.semantic_query == (
        "How does ldhA deletion improve succinate production in Escherichia coli?"
    )
    assert result.exact_terms == ("ldhA",)
    assert "E. coli" in result.lexical_query
    assert "succinic acid" in result.lexical_query
    assert "knockout" in result.lexical_query
    assert "ldhA" in result.bm25_query


def test_english_query_needs_no_model_call():
    query = "How does CRISPRi improve succinate production in E. coli?"
    result = QueryProcessor().process(query)

    assert result.original_query == query
    assert result.semantic_query == query
    assert result.exact_terms == ("CRISPRi",)
    assert "CRISPR interference" in result.lexical_query


def test_scientific_entities_survive_model_generation_exactly():
    entities = [
        "ldhA",
        "ppc",
        "pckA",
        "CRISPRi",
        "dCas9",
        "pET-28a(+)",
        "BBa_J23100",
        "MG1655",
        "OD600",
        "IPTG",
    ]
    processor = QueryProcessor()
    original = "比较 " + "、".join(entities) + " 在工程菌中的作用"
    preparation = processor.prepare(original)
    placeholders = re.findall(r"ZXQENTITY\d+QXZ", preparation.protected_query)
    generated = json.dumps(
        {
            "semantic_query": f"Compare {' '.join(placeholders)} in engineered cells",
            "lexical_query": " ".join(placeholders),
            "exact_terms": placeholders,
        }
    )

    result = processor.process_model_output(original, generated)

    assert list(result.exact_terms) == entities
    for entity in entities:
        assert entity in result.semantic_query
        assert entity in result.bm25_query


def test_chinese_query_cannot_bypass_base_model_generation():
    with pytest.raises(QueryProcessingError, match="must pass through"):
        QueryProcessor().process("如何提高丁二酸产量？")


def test_invalid_model_output_is_rejected():
    with pytest.raises(QueryProcessingError, match="not a valid JSON object"):
        QueryProcessor().process_model_output("如何提高丁二酸产量？", "not json")


def test_json_is_extracted_from_qwen_reasoning_wrapper():
    output = (
        "<think>Use one retrieval query.</think>\n```json\n"
        '{"semantic_query":"How can succinate production be improved?",'
        '"lexical_query":"succinate production yield","exact_terms":[]}\n```'
    )

    result = QueryProcessor().process_model_output("如何提高丁二酸产量？", output)

    assert result.semantic_query == "How can succinate production be improved?"
    assert "succinic acid" in result.lexical_query


def test_model_can_route_conversation_without_fabricating_a_query():
    result = QueryProcessor().process_model_output(
        "你好", {"route": "chat", "semantic_query": "", "lexical_query": ""}
    )

    assert result.retrieval_required is False
    assert result.semantic_query == ""
    assert result.lexical_query == ""


def test_scientific_route_still_requires_a_valid_query():
    result = QueryProcessor().process_model_output(
        "如何提高丁二酸产量？",
        {
            "route": "retrieve",
            "semantic_query": "How can succinate production be improved?",
            "lexical_query": "succinate production yield",
        },
    )

    assert result.retrieval_required is True
    assert result.semantic_query == "How can succinate production be improved?"
