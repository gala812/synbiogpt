import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/paper_routing.py"
)
SPEC = importlib.util.spec_from_file_location("paper_routing", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
parse_paper_request = MODULE.parse_paper_request


@pytest.mark.parametrize(
    ("message", "intent", "identifier_type", "value"),
    [
        ("推荐与 PMID: 32064678 相关的论文", "related_papers", "pmid", "32064678"),
        (
            "推荐 https://pubmed.ncbi.nlm.nih.gov/29917318/ 的相关论文",
            "related_papers",
            "pmid",
            "29917318",
        ),
        (
            "推荐与《Three-Dimensional Printed Cellulose for Wound Dressing Applications》相关的论文",
            "related_papers",
            "title",
            "Three-Dimensional Printed Cellulose for Wound Dressing Applications",
        ),
        (
            "总结论文《Impact of CRISPR interference on strain development in biotechnology》",
            "paper_summary",
            "title",
            "Impact of CRISPR interference on strain development in biotechnology",
        ),
        (
            "查找标题为 A synthetic biology framework for engineering living materials",
            "paper_search",
            "title",
            "A synthetic biology framework for engineering living materials",
        ),
        (
            "查找论文 A synthetic biology framework for engineering living materials",
            "paper_search",
            "title",
            "A synthetic biology framework for engineering living materials",
        ),
        ("推荐一些CRISPRi代谢工程论文", "paper_search", "topic", "推荐一些CRISPRi代谢工程论文"),
        ("找关于“CRISPRi”的论文", "paper_search", "topic", "找关于“CRISPRi”的论文"),
        ("推荐一些关于“succinate”的论文", "paper_search", "topic", "推荐一些关于“succinate”的论文"),
    ],
)
def test_explicit_paper_routes(message, intent, identifier_type, value):
    request = parse_paper_request(message)
    assert request is not None
    assert request.to_dict() == {
        "intent": intent,
        "identifier_type": identifier_type,
        "identifier_value": value,
    }


@pytest.mark.parametrize(
    "message",
    [
        "CRISPRi如何调控代谢通路？",
        "请概括CRISPRi的代谢调控机制",
        "Give an overview of CRISPR interference mechanisms",
        "请解释论文中常见的统计方法",
        "谢谢，继续分析这个机制",
        "",
        None,
    ],
)
def test_ordinary_questions_do_not_enter_paper_route(message):
    assert parse_paper_request(message) is None
