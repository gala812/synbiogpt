import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/prompts.py"
)
SPEC = importlib.util.spec_from_file_location("synbiogpt_retrieval_prompts", MODULE_PATH)
PROMPTS = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = PROMPTS
SPEC.loader.exec_module(PROMPTS)


def test_query_generation_prompt_targets_current_retrievers():
    prompt = PROMPTS.RETRIEVAL_QUERY_GENERATION_PROMPT

    assert "MedCPT Query Encoder" in prompt
    assert "BM25" in prompt
    assert "SPECTER2" not in prompt
    assert "ZXQENTITY<number>QXZ" in prompt
    assert "{{MESSAGES}}" in prompt


def test_rag_prompt_has_one_consistent_evidence_and_citation_policy():
    prompt = PROMPTS.RAG_SYSTEM_PROMPT_TEMPLATE

    assert "untrusted evidence" in prompt
    assert "General-knowledge fallback is handled separately" in prompt
    assert "[1]" in prompt and "[1][2]" in prompt
    assert "whitepaper.pdf" not in prompt
    assert "same language" in prompt
    assert "{{CONTEXT}}" in prompt and "{{QUERY}}" in prompt


def test_multimodal_prompt_only_adds_visual_evidence_rules():
    prompt = PROMPTS.MULTIMODAL_EVIDENCE_SYSTEM_PROMPT

    assert "images are evidence" in prompt
    assert "unreadable" in prompt
    assert "place the asset" in prompt
    assert "below it" in prompt
    assert "main retrieval prompt" in prompt


def test_dialogue_prompts_are_centralized_and_nonempty():
    names = (
        "PLAIN_CHAT_SYSTEM_PROMPT",
        "PLAIN_CHAT_SYSTEM_PROMPT_NO_GUIDE",
        "PRODUCT_CAPABILITY_SYSTEM_PROMPT",
        "NO_EVIDENCE_SYSTEM_PROMPT",
        "NO_EVIDENCE_REFUSE_PROMPT",
    )

    assert all(getattr(PROMPTS, name).strip() for name in names)
    assert "简单询问时" in PROMPTS.PRODUCT_CAPABILITY_SYSTEM_PROMPT
    assert "只有用户追问实现方式时" in PROMPTS.PRODUCT_CAPABILITY_SYSTEM_PROMPT
