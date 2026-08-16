import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1] / "backend/open_webui/apps/retrieval/synbio"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROUTING = load("synbio_dialogue_routing", ROOT / "routing.py")
DIALOGUE = load("synbio_dialogue_prompts", ROOT / "dialogue.py")


@pytest.mark.parametrize(
    "message",
    [
        "你好",
        "您好",
        "你好呀",
        "您好呀",
        "哈喽",
        "哈啰",
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "谢谢",
        "谢谢啦",
        "多谢",
        "好的",
        "嗯嗯",
        "ok",
        "okay",
        "在吗",
        "在不在",
        "早上好",
        "晚上好",
        "再见",
        "拜拜",
        "你是谁",
        "你能做什么",
        "你能帮我做什么",
    ],
)
def test_plain_chat_whitelist_matches_complete_messages(message):
    assert ROUTING.is_explicit_plain_chat(message)


@pytest.mark.parametrize(
    "message",
    [
        "你好！！！",
        "您好。",
        "HeLLo???",
        "THANK YOU!",
        "ＨＥＬＬＯ！",
        "ＯＫ？",
        "  谢谢啦。  ",
    ],
)
def test_plain_chat_normalizes_case_width_whitespace_and_trailing_punctuation(
    message,
):
    assert ROUTING.is_explicit_plain_chat(message)


def test_plain_chat_route_flag_is_explicit():
    assert ROUTING.route_flags("你好") == {"plain_chat": True}


@pytest.mark.parametrize(
    "message",
    [
        "你好，帮我查一下CRISPRi",
        "你好，请问CRISPRi是什么？",
        "谢谢，帮我查一下CRISPRi",
        "好的，那这个机制是什么",
        "Docker是什么？",
    ],
)
def test_non_plain_chat_messages_do_not_route_to_plain_chat(message):
    assert not ROUTING.is_explicit_plain_chat(message)
    assert ROUTING.route_flags(message) == {}


@pytest.mark.parametrize("message", ["", None, ["你好"]])
def test_non_string_or_empty_messages_are_not_plain_chat(message):
    assert not ROUTING.is_explicit_plain_chat(message)
    assert ROUTING.route_flags(message) == {}


def test_plain_chat_prompt_guides_only_on_first_user_message():
    first_turn = [{"role": "user", "content": "你好"}]
    later_turn = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "谢谢"},
    ]

    first_prompt = DIALOGUE.get_plain_chat_prompt(first_turn)
    later_prompt = DIALOGUE.get_plain_chat_prompt(later_turn)

    assert first_prompt == DIALOGUE.PLAIN_CHAT_SYSTEM_PROMPT
    assert later_prompt == DIALOGUE.PLAIN_CHAT_SYSTEM_PROMPT_NO_GUIDE
    assert "引导" in first_prompt
    assert "引导" not in later_prompt


def test_first_user_message_is_shared_with_prompt_selection():
    messages = [{"role": "user", "content": "你好"}]

    assert DIALOGUE.is_first_user_message(messages)
    assert not DIALOGUE.is_first_user_message(messages + [{"role": "user"}])


def test_no_evidence_prompts_use_distinct_defaults():
    answer = DIALOGUE.get_no_evidence_prompt("answer")
    refuse = DIALOGUE.get_no_evidence_prompt("refuse")

    assert answer == DIALOGUE.NO_EVIDENCE_SYSTEM_PROMPT
    assert refuse == DIALOGUE.NO_EVIDENCE_REFUSE_PROMPT
    assert answer and refuse and answer != refuse


def test_no_evidence_mode_is_normalized_in_dialogue_module(monkeypatch):
    assert DIALOGUE.get_no_evidence_mode() == "answer"

    monkeypatch.setattr(DIALOGUE, "NO_EVIDENCE_MODE", " REFUSE ")
    assert DIALOGUE.get_no_evidence_mode() == "refuse"
    monkeypatch.setattr(DIALOGUE, "NO_EVIDENCE_MODE", "unexpected")
    assert DIALOGUE.get_no_evidence_mode() == "answer"


@pytest.mark.parametrize(
    "message",
    [
        "请结合文献图片说明细菌纤维素的应用",
        "这些图中哪个展示伤口愈合过程？",
        "相关表格说明了哪些性能？",
        "Explain the pathway in Figure 2.",
        "Show the relevant table evidence.",
    ],
)
def test_explicit_visual_requests_enable_visual_evidence(message):
    assert ROUTING.requests_visual_evidence(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "ldhA 敲除如何影响丁二酸产量？",
        "如何控制 dCas9 表达强度？",
        "试图提高产量是否合理？",
        None,
    ],
)
def test_non_visual_questions_do_not_expand_images(message):
    assert ROUTING.requests_visual_evidence(message) is False
