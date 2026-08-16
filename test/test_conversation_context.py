import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).parents[1] / "backend/open_webui/apps/retrieval"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package in (
    "open_webui",
    "open_webui.apps",
    "open_webui.apps.retrieval",
    "open_webui.apps.retrieval.synbio",
):
    sys.modules.setdefault(package, ModuleType(package))

load(
    "open_webui.apps.retrieval.query_processor",
    ROOT / "query_processor.py",
)
CONVERSATION = load(
    "open_webui.apps.retrieval.synbio.conversation",
    ROOT / "synbio/conversation.py",
)


def test_query_window_is_token_bounded_instead_of_six_message_bounded():
    messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message {index}",
        }
        for index in range(10)
    ]

    window = CONVERSATION.select_message_window(messages, token_budget=1_000)

    assert window.messages == messages
    assert window.omitted_messages == 0


def test_window_keeps_a_contiguous_recent_suffix_and_system_message():
    system = {"role": "system", "content": "User Context: works on E. coli"}
    messages = [
        system,
        {"role": "user", "content": "old " * 100},
        {"role": "assistant", "content": "middle " * 100},
        {"role": "user", "content": "latest question"},
    ]

    window = CONVERSATION.select_message_window(messages, token_budget=80)

    assert window.messages == [system, messages[-1]]
    assert window.omitted_messages == 2


def test_explicit_follow_up_inherits_prior_user_entities_only():
    messages = [
        {"role": "user", "content": "在 E. coli 中敲除 ldhA"},
        {"role": "assistant", "content": "模型回答中出现 pcnB"},
        {"role": "user", "content": "同时抑制 ppc 会有什么影响？"},
    ]

    assert CONVERSATION.inherited_exact_terms(
        messages, "同时抑制 ppc 会有什么影响？"
    ) == ("E. coli", "ldhA")


def test_new_topic_does_not_inherit_stale_entities():
    messages = [
        {"role": "user", "content": "在 E. coli 中敲除 ldhA"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "细菌纤维素有哪些用途？"},
    ]

    assert CONVERSATION.inherited_exact_terms(
        messages, "细菌纤维素有哪些用途？"
    ) == ()


def test_entity_fallback_is_english_and_conservative():
    assert CONVERSATION.fallback_semantic_query("为什么？", ("CRISPRi", "ldhA")) == (
        "Scientific literature about CRISPRi ldhA"
    )
    assert CONVERSATION.fallback_semantic_query("为什么？", ()) is None


def test_multimodal_images_are_included_in_answer_budget():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "请解释图片"},
            {"type": "image_url", "image_url": {"url": "http://assets/1"}},
            {"type": "image_url", "image_url": {"url": "http://assets/2"}},
        ],
    }

    assert CONVERSATION.estimate_tokens(message) >= 1_536


def test_recent_query_cache_is_bounded_and_conversation_scoped():
    cache = CONVERSATION.RecentQueryCache(max_entries=2, ttl_seconds=60)
    cache.put("user:chat-1", "first")
    cache.put("user:chat-2", "second")
    cache.put("user:chat-3", "third")

    assert cache.get("user:chat-1") is None
    assert cache.get("user:chat-2") == "second"
    assert CONVERSATION.conversation_key(
        "user", {"chat_id": "chat-2"}
    ) == "user:chat-2"
    assert CONVERSATION.conversation_key("user", {}) == ""
