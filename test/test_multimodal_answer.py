import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/multimodal_answer.py"
)
SPEC = importlib.util.spec_from_file_location("multimodal_answer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_collects_deduplicated_retrieval_images_with_limit():
    sources = [
        {"metadata": [{"image_urls": ["http://assets/a", "http://assets/b"]}]},
        {"metadata": [{"image_urls": ["http://assets/a", "http://assets/c"]}]},
    ]

    assert MODULE.collect_retrieval_image_urls(sources, max_images=2) == [
        "http://assets/a",
        "http://assets/b",
    ]


def test_builds_assistant_image_files():
    assert MODULE.build_retrieval_image_files(
        ["http://assets/a", "  ", "http://assets/b"]
    ) == [
        {"type": "image", "url": "http://assets/a"},
        {"type": "image", "url": "http://assets/b"},
    ]


def test_converts_string_content_and_preserves_original_chinese_query():
    messages = [{"role": "user", "content": "大肠杆菌中 ldhA 有什么作用？"}]

    injected = MODULE.inject_images_into_last_user_message(
        messages, ["http://assets/a"]
    )

    assert injected == 1
    assert messages[0]["content"] == [
        {"type": "text", "text": "大肠杆菌中 ldhA 有什么作用？"},
        {"type": "image_url", "image_url": {"url": "http://assets/a"}},
    ]


def test_preserves_existing_content_and_deduplicates_images():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "解释这张图"},
                {"type": "image_url", "image_url": {"url": "http://assets/a"}},
            ],
        }
    ]

    injected = MODULE.inject_images_into_last_user_message(
        messages,
        ["http://assets/a", "http://assets/b", "http://assets/c"],
        max_images=2,
    )

    assert injected == 1
    assert len(messages[0]["content"]) == 3
    assert messages[0]["content"][-1]["image_url"]["url"] == "http://assets/b"


def test_does_not_change_messages_without_retrieved_images():
    messages = [{"role": "user", "content": "没有图片的查询"}]

    assert MODULE.inject_images_into_last_user_message(messages, []) == 0
    assert messages == [{"role": "user", "content": "没有图片的查询"}]
