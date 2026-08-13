import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/synbio/routing.py"
)
SPEC = importlib.util.spec_from_file_location("synbio_routing", MODULE_PATH)
ROUTING = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ROUTING)


def route(files=None, metadata=None, *, enabled=True):
    return ROUTING.add_default_knowledge(
        files,
        metadata,
        enabled=enabled,
        collection="fulltext_medcpt_ip_v1",
    )


def test_plain_chat_uses_default_knowledge():
    assert route()[0]["id"] == "fulltext_medcpt_ip_v1"
    assert route()[0]["type"] == "collection"


def test_uploaded_files_are_kept_after_default_knowledge():
    upload = {"id": "upload-1", "name": "notes.pdf", "type": "file"}
    assert route([upload])[1] == upload


def test_default_collection_is_not_duplicated():
    selected = {"id": "fulltext_medcpt_ip_v1", "type": "collection"}
    assert route([selected]) == [selected]


def test_internal_tasks_and_disabled_mode_do_not_retrieve():
    assert route(metadata={"task": "title_generation"}) == []
    assert route(enabled=False) == []
