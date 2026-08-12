import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/search/opensearch_bm25.py"
)
SPEC = importlib.util.spec_from_file_location("opensearch_bm25", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeClient:
    def __init__(self):
        self.index = None
        self.body = None

    def search(self, *, index, body):
        self.index = index
        self.body = body
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "stored-id",
                        "_score": 8.5,
                        "_source": {
                            "doc_id": "PMC1_results_0001",
                            "text": "succinate production increased",
                            "collection_name": "fulltext_medcpt_ip_v1",
                            "metadata": {"pmid": "1", "pmcid": "PMC1"},
                        },
                    }
                ]
            }
        }

    def mget(self, *, index, body, _source):
        self.index = index
        self.body = body
        self.source_fields = _source
        return {
            "docs": [
                {
                    "found": True,
                    "_source": {
                        "doc_id": "PMC1_results_0001",
                        "text": "succinate production increased",
                        "collection_name": "fulltext_medcpt_ip_v1",
                        "pmcid": "PMC1",
                        "metadata": {
                            "chunk_id": "PMC1_results_0001",
                            "pmcid": "PMC1",
                            "parent_chunk_id": "PMC1_parent_0001",
                        },
                    },
                }
            ]
        }


def test_fulltext_collection_uses_production_bm25_index(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(MODULE, "_get_client", lambda: client)
    monkeypatch.delenv("MEDCPT_BM25_INDEX", raising=False)

    hits = MODULE.search_bm25(
        ["fulltext_medcpt_ip_v1"],
        "Escherichia coli ldhA succinate deletion knockout",
        top_k=100,
        exact_terms=["ldhA"],
    )

    assert client.index == "fulltext_bm25_v1"
    assert client.body["size"] == 100
    multi_match = client.body["query"]["bool"]["must"][0]["multi_match"]
    assert multi_match["query"] == ("Escherichia coli ldhA succinate deletion knockout")
    assert multi_match["minimum_should_match"] == "30%"
    assert client.body["query"]["bool"]["minimum_should_match"] == 1
    exact_match = client.body["query"]["bool"]["should"][0]["multi_match"]
    assert exact_match["query"] == "ldhA"
    assert exact_match["type"] == "phrase"
    assert hits[0]["doc_id"] == "PMC1_results_0001"


def test_generic_collection_keeps_generic_bm25_index(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(MODULE, "_get_client", lambda: client)
    monkeypatch.delenv("OPENSEARCH_BM25_INDEX", raising=False)

    MODULE.search_bm25(["file-user-upload"], "synthetic biology", top_k=10)

    assert client.index == "open_webui_bm25"


def test_fetch_chunks_by_ids_uses_deterministic_storage_ids(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(MODULE, "_get_client", lambda: client)

    documents = MODULE.fetch_chunks_by_ids(
        "fulltext_medcpt_ip_v1",
        ["PMC1_results_0001", "PMC1_results_0001"],
    )

    assert client.index == "fulltext_bm25_v1"
    assert client.body == {"ids": ["fulltext_medcpt_ip_v1:PMC1_results_0001"]}
    assert "metadata" in client.source_fields
    document = documents["PMC1_results_0001"]
    assert document.page_content == "succinate production increased"
    assert document.metadata["parent_chunk_id"] == "PMC1_parent_0001"


def test_fetch_chunks_by_ids_falls_back_across_legacy_id_prefix(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(MODULE, "_get_client", lambda: client)
    client.mget = lambda **_: {"docs": [{"found": False}]}

    documents = MODULE.fetch_chunks_by_ids(
        "fulltext_medcpt_ip_v1",
        ["PMC1_results_0001"],
    )

    assert client.body["query"]["terms"] == {"doc_id": ["PMC1_results_0001"]}
    assert documents["PMC1_results_0001"].metadata["pmcid"] == "PMC1"
