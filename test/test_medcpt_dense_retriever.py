import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/search/medcpt_dense.py"
)
SPEC = importlib.util.spec_from_file_location("medcpt_dense_retriever", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MedCPTDenseRetriever = MODULE.MedCPTDenseRetriever


class FakeEncoder:
    dimension = 3

    def encode(self, query):
        assert query == "synthetic biology"
        return [3.0, 4.0, 0.0]


class FakeClient:
    def __init__(self, *, distance="Dot", dimension=3):
        vectors = SimpleNamespace(size=dimension, distance=distance)
        params = SimpleNamespace(vectors=vectors)
        config = SimpleNamespace(params=params)
        self.info = SimpleNamespace(
            config=config,
            status="green",
            points_count=2,
            indexed_vectors_count=2,
        )
        self.query = None

    def get_collection(self, collection_name):
        assert collection_name == "fulltext_medcpt_ip_v1"
        return self.info

    def query_points(self, **kwargs):
        self.query = kwargs
        point = SimpleNamespace(
            id="point-1",
            score=12.5,
            payload={
                "text": "Engineered bacteria produced the target metabolite.",
                "metadata": {
                    "chunk_id": "PMC1_results_0001",
                    "pmid": "1",
                    "pmcid": "PMC1",
                },
            },
        )
        return SimpleNamespace(points=[point])


def _retriever(client):
    return MedCPTDenseRetriever(
        url="http://qdrant",
        collection_name="fulltext_medcpt_ip_v1",
        encoder=FakeEncoder(),
        client=client,
    )


def test_retriever_validates_dot_collection_and_returns_payload():
    client = FakeClient()
    retriever = _retriever(client)

    assert retriever.validate_collection()["distance"] == "dot"
    vector = retriever.encode_query("synthetic biology")
    hits = retriever.search_vector(vector, limit=5)

    assert vector == [3.0, 4.0, 0.0]
    assert hits[0].score == 12.5
    assert hits[0].metadata["pmid"] == "1"
    assert client.query["query"] == vector
    assert client.query["limit"] == 5
    assert client.query["query_filter"] is None


@pytest.mark.parametrize(
    ("distance", "dimension", "message"),
    [("Cosine", 3, "expected dot"), ("Dot", 4, "encoder dimension")],
)
def test_retriever_rejects_incompatible_collection(distance, dimension, message):
    with pytest.raises(RuntimeError, match=message):
        _retriever(FakeClient(distance=distance, dimension=dimension)).validate_collection()


def test_retriever_rejects_empty_query_and_wrong_vector_size():
    retriever = _retriever(FakeClient())

    with pytest.raises(ValueError, match="empty"):
        retriever.encode_query("  ")
    with pytest.raises(ValueError, match="expected 3"):
        retriever.search_vector([1.0, 2.0])
