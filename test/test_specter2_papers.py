import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/search/specter2_papers.py"
)
SPEC = importlib.util.spec_from_file_location("specter2_papers", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PmidPmcidMapper = MODULE.PmidPmcidMapper
Specter2PaperRetriever = MODULE.Specter2PaperRetriever
normalize_paper_title = MODULE.normalize_paper_title

ENCODER_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/models/specter2.py"
)
ENCODER_SPEC = importlib.util.spec_from_file_location("specter2_encoder", ENCODER_PATH)
assert ENCODER_SPEC and ENCODER_SPEC.loader
ENCODER_MODULE = importlib.util.module_from_spec(ENCODER_SPEC)
ENCODER_SPEC.loader.exec_module(ENCODER_MODULE)
Specter2PaperEncoder = ENCODER_MODULE.Specter2PaperEncoder


class FakeEncoder:
    dimension = 3

    def encode(self, query):
        assert query == "CRISPR interference"
        return [1.0, 0.0, 0.0]


class FakeClient:
    def __init__(self):
        vectors = SimpleNamespace(size=3, distance="Cosine")
        self.info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)),
            status="green",
            points_count=3,
        )
        self.query = None

    def get_collection(self, collection_name):
        assert collection_name == "synbiogpt_papers_specter2"
        return self.info

    def query_points(self, **kwargs):
        self.query = kwargs
        points = [
            SimpleNamespace(
                id="p1",
                score=0.91,
                payload={
                    "doc_id": "123",
                    "title": "CRISPRi engineering",
                    "document": "Abstract text",
                    "journal": "Synthetic Biology",
                    "publication_date": "2025-01-01",
                },
            ),
            SimpleNamespace(
                id="p2", score=0.90, payload={"doc_id": "123", "title": "duplicate"}
            ),
            SimpleNamespace(
                id="p3", score=0.85, payload={"doc_id": "456", "title": "Second"}
            ),
        ]
        return SimpleNamespace(points=points)


def test_encoder_uses_normalized_pooler_output():
    class Tensor:
        def __init__(self, values=None):
            self.values = values

        def to(self, device):
            assert device == "cpu"
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    class Functional:
        @staticmethod
        def normalize(value, *, p, dim):
            assert value.values == [[3.0, 4.0]]
            assert (p, dim) == (2, 1)
            return Tensor([[0.6, 0.8]])

    class Torch:
        nn = SimpleNamespace(functional=Functional())

        @staticmethod
        def inference_mode():
            return InferenceMode()

    class Tokenizer:
        def __call__(self, values, **kwargs):
            assert values == ["paper query"]
            assert kwargs["max_length"] == 512
            return {"input_ids": Tensor()}

    class Model:
        def __call__(self, **_):
            return SimpleNamespace(pooler_output=Tensor([[3.0, 4.0]]))

    encoder = object.__new__(Specter2PaperEncoder)
    encoder._torch = Torch()
    encoder._device = "cpu"
    encoder._max_tokens = 512
    encoder.tokenizer = Tokenizer()
    encoder.model = Model()

    assert encoder.encode("paper query") == pytest.approx([0.6, 0.8])


def test_search_uses_alias_and_deduplicates_pmids(tmp_path):
    database = tmp_path / "mapping.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE paper_id_mapping (pmid TEXT, pmcid TEXT)")
        connection.executemany(
            "INSERT INTO paper_id_mapping VALUES (?, ?)",
            [("123", "PMC123"), ("456", "PMC456")],
        )
    client = FakeClient()
    retriever = Specter2PaperRetriever(
        url="http://qdrant",
        encoder=FakeEncoder(),
        mapper=PmidPmcidMapper(database),
        client=client,
    )

    assert retriever.validate_collection()["distance"] == "cosine"
    hits = retriever.search("CRISPR interference", limit=2)

    assert [hit.pmid for hit in hits] == ["123", "456"]
    assert [hit.pmcid for hit in hits] == ["PMC123", "PMC456"]
    assert hits[0].to_dict()["pubmed_url"].endswith("/123/")
    assert client.query["collection_name"] == "synbiogpt_papers_specter2"
    assert client.query["query"] == [1.0, 0.0, 0.0]


def test_search_rejects_invalid_input():
    retriever = Specter2PaperRetriever(
        url="http://qdrant", encoder=FakeEncoder(), client=FakeClient()
    )
    with pytest.raises(ValueError, match="empty"):
        retriever.search(" ")
    with pytest.raises(ValueError, match="between 1 and 50"):
        retriever.search("CRISPR interference", limit=51)
    with pytest.raises(ValueError, match="expected 3"):
        retriever.search_vector([1.0, 0.0])


def test_pmid_lookup_and_recommendation_do_not_encode_a_query(monkeypatch):
    class QueryModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        SimpleNamespace(
            models=SimpleNamespace(
                Filter=QueryModel,
                FieldCondition=QueryModel,
                MatchValue=QueryModel,
            )
        ),
    )

    class NoQueryEncoder:
        dimension = 3

        def encode(self, _):
            raise AssertionError("PMID routes must use the indexed paper directly")

    class LookupClient(FakeClient):
        def scroll(self, **kwargs):
            point = SimpleNamespace(
                id="p1",
                payload={
                    "doc_id": "123",
                    "title": "Seed paper",
                    "document": "Seed abstract",
                },
                vector=[1.0, 0.0, 0.0],
            )
            return [point], None

    retriever = Specter2PaperRetriever(
        url="http://qdrant", encoder=NoQueryEncoder(), client=LookupClient()
    )

    assert retriever.paper("123").title == "Seed paper"
    assert [hit.pmid for hit in retriever.related("123", limit=2)] == ["456"]


def test_title_resolution_prefers_normalized_exact_match():
    class TitleEncoder:
        dimension = 3

        def __init__(self):
            self.query = None

        def encode(self, query):
            self.query = query
            return [1.0, 0.0, 0.0]

    class TitleClient(FakeClient):
        def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="p1",
                        score=0.95,
                        payload={"doc_id": "111", "title": "A Related Cellulose Paper"},
                    ),
                    SimpleNamespace(
                        id="p2",
                        score=0.93,
                        payload={
                            "doc_id": "222",
                            "title": "Three-Dimensional Printed Cellulose: For Wound Dressing Applications.",
                        },
                    ),
                ]
            )

    supplied_title = (
        "three-dimensional printed cellulose for wound dressing applications"
    )
    encoder = TitleEncoder()
    retriever = Specter2PaperRetriever(
        url="http://qdrant", encoder=encoder, client=TitleClient()
    )
    resolution = retriever.resolve_title(supplied_title)

    assert resolution.status == "exact"
    assert resolution.matched.pmid == "222"
    assert encoder.query == supplied_title
    assert normalize_paper_title("A: Paper.") == "a paper"


def test_title_resolution_requires_confirmation_for_duplicate_exact_titles():
    class TitleEncoder:
        dimension = 3

        def encode(self, _):
            return [1.0, 0.0, 0.0]

    class DuplicateTitleClient(FakeClient):
        def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="p1",
                        score=0.96,
                        payload={"doc_id": "111", "title": "An Exact Paper Title"},
                    ),
                    SimpleNamespace(
                        id="p2",
                        score=0.95,
                        payload={"doc_id": "222", "title": "An exact paper title."},
                    ),
                ]
            )

    retriever = Specter2PaperRetriever(
        url="http://qdrant", encoder=TitleEncoder(), client=DuplicateTitleClient()
    )
    resolution = retriever.resolve_title("An Exact Paper Title")

    assert resolution.status == "ambiguous"
    assert resolution.matched is None
    assert [hit.pmid for _, hit in resolution.candidates] == ["111", "222"]
