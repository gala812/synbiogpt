import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/models/medcpt.py"
)
SPEC = importlib.util.spec_from_file_location("medcpt_query_encoder", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CollectionEmbeddingRouter = MODULE.CollectionEmbeddingRouter
CollectionRerankerRouter = MODULE.CollectionRerankerRouter
MedCPTCrossEncoder = MODULE.MedCPTCrossEncoder
build_medcpt_rerank_text = MODULE.build_medcpt_rerank_text
raw_cls = MODULE._raw_cls


def test_collection_embedding_router_uses_medcpt_only_for_fulltext():
    default_calls = []
    medcpt_calls = []

    def default(value):
        default_calls.append(value)
        return [1.0]

    def medcpt(value):
        medcpt_calls.append(value)
        return [2.0]

    router = CollectionEmbeddingRouter(
        default,
        {"fulltext_medcpt_ip_v1": medcpt},
    )

    assert router("document uploaded by a user") == [1.0]
    assert router.for_collection("other_collection")("query") == [1.0]
    assert router.for_collection("fulltext_medcpt_ip_v1")("query") == [2.0]
    assert default_calls == ["document uploaded by a user", "query"]
    assert medcpt_calls == ["query"]


def test_query_encoder_uses_raw_cls_without_normalization():
    class HiddenState:
        def __getitem__(self, key):
            assert key == (slice(None), 0, slice(None))
            return self

        def float(self):
            return [6.0, 8.0]

    assert raw_cls(HiddenState()) == [6.0, 8.0]


def test_collection_reranker_router_limits_medcpt_to_fulltext():
    default = object()
    medcpt = object()
    router = CollectionRerankerRouter(
        default, {"fulltext_medcpt_ip_v1": medcpt}
    )

    assert router.for_collection("fulltext_medcpt_ip_v1") is medcpt
    assert router.for_collection("user-upload") is default


def test_rerank_text_includes_title_section_and_original_text():
    result = build_medcpt_rerank_text(
        "The ldhA deletion increased succinate production.",
        {
            "paper_title": "Engineering Escherichia coli",
            "section_path": ["Results", "Fermentation"],
        },
    )

    assert result == (
        "Title: Engineering Escherichia coli\n"
        "Section: Results > Fermentation\n"
        "Text: The ldhA deletion increased succinate production."
    )


def test_cross_encoder_batches_pairs_and_returns_raw_logits():
    class Context:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    class Torch:
        @staticmethod
        def inference_mode():
            return Context()

    class Tensor:
        def to(self, device):
            assert device == "cuda:0"
            return self

    class Logits:
        def __init__(self, values):
            self.values = values

        def squeeze(self, dim):
            assert dim == 1
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class Tokenizer:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, pairs, **kwargs):
            self.batch_sizes.append(len(pairs))
            assert kwargs["max_length"] == 512
            return {"input_ids": Tensor()}

    class Model:
        def __init__(self):
            self.offset = 0

        def __call__(self, **_):
            values = [6.5, -2.0] if self.offset == 0 else [1.25]
            self.offset += 1
            return SimpleNamespace(logits=Logits(values))

    encoder = object.__new__(MedCPTCrossEncoder)
    encoder._torch = Torch()
    encoder._device = "cuda:0"
    encoder._max_tokens = 512
    encoder._batch_size = 2
    encoder.tokenizer = Tokenizer()
    encoder.model = Model()

    scores = encoder.score("succinate production", ["a", "b", "c"])

    assert scores == [6.5, -2.0, 1.25]
    assert encoder.tokenizer.batch_sizes == [2, 1]
