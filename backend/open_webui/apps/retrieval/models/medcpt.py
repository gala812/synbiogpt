from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _raw_cls(last_hidden_state):
    return last_hidden_state[:, 0, :].float()


class CollectionEmbeddingRouter:
    """Use collection-specific query encoders without changing upload embeddings."""

    def __init__(
        self,
        default: Callable[[Any], Any],
        overrides: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        self.default = default
        self.overrides = dict(overrides or {})

    def __call__(self, value: Any) -> Any:
        return self.default(value)

    def for_collection(self, collection_name: str) -> Callable[[Any], Any]:
        return self.overrides.get(collection_name, self.default)


class CollectionRerankerRouter:
    """Use MedCPT reranking only for its full-text collection."""

    def __init__(self, default: Any = None, overrides: Mapping[str, Any] | None = None):
        self.default = default
        self.overrides = dict(overrides or {})

    def for_collection(self, collection_name: str):
        return self.overrides.get(collection_name, self.default)

    def predict(self, pairs):
        if self.default is None:
            raise RuntimeError("No default reranker is configured for this collection")
        return self.default.predict(pairs)


class MedCPTQueryEncoder:
    """MedCPT Query Encoder using the model's raw CLS representation."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        dtype: str = "auto",
        max_tokens: int = 64,
        local_files_only: bool = False,
        expected_dimension: int = 768,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MedCPT query encoding requires PyTorch and transformers"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is not available: {device}")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        resolved_dtype = dtype
        if dtype == "auto":
            resolved_dtype = "float16" if device.startswith("cuda") else "float32"
        dtype_by_name = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if resolved_dtype not in dtype_by_name:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self._torch = torch
        self._device = device
        self._max_tokens = max_tokens
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            dtype=dtype_by_name[resolved_dtype],
        ).to(device)
        self.model.eval()
        self.dimension = int(getattr(self.model.config, "hidden_size", 0))
        if self.dimension != expected_dimension:
            raise RuntimeError(
                f"Expected {expected_dimension}-dimensional MedCPT query vectors, "
                f"but model reports {self.dimension}"
            )

    def encode(self, texts: str | Sequence[str]) -> list[float] | list[list[float]]:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        if not values:
            return []

        encoded = self.tokenizer(
            values,
            padding=True,
            truncation=True,
            max_length=self._max_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            vectors = _raw_cls(self.model(**encoded).last_hidden_state)
        result = vectors.cpu().tolist()
        return result[0] if single else result


def build_medcpt_rerank_text(text: str, metadata: Mapping[str, Any]) -> str:
    """Add document context while preserving the original chunk text."""

    title = str(metadata.get("paper_title") or metadata.get("title") or "").strip()
    section_path = metadata.get("section_path")
    if isinstance(section_path, Sequence) and not isinstance(section_path, str):
        section = " > ".join(str(value).strip() for value in section_path if value)
    else:
        section = str(section_path or metadata.get("section") or "").strip()
        subsection = str(metadata.get("subsection") or "").strip()
        if subsection and subsection.lower() not in section.lower():
            section = " > ".join(value for value in (section, subsection) if value)

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if section:
        parts.append(f"Section: {section}")
    parts.append(f"Text: {str(text or '').strip()}")
    return "\n".join(parts)


class MedCPTCrossEncoder:
    """Official MedCPT Cross Encoder using raw relevance logits."""

    is_medcpt_cross_encoder = True
    uses_raw_logits = True

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        dtype: str = "auto",
        max_tokens: int = 512,
        batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MedCPT cross-encoding requires PyTorch and transformers"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is not available: {device}")
        if not 1 <= max_tokens <= 512:
            raise ValueError("max_tokens must be between 1 and 512")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        resolved_dtype = dtype
        if dtype == "auto":
            resolved_dtype = "float16" if device.startswith("cuda") else "float32"
        dtype_by_name = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if resolved_dtype not in dtype_by_name:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self._torch = torch
        self._device = device
        self._max_tokens = max_tokens
        self._batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            dtype=dtype_by_name[resolved_dtype],
        ).to(device)
        self.model.eval()
        if int(getattr(self.model.config, "num_labels", 0)) != 1:
            raise RuntimeError("MedCPT Cross Encoder must output exactly one logit")

    def _score_pairs(self, pairs: Sequence[Sequence[str]]) -> list[float]:
        pairs = list(pairs)
        scores: list[float] = []
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            encoded = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self._max_tokens,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits.squeeze(dim=1).float()
            scores.extend(float(value) for value in logits.cpu().tolist())
        return scores

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        return self._score_pairs([[query, document] for document in documents])

    def predict(self, pairs: Sequence[Sequence[str]]):
        pairs = list(pairs)
        if not pairs:
            return self._torch.tensor([], dtype=self._torch.float32)
        return self._torch.tensor(
            self._score_pairs(pairs), dtype=self._torch.float32
        )
