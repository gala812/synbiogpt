from __future__ import annotations

from collections.abc import Sequence


class MedCPTArticleEncoder:
    """CLS-pooled, L2-normalized MedCPT Article Encoder."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype: str = "auto",
        max_tokens: int = 448,
        local_files_only: bool = False,
        expected_dimension: int = 768,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MedCPT indexing requires PyTorch and transformers"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is not available: {device}")

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
            model_name, local_files_only=local_files_only
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
                f"Expected {expected_dimension}-dimensional MedCPT vectors, "
                f"but model reports {self.dimension}"
            )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        torch = self._torch
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded).last_hidden_state[:, 0, :]
            output = torch.nn.functional.normalize(output.float(), p=2, dim=1)
        return output.cpu().tolist()
