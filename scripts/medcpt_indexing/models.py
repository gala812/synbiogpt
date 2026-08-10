from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(slots=True)
class IndexDocument:
    point_id: str
    chunk_id: str
    text: str
    embedding_text: str
    metadata: dict


@dataclass(slots=True)
class IndexingConfig:
    chunks_dir: Path
    mapping_db: Path
    state_file: Path
    collection_name: str = "fulltext_medcpt_v1"
    bm25_index_name: str = "fulltext_bm25_v1"
    encode_batch_size: int = 128
    upload_batch_size: int = 1024
    max_tokens: int = 448
    limit_shards: int = 0
    log_every: int = 10_000


class Encoder(Protocol):
    dimension: int
    model_name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorSink(Protocol):
    def ensure_ready(self, dimension: int) -> None: ...

    def write(
        self, documents: Sequence[IndexDocument], vectors: Sequence[Sequence[float]]
    ) -> int: ...

    def count_shard(self, source_shard: str) -> int: ...


class KeywordSink(Protocol):
    def ensure_ready(self) -> None: ...

    def write(self, documents: Sequence[IndexDocument]) -> int: ...

    def count_shard(self, source_shard: str) -> int: ...
