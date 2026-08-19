from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPECTER2_PAPER_COLLECTION = "synbiogpt_papers_specter2"


@dataclass(frozen=True, slots=True)
class PaperSearchHit:
    rank: int
    pmid: str
    pmcid: str | None
    title: str
    abstract: str
    journal: str
    publication_date: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pubmed_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return result


class PmidPmcidMapper:
    """Read PMID-to-PMCID mappings from the existing audited SQLite file."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        if not self.database.is_file():
            raise FileNotFoundError(f"PMID mapping database not found: {self.database}")
        with self._connect() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        self.table = next(
            (name for name in ("paper_id_mapping", "mapping") if name in tables),
            None,
        )
        if self.table is None:
            raise ValueError("PMID mapping database has no supported mapping table")

    def _connect(self):
        uri = f"file:{self.database.resolve().as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def lookup_many(self, pmids: list[str]) -> dict[str, str]:
        values = list(dict.fromkeys(value for value in pmids if value))
        if not values:
            return {}
        placeholders = ",".join("?" for _ in values)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT pmid, pmcid FROM {self.table} "
                f"WHERE pmid IN ({placeholders})",
                values,
            )
            return {str(pmid): str(pmcid).upper() for pmid, pmcid in rows}


class Specter2PaperRetriever:
    """Search and recommend papers without entering the full-text RAG path."""

    def __init__(
        self,
        *,
        url: str,
        encoder,
        api_key: str | None = None,
        collection_name: str = SPECTER2_PAPER_COLLECTION,
        mapper: PmidPmcidMapper | None = None,
        timeout: int = 120,
        client=None,
    ) -> None:
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Install qdrant-client for paper search") from exc
            client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)
        self.client = client
        self.collection_name = collection_name
        self.encoder = encoder
        self.mapper = mapper

    @staticmethod
    def _distance_name(value: Any) -> str:
        return str(getattr(value, "value", value)).lower()

    def validate_collection(self) -> dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        dimension = getattr(vectors, "size", None)
        distance = self._distance_name(getattr(vectors, "distance", None))
        if dimension != self.encoder.dimension:
            raise RuntimeError(
                f"Collection dimension is {dimension}, encoder dimension is "
                f"{self.encoder.dimension}"
            )
        if distance != "cosine":
            raise RuntimeError(
                f"Collection {self.collection_name!r} uses {distance}, expected cosine"
            )
        return {
            "collection_name": self.collection_name,
            "status": str(getattr(info, "status", "unknown")),
            "points_count": int(getattr(info, "points_count", 0)),
            "dimension": dimension,
            "distance": distance,
        }

    def _hits(self, points, *, limit: int, excluded_pmid: str = ""):
        unique = []
        seen_pmids: set[str] = set()
        for point in points:
            payload = dict(point.payload or {})
            pmid = str(payload.get("doc_id") or payload.get("id") or "").strip()
            if not pmid or pmid == excluded_pmid or pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            unique.append((point, payload, pmid))
            if len(unique) == limit:
                break

        mappings = (
            self.mapper.lookup_many([item[2] for item in unique])
            if self.mapper
            else {}
        )
        return [
            PaperSearchHit(
                rank=rank,
                pmid=pmid,
                pmcid=mappings.get(pmid),
                title=str(payload.get("title") or "").strip(),
                abstract=str(payload.get("document") or "").strip(),
                journal=str(payload.get("journal") or "").strip(),
                publication_date=str(payload.get("publication_date") or "").strip(),
                score=float(point.score),
            )
            for rank, (point, payload, pmid) in enumerate(unique, 1)
        ]

    def search_vector(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        excluded_pmid: str = "",
    ) -> list[PaperSearchHit]:
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        if len(vector) != self.encoder.dimension:
            raise ValueError(
                f"Query vector has {len(vector)} values, expected "
                f"{self.encoder.dimension}"
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=min(100, max(limit * 3, limit + 1)),
            with_payload=True,
            with_vectors=False,
        )
        return self._hits(
            response.points, limit=limit, excluded_pmid=excluded_pmid
        )

    def search(self, semantic_query: str, *, limit: int = 10):
        semantic_query = semantic_query.strip()
        if not semantic_query:
            raise ValueError("semantic_query cannot be empty")
        return self.search_vector(self.encoder.encode(semantic_query), limit=limit)

    def related(self, pmid: str, *, limit: int = 10):
        pmid = pmid.strip()
        if not pmid.isdigit():
            raise ValueError("pmid must contain digits only")
        from qdrant_client import models

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=pmid)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=True,
        )
        if not points:
            raise LookupError(f"PMID {pmid} is not present in the paper index")
        vector = points[0].vector
        if isinstance(vector, dict):
            vector = next(iter(vector.values()))
        return self.search_vector(vector, limit=limit, excluded_pmid=pmid)
