"""SynBioGPT retrieval pipeline and its stable configuration boundary."""

from .config import RetrievalConfig
from .pipeline import RetrievalPipeline

__all__ = ["RetrievalConfig", "RetrievalPipeline"]
