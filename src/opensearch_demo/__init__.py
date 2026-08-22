"""OpenSearch hybrid-search demo — built from the 2026-08-09 knowledge-sharing call."""
from .client import get_client, wait_for_cluster
from .config import HNSWParams, PipelineParams
from .schema import create_index
from .ingest import index_docs, prepare
from .search import hybrid_search, lexical_search, neural_search, reciprocal_rank_fusion
from .rerank import rerank
from . import amazon, corpus, pipeline

__all__ = [
    "get_client", "wait_for_cluster", "HNSWParams", "PipelineParams",
    "create_index", "prepare", "index_docs", "lexical_search", "neural_search",
    "hybrid_search", "reciprocal_rank_fusion", "rerank", "amazon", "corpus", "pipeline",
]
