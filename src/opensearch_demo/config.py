from __future__ import annotations

"""Central knobs. Everything tunable in the meeting's walkthrough lives here."""
from dataclasses import dataclass, field
import os

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "articles")

# all-MiniLM-L6-v2: 384 dims, 256-token window. That window is exactly the
# truncation limit Rashid flagged — see embed.build_embedding_text.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_MAX_TOKENS = 256

# Cross-encoder scores (query, doc) jointly, so it cannot be precomputed at
# index time — that is why it sits in stage 3 over a trimmed set, never over
# the whole corpus.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class HNSWParams:
    """The three knobs from the HNSW segment of the call.

    m               max bidirectional links per node. Higher = denser graph,
                    better recall, more memory. Fixed at build time.
    ef_construction candidates held while inserting a node. Higher = better
                    graph quality, slower indexing. Fixed at build time.
    ef_search       candidates held while querying. Higher = better recall,
                    slower queries. The only one tunable after indexing.
    """
    m: int = 16
    ef_construction: int = 128
    ef_search: int = 100
    # faiss = GPU-optimised, lucene = CPU-optimised. On this laptop: lucene.
    engine: str = "lucene"
    space_type: str = "cosinesimil"


@dataclass
class PipelineParams:
    """Fan-out at each stage. The funnel shape is the whole point:
    retrieve wide and cheap, then narrow with progressively costlier models."""
    # Sized for the 27-document demo corpus so each stage visibly narrows.
    # Production shape is the same funnel an order of magnitude wider:
    # ~1000 candidates -> ~100 trimmed -> ~50 reranked -> ~10 final.
    candidates: int = 20    # stage 1 — retrieve wide and cheap
    after_trim: int = 12    # stage 2 — must/should filters
    rerank_top_k: int = 10  # stage 3 — cross-encoder input
    final_k: int = 5        # stage 4 — what an LLM would actually read
    rrf_k: int = 60         # RRF smoothing constant (60 is the paper default)
