from __future__ import annotations
"""Stage 3: cross-encoder reranking.

A bi-encoder (what we indexed with) embeds query and document independently,
so document vectors can be precomputed. That is what makes retrieval fast, and
also what limits it: the model never sees the pair together.

A cross-encoder reads (query, document) as one input, which is far more accurate
and completely unprecomputable — cost is O(candidates) model calls per query.
Hence the funnel: it only ever sees the shortlist.
"""
from functools import lru_cache
from typing import Dict, List

from .config import RERANK_MODEL
from .embed import build_embedding_text


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str = RERANK_MODEL):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query: str, docs: List[Dict], top_k: int | None = None) -> List[Dict]:
    """Rescore `docs` against `query` and re-sort. Adds `_ce_score` and keeps
    `_retrieval_rank` so you can see how far the reranker moved each document."""
    if not docs:
        return []
    model = get_cross_encoder()
    pairs = [(query, build_embedding_text(d)) for d in docs]
    scores = model.predict(pairs, show_progress_bar=len(pairs) > 64)

    scored = [
        {**doc, "_ce_score": float(score), "_retrieval_rank": rank}
        for rank, (doc, score) in enumerate(zip(docs, scores), start=1)
    ]
    scored.sort(key=lambda d: d["_ce_score"], reverse=True)
    for new_rank, doc in enumerate(scored, start=1):
        doc["_rank_delta"] = doc["_retrieval_rank"] - new_rank
    return scored[:top_k] if top_k else scored
