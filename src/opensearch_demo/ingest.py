from __future__ import annotations
"""Bulk indexing: embed once, write once."""
from typing import Dict, Iterable, List

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from .config import INDEX_NAME
from .embed import build_embedding_text, encode, warn_on_truncation


def prepare(docs: List[Dict], report_truncation: bool = True) -> List[Dict]:
    """Attach an embedding to every document.

    Embeddings are computed here, in the client. The alternative shown in the
    call is to register the model inside OpenSearch (ML Commons: create a model
    group, get its ID, deploy the model, attach an ingest pipeline) so the
    cluster embeds on write. That removes the client-side model but couples
    indexing throughput to cluster resources. See README for the trade-off.
    """
    texts = [build_embedding_text(d) for d in docs]

    if report_truncation:
        over = warn_on_truncation(texts)
        if over:
            print(f"⚠ {len(over)}/{len(texts)} documents exceed the "
                  f"embedding window and will be truncated. "
                  f"First few: {[docs[i]['id'] for i in over[:5]]}")

    vectors = encode(texts)
    return [{**d, "embedding": vectors[i].tolist()} for i, d in enumerate(docs)]


def index_docs(
    client: OpenSearch,
    docs: List[Dict],
    index: str = INDEX_NAME,
    refresh: bool = True,
) -> tuple[int, list]:
    """Bulk-write prepared documents. `refresh=True` makes them immediately
    searchable — convenient in a notebook, wasteful in production, where the
    refresh interval exists precisely to batch this work."""
    actions: Iterable[Dict] = (
        {"_index": index, "_id": d["id"], "_source": d} for d in docs
    )
    succeeded, errors = bulk(client, actions, refresh=refresh, stats_only=False)
    return succeeded, errors
