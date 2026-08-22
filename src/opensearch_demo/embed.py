from __future__ import annotations
"""Embedding the documents.

Two choices from the call are encoded here:

1. Concatenate the fields you want searchable into ONE string before encoding.
   Embedding title and body separately, then averaging, blurs both.

2. Watch the token window. MiniLM truncates at 256 tokens — silently. A long
   document loses its tail, and the tail is often where the answer was.
   `build_embedding_text` puts the title first so the most identifying text
   survives truncation, and `warn_on_truncation` makes the loss visible.
"""
from functools import lru_cache
from typing import Iterable, List, Sequence

from .config import EMBED_MAX_TOKENS, EMBED_MODEL


@lru_cache(maxsize=2)
def get_encoder(model_name: str = EMBED_MODEL):
    """Loaded lazily and cached — the first call downloads ~90 MB."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_embedding_text(doc: dict) -> str:
    """Title, then features, then description — so truncation eats the least
    important end. Median Appliances doc is ~150 tokens; the long tail truncates."""
    parts = [doc.get("title", ""), doc.get("features", ""), doc.get("text", "")]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def warn_on_truncation(texts: Sequence[str], model_name: str = EMBED_MODEL) -> List[int]:
    """Return indices of texts that exceed the window, so truncation is a fact
    you can see rather than a silent quality loss."""
    tok = get_encoder(model_name).tokenizer
    over = []
    for i, t in enumerate(texts):
        n = len(tok.encode(t, add_special_tokens=True))
        if n > EMBED_MAX_TOKENS:
            over.append(i)
    return over


def encode(texts: Iterable[str], batch_size: int = 32, normalize: bool = True):
    """Encode to vectors. normalize=True pairs with cosine space so the HNSW
    graph compares unit vectors."""
    texts = list(texts)
    return get_encoder().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
    )


def encode_query(query: str):
    """One query -> one vector. Same model as the documents, necessarily."""
    return encode([query])[0]
