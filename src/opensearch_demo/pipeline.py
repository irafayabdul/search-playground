from __future__ import annotations
"""The full funnel, end to end.

    Candidate Generation  ->  Trim  ->  Rerank  ->  (Answer)
        ~100 docs            ~50      cross-enc     top 5
        cheap, wide          filters   expensive    what an LLM reads

Each stage is cheaper per document than the next and sees more of them. Invert
that and you are running a cross-encoder over the corpus.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from .config import INDEX_NAME, PipelineParams
from .rerank import rerank
from .search import hybrid_search, lexical_search, neural_search


@dataclass
class StageTrace:
    """What each stage did — the object the notebook prints."""
    name: str
    n_in: int
    n_out: int
    top_ids: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name:<22} {self.n_in:>4} -> {self.n_out:<4}  top: {', '.join(self.top_ids[:3])}"


@dataclass
class SearchResult:
    query: str
    documents: List[Dict[str, Any]]
    trace: List[StageTrace]

    def explain(self) -> str:
        return "\n".join(str(t) for t in self.trace)


def run(
    client: OpenSearch,
    query: str,
    mode: str = "hybrid",
    params: Optional[PipelineParams] = None,
    index: str = INDEX_NAME,
    must: Optional[Dict] = None,
    should: Optional[Dict] = None,
    geo: Optional[Dict] = None,
    do_rerank: bool = True,
) -> SearchResult:
    params = params or PipelineParams()
    trace: List[StageTrace] = []
    filters = {"must": must, "should": should, "geo": geo}

    # --- Stage 1: candidate generation -----------------------------------
    retrievers = {"lexical": lexical_search, "neural": neural_search, "hybrid": hybrid_search}
    if mode not in retrievers:
        raise ValueError(f"mode must be one of {sorted(retrievers)}, got {mode!r}")
    kwargs = {"params": params} if mode == "hybrid" else {}
    candidates = retrievers[mode](
        client, query, k=params.candidates, index=index, **filters, **kwargs
    )
    trace.append(StageTrace(f"1. retrieve ({mode})", params.candidates,
                            len(candidates), [d["id"] for d in candidates]))

    # --- Stage 2: trim ----------------------------------------------------
    # Hard filters already ran inside the query. This is the second cut: keep
    # the funnel narrow enough that stage 3 stays affordable.
    trimmed = candidates[: params.after_trim]
    trace.append(StageTrace("2. trim", len(candidates), len(trimmed),
                            [d["id"] for d in trimmed]))

    # --- Stage 3: rerank --------------------------------------------------
    if do_rerank:
        shortlist = trimmed[: params.rerank_top_k]
        reranked = rerank(query, shortlist)
        trace.append(StageTrace("3. rerank (cross-enc)", len(shortlist),
                                len(reranked), [d["id"] for d in reranked]))
    else:
        reranked = trimmed
        trace.append(StageTrace("3. rerank (skipped)", len(trimmed), len(reranked),
                                [d["id"] for d in reranked]))

    final = reranked[: params.final_k]
    trace.append(StageTrace("4. final", len(reranked), len(final),
                            [d["id"] for d in final]))

    # Stage 5 would hand `final` to an LLM as context. Deliberately not wired
    # up here — that is the RAG session Rashid scheduled for next time.
    return SearchResult(query=query, documents=final, trace=trace)
