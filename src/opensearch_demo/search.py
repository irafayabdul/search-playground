from __future__ import annotations
"""Stage 1 (retrieval) and the fusion step.

Three retrieval modes, deliberately kept separate so you can run them on the
same query and watch them disagree:

  lexical  BM25 over the analysed text. Exact terms, identifiers, rare words.
  neural   HNSW k-NN over embeddings. Paraphrase and synonymy.
  hybrid   Both, fused by Reciprocal Rank Fusion.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from .config import INDEX_NAME, PipelineParams
from .embed import encode_query

# return everything except the vector — works for both the toy and the
# product schema, and keeps result payloads small
_SOURCE = {"excludes": ["embedding"]}


def _filter_clauses(
    must: Optional[Dict[str, Any]] = None,
    should: Optional[Dict[str, Any]] = None,
    geo: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """Translate plain dicts into OpenSearch bool clauses.

    must   -> hard constraint. A document that fails it is gone.
    should -> soft preference. Failing it costs score, not membership.
    """
    clauses: Dict[str, list] = {"filter": [], "should": []}
    for field, value in (must or {}).items():
        if isinstance(value, dict) and {"gte", "lte"} & value.keys():
            clauses["filter"].append({"range": {field: value}})
        elif isinstance(value, (list, tuple)):
            clauses["filter"].append({"terms": {field: list(value)}})
        else:
            clauses["filter"].append({"term": {field: value}})
    for field, value in (should or {}).items():
        if isinstance(value, (list, tuple)):
            clauses["should"].append({"terms": {field: list(value)}})
        else:
            clauses["should"].append({"term": {field: value}})
    if geo:
        clauses["filter"].append({
            "geo_distance": {
                "distance": geo["distance"],
                "location": {"lat": geo["lat"], "lon": geo["lon"]},
            }
        })
    return clauses


def _hits(response: dict) -> List[Dict]:
    return [
        {**h["_source"], "_score": h["_score"], "_id": h["_id"]}
        for h in response["hits"]["hits"]
    ]


def lexical_search(
    client: OpenSearch,
    query: str,
    k: int = 100,
    index: str = INDEX_NAME,
    must: Optional[Dict] = None,
    should: Optional[Dict] = None,
    geo: Optional[Dict] = None,
) -> List[Dict]:
    """BM25. title is boosted — a term in the title is a stronger signal."""
    clauses = _filter_clauses(must, should, geo)
    body = {
        "size": k,
        "_source": _SOURCE,
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "features", "text"],
                        "type": "best_fields",
                    }
                }],
                **clauses,
            }
        },
    }
    return _hits(client.search(index=index, body=body))


def neural_search(
    client: OpenSearch,
    query: str,
    k: int = 100,
    index: str = INDEX_NAME,
    must: Optional[Dict] = None,
    should: Optional[Dict] = None,
    geo: Optional[Dict] = None,
) -> List[Dict]:
    """HNSW k-NN.

    Note the filter goes *inside* the knn clause. That matters: OpenSearch then
    applies it during graph traversal rather than after, so you still get k
    results. Filtering after the fact silently returns fewer than k whenever the
    filter is selective.
    """
    clauses = _filter_clauses(must, should, geo)
    knn: Dict[str, Any] = {
        "vector": encode_query(query).tolist(),
        "k": k,
    }
    if clauses["filter"]:
        knn["filter"] = {"bool": {"filter": clauses["filter"]}}

    body = {
        "size": k,
        "_source": _SOURCE,
        "query": {"knn": {"embedding": knn}},
    }
    return _hits(client.search(index=index, body=body))


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Dict]:
    """Fuse ranked lists on RANK, not score.

    score(d) = Σ_lists  w_i / (k + rank_i(d))     [rank is 1-based]

    Using rank is the point: BM25 scores are unbounded and corpus-dependent,
    cosine similarities sit in [-1, 1]. They are not comparable, and normalising
    them is guesswork. Rank position is comparable by construction.

    k dampens the advantage of the very top positions — with k=60 the gap
    between rank 1 and rank 2 is small, so a document both lists rank highly
    beats one that a single list ranks first.
    """
    weights = weights or [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must have one entry per result list")

    fused: Dict[str, float] = defaultdict(float)
    by_id: Dict[str, Dict] = {}
    contributions: Dict[str, Dict[int, int]] = defaultdict(dict)

    for list_idx, (results, weight) in enumerate(zip(result_lists, weights)):
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            fused[doc_id] += weight / (k + rank)
            contributions[doc_id][list_idx] = rank
            by_id.setdefault(doc_id, doc)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {**by_id[doc_id],
         "_rrf_score": score,
         "_ranks": contributions[doc_id]}    # which list found it, and where
        for doc_id, score in ordered
    ]


def hybrid_search(
    client: OpenSearch,
    query: str,
    k: int = 100,
    index: str = INDEX_NAME,
    params: Optional[PipelineParams] = None,
    weights: Optional[List[float]] = None,
    **filters,
) -> List[Dict]:
    """Run both retrievers and fuse. This is stage 1 of the funnel."""
    params = params or PipelineParams()
    lex = lexical_search(client, query, k=k, index=index, **filters)
    vec = neural_search(client, query, k=k, index=index, **filters)
    return reciprocal_rank_fusion([lex, vec], k=params.rrf_k, weights=weights)
