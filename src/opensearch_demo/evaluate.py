from __future__ import annotations
"""Offline relevance metrics.

Implemented from the formulas rather than pulled from a library, because the
point of the next session is understanding what the numbers mean. `ranx` (MIT)
is the production choice and a good cross-check — see notebook 03.

NDCG@k
    DCG@k  = Σ (2^rel_i − 1) / log2(i + 1)          i is the 1-based rank
    IDCG@k = the same over the best possible ordering of the judged set
    NDCG@k = DCG@k / IDCG@k                          → 1.0 is a perfect ranking

The exponential gain (2^rel − 1) is the common convention: it makes the
difference between "exact match" and "substitute" count for much more than the
difference between "substitute" and "complement".

A note on what these numbers mean HERE: our corpus contains products that were
never judged for a given query. Standard practice counts unjudged as
non-relevant, so NDCG reads as a floor. Fine for comparing two of our own
pipelines against each other; not comparable to a published leaderboard.
"""
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

# ESCI labels -> graded relevance
DEFAULT_GRADES = {"E": 3, "S": 2, "C": 1, "I": 0}
# What counts as "relevant" for recall. E only is strict; {E,S} is the looser
# reading used by some ESCI papers.
DEFAULT_RELEVANT = {"E"}


def dcg(relevances: Sequence[float], k: Optional[int] = None) -> float:
    """Discounted cumulative gain over a ranked list of relevance grades."""
    rels = list(relevances)[: k] if k else list(relevances)
    return sum((2 ** rel - 1) / math.log2(i + 1)
               for i, rel in enumerate(rels, start=1))


def ndcg_at_k(ranked_grades: Sequence[float],
              all_grades: Sequence[float],
              k: int = 10) -> float:
    """ranked_grades: grades of the docs we returned, in our order.
    all_grades:     grades of every judged doc for this query (for the ideal)."""
    ideal = sorted(all_grades, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:                      # no relevant document exists to find
        return 0.0
    return dcg(ranked_grades, k) / idcg


def recall_at_k(retrieved_ids: Sequence[str],
                relevant_ids: Iterable[str],
                k: int = 100) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for d in list(retrieved_ids)[:k] if d in relevant)
    return hits / len(relevant)


def evaluate_query(retrieved_ids: Sequence[str],
                   judgments: Dict[str, dict],
                   ks: Sequence[int] = (5, 10),
                   recall_ks: Sequence[int] = (10, 100),
                   grades: Dict[str, int] = None,
                   relevant_labels: set = None) -> Dict[str, float]:
    """Score one query's result list against its judgments.

    judgments: {doc_id: {"label": "E"|"S"|"C"|"I", ...}} from esci.load_qrels()
    """
    grades = grades or DEFAULT_GRADES
    relevant_labels = relevant_labels or DEFAULT_RELEVANT

    graded = {d: grades[j["label"]] for d, j in judgments.items()}
    # Anything we retrieved that nobody judged counts as 0 — the closed-world
    # assumption. It is why these numbers are a floor.
    ranked = [graded.get(d, 0) for d in retrieved_ids]
    all_grades = list(graded.values())
    relevant_ids = [d for d, j in judgments.items() if j["label"] in relevant_labels]

    out = {f"ndcg@{k}": ndcg_at_k(ranked, all_grades, k) for k in ks}
    out.update({f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k)
                for k in recall_ks})
    return out


def evaluate_run(run: Dict[int, List[str]],
                 qrels: Dict[int, dict],
                 **kw) -> Dict[str, float]:
    """Mean metrics over many queries.

    run:   {query_id: [doc_id, ...]}  what our pipeline returned
    qrels: {query_id: {"query":..., "judgments": {...}}}
    """
    rows = [evaluate_query(ids, qrels[qid]["judgments"], **kw)
            for qid, ids in run.items() if qid in qrels]
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
