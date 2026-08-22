from __future__ import annotations
"""A small hand-written corpus, committed to the repo so there is no download.

Every document is distinct, and the set is shaped so the retrieval modes
disagree in ways you can actually see:

  * LEXICAL WINS  — docs keyed by exact identifiers (`ef_search`,
    `circuit_breaking_exception`, `429`). Embeddings blur these into their
    neighbourhood; BM25 matches the literal token.
  * NEURAL WINS   — docs that answer a question without sharing its vocabulary
    ("trim the tail of a slow query" vs "make search faster"). BM25 scores zero.
  * BOTH DISAGREE — near-synonymous topics split across categories, where the
    fused ranking beats either list alone.

Query `probe_queries()` for examples chosen to expose each case.
"""
import json
from pathlib import Path
from typing import Dict, List

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "articles.jsonl"

# (title, category, tags, year, rating, city, body)
_RAW = [
    # ---- vector-search ---------------------------------------------------
    ("Tuning ef_search without rebuilding your index", "vector-search",
     ["tuning", "recall", "latency"], 2025, 4.6, "Karachi",
     "Of the three HNSW parameters, only ef_search can be changed on a live index. "
     "m and ef_construction are baked into the graph when the vectors are written, so "
     "changing them means a full reindex. Raising ef_search widens the candidate queue "
     "during traversal: recall climbs quickly at first, then flattens while latency keeps "
     "growing. Sweep it against a judgement set and stop at the knee of the curve."),

    ("Why your k-NN query returns fewer than k results", "vector-search",
     ["recall", "production"], 2024, 4.4, "Bengaluru",
     "A filter applied after graph traversal removes documents that were already chosen, "
     "so a selective filter silently shrinks the result set. Push the filter into the k-NN "
     "clause instead. The engine then applies it during traversal and keeps walking until it "
     "has k surviving neighbours."),

    ("Product quantization: fitting a large index in memory", "vector-search",
     ["memory", "theory"], 2025, 4.3, "Berlin",
     "Full-precision vectors dominate the memory footprint of a large index. Quantization "
     "splits each vector into subspaces and replaces every subspace with a codebook entry, "
     "trading a small, measurable recall loss for a large reduction in resident memory. "
     "The compression ratio is a deployment decision, not a default."),

    ("Choosing between the faiss and lucene engines", "vector-search",
     ["memory", "tuning"], 2026, 4.1, "Dublin",
     "The faiss engine is optimised for GPU execution and offers the wider set of "
     "quantization options. The lucene engine runs on the CPU, integrates directly with the "
     "existing segment machinery, and avoids a native memory pool sitting outside the JVM "
     "heap. On a laptop or a CPU-only node, lucene is the simpler operational story."),

    ("Approximate versus exact nearest neighbour", "vector-search",
     ["theory", "latency"], 2024, 4.0, "Toronto",
     "Exact k-NN compares the query against every stored vector, so cost grows linearly "
     "with the corpus and becomes untenable past a few hundred thousand documents. "
     "Approximate methods trade a bounded, measurable loss in recall for sublinear lookup. "
     "Below roughly ten thousand documents the exact scan is often faster than the graph."),

    ("Reading the memory circuit breaker in vector search", "vector-search",
     ["memory", "production"], 2025, 3.9, "Singapore",
     "Native HNSW graphs live outside the JVM heap in a separate cache. When that cache "
     "fills, the node raises circuit_breaking_exception and rejects queries rather than "
     "swapping. The fix is rarely a larger limit — it is usually quantization, fewer graph "
     "links, or more nodes."),

    # ---- search-internals -------------------------------------------------
    ("How an inverted index answers a query", "search-internals",
     ["retrieval", "theory"], 2023, 4.5, "Karachi",
     "A term dictionary maps each token to a postings list of the documents containing it. "
     "Answering a query means intersecting or unioning a handful of those lists, which is "
     "why lexical retrieval stays fast on corpora far too large to scan. The cost is "
     "vocabulary-bound, not corpus-bound."),

    ("BM25: term saturation and length normalisation", "search-internals",
     ["retrieval", "theory"], 2023, 4.7, "Bengaluru",
     "BM25 improves on raw term frequency in two ways. Saturation means the tenth occurrence "
     "of a word adds far less than the second. Length normalisation stops long documents "
     "from scoring highly through sheer volume. The k1 and b constants control how sharply "
     "each applies, and the defaults are rarely optimal for short catalogue text."),

    ("Analysers decide what your index can match", "search-internals",
     ["tuning", "retrieval"], 2024, 4.2, "Berlin",
     "Tokenisation, lowercasing, and stemming happen at write time, and a query can only "
     "match the tokens that survived. Mapping a field as keyword rather than text skips "
     "analysis entirely, which is correct for identifiers and filters and wrong for prose. "
     "Most puzzling zero-result queries are an analyser mismatch."),

    ("What the refresh interval actually delays", "search-internals",
     ["production", "latency"], 2025, 3.8, "Dublin",
     "An indexed document is not searchable until the next refresh builds a new segment. "
     "The default one-second gap batches that work. Forcing a refresh on every write, as "
     "convenient as it is in a notebook, turns a batched cost into a per-document one and "
     "collapses indexing throughput."),

    # ---- ranking ----------------------------------------------------------
    ("Reciprocal rank fusion, and why it uses rank", "ranking",
     ["retrieval", "theory"], 2025, 4.8, "Toronto",
     "BM25 scores are unbounded and depend on corpus statistics; cosine similarities sit in "
     "a fixed interval. Comparing them directly is meaningless and normalising them is "
     "guesswork. Fusion on rank position sidesteps the problem: each list contributes "
     "1/(k+rank), and the constant k damps the top positions so a document both lists like "
     "outranks one that a single list adores."),

    ("Cross-encoders cannot be precomputed", "ranking",
     ["retrieval", "latency"], 2026, 4.6, "Singapore",
     "A bi-encoder embeds the query and the document separately, which is what lets document "
     "vectors be written once at index time. A cross-encoder reads the pair jointly and is "
     "substantially more accurate for it, but nothing can be cached, so cost scales with the "
     "number of candidates scored. It belongs on a shortlist, never on a corpus."),

    ("Deciding how deep to rerank", "ranking",
     ["tuning", "latency"], 2025, 4.4, "Karachi",
     "Reranking depth is a straight latency-for-quality trade. Score the top twenty and you "
     "pay twenty model calls; score two hundred and quality improves marginally while the "
     "p99 becomes unacceptable. Measure where your own reranker stops changing the top five "
     "and cut there."),

    ("Boosting fields without breaking relevance", "ranking",
     ["tuning"], 2024, 3.7, "Bengaluru",
     "Weighting the title above the body encodes a reasonable prior: a term in a title is "
     "usually more about the document than the same term buried in prose. Pushed too far, the "
     "boost drowns out the body entirely and every query collapses onto title keyword "
     "matching."),

    # ---- embeddings --------------------------------------------------------
    ("The token window truncates in silence", "embeddings",
     ["production", "recall"], 2025, 4.9, "Berlin",
     "An embedding model with a 256-token window does not warn you when a document runs "
     "long. It encodes the beginning and discards the rest, and the discarded tail is often "
     "where the specifics live. Put the most identifying text first, and count tokens at "
     "ingest so the loss is visible rather than assumed."),

    ("Concatenate fields before embedding, not after", "embeddings",
     ["tuning", "theory"], 2025, 4.5, "Dublin",
     "Embedding a title and a body separately and averaging the two vectors produces "
     "something that represents neither well. Joining the fields into one string first lets "
     "the model attend across them and yields a single coherent representation. The ordering "
     "of the concatenation then matters, because of truncation."),

    ("Contrastive objectives and what negatives teach", "embeddings",
     ["theory", "evaluation"], 2026, 4.3, "Toronto",
     "Training pulls matching pairs together and pushes mismatched pairs apart. The quality "
     "of the negatives decides what the model learns: random negatives are trivially "
     "separable and teach little, while hard negatives mined from near-misses force the "
     "distinctions that retrieval actually depends on."),

    ("Query and document must share an embedding model", "embeddings",
     ["production"], 2024, 4.1, "Singapore",
     "Vectors from two different models occupy unrelated spaces, and their cosine similarity "
     "is noise. Changing the embedding model therefore means reindexing every document, not "
     "just redeploying the query path. Version the model alongside the index."),

    # ---- infrastructure -----------------------------------------------------
    ("Vector queries must fan out to every shard", "infrastructure",
     ["retrieval", "latency"], 2025, 4.2, "Karachi",
     "Sharding a lexical index lets a routing key send a query to one shard. Similarity has "
     "no such key: the nearest neighbour may live anywhere, so every shard is searched in "
     "parallel and the partial results are merged. Adding shards therefore adds coordination "
     "cost to vector queries even as it spreads the memory."),

    ("Sizing shards for a vector index", "infrastructure",
     ["memory", "tuning"], 2026, 3.9, "Bengaluru",
     "Each shard holds its own HNSW graph, so a document count that fits comfortably in one "
     "shard's cache may thrash when split across many. Size by resident graph memory rather "
     "than by document count, and remember that replicas multiply the requirement."),

    ("Running OpenSearch 2.19 as a single node", "infrastructure",
     ["production"], 2026, 3.6, "Berlin",
     "A single-node cluster reports green only when replicas are set to zero, since there is "
     "nowhere to place a copy. Disabling the security plugin is acceptable for local work "
     "and unacceptable anywhere reachable. Bind the port to loopback and treat the setup as "
     "disposable."),

    ("Embedding in the cluster with ML Commons", "infrastructure",
     ["production", "tuning"], 2026, 4.0, "Dublin",
     "Registering a model group returns a group ID, under which a model is registered and "
     "deployed into the cluster itself. An ingest pipeline can then embed documents on write, "
     "removing the client-side model. The cost is that indexing throughput now competes with "
     "query traffic for the same nodes."),

    # ---- operations ---------------------------------------------------------
    ("Splitting a latency budget across the funnel", "operations",
     ["latency", "production"], 2025, 4.7, "Toronto",
     "Give each stage a share of the p99 target and measure them separately. Retrieval is "
     "usually cheap, filtering nearly free, and reranking dominates. When the budget is "
     "missed it is almost always because the shortlist handed to the reranker grew without "
     "anyone deciding that it should."),

    ("Handling 429 rejections under load", "operations",
     ["production"], 2024, 3.5, "Singapore",
     "A 429 means a queue is full, not that the cluster is broken. Bulk indexing during peak "
     "query hours is the usual cause. Back off exponentially, shrink the bulk batch, and move "
     "the ingest window; raising the queue size mostly defers the same failure."),

    ("Building a judgement set you can trust", "operations",
     ["evaluation"], 2025, 4.5, "Karachi",
     "Relevance metrics are only as good as the labels beneath them. Sample real queries "
     "including the long tail, have two people judge each pair, and measure agreement before "
     "believing any number. A judgement set drawn only from popular queries will report "
     "improvements your users never see."),

    ("Reading NDCG without fooling yourself", "operations",
     ["evaluation", "theory"], 2026, 4.4, "Bengaluru",
     "NDCG rewards putting relevant documents near the top, discounting by position. It says "
     "nothing about documents your retrieval never surfaced, so a system with poor recall can "
     "post a healthy score. Track recall at the candidate stage alongside it."),

    ("Trimming the tail of a slow query path", "operations",
     ["latency", "tuning"], 2025, 4.6, "Berlin",
     "Median latency hides the problem; the tail is where users feel it. Profile a slow "
     "request end to end before tuning anything, because the usual culprit is an oversized "
     "candidate set flowing into an expensive stage rather than the search engine itself."),

    # ---- distractors -------------------------------------------------------
    # These use query vocabulary in an unrelated sense. They exist to make
    # lexical retrieval fail visibly: BM25 sees the shared token, the embedding
    # sees a different subject. Without competitors like these, a toy corpus
    # makes every retrieval mode look equally good.
    ("Making your CI build faster", "engineering",
     ["tuning", "production"], 2025, 3.4, "Dublin",
     "Test suites grow until someone measures them. Cache dependencies between runs, "
     "parallelise the slowest stages, and stop rebuilding container images that have not "
     "changed. Most pipelines spend the majority of their wall clock waiting rather than "
     "computing."),

    ("Filtering noise out of production dashboards", "engineering",
     ["production", "evaluation"], 2024, 3.2, "Toronto",
     "A dashboard nobody reads is worse than no dashboard. Drop panels that have never "
     "triggered an action, and filter the results down to the handful of signals that "
     "actually precede an incident. Alert fatigue is a design failure, not an operator one."),

    ("What a search team costs to run", "management",
     ["production"], 2026, 3.0, "Berlin",
     "Headcount dominates, but the infrastructure line grows faster than most plans assume "
     "once vector indices enter the picture. Budget separately for the annotation effort "
     "behind evaluation, which is recurring work rather than a one-off project cost."),

    ("Reranking the product backlog each quarter", "management",
     ["evaluation"], 2025, 2.9, "Singapore",
     "Sort by expected value rather than by whoever argued most recently. The exercise is "
     "worth the cost mainly because it forces the team to say out loud which items they have "
     "quietly stopped believing in."),

    ("Why my numbers disappeared from the weekly report", "engineering",
     ["production"], 2024, 2.8, "Karachi",
     "A silent schema change upstream is the usual explanation. The results were not deleted; "
     "the join that produced them started returning nothing after a column was renamed, and "
     "no one had a test asserting the row count."),

    ("Speeding up a slow dashboard query", "engineering",
     ["latency", "tuning"], 2025, 3.3, "Bengaluru",
     "Add the index the planner is asking for before rewriting anything by hand. Most slow "
     "analytical queries are a missing index or an accidental cross join, and no amount of "
     "clever restructuring compensates for either."),
]

_CITY_COORDS = {
    "Karachi": (24.8607, 67.0011), "Bengaluru": (12.9716, 77.5946),
    "Berlin": (52.5200, 13.4050), "Dublin": (53.3498, -6.2603),
    "Toronto": (43.6532, -79.3832), "Singapore": (1.3521, 103.8198),
}


def build() -> List[Dict]:
    docs = []
    for i, (title, category, tags, year, rating, city, body) in enumerate(_RAW):
        lat, lon = _CITY_COORDS[city]
        docs.append({
            "id": f"doc-{i:04d}", "title": title, "text": body,
            "category": category, "tags": tags, "published_year": year,
            "rating": rating, "city": city,
            "location": {"lat": lat, "lon": lon},
        })
    return docs


def probe_queries() -> List[Dict[str, str]]:
    """Queries chosen to make the retrieval modes disagree."""
    return [
        {"query": "ef_search", "expect": "lexical",
         "why": "an exact parameter name — BM25 matches the token, vectors blur it"},
        {"query": "circuit_breaking_exception", "expect": "lexical",
         "why": "a literal error string that appears in exactly one document"},
        {"query": "how do I make my search faster", "expect": "neural",
         "why": "no document uses this phrasing; the latency docs share almost no terms"},
        {"query": "why did my results disappear after filtering", "expect": "neural",
         "why": "describes a symptom; the answering document names neither word"},
        {"query": "reranking cost", "expect": "hybrid",
         "why": "split across ranking and operations — fusion beats either list"},
    ]


def write(path: Path = DATA_PATH) -> Path:
    docs = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    return path


def load(path: Path = DATA_PATH) -> List[Dict]:
    if not path.exists():
        write(path)
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    p = write()
    print(f"wrote {sum(1 for _ in p.open())} documents -> {p}")
