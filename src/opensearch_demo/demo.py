from __future__ import annotations
"""Headless end-to-end run: build the index if needed, then search.

    python -m opensearch_demo.demo "your query here"
"""
import argparse
import sys

from . import amazon, corpus, pipeline
from .client import get_client, wait_for_cluster
from .config import INDEX_NAME, PipelineParams
from .ingest import index_docs, prepare
from .schema import create_index


def ensure_index(client, index: str = INDEX_NAME, recreate: bool = False,
                 dataset: str = "toy", category: str = amazon.DEFAULT_CATEGORY,
                 limit: int | None = None) -> int:
    if not recreate and client.indices.exists(index=index):
        count = client.count(index=index)["count"]
        if count:
            print(f"index '{index}' already holds {count} documents")
            return count

    print(f"building index '{index}' ({dataset}) ...")
    create_index(client, index=index, recreate=True,
                 mapping="products" if dataset == "amazon" else "toy")
    docs = amazon.load(category, limit=limit) if dataset == "amazon" else corpus.load()
    print(f"  embedding {len(docs)} documents ...")
    prepared = prepare(docs)
    succeeded, errors = index_docs(client, prepared, index=index)
    if errors:
        print(f"  ⚠ {len(errors)} documents failed to index", file=sys.stderr)
    print(f"  indexed {succeeded} documents")
    return succeeded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="?",
                    default="how do I make vector search faster")
    ap.add_argument("--mode", choices=["lexical", "neural", "hybrid"], default="hybrid")
    ap.add_argument("--rebuild", action="store_true", help="drop and rebuild the index")
    ap.add_argument("--dataset", choices=["toy", "amazon"], default="toy")
    ap.add_argument("--amazon-category", default=amazon.DEFAULT_CATEGORY)
    ap.add_argument("--limit", type=int, default=None,
                    help="index only the first N documents (amazon)")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--category", help="hard filter, e.g. --category vector-search")
    args = ap.parse_args()

    client = get_client()
    health = wait_for_cluster(client)
    print(f"cluster: {health['cluster_name']} / {health['status']}\n")

    index = (f"products-{args.amazon_category.lower()}"
             if args.dataset == "amazon" else INDEX_NAME)
    ensure_index(client, index=index, recreate=args.rebuild, dataset=args.dataset,
                 category=args.amazon_category, limit=args.limit)

    must = {"category": args.category} if args.category else None
    result = pipeline.run(
        client, args.query, mode=args.mode, index=index,
        params=PipelineParams(), must=must, do_rerank=not args.no_rerank,
    )

    print(f"\nquery: {result.query!r}   mode: {args.mode}")
    print("-" * 72)
    print(result.explain())
    print("-" * 72)
    for i, doc in enumerate(result.documents, start=1):
        bits = []
        if "_ce_score" in doc:
            bits.append(f"ce={doc['_ce_score']:+.3f}")
        if "_rrf_score" in doc:
            bits.append(f"rrf={doc['_rrf_score']:.4f}")
        if "_rank_delta" in doc and doc["_rank_delta"]:
            bits.append(f"moved {doc['_rank_delta']:+d}")
        print(f"{i}. {doc['title']}")
        cat = doc.get("category") or (doc.get("categories") or ["?"])[-1]
        meta = f"${doc['price']}" if doc.get("price") else ""
        if doc.get("average_rating"):
            meta += f"  ★{doc['average_rating']} ({doc.get('rating_number', 0)})"
        print(f"   {doc['id']}  [{cat}]  {meta}  {'  '.join(bits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
