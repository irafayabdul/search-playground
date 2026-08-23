from __future__ import annotations

"""Index mapping.

The call's framing: you hand-write the schema, and every field earns its place
by being something you will actually filter, score, or retrieve on. Columns
from the source DB that serve none of those purposes do not belong in the index.
"""
from typing import Any, Dict

from opensearchpy import OpenSearch

from .config import EMBED_DIM, INDEX_NAME, HNSWParams


def build_mapping(hnsw: HNSWParams | None = None) -> Dict[str, Any]:
    hnsw = hnsw or HNSWParams()
    return {
        "settings": {
            "index": {
                "knn": True,                       # enables the k-NN field type
                "knn.algo_param.ef_search": hnsw.ef_search,
                # 1 shard because this is one node. Sharding splits the index for
                # scale — but a vector query must hit *every* shard, since
                # similarity has no shard key to route on.
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                # --- retrievable / lexically searchable -----------------------
                # `text` is analysed for BM25; the `.keyword` sub-field stays raw
                # for exact-match filters and aggregations.
                "title": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "text": {"type": "text"},

                # --- filter fields -------------------------------------------
                # keyword, not text: filters need exact terms, not analysed ones.
                "category": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "published_year": {"type": "integer"},
                "rating": {"type": "float"},
                # the special data type from the call — enables geo_distance
                "location": {"type": "geo_point"},

                # --- the vector ----------------------------------------------
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBED_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": hnsw.space_type,
                        "engine": hnsw.engine,
                        "parameters": {
                            "m": hnsw.m,
                            "ef_construction": hnsw.ef_construction,
                        },
                    },
                },
            }
        },
    }


def create_index(
    client: OpenSearch,
    index: str = INDEX_NAME,
    hnsw: HNSWParams | None = None,
    recreate: bool = False,
    mapping: str = "toy",
) -> dict:
    """Create the index. `recreate=True` drops it first — the notebook uses this
    to rebuild with different HNSW params, since m/ef_construction are baked in
    at build time and cannot be changed on a live index."""
    if client.indices.exists(index=index):
        if not recreate:
            return {"acknowledged": True, "already_existed": True}
        client.indices.delete(index=index)
    builder = {"products": build_product_mapping,
               "esci": build_esci_mapping}.get(mapping, build_mapping)
    return client.indices.create(index=index, body=builder(hnsw))

def build_product_mapping(hnsw: HNSWParams | None = None) -> Dict[str, Any]:
    """Mapping for the Amazon Appliances corpus (see amazon.py).

    Same design rule as the toy mapping: every field earns its place.
    text/features are analysed for BM25; store and categories are keyword
    because filters need exact terms; price/rating are numeric for ranges.
    """
    hnsw = hnsw or HNSWParams()
    base = build_mapping(hnsw)
    base["mappings"] = {
        "properties": {
            "title": {"type": "text",
                      "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
            "features": {"type": "text"},
            "text": {"type": "text"},
            "main_category": {"type": "keyword"},
            # the category *path* — ["Appliances", "Refrigerators...", "Ice Makers"].
            # keyword on an array: a filter matches any level of the tree.
            "categories": {"type": "keyword"},
            "store": {"type": "keyword"},
            "price": {"type": "float"},
            "average_rating": {"type": "float"},
            "rating_number": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": hnsw.space_type,
                    "engine": hnsw.engine,
                    "parameters": {"m": hnsw.m,
                                   "ef_construction": hnsw.ef_construction},
                },
            },
        }
    }
    return base


def build_esci_mapping(hnsw: HNSWParams | None = None) -> Dict[str, Any]:
    """Mapping for the ESCI corpus (see esci.py).

    Thinner filter metadata than the Amazon-Reviews corpus by design: ESCI
    natively carries only brand and color. Price/rating/category require the
    ESCI-S enrichment, which is a separate 3.4 GB download and not wired here.
    """
    hnsw = hnsw or HNSWParams()
    base = build_mapping(hnsw)
    base["mappings"] = {
        "properties": {
            "title": {"type": "text",
                      "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
            "features": {"type": "text"},
            "text": {"type": "text"},
            "brand": {"type": "keyword"},
            "color": {"type": "keyword"},
            "locale": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": hnsw.space_type,
                    "engine": hnsw.engine,
                    "parameters": {"m": hnsw.m,
                                   "ef_construction": hnsw.ef_construction},
                },
            },
        }
    }
    return base
