from __future__ import annotations
"""Loader for the Amazon Shopping Queries (ESCI) dataset.

Two files, both downloaded whole regardless of which slice you index:
  examples.parquet   49 MB   — every (query, product, label) judgment
  products.parquet  1.06 GB  — every product's text

The slice only decides how many products get EMBEDDED, which is where the
time and disk actually go.

Three independent filters select a slice:
  locale   us / jp / es                      -> we use `us`
  version  small_version / large_version     -> `small` is Amazon's official
                                                Task 1 subset, not a sample
  split    train / test                      -> the official benchmark division

We index BOTH splits and evaluate only on `test` queries. Indexing train-split
products gives the test queries a realistic haystack to be found in; scoring
stays clean because the split governs which QUERIES you score, not which
DOCUMENTS exist. Those extra products are unjudged rather than known-irrelevant,
so NDCG reads as a floor — standard practice, but do not compare the number to
a published leaderboard computed over a different corpus.

Labels: E(xact) / S(ubstitute) / C(omplement) / I(rrelevant).
"""
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw"

BASE = ("https://media.githubusercontent.com/media/amazon-science/esci-data"
        "/main/shopping_queries_dataset/shopping_queries_dataset_{name}.parquet")

EXAMPLES = RAW / "esci_examples.parquet"
PRODUCTS = RAW / "esci_products.parquet"
CLEAN = DATA_DIR / "esci_us_small.jsonl"
QRELS = DATA_DIR / "esci_us_small_qrels.jsonl"

INDEX_NAME = "products-esci"

# Gain values for NDCG. The paper's own weighting; 3/2/1/0 is the other common
# choice and changes absolute numbers but rarely the ranking of two systems.
LABEL_GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
LABEL_GRADED = {"E": 3, "S": 2, "C": 1, "I": 0}


def download(name: str, dest: Path) -> Path:
    """Stream one parquet from the ESCI repo's Git-LFS media host."""
    import urllib.request
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE.format(name=name)
    print(f"downloading {name} -> {dest}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.rename(dest)
    return dest


def load_examples(locale: str = "us", small: bool = True):
    """The judgment table, filtered to one locale/version."""
    import pandas as pd
    download("examples", EXAMPLES)
    df = pd.read_parquet(EXAMPLES)
    df = df[df.product_locale == locale]
    if small:
        df = df[df.small_version == 1]
    return df


def clean_record(raw: Dict) -> Optional[Dict]:
    """One products.parquet row -> an index document."""
    title = (raw.get("product_title") or "").strip()
    pid = raw.get("product_id")
    if not pid or len(title) < 3:
        return None

    def txt(key):
        v = raw.get(key)
        return v.strip() if isinstance(v, str) else ""

    return {
        "id": pid,
        "title": title,
        # bullet points are the feature list; keep them separate from the prose
        # so lexical search can weight them independently
        "features": txt("product_bullet_point"),
        "text": txt("product_description"),
        "brand": txt("product_brand") or None,
        "color": txt("product_color") or None,
        "locale": raw.get("product_locale"),
    }


def prepare_files(locale: str = "us", small: bool = True,
                  splits: tuple = ("train", "test"), force: bool = False) -> tuple:
    """Write the index-ready corpus and the qrels file. Returns both paths."""
    import pandas as pd
    if CLEAN.exists() and QRELS.exists() and not force:
        return CLEAN, QRELS

    ex = load_examples(locale, small)
    ex = ex[ex.split.isin(splits)]
    wanted = set(ex.product_id.unique())
    print(f"{len(ex):,} judgments | {ex.query_id.nunique():,} queries "
          f"| {len(wanted):,} unique products")

    download("products", PRODUCTS)
    print("reading products.parquet ...")
    # Push locale + column selection into the parquet reader: loading all
    # 1.8M rows x all columns first would peak several GB of RAM.
    prod = pd.read_parquet(
        PRODUCTS,
        columns=["product_id", "product_title", "product_description",
                 "product_bullet_point", "product_brand", "product_color",
                 "product_locale"],
        filters=[("product_locale", "==", locale)],
    )
    prod = prod[prod.product_id.isin(wanted)]

    kept = 0
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN.open("w") as fh:
        for row in prod.to_dict("records"):
            doc = clean_record(row)
            if doc:
                fh.write(json.dumps(doc) + "\n")
                kept += 1
    print(f"corpus: {kept:,} documents -> {CLEAN}")

    # qrels: only the TEST split is scored, whatever we indexed
    test = ex[ex.split == "test"]
    with QRELS.open("w") as fh:
        for r in test[["query_id", "query", "product_id", "esci_label"]].to_dict("records"):
            fh.write(json.dumps({
                "query_id": int(r["query_id"]), "query": r["query"],
                "doc_id": r["product_id"], "label": r["esci_label"],
                "gain": LABEL_GAIN[r["esci_label"]],
                "graded": LABEL_GRADED[r["esci_label"]],
            }) + "\n")
    print(f"qrels ({test.query_id.nunique():,} test queries): {QRELS}")
    return CLEAN, QRELS


def iter_corpus(limit: Optional[int] = None) -> Iterator[Dict]:
    if not CLEAN.exists():
        prepare_files()
    with CLEAN.open() as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                return
            yield json.loads(line)


def load(limit: Optional[int] = None) -> List[Dict]:
    return list(iter_corpus(limit))


def load_qrels() -> Dict[int, Dict]:
    """query_id -> {query, judgments: {doc_id: label/gain}}."""
    if not QRELS.exists():
        prepare_files()
    out: Dict[int, Dict] = {}
    with QRELS.open() as fh:
        for line in fh:
            r = json.loads(line)
            q = out.setdefault(r["query_id"],
                               {"query": r["query"], "judgments": {}})
            q["judgments"][r["doc_id"]] = r
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    prepare_files(splits=tuple(a.splits.split(",")), force=a.force)
