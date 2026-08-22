from __future__ import annotations
"""Loader for one category of McAuley-Lab/Amazon-Reviews-2023 product metadata.

The full dataset is ~48M products across 33 categories — far beyond a laptop.
One category is the standard move: rich text (title/features/description),
real filterable metadata (price, rating, category tree, store), local scale.

Category choice (measured on 8MB samples, 2026-08-22):
  Appliances chosen — 94k products, 68% have descriptions, 85% features,
  96% category trees, median 115 words. All_Beauty/Health were rejected for
  thin text (~22 words, no tree); Software for weak tree; the others for size.
"""
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional

HF_URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
          "/resolve/main/raw/meta_categories/meta_{category}.jsonl")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CATEGORY = "Appliances"


def raw_path(category: str = DEFAULT_CATEGORY) -> Path:
    return DATA_DIR / "raw" / f"meta_{category}.jsonl"


def clean_path(category: str = DEFAULT_CATEGORY) -> Path:
    return DATA_DIR / f"amazon_{category.lower()}.jsonl"


def download(category: str = DEFAULT_CATEGORY, force: bool = False) -> Path:
    """Stream the category metadata file from HF (hundreds of MB — be patient)."""
    dest = raw_path(category)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = HF_URL.format(category=category)
    print(f"downloading {url}\n         -> {dest}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done % (50 << 20) < (1 << 20):
                print(f"  {done / 1048576:,.0f} / {total / 1048576:,.0f} MB", file=sys.stderr)
    tmp.rename(dest)
    return dest


def _parse_price(value) -> Optional[float]:
    """Prices arrive as float, '13.99', '$13.99', or junk. None if unparseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lstrip("$").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def clean_record(raw: Dict) -> Optional[Dict]:
    """Raw metadata record -> index document, or None if unusable."""
    title = (raw.get("title") or "").strip()
    if len(title) < 3:
        return None
    doc_id = raw.get("parent_asin")
    if not doc_id:
        return None
    features = [f.strip() for f in (raw.get("features") or []) if f and f.strip()]
    description = [d.strip() for d in (raw.get("description") or []) if d and d.strip()]
    return {
        "id": doc_id,
        "title": title,
        "features": " ".join(features),
        "text": " ".join(description),
        "main_category": raw.get("main_category"),
        "categories": raw.get("categories") or [],
        "store": raw.get("store"),
        "price": _parse_price(raw.get("price")),
        "average_rating": raw.get("average_rating"),
        "rating_number": raw.get("rating_number") or 0,
    }


def iter_clean(category: str = DEFAULT_CATEGORY, limit: Optional[int] = None) -> Iterator[Dict]:
    n = 0
    with raw_path(category).open(errors="ignore") as fh:
        for line in fh:
            try:
                doc = clean_record(json.loads(line))
            except json.JSONDecodeError:
                continue
            if doc is None:
                continue
            yield doc
            n += 1
            if limit and n >= limit:
                return


def prepare_file(category: str = DEFAULT_CATEGORY, limit: Optional[int] = None,
                 force: bool = False) -> Path:
    """Download if needed, clean, write the index-ready JSONL. Returns its path."""
    out = clean_path(category)
    if out.exists() and not force and limit is None:
        return out
    download(category)
    kept = 0
    with out.open("w") as fh:
        for doc in iter_clean(category, limit=limit):
            fh.write(json.dumps(doc) + "\n")
            kept += 1
    print(f"kept {kept:,} documents -> {out}")
    return out


def load(category: str = DEFAULT_CATEGORY, limit: Optional[int] = None) -> List[Dict]:
    path = clean_path(category)
    if not path.exists():
        prepare_file(category, limit=limit)
    with path.open() as fh:
        docs = [json.loads(line) for line in fh if line.strip()]
    return docs[:limit] if limit else docs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=DEFAULT_CATEGORY)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prepare_file(args.category, limit=args.limit, force=args.force)
