# OpenSearch Hybrid Search — Demo

A runnable walkthrough of the search stack covered in the **Knowledge Sharing –
Search** session (2026-08-09, Rashid Nizamani / Abdul Rafay / Shashank
Priyadarshi). Meeting notes and transcript live in [`meetings/`](meetings/).

Everything runs locally: OpenSearch in Docker, embeddings on the CPU, no cloud
services and no API keys.

---

## What it demonstrates

The retrieval funnel from the call, one stage per idea:

```
Candidate Generation  ──▶  Trim  ──▶  Rerank  ──▶  Answer
   ~100 docs              ~50        cross-encoder   top 5
   BM25 + k-NN, fused     filters    expensive       what an LLM reads
   cheap, wide            must/should  accurate
```

Each stage is more expensive per document than the last and sees fewer of them.
That ordering is the entire design.

| Concept from the call | Where it lives |
|---|---|
| Index schema, `must` / `should` filters, `geo_point` | `src/opensearch_demo/schema.py` |
| Concatenate fields before embedding; 256-token truncation | `src/opensearch_demo/embed.py` |
| HNSW `m`, `ef_construction`, `ef_search`; faiss vs lucene | `src/opensearch_demo/config.py` |
| BM25 (lexical) vs k-NN (neural) | `src/opensearch_demo/search.py` |
| Reciprocal Rank Fusion | `search.reciprocal_rank_fusion` |
| Cross-encoder reranking | `src/opensearch_demo/rerank.py` |
| The staged funnel | `src/opensearch_demo/pipeline.py` |

---

## Setup

**1. Start OpenSearch** (needs Docker Desktop running):

```bash
docker compose -f docker/docker-compose.yml up -d
curl localhost:9200/_cluster/health          # expect "status":"green"
```

Dashboards, for poking at the index by hand: <http://localhost:5601>

**2. Python environment** — needs 3.10–3.12 (torch wheels lag on 3.13+):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
```

**3. Run the notebooks**:

```bash
.venv/bin/jupyter lab
```

Start with `notebooks/01_index_and_search.ipynb`, then `02_hybrid_rrf_rerank`, then `03_evaluation` (needs the ESCI corpus).

**4. The evaluation corpus** — ESCI (Amazon Shopping Queries), 482k products with
8,956 judged test queries. This is what makes NDCG computable:

```bash
.venv/bin/python -m opensearch_demo.esci                                     # ~5 min
.venv/bin/python -m opensearch_demo.demo --dataset esci --rebuild "warm up"  # ~90 min
```

**5. The demo corpus** — one category of
[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(default: Appliances, 94k products — chosen after measuring text richness across
six candidate categories; see `docs/research/datasets.html`):

```bash
.venv/bin/python -m opensearch_demo.amazon                       # download + clean (~272 MB)
.venv/bin/python -m opensearch_demo.demo --dataset amazon --rebuild "warm up"   # embed + index (~20 min)
```

The toy corpus (`data/articles.jsonl`, 33 hand-written docs) still works offline
for a quick start — it is the default when `--dataset` is omitted.

Or headless, to check everything works end to end:

```bash
.venv/bin/python -m opensearch_demo.demo "how do I make vector search faster"
```

### Teardown and rebuild

Nothing here is precious — every byte is reproducible from this repo. Reclaim
disk after a session:

```bash
scripts/teardown.sh              # containers + index volume        (~1.1 GB)
scripts/teardown.sh --data       # + downloaded datasets            (~0.4 GB)
scripts/teardown.sh --venv       # + the virtualenv                 (~1.2 GB)
scripts/teardown.sh --images     # + the two OpenSearch images      (~2.8 GB)
scripts/teardown.sh --models     # + HuggingFace cache              (~0.6 GB)
scripts/teardown.sh --all        # everything
```

Two tiers are opt-in because they touch machine-wide state other projects share:
`--models` clears the HuggingFace cache (anything else on this machine
re-downloads its models), and `--images` removes Docker images. `--images` is
deliberately scoped to our two tags rather than `docker system prune -a`, which
would take unrelated images with it.

Coming back:

```bash
scripts/rebuild.sh            # venv + deps + cluster + toy corpus
scripts/rebuild.sh amazon     # ... with the 94k Appliances corpus
```

---

## System design & roadmap

`docs/system-design.html` is the blueprint: the full architecture with sequence
diagrams and the complete data graph, every store's contract, the measured
evaluation, how training data gets prepared (both the benchmark path and the
production methodology), and the phased roadmap — what is built, what is next
(training the retriever on ESCI), and where this converges with the GraphRAG
project.

## Understanding the code

`docs/code-walkthrough.html` walks every module in the order data moves through it —
index-time flow, query-time funnel, every tunable knob, and the reasoning behind each
decision. Read it alongside the notebooks.

## Why both `.py` and `.ipynb`

The logic is in `src/opensearch_demo/`; the notebooks import it and narrate.
Notebooks diff badly in git — they are JSON with embedded outputs, so a review
comment on a two-line change lands in an unreadable blob. Keeping the logic
importable means it can be reviewed in a PR, reused headlessly, and tested.
The notebooks stay thin: run a stage, look at the table, change a parameter,
run it again.

### Committing notebooks

Strip outputs before committing, so diffs stay reviewable:

```bash
.venv/bin/jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

---

## Two things this deliberately does not do

**Embeddings are computed client-side.** The alternative shown in the call is
ML Commons: register a model group, take its group ID, deploy the model into
the cluster, and attach an ingest pipeline so OpenSearch embeds on write. That
removes the Python-side model but ties indexing throughput to cluster
resources. The hook is marked in `ingest.prepare`. *(Rashid was going to send
the model-group code and group ID — once that arrives it slots in there.)*

**No LLM answer generation.** Stage 5 hands the final documents to a model as
context. That is the RAG session scheduled for next time, so the pipeline stops
at the ranked list.

## Security note

`docker-compose.yml` disables the OpenSearch security plugin — plain HTTP, no
auth — because this is a single-node learning setup. Ports bind to `127.0.0.1`
only. Do not reuse this compose file for anything reachable from a network.
