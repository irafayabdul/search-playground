#!/usr/bin/env bash
# Restore a working session from nothing. Complements scripts/teardown.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-toy}"     # toy | amazon | esci

[ -d .venv ] || { echo "• creating venv"; uv venv --python 3.11 .venv; }
echo "• installing dependencies"
uv pip install --python .venv/bin/python -q -e .

echo "• starting OpenSearch (first run re-pulls images, ~2.8 GB)"
docker compose -f docker/docker-compose.yml up -d
until curl -sf localhost:9200/_cluster/health >/dev/null 2>&1; do sleep 3; done
echo "  cluster up"

case "$DATASET" in
  toy)    .venv/bin/python -m opensearch_demo.demo --rebuild "warm up" ;;
  amazon) .venv/bin/python -m opensearch_demo.demo --dataset amazon --rebuild "warm up" ;;
  esci)   echo "  esci loader not wired yet — see docs/research/datasets.html" ;;
esac
echo "done. notebooks: .venv/bin/jupyter lab"
