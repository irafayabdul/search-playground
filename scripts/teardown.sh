#!/usr/bin/env bash
# Reclaim disk after a working session. Everything removed here is reproducible
# from the repo — see scripts/rebuild.sh.
#
# Safe by default: stops containers and drops the OpenSearch volume only.
# Each extra tier is opt-in, because the last two touch things OTHER projects
# on this machine may share.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=0; MODELS=0; IMAGES=0; VENV=0
for arg in "$@"; do
  case "$arg" in
    --data)   DATA=1 ;;
    --models) MODELS=1 ;;
    --images) IMAGES=1 ;;
    --venv)   VENV=1 ;;
    --all)    DATA=1; MODELS=1; IMAGES=1; VENV=1 ;;
    -h|--help)
      sed -n '2,10p' "$0"; echo
      echo "Usage: $0 [--data] [--models] [--images] [--venv] [--all]"
      exit 0 ;;
    *) echo "unknown flag: $arg (try --help)"; exit 1 ;;
  esac
done

before=$(df -k / | tail -1 | awk '{print $4}')
echo "── tearing down ──────────────────────────────"

# always: containers + index volume
echo "• stopping containers, dropping index volume"
docker compose -f docker/docker-compose.yml down -v 2>/dev/null || true

if [ "$DATA" = 1 ]; then
  echo "• removing downloaded datasets (data/raw, generated jsonl)"
  rm -rf data/raw
  rm -f data/amazon_*.jsonl data/esci_*.jsonl
  # data/articles.jsonl is the 17 KB toy corpus and is tracked in git — keep it.
fi

if [ "$MODELS" = 1 ]; then
  echo "• removing HuggingFace model cache"
  echo "  ⚠ this cache is shared machine-wide — other projects re-download after this"
  read -r -p "  proceed? [y/N] " ok
  [ "${ok:-n}" = "y" ] && rm -rf ~/.cache/huggingface || echo "  skipped"
fi

if [ "$IMAGES" = 1 ]; then
  echo "• removing the OpenSearch images used by this project"
  # Scoped on purpose: `docker system prune -a` would take unrelated images too.
  docker rmi opensearchproject/opensearch:2.19.0 \
             opensearchproject/opensearch-dashboards:2.19.0 2>/dev/null || true
  echo '  note: docker system prune -a reclaims more, but affects other projects'
fi

if [ "$VENV" = 1 ]; then
  echo "• removing .venv"
  rm -rf .venv
fi

after=$(df -k / | tail -1 | awk '{print $4}')
echo "──────────────────────────────────────────────"
awk -v b="$before" -v a="$after" \
  'BEGIN { printf "reclaimed: %.2f GB   (free now: %.1f GB)\n", (a-b)/1048576, a/1048576 }'
echo "rebuild any time with: scripts/rebuild.sh"
