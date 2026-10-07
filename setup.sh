#!/usr/bin/env bash
# Set up a new clone: uv, the locked Python environment, HealthBench, and the response cache.
# You can run it again. It skips each step that is done.
set -euo pipefail
cd "$(dirname "$0")"

# 1. uv, which installs the environment in uv.lock.
if ! command -v uv > /dev/null; then
  echo "== installing uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. The Python environment, in .venv.
echo "== python environment"
uv sync --frozen --no-dev

# 3. The HealthBench oss_eval and hard splits, into healthbench_data/.
echo "== HealthBench"
uv run --frozen --no-dev python dataset/load_healthbench.py

# 4. All model responses that the paper uses, into cache/. If you set CACHE_URL, the archive comes from that URL.
if [ ! -d cache ]; then
  echo "== response cache"
  if [ -n "${CACHE_URL:-}" ]; then
    curl -L "$CACHE_URL" -o cache.tar.gz
  fi
  tar -xzf cache.tar.gz
else
  echo "== response cache: already extracted"
fi
