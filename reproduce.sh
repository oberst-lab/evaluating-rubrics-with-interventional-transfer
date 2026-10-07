#!/usr/bin/env bash
# Make all generated figures, tables and prompt listings of the paper from the response cache. This takes a few minutes and makes no API calls.
# The output goes to $PAPER_INPUTS (default out/inputs/), with the same layout as the inputs/ folder of the paper.
# With the default, and if latexmk is installed, the script also makes out/figures.pdf from figures.tex.
set -euo pipefail
cd "$(dirname "$0")"

./setup.sh

# Some statistics add floats in set order, and Python changes that order in each process. A fixed seed makes each tables.json the same in each run.
export PYTHONHASHSEED=0
export PAPER_INPUTS="${PAPER_INPUTS:-$PWD/out/inputs}"
E=experiments
run() { echo "== $*"; uv run --frozen --no-dev python "$@"; }

# 1. The interventional metrics, for GPT-5.6-Terra and the eight panel models.
run $E/interventional_metrics/main.py $E/interventional_metrics/config.json

# 2. The two existing approaches, RubricRAG and GenRubric.
run $E/existing_approaches/main.py $E/existing_approaches/config.json

# 3. The single-criterion interventions. They start from the GPT-5.6-Terra run of step 1.
run $E/criteria_interventions/run.py $E/criteria_interventions/config.json
run $E/criteria_interventions/score_own.py $E/criteria_interventions/config.json
run $E/criteria_interventions/evaluate.py $E/criteria_interventions/config.json

# 4. The appendix experiments.
run $E/appendix/main.py $E/appendix/config.json

# 5. All generated figures, tables and prompt listings, into $PAPER_INPUTS.
run $E/prompts/prompts_to_latex.py
run $E/existing_approaches/figures.py
run $E/existing_approaches/matching_pass_fail.py
run $E/interventional_metrics/figures.py
run $E/criteria_interventions/figures.py
run $E/appendix/figures.py

echo "== done: the paper's generated inputs are in $PAPER_INPUTS"

# 6. One PDF with all generated figures, tables and prompts. figures.tex reads out/inputs/, so this step needs the default PAPER_INPUTS.
if [ "$PAPER_INPUTS" != "$PWD/out/inputs" ]; then
  echo "== skipped out/figures.pdf: PAPER_INPUTS is not out/inputs"
elif ! command -v latexmk > /dev/null; then
  echo "== skipped out/figures.pdf: latexmk is not installed"
else
  echo "== out/figures.pdf"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=out/latex figures.tex > /dev/null
  cp out/latex/figures.pdf out/figures.pdf
fi
