# Evaluating Rubric Generation with Interventional Transfer

This repository contains the code and the cached model outputs for the paper *Evaluating Rubric Generation with Interventional Transfer* (arXiv link to come).

Run this command:

```bash
./reproduce.sh
```

The script uses only the cached responses. It makes no API calls and needs no API keys. It does these steps:

1. It downloads HealthBench.
2. It runs the code for each figure, table and prompt listing in the paper, and writes the LaTeX to `out/inputs/`.
3. It compiles `figures.tex` into `out/figures.pdf`. This PDF shows all the files in `out/inputs/`. The caption of each float is its file name. This step needs `latexmk`.

The script takes a few minutes.

## Setup

`reproduce.sh` runs `setup.sh` first, so you do not have to run `setup.sh` yourself. `setup.sh` does these steps:
1. It installs [uv](https://docs.astral.sh/uv/), if uv is not installed.
2. It makes `.venv/` from `uv.lock` (Python 3.13, with numpy, pydantic, requests and tqdm).
3. It downloads the HealthBench `oss_eval` and `hard` splits from Hugging Face into `healthbench_data/`.
4. It extracts `cache.tar.gz` into `cache/`. The archive contains the 113,878 model responses that the experiments use. Its size is 25 MB.

## The output

The name of each output file starts with the number of its float in the paper. Body floats start at 01, and appendix floats start at A01.

| Output file in `out/inputs/` | Label | Script |
|---|---|---|
| `figures/04_existing_approaches.tex` | `fig:existing_approaches` | `existing_approaches/figures.py` |
| `tables/A04_existing_differences.tex` | `tab:existing_differences` | `existing_approaches/figures.py` |
| `tables/A06_matching_pass_fail.tex` | `tab:matching_pass_fail` | `existing_approaches/matching_pass_fail.py` |
| `tables/A07_matching_cutoffs.tex` | `tab:matching_cutoffs` | `existing_approaches/matching_pass_fail.py` |
| `figures/05_interventional_metrics.tex` | `fig:interventional_metrics` | `interventional_metrics/figures.py` |
| `tables/A08_it_differences.tex` | `tab:it_differences` | `interventional_metrics/figures.py` |
| `figures/06_criterion_composition.tex` | `fig:criterion_composition` | `criteria_interventions/figures.py` |
| `tables/A10_perturbation_text_stats.tex` | `tab:perturbation_text_stats` | `criteria_interventions/figures.py` |
| `figures/A01_matching_framings.tex` | `fig:matching_framings` | `appendix/figures.py` |
| `figures/A02_response_curations.tex` | `fig:response_curations` | `appendix/figures.py` |
| `figures/A04_it_panel.tex` | `fig:it_panel` | `appendix/figures.py` |
| `tables/A05_foreign_rubric.tex` | `tab:foreign_rubric` | `appendix/figures.py` |
| `prompts/00_all.tex` to `prompts/11_score_response.tex` | | `prompts/prompts_to_latex.py` |

The scripts are in `experiments/`.
The other figures and tables of the paper are drawn manually. They explain the method and do not show results: the setting, the three approaches, the intervention grid, the survey of prior metrics, the score slices and the model choices. The model-choices table uses the values in the `config.json` files.

The code does not write captions. Each generated figure has an empty `\caption{}`. If the output file exists, the script keeps the caption that is in it. To compare the output with the paper, write the output into a copy of the `inputs/` folder of the paper:

```bash
cp -R path/to/paper/inputs /tmp/paper_inputs
PAPER_INPUTS=/tmp/paper_inputs ./reproduce.sh
diff -r path/to/paper/inputs /tmp/paper_inputs
```

`PAPER_INPUTS` must be `out/inputs/` for the script to create `out/figures.pdf`.

## Repository layout

```
reproduce.sh, setup.sh       the two entry points
figures.tex                  one document with all generated figures, tables and prompts
cache.tar.gz                 all model responses that the experiments use
dataset/load_healthbench.py  downloads HealthBench
utils/                       the data model, the response cache, the bootstrap, and the cache-only LLM stub
experiments/
  llm_functions.py           all steps that call a model: fill a prompt, read the cache, parse the answer
  prompts/                   the prompt templates of those steps, with no changes
  common/                    the metrics, the pickled result records, the parallel-call helper, and the TikZ and table writers
  interventional_metrics/    the interventional transfer metrics (the method of the paper)
  existing_approaches/       RubricRAG and GenRubric, calculated as in their papers
  criteria_interventions/    the single-criterion interventions and the composition of each rubric
  appendix/                  the matching framings, the response panel subsets, and the foreign-rubric check
```

Each experiment folder has three stages. The stages exchange data through files:

1. `run.py` makes all model calls and pickles one record for each example to `experiments/out/pickle/`.
2. `evaluate.py` calculates the statistics with paired bootstrap intervals and writes `experiments/out/json/<experiment>/tables.json`. The interventional metrics write one `tables.json` for each response model, in `experiments/out/json/interventional_metrics/<model>/`.
3. `figures.py` reads only that JSON and writes the LaTeX.

`main.py` runs the first two stages. The `config.json` in each experiment folder contains all models, sample sizes and seeds.
Git tracks the `tables.json` files and `out/inputs/`. Thus, after `./reproduce.sh`, `git status` shows if a statistic, figure, table or prompt listing changed. Git does not track `out/figures.pdf`.

The single-criterion experiment uses the pickle of the interventional-metrics experiment. Thus, `reproduce.sh` runs the experiments in that order.

## The response cache

Each model call goes through a step in `llm_functions.py`. Each step looks for its answer in `cache/` first.
The path of a response is `cache/<model>/<hash of system prompt>/<hash of message template>/<hash of filled prompt>_temp_<temperature>.txt`. Each prompt folder also contains a readable `system.txt` and `message.txt`.

The reasoning effort and the number of the judge vote are part of the cache key. The score of a response is the majority value of each criterion over a maximum of three judgements. Each judgement has its own cache entry.

In this repository, `utils/llm.py` makes no API calls. If a step does not find its answer in the cache, it raises `CacheMiss`. The published cache contains all answers.
To make new calls, for example with a different model or prompt, write `LLM.generate` in `utils/llm.py` for a provider. Each step gives it the model id, the system prompt, the message, the reasoning effort and the tool schema of the answer. The step then writes the answer to the cache.

## Data

[HealthBench](https://huggingface.co/datasets/openai/healthbench) (OpenAI, 2025). The experiments use 105 conversations from `oss_eval`: 15 from each of the seven themes, selected with a fixed seed. Conversations that are also in the `hard` split are not used. The `hard` split gives the foreign rubrics for the RubricRAG bad responses.

## License and citation

The code has the MIT License. See `LICENSE`. If you use this code or the cached responses, please cite the paper (BibTeX to come with the arXiv link).
