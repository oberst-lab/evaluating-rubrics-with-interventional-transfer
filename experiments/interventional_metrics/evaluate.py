"""Stage 2: the four interventional metrics for each generator, and the paired differences between generators.

This file calls no model. With |C(R)| as the number of criteria of C that R passes, the metrics are:

    optimization  (|C*(R_do(Cg:0->1))| - |C*(R)|) / (|C*| - |C*(R)|)     expert criteria gained / expert criteria that failed
    improvement   (|Cg(R_do(C*:0->1))| - |Cg(R)|) / (|Cg| - |Cg(R)|)     generated criteria gained / generated criteria that failed
    degradation   (|Cg(R)| - |Cg(R_do(C*:1->0))|) / |Cg(R)|              generated criteria lost / generated criteria that passed
    rejection     (|C*(R)| - |C*(R_do(Cg:1->0))|) / |C*(R)|              expert criteria lost / expert criteria that passed

For degradation and rejection, `after` is the original response and `before` is the edit. Thus, a loss is a positive number.
All cells use the same bootstrap resample. Thus, the differences are paired over examples.
"""

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/: common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.bootstrap import bootstrap_cells
from utils.config import REPO_ROOT

from common.metrics import it_impact_share, pairwise_differences
from common.report import write_tables_json
from common.results import InterventionalResult


@dataclass
class EvaluateConfig:
    """No field has a default. Each value comes from config.json, so a missing key raises an error."""

    results_dir: Path  # run.py writes <results_dir>/<model>/results.pkl
    json_dir: Path  # this file writes <json_dir>/<model>/tables.json
    response_models: list[str]
    n_bootstrap: int
    confidence: float
    bootstrap_seed: int


def load_config(config_path: Path) -> EvaluateConfig:
    raw = json.loads(config_path.read_text())
    return EvaluateConfig(
        results_dir=REPO_ROOT / raw["paths"]["results_dir"],
        json_dir=REPO_ROOT / raw["paths"]["json_dir"],
        response_models=raw["run"]["response_models"],
        **raw["evaluate"],
    )


def metric_cells(
    records: list[InterventionalResult], rubric_models: list[str], response_model: str
) -> dict[tuple, dict]:
    """The table for bootstrap_cells.

    It has one row for each metric, by generator, and one ("difference", metric, population) row, by generator pair.
    """
    values = {
        ("optimization impact", response_model): {
            g: it_impact_share(
                after=[record.revised_true_scores[g].scores for record in records],
                before=[record.scores["true"].scores for record in records],
                # the expert criteria that the response failed
                headroom=[
                    len(record.scores["true"].scores)
                    - sum(record.scores["true"].scores.values())
                    for record in records
                ],
            )
            for g in rubric_models
        },
        ("improvement impact", response_model): {
            g: it_impact_share(
                after=[record.improved_scores[g].scores for record in records],
                before=[record.scores[g].scores for record in records],
                # the generated criteria that the response failed
                headroom=[
                    len(record.scores[g].scores) - sum(record.scores[g].scores.values())
                    for record in records
                ],
            )
            for g in rubric_models
        },
        ("degradation impact", response_model): {
            g: it_impact_share(
                after=[record.scores[g].scores for record in records],
                before=[record.degraded_scores[g].scores for record in records],
                # the generated criteria that the response passed
                headroom=[sum(record.scores[g].scores.values()) for record in records],
            )
            for g in rubric_models
        },
        ("rejection detector", response_model): {
            g: it_impact_share(
                after=[record.scores["true"].scores for record in records],
                before=[record.rejected_true_scores[g].scores for record in records],
                # the expert criteria that the response passed
                headroom=[
                    sum(record.scores["true"].scores.values()) for record in records
                ],
            )
            for g in rubric_models
        },
    }
    table = dict(values)
    for (metric, population), by_generator in values.items():
        table[("difference", metric, population)] = pairwise_differences(
            by_label=by_generator, order=rubric_models
        )
    return table


def main(cfg: EvaluateConfig, response_model: str) -> None:
    model_dir = response_model.replace("/", "_").replace(":", "_")
    results: list[InterventionalResult] = pickle.loads(
        (cfg.results_dir / model_dir / "results.pkl").read_bytes()
    )
    rubric_models = list(results[0].rubrics)

    table = bootstrap_cells(
        compute=lambda records: metric_cells(
            records=records, rubric_models=rubric_models, response_model=response_model
        ),
        records=results,
        n_bootstrap=cfg.n_bootstrap,
        confidence=cfg.confidence,
        seed=cfg.bootstrap_seed,
    )
    metrics, differences = {}, {}
    for key, cells in table.items():
        if key[0] == "difference":
            _, metric, population = key
            differences.setdefault(metric, {})[population] = cells
        else:
            metric, population = key
            metrics.setdefault(metric, {})[population] = cells

    write_tables_json(
        path=cfg.json_dir / model_dir / "tables.json",
        n_examples=len(results),
        rubric_models=rubric_models,
        response_models=[response_model],
        metrics=metrics,
        differences=differences,
        extra={"response_model": response_model},
    )


if __name__ == "__main__":
    cfg = load_config(
        config_path=Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "config.json"
    )
    for response_model in cfg.response_models:
        main(cfg=cfg, response_model=response_model)
