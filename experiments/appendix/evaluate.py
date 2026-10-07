"""Stage 2: the statistics of the appendix experiments.

This file calls no model.
Experiment 1: the matching F1 of each framing. It is the F1 of each example, then the mean, as in existing_approaches.
Experiment 2: the GenRubric score correlation on each subset of the panel. Only the responses change between subsets.
Experiment 3: how much the RubricRAG bad response decreases the expert score, for each response model. These rows are by response model, not by generator.
"""

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean as mean  # fmean is much faster than statistics.mean

sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/: common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.bootstrap import bootstrap_cells
from utils.config import REPO_ROOT

from common.metrics import fraction, framing_f1, genrubric_correlation, it_impact_share
from common.report import (
    FOREIGN_RUBRIC,
    NO_RESPONSES,
    ORDINARY,
    PANEL,
    write_tables_json,
)
from common.results import AppendixResult


@dataclass
class EvaluateConfig:
    """No field has a default. Each value comes from config.json, so a missing key raises an error."""

    results_path: Path
    json_path: Path
    response_curations: dict[
        str, list[str]
    ]  # subset name -> the panel models in the subset; the first subset is the full panel
    n_bootstrap: int
    confidence: float
    bootstrap_seed: int


def load_config(config_path: Path) -> EvaluateConfig:
    raw = json.loads(config_path.read_text())
    return EvaluateConfig(
        results_path=REPO_ROOT / raw["paths"]["results_path"],
        json_path=REPO_ROOT / raw["paths"]["json_path"],
        **raw["evaluate"],
    )


def metric_cells(
    records: list[AppendixResult],
    rubric_models: list[str],
    framings: list[str],
    response_curations: dict[str, list[str]],
    response_models: list[str],
) -> dict[tuple, dict]:
    """The table for bootstrap_cells.

    It has one row for each statistic. The rows are by generator, but the foreign-rubric rows are by response model.
    The name of each correlation row contains its subset.
    """
    values = {}
    for framing in framings:
        values[(f"{framing} f1", NO_RESPONSES)] = {
            g: mean(
                framing_f1(
                    matched=record.matches[framing][g],
                    n_true=len(record.true_rubric.criteria),
                )
                for record in records
            )
            for g in rubric_models
        }

    def correlation(curation: str, generator: str) -> float | None:
        models = response_curations[curation]
        return genrubric_correlation(
            generated=[
                [record.scores[generator][model].scores for model in models]
                for record in records
            ],
            expert=[
                [record.scores["true"][model].scores for model in models]
                for record in records
            ],
        )

    for curation in response_curations:
        values[(f"score correlation spearman {curation}", PANEL)] = {
            g: correlation(curation=curation, generator=g) for g in rubric_models
        }

    table = dict(values)

    # The effect of the bad response, for each response model.
    # The effect is the usual score minus the foreign-rubric score for each example, then the mean.
    table[("expert score", ORDINARY)] = {
        m: mean(
            fraction(score_map=record.scores["true"][m].scores) for record in records
        )
        for m in response_models
    }
    table[("expert score", FOREIGN_RUBRIC)] = {
        m: mean(
            fraction(score_map=record.bad_true_scores[m].scores) for record in records
        )
        for m in response_models
    }
    table[("foreign rubric effect", PANEL)] = {
        m: mean(
            fraction(score_map=record.scores["true"][m].scores)
            - fraction(score_map=record.bad_true_scores[m].scores)
            for record in records
        )
        for m in response_models
    }
    # The same effect as an IT fraction: the expert criteria lost, divided by the expert criteria that the usual response passed.
    table[("foreign rubric share lost", PANEL)] = {
        m: it_impact_share(
            after=[record.scores["true"][m].scores for record in records],
            before=[record.bad_true_scores[m].scores for record in records],
            headroom=[
                sum(record.scores["true"][m].scores.values()) for record in records
            ],
        )
        for m in response_models
    }

    return table


def main(cfg: EvaluateConfig) -> None:
    results: list[AppendixResult] = pickle.loads(cfg.results_path.read_bytes())
    rubric_models = list(results[0].rubrics)
    framings = list(results[0].matches)
    response_models = list(results[0].responses)
    for curation, models in cfg.response_curations.items():
        unknown = [model for model in models if model not in results[0].responses]
        assert not unknown, (
            f"curation {curation!r} names models the panel never answered with: {unknown}"
        )

    table = bootstrap_cells(
        compute=lambda records: metric_cells(
            records=records,
            rubric_models=rubric_models,
            framings=framings,
            response_curations=cfg.response_curations,
            response_models=response_models,
        ),
        records=results,
        n_bootstrap=cfg.n_bootstrap,
        confidence=cfg.confidence,
        seed=cfg.bootstrap_seed,
    )
    metrics = {}
    for (metric, population), cells in table.items():
        metrics.setdefault(metric, {})[population] = cells

    write_tables_json(
        path=cfg.json_path,
        n_examples=len(results),
        rubric_models=rubric_models,
        response_models=list(
            results[0].responses
        ),  # the full panel; the subsets are in extra
        metrics=metrics,
        differences={},
        extra={
            "framings": framings,
            "response_curations": cfg.response_curations,
        },
    )


if __name__ == "__main__":
    main(
        cfg=load_config(
            config_path=Path(sys.argv[1])
            if len(sys.argv) > 1
            else Path(__file__).parent / "config.json"
        )
    )
