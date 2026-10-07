"""Stage 2: the statistics of the existing approaches for each generator, and the paired differences between generators.

This file calls no model. It uses only the stored similarity grids and scores.
RubricRAG LLM-matching F1: their Section 3.3.1.
RubricRAG score correlation: their Table 3. It is the Spearman correlation across examples, on the physician response.
RubricRAG discrimination: their Section 3.3.3 ii. The expert rubric also gets a value.
The two physician metrics use only the examples that have a physician response.
GenRubric score correlation: their App B.1. We do not calculate their other statistics in that section.
All cells use the same bootstrap resample. Thus, the differences are paired over examples.
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

from common.metrics import (
    ScoreMap,
    genrubric_correlation,
    pairwise_differences,
    preference_alignment,
    rubricrag_correlation,
    rubricrag_precision_recall,
)
from common.report import (
    NO_RESPONSES,
    PANEL,
    PHYSICIAN,
    PHYSICIAN_VS_BAD,
    write_tables_json,
)
from common.results import ExistingApproachesResult


@dataclass
class EvaluateConfig:
    """No field has a default. Each value comes from config.json, so a missing key raises an error."""

    results_path: Path
    json_path: Path
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


def precision_recall(
    record: ExistingApproachesResult, generator: str
) -> tuple[float, float]:
    return rubricrag_precision_recall(
        grid=record.similarity[generator], n_true=len(record.true_rubric.criteria)
    )


def rubric_rag_f1(records: list[ExistingApproachesResult], generator: str) -> float:
    """The F1 of each example, then the mean over examples, as in the paper.

    This is not 2PR/(P+R) of the mean precision and recall.
    F1 is 0 when precision and recall are both 0.
    """
    f1s = []
    for record in records:
        precision, recall = precision_recall(record=record, generator=generator)
        f1s.append(
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return mean(f1s)


def panel_scores(
    record: ExistingApproachesResult, rubric: str, response_models: list[str]
) -> list[ScoreMap]:
    """The scores of `rubric` on the panel responses, in the order of `response_models`."""
    return [record.scores[rubric][model].scores for model in response_models]


def metric_cells(
    records: list[ExistingApproachesResult],
    rubric_models: list[str],
    response_models: list[str],
) -> dict[tuple, dict]:
    """The table for bootstrap_cells.

    It has one row for each statistic, by generator, and one ("difference", statistic, population) row, by generator pair.
    """
    expert = [
        panel_scores(record=record, rubric="true", response_models=response_models)
        for record in records
    ]
    correlation = {}
    for generator in rubric_models:
        generated = [
            panel_scores(
                record=record, rubric=generator, response_models=response_models
            )
            for record in records
        ]
        correlation[generator] = genrubric_correlation(
            generated=generated, expert=expert
        )

    # Select the examples with a physician response in each replicate. The bootstrap thus resamples all examples.
    with_physician = [
        record for record in records if record.physician_scores is not None
    ]
    expert_physician = [
        record.physician_scores["true"].scores for record in with_physician
    ]

    values = {
        ("f1", NO_RESPONSES): {
            g: rubric_rag_f1(records=records, generator=g) for g in rubric_models
        },
        ("score correlation", PANEL): correlation,
        ("score correlation spearman", PHYSICIAN): {
            g: None
            if not with_physician
            else rubricrag_correlation(
                generated=[
                    record.physician_scores[g].scores for record in with_physician
                ],
                expert=expert_physician,
            )
            for g in rubric_models
        },
        # This metric uses one rubric at a time. Thus, the expert rubric also gets a value.
        ("preference alignment", PHYSICIAN_VS_BAD): {
            rubric: preference_alignment(
                good=[
                    record.physician_scores[rubric].scores for record in with_physician
                ],
                bad=[record.bad_scores[rubric].scores for record in with_physician],
            )
            for rubric in ["true", *rubric_models]
        },
    }
    table = dict(values)
    for (metric, population), by_generator in values.items():
        table[("difference", metric, population)] = pairwise_differences(
            by_label=by_generator, order=rubric_models
        )
    return table


def main(cfg: EvaluateConfig) -> None:
    results: list[ExistingApproachesResult] = pickle.loads(
        cfg.results_path.read_bytes()
    )
    rubric_models = list(results[0].rubrics)
    response_models = list(results[0].responses)

    table = bootstrap_cells(
        compute=lambda records: metric_cells(
            records=records,
            rubric_models=rubric_models,
            response_models=response_models,
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
        path=cfg.json_path,
        n_examples=len(results),
        rubric_models=rubric_models,
        response_models=response_models,
        metrics=metrics,
        differences=differences,
        extra={
            "n_physician": sum(
                record.physician_scores is not None for record in results
            )
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
