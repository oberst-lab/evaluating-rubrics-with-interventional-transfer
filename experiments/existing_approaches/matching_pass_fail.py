"""Does the LLM-matching similarity of a generated criterion depend on whether the response passes it?

Each generated criterion gets its best similarity to an expert criterion, on the 0 to 9 scale.
We split the criteria by their score on the GPT-5.6-Terra response.
The similarities come from the existing_approaches run. The scores come from the interventional_metrics run.
tab:matching_pass_fail: the mean similarity of passing and failing criteria, and the difference.
tab:matching_cutoffs: the fraction of passing and failing criteria at or above each cutoff.
"""

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean as mean

sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/: common, llm_functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.config import PAPER_INPUTS
from utils.bootstrap import bootstrap_cells

from common.report import format_cell, short_name
from common.results import ExistingApproachesResult, InterventionalResult
from common.tikz import booktabs_table, write

PICKLES = Path(__file__).parent.parent / "out" / "pickle"


def split_similarities(
    matching: ExistingApproachesResult,
    intervention: InterventionalResult,
    generator: str,
) -> tuple[list[int], list[int]]:
    """(passing, failing): the best similarity of each generated criterion, split by its score on the response."""
    passed = intervention.scores[generator].scores
    best = {
        index: max(by_true.values())
        for index, by_true in matching.similarity[generator].items()
    }
    return [s for index, s in best.items() if passed[index] == 1], [
        s for index, s in best.items() if passed[index] == 0
    ]


def pass_fail_table(records: list[dict], generators: list[str], cells: dict) -> str:
    rows = []
    for generator in generators:
        passing = [s for record in records for s in record[generator][0]]
        failing = [s for record in records for s in record[generator][1]]
        rows.append(
            [
                short_name(model=generator),
                f"{mean(passing):.2f}",
                f"{len(passing)}",
                f"{mean(failing):.2f}",
                f"{len(failing)}",
                format_cell(cell=cells[generator]["difference"]),
            ]
        )
    return booktabs_table(
        first_header="rubric generator",
        columns=["passing", "$n$", "failing", "$n$", "passing $-$ failing"],
        rows=rows,
        group_header=None,
        label="tab:matching_pass_fail",
    )


def cutoffs_table(
    records: list[dict], generators: list[str], cutoffs: list[int]
) -> str:
    def share(values: list[int], cutoff: int) -> str:
        return rf"{round(100 * sum(v >= cutoff for v in values) / len(values))}\%"

    pooled = {
        g: (
            [s for r in records for s in r[g][0]],
            [s for r in records for s in r[g][1]],
        )
        for g in generators
    }
    return booktabs_table(
        first_header="similarity $\\geq$",
        columns=[short_name(model=g) for g in generators],
        rows=[
            [f"{cutoff}"]
            + [
                f"{share(pooled[g][0], cutoff)} / {share(pooled[g][1], cutoff)}"
                for g in generators
            ]
            for cutoff in cutoffs
        ],
        group_header="passing / failing",
        label="tab:matching_cutoffs",
    )


def main(
    tables_dir: Path,
    draft_order: dict[str, str],
    cutoffs: list[int],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> None:
    matching = {
        r.example.prompt_id: r
        for r in pickle.loads(
            (PICKLES / "existing_approaches" / "results.pkl").read_bytes()
        )
    }
    intervention = {
        r.example.prompt_id: r
        for r in pickle.loads(
            (
                PICKLES / "interventional_metrics" / "gpt-5.6-terra" / "results.pkl"
            ).read_bytes()
        )
    }
    assert set(matching) == set(intervention), (
        "the two runs must cover the same examples"
    )
    generators = list(next(iter(matching.values())).rubrics)
    for prompt_id in matching:
        for generator in generators:
            assert (
                matching[prompt_id].rubrics[generator].criteria
                == intervention[prompt_id].rubrics[generator].criteria
            ), f"{generator} rubric differs between runs on {prompt_id}"

    records = [
        {
            g: split_similarities(
                matching=matching[p], intervention=intervention[p], generator=g
            )
            for g in generators
        }
        for p in sorted(matching)
    ]

    def compute(sample: list[dict]) -> dict[str, dict[str, float | None]]:
        """The mean is over all criteria. Thus, an example with more criteria has more weight."""
        table = {}
        for generator in generators:
            passing = [s for record in sample for s in record[generator][0]]
            failing = [s for record in sample for s in record[generator][1]]
            table[generator] = {
                "difference": mean(passing) - mean(failing)
                if passing and failing
                else None
            }
        return table

    cells = bootstrap_cells(
        compute=compute,
        records=records,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    write(
        directory=tables_dir,
        stem="matching_pass_fail",
        text=pass_fail_table(records=records, generators=generators, cells=cells),
        draft_order=draft_order,
    )
    write(
        directory=tables_dir,
        stem="matching_cutoffs",
        text=cutoffs_table(records=records, generators=generators, cutoffs=cutoffs),
        draft_order=draft_order,
    )


if __name__ == "__main__":

    @dataclass
    class Config:
        tables_dir: Path = PAPER_INPUTS / "tables"
        # The number of each table in the paper is the prefix of its file name.
        draft_order: dict[str, str] = field(
            default_factory=lambda: {
                "matching_pass_fail": "A06",
                "matching_cutoffs": "A07",
            }
        )
        cutoffs: list[int] = field(default_factory=lambda: list(range(1, 10)))
        # The same values as the evaluate block of existing_approaches/config.json.
        n_bootstrap: int = 1000
        confidence: float = 0.95
        seed: int = 0

    cfg = Config()
    main(
        tables_dir=cfg.tables_dir,
        draft_order=cfg.draft_order,
        cutoffs=cfg.cutoffs,
        n_bootstrap=cfg.n_bootstrap,
        confidence=cfg.confidence,
        seed=cfg.seed,
    )
