"""The figure and table of this folder, for the main response model. They use the JSON from evaluate.py and go into PAPER_INPUTS.

Each write prints if the file changed.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/: common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.config import PAPER_INPUTS
from common.report import format_cell, read_tables_json, short_name
from common.tikz import Cluster, booktabs_table, cluster_bar_chart, write


def excludes_zero(cell: tuple[float, float, float]) -> bool:
    """True if the (low, point, high) interval does not contain zero."""
    low, _, high = cell
    return low > 0 or high < 0


def interventional_metrics_figure(table: dict) -> str:
    """The four interventional metrics, with one bar for each generator.

    All metrics are fractions. Thus, they use one axis.
    The two "pass" clusters and the two "fail" clusters each have a header, as in fig:intervention_grid.
    A bracket joins two generators when their paired difference in tab:it_differences does not contain zero.
    """
    response_model = table["response_model"]
    generators = table["rubric_models"]
    titles = [
        r"\footnotesize intervene on $\1C^g$ \\gain on $\1C^*$ ",
        r"\footnotesize intervene on $\1C^*$ \\gain on $\1C^g$",
        r"\footnotesize intervene on $\1C^*$ \\loss on $\1C^g$",
        r"\footnotesize intervene on $\1C^g$ \\loss on $\1C^*$",
    ]
    # (cluster, earlier bar, later bar) for each significant pair. The clusters are in the order of METRIC_NAMES.
    brackets = [
        (index, i, j)
        for index, metric in enumerate(METRIC_NAMES)
        for i, earlier in enumerate(generators)
        for j, later in enumerate(generators)
        if i < j
        and excludes_zero(
            cell=table["differences"][metric][response_model][(later, earlier)]
        )
    ]
    return cluster_bar_chart(
        clusters=[
            Cluster(
                title=title,
                bars=[
                    [table["metrics"][metric][response_model][g]] for g in generators
                ],
            )
            for title, metric in zip(titles, METRIC_NAMES)
        ],
        series=[short_name(model=g) for g in generators],
        colours=["barone", "bartwo", "barthree"],
        series_label="rubric generator:",
        y_label="$Q$",
        y_label_horizontal=True,
        x_unit_cm=1.05,  # four clusters of three bars fit the text width
        scale=0.9,
        brackets=brackets,
        cluster_groups=[("intervene to pass", 0, 1), ("intervene to fail", 2, 3)],
        label="fig:interventional_metrics",
    )


# The four metrics in the cluster order of fig:interventional_metrics, with their names in the paper.
METRIC_NAMES = {
    "optimization impact": "Training target",
    "improvement impact": "Detect good responses",
    "degradation impact": "Detect bad responses",
    "rejection detector": "Degradation flag",
}


def pairwise_differences_table(table: dict) -> str:
    """The paired differences between generators for fig:interventional_metrics.

    There is one row for each metric and one column for each pair of generators.
    Each cell is later minus earlier, in the order of rubric_models (weakest first).
    """
    response_model = table["response_model"]
    generators = table["rubric_models"]
    pairs = [
        (later, earlier)
        for i, earlier in enumerate(generators)
        for later in generators[i + 1 :]
    ]
    return booktabs_table(
        first_header="",
        columns=[
            rf"\shortstack{{{short_name(model=later)} \\ $-$ {short_name(model=earlier)}}}"
            for later, earlier in pairs
        ],
        rows=[
            [name]
            + [
                format_cell(cell=table["differences"][metric][response_model][pair])
                for pair in pairs
            ]
            for metric, name in METRIC_NAMES.items()
        ],
        group_header=None,
        label="tab:it_differences",
    )


def main(
    json_path: Path,
    inputs_dir: Path,
    draft_order: dict[str, str],
    table_order: dict[str, str],
) -> None:
    table = read_tables_json(json_path)
    write(
        directory=inputs_dir / "figures",
        stem="interventional_metrics",
        text=interventional_metrics_figure(table=table),
        draft_order=draft_order,
    )
    write(
        directory=inputs_dir / "tables",
        stem="it_differences",
        text=pairwise_differences_table(table=table),
        draft_order=table_order,
    )


if __name__ == "__main__":

    @dataclass
    class Config:
        json_path: Path = (
            Path(__file__).parent.parent
            / "out"
            / "json"
            / "interventional_metrics"
            / "gpt-5.6-terra"
            / "tables.json"
        )
        inputs_dir: Path = PAPER_INPUTS
        # The number of each float in the paper is the prefix of its file name. Body floats start at 01, appendix floats at A01.
        draft_order: dict[str, str] = field(
            default_factory=lambda: {"interventional_metrics": "05"}
        )
        table_order: dict[str, str] = field(
            default_factory=lambda: {"it_differences": "A08"}
        )

    cfg = Config()
    main(
        json_path=cfg.json_path,
        inputs_dir=cfg.inputs_dir,
        draft_order=cfg.draft_order,
        table_order=cfg.table_order,
    )
