"""The figure and table of this folder. They use the JSON from evaluate.py and go into PAPER_INPUTS.

Each write prints if the file changed.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/: common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.config import PAPER_INPUTS
from common.report import (
    NO_RESPONSES,
    PANEL,
    PHYSICIAN,
    PHYSICIAN_VS_BAD,
    format_cell,
    read_tables_json,
    short_name,
)
from common.tikz import Cluster, booktabs_table, cluster_bar_chart, write


def existing_approaches_figure(table: dict) -> str:
    """The four existing metrics, with one bar for each rubric generator.

    The expert rubric has a bar only in the discrimination cluster. In the other clusters, the expert rubric is the reference.
    """
    generators = table["rubric_models"]
    metrics = table["metrics"]
    none_for_expert = [[]]  # no expert bar in this cluster
    matching = [
        Cluster(
            title=r"RubricRAG \\LLM-matching \\$F_1$",
            bars=none_for_expert
            + [[metrics["f1"][NO_RESPONSES][g]] for g in generators],
        ),
    ]
    return cluster_bar_chart(
        clusters=[
            *matching,
            Cluster(
                title=rf"GenRubric \\Score Correlation \\({len(table['response_models'])} responses)",
                bars=none_for_expert
                + [[metrics["score correlation"][PANEL][g]] for g in generators],
            ),
            Cluster(
                title=rf"RubricRAG \\Score Correlation \\(physician) \\$n{{=}}{table['n_physician']}$",
                bars=none_for_expert
                + [
                    [metrics["score correlation spearman"][PHYSICIAN][g]]
                    for g in generators
                ],
            ),
            Cluster(
                title=rf"RubricRAG \\Discrimination \\High/Low Quality \\(physician vs bad) \\$n{{=}}{table['n_physician']}$",
                bars=[
                    [metrics["preference alignment"][PHYSICIAN_VS_BAD][rubric]]
                    for rubric in ["true", *generators]
                ],
            ),
        ],
        series=[short_name(model=rubric) for rubric in ["true", *generators]],
        colours=["barexpert", "barone", "bartwo", "barthree"],
        series_label="rubric:",
        y_label="metric value",
        x_unit_cm=1.05,  # four clusters of four bars fit the text width
        scale=0.9,
        label="fig:existing_approaches",
    )


def pairwise_differences_table(table: dict) -> str:
    """The paired differences between generators for fig:existing_approaches.

    There is one row for each cluster of the figure and one column for each pair of generators.
    Each cell is later minus earlier, in the order of rubric_models (weakest first).
    """
    generators = table["rubric_models"]
    pairs = [
        (later, earlier)
        for i, earlier in enumerate(generators)
        for later in generators[i + 1 :]
    ]
    # (row name, metric, population), in the cluster order of the figure
    rows = [
        ("LLM-matching $F_1$", "f1", NO_RESPONSES),
        ("Score Correlation", "score correlation", PANEL),
        ("Score Correlation (physician)", "score correlation spearman", PHYSICIAN),
        (
            r"\shortstack[l]{Discrimination between \\ high/low quality responses}",
            "preference alignment",
            PHYSICIAN_VS_BAD,
        ),
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
                format_cell(cell=table["differences"][metric][population][pair])
                for pair in pairs
            ]
            for name, metric, population in rows
        ],
        group_header=None,
        label="tab:existing_differences",
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
        stem="existing_approaches",
        text=existing_approaches_figure(table=table),
        draft_order=draft_order,
    )
    write(
        directory=inputs_dir / "tables",
        stem="existing_differences",
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
            / "existing_approaches"
            / "tables.json"
        )
        inputs_dir: Path = PAPER_INPUTS
        # The number of each float in the paper is the prefix of its file name. Body floats start at 01, appendix floats at A01.
        draft_order: dict[str, str] = field(
            default_factory=lambda: {"existing_approaches": "04"}
        )
        table_order: dict[str, str] = field(
            default_factory=lambda: {"existing_differences": "A04"}
        )

    cfg = Config()
    main(
        json_path=cfg.json_path,
        inputs_dir=cfg.inputs_dir,
        draft_order=cfg.draft_order,
        table_order=cfg.table_order,
    )
