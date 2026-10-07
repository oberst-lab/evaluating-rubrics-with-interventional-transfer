"""The criterion-composition figure and the text-statistics table. They use the JSON from evaluate.py and go into PAPER_INPUTS.

write() keeps the caption that is in the file now.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/: common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.config import PAPER_INPUTS
from common.report import read_tables_json, short_name
from common.tikz import ShareBlock, ShareRow, booktabs_table, share_bars, write


def criterion_composition_figure(table: dict) -> str:
    """The criteria of each rubric, sorted in two blocks.

    Top block: the effect on the other rubric (irrelevant, matched, general). The expert criteria have one row for each generated rubric.
    Bottom block: the effect on the own rubric. The expert rubric has one row.
    The failing and passing criteria are in different panels.
    The figure shows only point estimates. The intervals are in tables.json.
    """
    generators = table["rubric_models"]

    def row(label: str, name: str, categories: list[str]) -> ShareRow:
        return ShareRow(
            label=name,
            shares=[
                [
                    table["metrics"][f"composition {c}"][state][label][1]
                    for c in categories
                ]
                for state in ("failing", "passing")
            ],
        )

    across = ["irrelevant", "matched", "general"]
    within = ["not redundant", "redundant with one", "redundant"]
    return share_bars(
        blocks=[
            ShareBlock(
                rows=[
                    ShareRow(label=r"$\1C^*$ impact via $c^g$ from", shares=[]),
                    *(
                        row(label=g, name=short_name(model=g), categories=across)
                        for g in generators
                    ),
                    ShareRow(label=r"$\1C^g$ impact via $c^*$ from", shares=[]),
                    *(
                        row(
                            label=f"true@{g}",
                            name=short_name(model=g),
                            categories=across,
                        )
                        for g in generators
                    ),
                ],
                segments=["irrelevant", "matched", "general"],
                colours=["black!22", "barone", "barthree"],
                text_colours=["black!75", "white", "white"],
            ),
            ShareBlock(
                rows=[
                    *(
                        row(label=g, name=short_name(model=g), categories=within)
                        for g in generators
                    ),
                    row(
                        label=f"true@{generators[0]}", name="expert", categories=within
                    ),
                ],
                segments=["none", "one", "several"],
                colours=["black!22", "bartwo!70", "barfour"],
                text_colours=["black!75", "white", "white"],
                legend_label="own-rubric criteria moved:",
            ),
        ],
        panels=[r"intervene to pass: $R_{+c}$", r"intervene to fail: $R_{-c}$"],
        label="fig:criterion_composition",
    )


def perturbation_text_stats_table(table: dict) -> str:
    """How much each intervention changed the words of the response, for all generators together.

    There is one row for each task of fig:interventional_metrics, in the same order.
    The first column is the mean length of a targeted criterion. The whole-rubric and single-criterion edits target the same criteria.
    Then, for each scope: the words changed, the words changed / the words of the targeted criteria, and the length of the edit / the length of R.
    The "measured" column only names the task. The text statistics do not use the measured rubric.
    """
    stats = table["text_stats"]
    tasks = [
        ("generated", "pass", r"$\1C^g$", r"$\1C^*$", r"+"),
        ("expert", "pass", r"$\1C^*$", r"$\1C^g$", r"+"),
        ("expert", "fail", r"$\1C^*$", r"$\1C^g$", r"-"),
        ("generated", "fail", r"$\1C^g$", r"$\1C^*$", r"-"),
    ]
    rows = []
    for rubric, direction, target, measured, arrow in tasks:
        cells = [
            f"${stats[f'criterion|{rubric}|{direction}']['words per target criterion']:.0f}$"
        ]
        for scope in ("whole", "criterion"):
            s = stats[f"{scope}|{rubric}|{direction}"]
            cells += [
                f"${s['changed']:.0f}$",
                f"${s['per target word']:.2f}$",
                f"${s['length']:.2f}$",
            ]
        rows.append([f"{target}, ${arrow}$", measured] + cells)
    return booktabs_table(
        first_header="target",
        columns=[
            "measured",
            r"$|c|$",
            r"$R' - R$",
            r"$\frac{R' - R}{|\1C|}$",
            r"$\frac{|R'|}{|R|}$",
            r"$R' - R$",
            r"$\frac{R' - R}{|c|}$",
            r"$\frac{|R'|}{|R|}$",
        ],
        rows=rows,
        group_header=[
            ("", 2),
            (r"whole rubric, $R_{\pm\1C}$", 3),
            (r"single criterion, $R_{\pm c}$", 3),
        ],
        label="tab:perturbation_text_stats",
    )


if __name__ == "__main__":

    @dataclass
    class Config:
        json_path: Path = (
            Path(__file__).parent.parent
            / "out"
            / "json"
            / "criteria_interventions"
            / "tables.json"
        )
        inputs_dir: Path = PAPER_INPUTS
        # The number of each float in the paper is the prefix of its file name. Body floats start at 01, appendix floats at A01.
        draft_order: dict[str, str] = field(
            default_factory=lambda: {"criterion_composition": "06"}
        )
        table_order: dict[str, str] = field(
            default_factory=lambda: {"perturbation_text_stats": "A10"}
        )

    cfg = Config()
    table = read_tables_json(cfg.json_path)
    write(
        directory=cfg.inputs_dir / "figures",
        stem="criterion_composition",
        text=criterion_composition_figure(table=table),
        draft_order=cfg.draft_order,
    )
    write(
        directory=cfg.inputs_dir / "tables",
        stem="perturbation_text_stats",
        text=perturbation_text_stats_table(table=table),
        draft_order=cfg.table_order,
    )
