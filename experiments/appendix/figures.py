"""The appendix figures and table of this folder. They use the JSON from evaluate.py and go into PAPER_INPUTS.

Each write prints if the file changed.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/: common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.config import PAPER_INPUTS
from common.report import (
    FOREIGN_RUBRIC,
    NO_RESPONSES,
    ORDINARY,
    PANEL,
    format_cell,
    read_tables_json,
    short_name,
)
from common.tikz import (
    Cluster,
    Panel,
    booktabs_table,
    cluster_bar_chart,
    stacked_cluster_bar_charts,
    write,
)

# The label under each cluster: the name of the framing, then its question.
QUESTIONS = {
    "match": r"Match \\measures the \\same thing",
    "simulate": r"Simulate \\would move \\with it",
    "implies": r"Implies \\logically \\entails it",
}


def matching_framings_figure(table: dict) -> str:
    """The matching F1 of our three framings, with one bar for each generator.

    Only the question changes between clusters.
    """
    generators = table["rubric_models"]
    return cluster_bar_chart(
        clusters=[
            Cluster(
                title=QUESTIONS[framing],
                bars=[
                    [table["metrics"][f"{framing} f1"][NO_RESPONSES][g]]
                    for g in generators
                ],
            )
            for framing in table["framings"]
        ],
        series=[short_name(model=g) for g in generators],
        colours=["barone", "bartwo", "barthree"],
        series_label="rubric generator:",
        y_label="matching $F_1$",
        x_unit_cm=1.3,
        placement="h",
        label="fig:matching_framings",
    )


# The label under each subset cluster. The figure adds the number of models.
CURATION_LABELS = {
    "all eight": r"all eight \\(published)",
    "open weight only": r"open \\weight only",
    "no GPT family": r"no GPT \\family",
    "drop the weakest": r"drop the \\weakest",
    "drop two weakest": r"drop two \\weakest",
    "frontier only": r"frontier \\only",
}


def response_curations_figure(
    table: dict, curations: list[str], rubric_models: list[str]
) -> str:
    """The GenRubric score correlation on six subsets of the response panel.

    Only the responses change between clusters. The subsets have different sizes.
    Each interval is for one subset. It does not tell if two subsets are different.
    """
    return cluster_bar_chart(
        clusters=[
            Cluster(
                title=rf"{CURATION_LABELS[curation]} \\{len(table['response_curations'][curation])} models",
                bars=[
                    [
                        table["metrics"][f"score correlation spearman {curation}"][
                            PANEL
                        ][g]
                    ]
                    for g in rubric_models
                ],
            )
            for curation in curations
        ],
        series=[short_name(model=g) for g in rubric_models],
        colours=["barone", "bartwo", "barthree"],
        series_label="rubric generator:",
        y_label="Spearman correlation",
        x_unit_cm=0.88,  # six clusters of three bars fit the text width
        label="fig:response_curations",
    )


def foreign_rubric_table(table: dict) -> str:
    """How much the RubricRAG bad response decreases the expert score, for each response model.

    Column 3 is the paired difference for each example, then the mean.
    Column 4 is the same decrease as an IT fraction: the passed expert criteria that the bad response lost.
    """
    return booktabs_table(
        first_header="response model",
        columns=[
            "ordinary",
            "foreign rubric",
            "difference",
            r"share of passing $\1C^*$ lost",
        ],
        rows=[
            [
                short_name(model=model),
                format_cell(cell=table["metrics"]["expert score"][ORDINARY][model]),
                format_cell(
                    cell=table["metrics"]["expert score"][FOREIGN_RUBRIC][model]
                ),
                format_cell(
                    cell=table["metrics"]["foreign rubric effect"][PANEL][model]
                ),
                format_cell(
                    cell=table["metrics"]["foreign rubric share lost"][PANEL][model]
                ),
            ]
            for model in table["response_models"]
        ],
        group_header="expert score",
        label="tab:foreign_rubric",
    )


# The four interventional metrics, in the order of fig:interventional_metrics, with the names of tab:it_differences.
TASK_AXES = {
    "optimization impact": r"Training\\target",
    "improvement impact": r"Detect good\\responses",
    "degradation impact": r"Detect bad\\responses",
    "rejection detector": r"Degradation\\flag",
}


def it_panel_figure(tables: dict[str, dict], panel: list[str]) -> str:
    """The four interventional metrics on each response model, with one panel for each metric.

    This is fig:interventional_metrics for each response model. GPT-5.6-Terra, the main response model, is last.
    The models are in the same order as in tab:foreign_rubric.
    """

    def family(model: str) -> str:
        """A short model name, so that nine names fit on one axis.

        A name with two parts keeps both parts. The second part is a version.
        """
        name = short_name(model=model)
        # The paper calls this model Terra.
        if name == "GPT-5.6-terra":
            return "Terra"
        # The full name touches the next name.
        if name == "Grok-4.20":
            return "Grok"
        # The full name touches the next name.
        if name == "Sonnet-4.6":
            return "Sonnet"
        return name if name.count("-") <= 1 else name.split("-")[0]

    return stacked_cluster_bar_charts(
        panels=[
            Panel(
                y_label=y_label,
                clusters=[
                    Cluster(
                        title=family(model=model),
                        bars=[
                            [tables[model]["metrics"][metric][model][g]]
                            for g in tables[model]["rubric_models"]
                        ],
                    )
                    for model in panel
                ],
            )
            for metric, y_label in TASK_AXES.items()
        ],
        series=[short_name(model=g) for g in tables[panel[0]]["rubric_models"]],
        colours=["barone", "bartwo", "barthree"],
        series_label="rubric generator:",
        x_unit_cm=0.53,  # nine clusters and the level y labels fit the 13.97 cm text width
        panel_height_cm=2.6,
        gap_cm=0.8,
        label="fig:it_panel",
    )


def main(
    json_path: Path,
    sweep_json_dir: Path,
    sweep_panel: list[str],
    inputs_dir: Path,
    draft_order: dict[str, str],
    table_order: dict[str, str],
) -> None:
    table = read_tables_json(json_path)
    # The figure also needs the subset definitions. Thus, we read the raw JSON too.
    table_json = json.loads(json_path.read_text())
    rubric_models = table_json["rubric_models"]
    # One tables.json for each response model, from interventional_metrics/evaluate.py.
    sweep = {
        model: read_tables_json(
            sweep_json_dir / model.replace("/", "_").replace(":", "_") / "tables.json"
        )
        for model in sweep_panel
    }
    write(
        directory=inputs_dir / "figures",
        stem="matching_framings",
        text=matching_framings_figure(table=table),
        draft_order=draft_order,
    )
    write(
        directory=inputs_dir / "figures",
        stem="response_curations",
        text=response_curations_figure(
            table={**table_json, "metrics": table["metrics"]},
            curations=list(table_json["response_curations"]),
            rubric_models=rubric_models,
        ),
        draft_order=draft_order,
    )
    write(
        directory=inputs_dir / "tables",
        stem="foreign_rubric",
        text=foreign_rubric_table(table=table),
        draft_order=table_order,
    )
    write(
        directory=inputs_dir / "figures",
        stem="it_panel",
        text=it_panel_figure(tables=sweep, panel=sweep_panel),
        draft_order=draft_order,
    )


if __name__ == "__main__":

    @dataclass
    class Config:
        json_path: Path = (
            Path(__file__).parent.parent / "out" / "json" / "appendix" / "tables.json"
        )
        sweep_json_dir: Path = (
            Path(__file__).parent.parent / "out" / "json" / "interventional_metrics"
        )
        # The eight panel models in the order of tab:foreign_rubric, then GPT-5.6-Terra.
        sweep_panel: list[str] = field(
            default_factory=lambda: [
                "workers-ai/@cf/qwen/qwen3-30b-a3b-fp8",
                "workers-ai/@cf/meta/llama-4-scout-17b-16e-instruct",
                "workers-ai/@cf/qwen/qwq-32b",
                "workers-ai/@cf/zai-org/glm-5.3-flash",
                "grok/grok-4.20-0309-non-reasoning",
                "claude-sonnet-4-6",
                "workers-ai/@cf/deepseek-ai/deepseek-v4-flash-0731",
                "gpt-5",
                "gpt-5.6-terra",
            ]
        )
        inputs_dir: Path = PAPER_INPUTS
        # The number of each float in the paper is the prefix of its file name. Body floats start at 01, appendix floats at A01.
        draft_order: dict[str, str] = field(
            default_factory=lambda: {
                "matching_framings": "A01",
                "response_curations": "A02",
                "it_panel": "A04",
            }
        )
        table_order: dict[str, str] = field(
            default_factory=lambda: {"foreign_rubric": "A05"}
        )

    cfg = Config()
    main(
        json_path=cfg.json_path,
        sweep_json_dir=cfg.sweep_json_dir,
        sweep_panel=cfg.sweep_panel,
        inputs_dir=cfg.inputs_dir,
        draft_order=cfg.draft_order,
        table_order=cfg.table_order,
    )
