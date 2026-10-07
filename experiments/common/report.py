"""The JSON file that each evaluate.py writes and each figures.py reads, and the model names.

All evaluate.py files write the same shape: metric -> population -> label -> [low, point, high].
"""

import json
import re
from pathlib import Path

NO_RESPONSES = "no responses"  # a metric that uses only the rubric text
PANEL = "panel"  # one response from each panel model
PHYSICIAN = "physician"  # the HealthBench physician response
PHYSICIAN_VS_BAD = (
    "physician vs bad"  # the physician response and the RubricRAG bad response
)
ORDINARY = "ordinary"  # the usual answer of each panel model
FOREIGN_RUBRIC = "foreign rubric"  # the answer of each panel model when told to satisfy the rubric of a different example

Cell = tuple[float, float, float] | None
Cells = dict[str, Cell]  # label (a generator or a response model) -> (low, point, high)
DifferenceCells = dict[
    tuple[str, str], Cell
]  # (later, earlier) -> (low, later - earlier, high)


def format_cell(cell: Cell) -> str:
    """Write a cell as "point (low, high)". Write `--` for None."""
    if cell is None:
        return "--"
    left, point, right = cell
    return f"{point:.2f} ({left:.2f}, {right:.2f})"


def short_name(model: str) -> str:
    """The name of a model as the paper shows it. For example, claude-sonnet-4-6 -> Sonnet-4.6 and "true" -> expert.

    Only the last part of the model id is used. The prefix tells the provider, not the model.
    """
    if model == "true":
        return "expert"
    name = model.split("/")[-1]
    # Short names that the paper uses.
    paper_name = {
        "qwen3-30b-a3b-fp8": "Qwen",
        "grok-4.20-0309-non-reasoning": "Grok-4.20",
    }.get(name)
    if paper_name:
        return paper_name
    if name.startswith("claude-"):
        family, *version = name.replace("claude-", "").split("-")
        return (
            f"{family.capitalize()}-{'.'.join(version)}"
            if version
            else family.capitalize()
        )
    if name.startswith("gpt-"):
        return "GPT-" + name[len("gpt-") :]
    # Open-weight ids: llama-4-scout-17b-16e-instruct -> Llama-4-Scout-17B. Remove the quantization, expert count and date parts.
    parts = [
        p
        for p in name.replace("-instruct", "").split("-")
        if not re.fullmatch(r"\d+e|fp8|awq|int4|int8|it|maas|\d{4}", p)
    ]
    return "-".join(
        p.upper() if re.fullmatch(r"\d+b|a\d+b", p) else p.capitalize() for p in parts
    )


def write_tables_json(
    path: Path,
    n_examples: int,
    rubric_models: list[str],
    response_models: list[str],
    metrics: dict[str, dict[str, Cells]],
    differences: dict[str, dict[str, DifferenceCells]],
    extra: dict,
) -> None:
    """Write the tables to JSON.

    `metrics` and `differences` are metric -> population -> cells. The population tells which responses the metric uses.
    A tuple key (later, earlier) becomes the string "later|earlier".
    `extra` holds other values for the figures, such as a second sample size.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n_examples": n_examples,
                "rubric_models": rubric_models,
                "response_models": response_models,
                "metrics": metrics,
                "differences": {
                    metric: {
                        population: {
                            f"{later}|{earlier}": cell
                            for (later, earlier), cell in cells.items()
                        }
                        for population, cells in by_population.items()
                    }
                    for metric, by_population in differences.items()
                },
                **extra,
            },
            indent=2,
        )
    )
    print(f"wrote {path}")


def read_tables_json(path: Path) -> dict:
    """Read a file from write_tables_json. Cells become tuples again, and difference keys become (later, earlier)."""
    raw = json.loads(path.read_text())
    raw["metrics"] = {
        metric: {
            population: {
                label: None if cell is None else tuple(cell)
                for label, cell in cells.items()
            }
            for population, cells in by_population.items()
        }
        for metric, by_population in raw["metrics"].items()
    }
    raw["differences"] = {
        metric: {
            population: {
                tuple(key.split("|")): None if cell is None else tuple(cell)
                for key, cell in cells.items()
            }
            for population, cells in by_population.items()
        }
        for metric, by_population in raw["differences"].items()
    }
    return raw
