"""The composition of each rubric from the single-criterion interventions, and how much the edits changed the response text.

This file calls no model.
Each target goes into a category by the number of criteria of the other rubric that moved with it: irrelevant (0), matched (1) or general (more than 1).
We split the targets by their start state on R, which is also the direction of the edit.
A target whose own edit did not flip it is in no category.
We sort the expert criteria once for each generated rubric.
Redundancy uses the same counts in the own rubric of the target: not redundant (0), redundant with one (1) or redundant (more than 1).
"""

import difflib
import json
import pickle
import sys
from statistics import fmean
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/: llm_functions, common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.bootstrap import bootstrap_cells
from utils.config import REPO_ROOT

from common.report import write_tables_json
from common.results import CriteriaResult, InterventionalResult

COMPOSITION = [
    "irrelevant",
    "matched",
    "general",
    "not redundant",
    "redundant with one",
    "redundant",
]
STATES = {
    "pass": "failing",
    "fail": "passing",
}  # direction -> the state on R of its targets


def composition_label(side: str, generator: str) -> str:
    """The row of a target: the generator for a generated criterion, "true@<generator>" for an expert criterion sorted against that generated rubric.

    We use @ because write_tables_json uses | in difference keys.
    """
    return generator if side == "generated" else f"true@{generator}"


@dataclass
class EvaluateConfig:
    """No field has a default. Each value comes from config.json, so a missing key raises an error."""

    source_results_path: Path
    results_dir: Path  # run.py and score_own.py write results_<direction>.pkl and results_<direction>_own.pkl here
    json_path: Path
    generators: list[str]  # weakest first
    response_model: str
    n_bootstrap: int
    confidence: float
    bootstrap_seed: int


def load_config(config_path: Path) -> EvaluateConfig:
    raw = json.loads(config_path.read_text())
    return EvaluateConfig(
        source_results_path=REPO_ROOT / raw["paths"]["source_results_path"],
        results_dir=REPO_ROOT / raw["paths"]["results_dir"],
        json_path=REPO_ROOT / raw["paths"]["json_path"],
        generators=raw["run"]["generators"],
        response_model=raw["run"]["response_model"],
        **raw["evaluate"],
    )


def moved_with(
    before: dict[int, int],
    after: dict[int, int],
    start: int,
    exclude: int | None = None,
) -> int:
    """The number of criteria that went from `start` to 1 - start."""
    return sum(
        1
        for b, v in before.items()
        if b != exclude and v == start and after[b] == 1 - start
    )


def composition_counts(
    source: InterventionalResult,
    by_direction: dict[str, tuple[CriteriaResult, dict]],
    generators: list[str],
) -> dict:
    """The counts of one example: (state, label) -> [targets, edit failed, irrelevant, matched, general, not redundant, redundant with one, redundant]."""
    counts = {}
    for direction, state in STATES.items():
        record, own = by_direction[direction]
        for g in generators:
            for side in ("generated", "expert"):
                tally = counts.setdefault(
                    (state, composition_label(side=side, generator=g)), [0] * 8
                )
                if side == "generated":
                    items = [
                        ((g, i), scores.scores, "true")
                        for i, (_, scores) in record.on_generated[g].items()
                    ]
                else:
                    items = [
                        (("true", j), by_generator[g].scores, g)
                        for j, (_, by_generator) in record.on_expert.items()
                    ]
                for (target_rubric, index), other_after, other_rubric in items:
                    own_after = own[(target_rubric, index)].scores
                    start = source.scores[target_rubric].scores[index]
                    if own_after[index] != 1 - start:
                        row = [1, 1, 0, 0, 0, 0, 0, 0]
                    else:
                        across = moved_with(
                            before=source.scores[other_rubric].scores,
                            after=other_after,
                            start=start,
                        )
                        within = moved_with(
                            before=source.scores[target_rubric].scores,
                            after=own_after,
                            start=start,
                            exclude=index,
                        )
                        row = [
                            1,
                            0,
                            int(across == 0),
                            int(across == 1),
                            int(across > 1),
                            int(within == 0),
                            int(within == 1),
                            int(within > 1),
                        ]
                    tally[:] = [a + b for a, b in zip(tally, row)]
    return counts


def composition_cells(records: list[dict], generators: list[str]) -> dict[tuple, dict]:
    """(category, state) -> label -> fraction.

    Each fraction is of the targets whose edit flipped them. Thus, the three categories add to one, and the three redundancy levels add to one.
    """
    totals = {}
    for record in records:
        for key, tally in record.items():
            totals[key] = [a + b for a, b in zip(totals.get(key, [0] * 8), tally)]
    table = {}
    for state in STATES.values():
        for category in COMPOSITION:
            cells = {}
            for g in generators:
                for side in ("generated", "expert"):
                    label = composition_label(side=side, generator=g)
                    (
                        targets,
                        failed,
                        irrelevant,
                        matched,
                        general,
                        alone,
                        with_one,
                        redundant,
                    ) = totals[(state, label)]
                    worked = targets - failed
                    value = {
                        "irrelevant": irrelevant,
                        "matched": matched,
                        "general": general,
                        "not redundant": alone,
                        "redundant with one": with_one,
                        "redundant": redundant,
                    }.get(category)
                    cells[label] = value / worked if worked else None
            table[(category, state)] = cells
    return table


def words(text: str) -> list[str]:
    return text.split()


def words_changed(before: str, after: str) -> int:
    """The words deleted plus the words added to change `before` into `after`, from a word diff.

    This is an edit distance, not a change in length. A new sentence of the same length thus counts as a change.
    """
    matcher = difflib.SequenceMatcher(a=words(before), b=words(after), autojunk=False)
    return sum(
        (i2 - i1) + (j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def text_stats(
    sources: list[InterventionalResult], by_direction: dict, generators: list[str]
) -> dict:
    """How much each type of intervention changed the words of the response, for all generators together.

    There are eight rows: (whole rubric or single criterion) x (generated or expert) x (pass or fail).
    The targeted criteria of an edit are the criteria that it must flip. A whole-rubric edit with no targeted criteria is not used.
    `length`: the mean of (words in edit) / (words in response).
    `changed`: the mean words changed in each edit.
    `per target word`: the total words changed / the total words of targeted criteria.
    There is no bootstrap. These values describe the edits.
    """
    rows: dict[
        tuple[str, str, str], list[tuple[str, str, list[str]]]
    ] = {}  # (scope, rubric, direction) -> [(response, edit, targeted criteria)]

    def add(
        scope: str,
        rubric: str,
        direction: str,
        response: str,
        rewrite: str,
        targeted: list[str],
    ) -> None:
        if targeted:
            rows.setdefault((scope, rubric, direction), []).append(
                (response, rewrite, targeted)
            )

    for index, source in enumerate(sources):
        R = source.response

        def state(name: str, start: int) -> list[str]:
            rubric = source.true_rubric if name == "true" else source.rubrics[name]
            return [
                rubric.criteria[i]
                for i, v in sorted(source.scores[name].scores.items())
                if v == start
            ]

        for g in generators:
            add(
                "whole",
                "generated",
                "pass",
                R,
                source.revised_responses[g],
                state(name=g, start=0),
            )
            add(
                "whole",
                "generated",
                "fail",
                R,
                source.rejected_responses[g],
                state(name=g, start=1),
            )
        add(
            "whole",
            "expert",
            "pass",
            R,
            source.improved_response,
            state(name="true", start=0),
        )
        add(
            "whole",
            "expert",
            "fail",
            R,
            source.degraded_response,
            state(name="true", start=1),
        )
        for direction, (interventions, _) in by_direction.items():
            record = interventions[index]
            for g in generators:
                for i, (rewrite, _) in record.on_generated[g].items():
                    add(
                        "criterion",
                        "generated",
                        direction,
                        R,
                        rewrite,
                        [source.rubrics[g].criteria[i]],
                    )
            # Count each edit on an expert criterion one time, not one time for each generated rubric.
            for j, (rewrite, _) in record.on_expert.items():
                add(
                    "criterion",
                    "expert",
                    direction,
                    R,
                    rewrite,
                    [source.true_rubric.criteria[j]],
                )

    stats = {}
    for (scope, rubric, direction), items in rows.items():
        changed = [words_changed(before=r, after=w) for r, w, _ in items]
        stats[f"{scope}|{rubric}|{direction}"] = {
            "length": fmean(len(words(w)) / len(words(r)) for r, w, _ in items),
            "changed": fmean(changed),
            "per target word": sum(changed)
            / sum(len(words(c)) for _, _, targeted in items for c in targeted),
            "words per target criterion": fmean(
                len(words(c)) for _, _, targeted in items for c in targeted
            ),
        }
    return stats


def main(cfg: EvaluateConfig) -> None:
    sources: list[InterventionalResult] = pickle.loads(
        cfg.source_results_path.read_bytes()
    )
    by_direction = {}
    for direction in STATES:
        interventions = {
            r.source_index: r
            for r in pickle.loads(
                (cfg.results_dir / f"results_{direction}.pkl").read_bytes()
            )
        }
        own = {
            o.source_index: o.own
            for o in pickle.loads(
                (cfg.results_dir / f"results_{direction}_own.pkl").read_bytes()
            )
        }
        by_direction[direction] = (interventions, own)
    composition_records = [
        composition_counts(
            source=source,
            by_direction={
                d: (interventions[i], own[i])
                for d, (interventions, own) in by_direction.items()
            },
            generators=cfg.generators,
        )
        for i, source in enumerate(sources)
    ]
    metrics = {}
    composition = bootstrap_cells(
        compute=lambda resample: composition_cells(
            records=resample, generators=cfg.generators
        ),
        records=composition_records,
        n_bootstrap=cfg.n_bootstrap,
        confidence=cfg.confidence,
        seed=cfg.bootstrap_seed,
    )
    for (category, state), cells in composition.items():
        metrics.setdefault(f"composition {category}", {})[state] = cells

    write_tables_json(
        path=cfg.json_path,
        n_examples=len(sources),
        rubric_models=cfg.generators,
        response_models=[cfg.response_model],
        metrics=metrics,
        differences={},
        extra={
            "response_model": cfg.response_model,
            "text_stats": text_stats(
                sources=sources,
                by_direction={
                    d: (interventions, None)
                    for d, (interventions, _) in by_direction.items()
                },
                generators=cfg.generators,
            ),
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
