"""All model calls for the single-criterion interventions, for one direction. Pickles one CriteriaResult for each example.

    direction "pass"   do(c: 0 -> 1) for each failing criterion
    direction "fail"   do(c: 1 -> 0) for each passing criterion

An edit on a generated criterion is scored with the expert rubric. An edit on an expert criterion is scored with all generated rubrics.
The input is the interventional_metrics pickle of the main response model.
A "pass" edit uses edit_response(edit="revise"). The prompt shows the target criterion and the criteria that the response satisfies.
A "fail" edit uses edit_response(edit="degrade"). The prompt shows the target criterion and the criteria that the response fails.
The prompt does not show the other criteria of the rubric.
"""

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/: llm_functions, common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from tqdm import tqdm

from utils.config import REPO_ROOT
from utils.cacher import Cacher
from utils.llm import LLM

from common.concurrency import parallel_map
from llm_functions import edit_response, score_response
from common.results import CriteriaResult, InterventionalResult


@dataclass
class RunConfig:
    """No field has a default. Each value comes from config.json, so a missing key raises an error."""

    model_reasoning_effort: dict[str, str]
    cache_dir: Path
    source_results_path: Path
    results_dir: Path  # each direction writes <results_dir>/results_<direction>.pkl
    generators: list[str]
    response_model: str  # the model that wrote R; it also does the edits
    scoring_model: str
    scoring_reasoning_effort: str
    judge_samples: int
    directions: list[str]  # "pass", "fail" or both
    temperature: float | None
    max_workers: int


def load_config(config_path: Path) -> RunConfig:
    raw = json.loads(config_path.read_text())
    return RunConfig(
        cache_dir=REPO_ROOT / raw["paths"]["cache_dir"],
        source_results_path=REPO_ROOT / raw["paths"]["source_results_path"],
        results_dir=REPO_ROOT / raw["paths"]["results_dir"],
        **raw["run"],
    )


def targets_of(
    source: InterventionalResult, generators: list[str], direction: str
) -> list[tuple[str, int]]:
    """(rubric name, criterion number) for each criterion that R fails ("pass") or passes ("fail"). "true" is the expert rubric."""
    start = {"pass": 0, "fail": 1}[direction]
    return [
        (name, index)
        for name in ["true", *generators]
        for index, score in sorted(source.scores[name].scores.items())
        if score == start
    ]


def main(cfg: RunConfig, direction: str) -> list[CriteriaResult]:
    sources: list[InterventionalResult] = pickle.loads(
        cfg.source_results_path.read_bytes()
    )
    indices = list(range(len(sources)))

    llm = LLM(model_reasoning_effort=cfg.model_reasoning_effort)
    cacher = Cacher(cache_dir=cfg.cache_dir)

    results = []
    for source_index in tqdm(indices):
        source = sources[source_index]
        conversation = source.example.conversation
        rubrics = {
            "true": source.true_rubric,
            **{g: source.rubrics[g] for g in cfg.generators},
        }

        def rewrite(target: tuple[str, int]) -> tuple[tuple[str, int], str | None]:
            name, index = target
            on_r = source.scores[name].scores
            # The prompt shows the target and the criteria that are already in the target state.
            kept = 1 if direction == "pass" else 0
            return target, edit_response(
                edit="revise" if direction == "pass" else "degrade",
                conversation=conversation,
                response=source.response,
                rubric=rubrics[name],
                scores={i: s for i, s in on_r.items() if s == kept} | {index: 1 - kept},
                response_model=cfg.response_model,
                llm=llm,
                cacher=cacher,
                temperature=cfg.temperature,
            ).result

        rewrites = {}
        for target, result in parallel_map(
            work=targets_of(
                source=source, generators=cfg.generators, direction=direction
            ),
            task=rewrite,
            max_workers=cfg.max_workers,
        ):
            if result is None:
                print(
                    f"{'-' * 50}\n rewrite refusal: {source_index} {source.example.prompt_id}: {target}\n{'-' * 50}"
                )
                continue
            rewrites[target] = result

        # The expert rubric scores an edit on a generated criterion. All generated rubrics score an edit on an expert criterion.
        readings = [
            (target, measured)
            for target in rewrites
            for measured in (cfg.generators if target[0] == "true" else ["true"])
        ]

        def score(reading: tuple[tuple[str, int], str]):
            target, measured = reading
            return reading, score_response(
                conversation=conversation,
                response=rewrites[target],
                rubric=rubrics[measured],
                llm=llm,
                cacher=cacher,
                scoring_model=cfg.scoring_model,
                temperature=cfg.temperature,
                reasoning_effort=cfg.scoring_reasoning_effort,
                samples=cfg.judge_samples,
            ).result

        scored = {}
        for reading, result in parallel_map(
            work=readings, task=score, max_workers=cfg.max_workers
        ):
            if result is None:
                print(
                    f"{'-' * 50}\n scoring refusal: {source_index} {source.example.prompt_id}: {reading}\n{'-' * 50}"
                )
                continue
            scored[reading] = result

        # Keep a target only if it has all its scores. A refusal removes only that criterion.
        on_generated = {g: {} for g in cfg.generators}
        on_expert = {}
        for (name, index), text in rewrites.items():
            if name == "true":
                by_generator = {
                    g: scored.get(((name, index), g)) for g in cfg.generators
                }
                if all(v is not None for v in by_generator.values()):
                    on_expert[index] = (text, by_generator)
            elif ((name, index), "true") in scored:
                on_generated[name][index] = (text, scored[((name, index), "true")])

        results.append(
            CriteriaResult(
                prompt_id=source.example.prompt_id,
                source_index=source_index,
                on_generated=on_generated,
                on_expert=on_expert,
            )
        )

    results_path = cfg.results_dir / f"results_{direction}.pkl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_bytes(pickle.dumps(results))
    print(f"wrote {len(results)} examples to {results_path}")
    return results


if __name__ == "__main__":
    cfg = load_config(
        config_path=Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "config.json"
    )
    for direction in cfg.directions:
        main(cfg=cfg, direction=direction)
