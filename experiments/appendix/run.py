"""Stage 1: all model calls for the appendix experiments. Pickles one AppendixResult for each conversation.

Experiment 1: our LLM-matching with three framings of the question (match, simulate, implies). Only the question changes.
Experiment 2: the GenRubric score correlation on subsets of the response panel. evaluate.py selects the subsets.
Experiment 3: the RubricRAG bad response from each panel model, scored with the expert rubric.
Our framings are different from RubricRAG LLM-matching. They show the conversation and both full rubrics in one call.
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
from utils.data_model import load_dataset, Example, Rubric
from utils.cacher import Cacher
from utils.llm import StepOutput, LLM

from common.concurrency import parallel_map
from common.rubrics import gen_rubric_panel, generated_rubrics
from llm_functions import (
    ask_llm_map_rubrics,
    convert_negative_criteria,
    respond_with_rubric,
    score_response,
)
from common.results import AppendixResult


@dataclass
class RunConfig:
    """No field has a default. Each value comes from config.json, so a missing key raises an error."""

    typical_data_path: Path
    hard_data_path: Path
    model_reasoning_effort: dict[
        str, str
    ]  # reasoning effort of each generator and response model; a model that is not in the map uses its default
    cache_dir: Path
    results_path: Path
    per_theme: int  # examples from each HealthBench theme
    sample_seed: int  # the seed of the shuffle in each theme
    convert_model: str  # writes the negative expert criteria as positive
    convert_reasoning_effort: str
    rubric_gen_models: list[
        str
    ]  # weakest first; evaluate.py calculates "later minus earlier"
    match_model: str  # answers all three framings
    match_reasoning_effort: str
    response_models: list[
        str
    ]  # the full GenRubric panel; evaluate.py selects the subsets
    foreign_rubric_prompt_id: str  # the hard-split example whose expert rubric all bad responses use; the same for all examples
    scoring_model: str  # the judge
    scoring_reasoning_effort: str
    judge_samples: int  # judgements for each score; the score is the majority, so use an odd number
    temperature: float | None
    skip_prompt_ids: list[str]  # examples that a model refused
    max_workers: int  # parallel calls in one example


def load_config(config_path: Path) -> RunConfig:
    """The paths in config.json are relative to the repository root."""
    raw = json.loads(config_path.read_text())
    return RunConfig(
        typical_data_path=REPO_ROOT / raw["paths"]["typical_data_path"],
        hard_data_path=REPO_ROOT / raw["paths"]["hard_data_path"],
        cache_dir=REPO_ROOT / raw["paths"]["cache_dir"],
        results_path=REPO_ROOT / raw["paths"]["results_path"],
        **raw["run"],
    )


def matching_framings(
    example: Example,
    index: int,
    true_rubric: Rubric,
    rubrics: dict[str, Rubric],
    llm: LLM,
    cacher: Cacher,
    match_model: str,
    match_reasoning_effort: str,
    temperature: float | None,
    max_workers: int,
) -> dict[str, dict[str, dict[int, list[int]]]] | None:
    """The answer of each framing for each generator: the expert criteria that each generated criterion matches.

    There is one call for each (framing, generator) pair. evaluate.py calculates precision, recall and F1.
    None if one call fails. Then all framings use the same examples.
    """
    # The three framings, in the order of the figure. The most strict is last.
    # Each pair is (the name in the results, the framing of ask_llm_map_rubrics).
    work = [
        (name, framing, generator)
        for name, framing in [
            ("match", "match_rubric"),
            ("simulate", "simulate_intervention"),
            ("implies", "implies"),
        ]
        for generator in rubrics
    ]
    answered = parallel_map(
        work=work,
        task=lambda item: (
            item,
            ask_llm_map_rubrics(
                framing=item[1],
                conversation=example.conversation,
                rubric=rubrics[item[2]],
                true_rubric=true_rubric,
                llm=llm,
                cacher=cacher,
                match_model=match_model,
                temperature=temperature,
                reasoning_effort=match_reasoning_effort,
            ).result,
        ),
        max_workers=max_workers,
    )

    matches = {}
    for (name, _, generator), result in answered:
        if result is None:
            print(
                f"{'-' * 50}\n {name} refusal: {index} {example.prompt_id}: {generator}\n{'-' * 50}"
            )
            return None
        matches.setdefault(name, {})[generator] = result.as_indices()
    return matches


def foreign_rubric_panel(
    example: Example,
    index: int,
    true_rubric: Rubric,
    foreign_rubric: Rubric,
    llm: LLM,
    cacher: Cacher,
    response_models: list[str],
    scoring_model: str,
    scoring_reasoning_effort: str,
    judge_samples: int,
    temperature: float | None,
    max_workers: int,
):
    """The RubricRAG bad response from each panel model, and the expert scores on it.

    All examples use the same foreign rubric.
    evaluate.py compares these scores with the scores of the usual panel responses from gen_rubric_panel.
    None if a model refuses. Then all models use the same examples.
    """
    bad_responses = dict(
        parallel_map(
            work=response_models,
            task=lambda model: (
                model,
                respond_with_rubric(
                    conversation=example.conversation,
                    rubric=foreign_rubric,
                    response_model=model,
                    llm=llm,
                    cacher=cacher,
                    temperature=temperature,
                ).result,
            ),
            max_workers=max_workers,
        )
    )
    refused = [model for model, bad in bad_responses.items() if bad is None]
    if refused:
        print(
            f"{'-' * 50}\n bad response refusal: {index} {example.prompt_id}: {refused}\n{'-' * 50}"
        )
        return None

    scored = parallel_map(
        work=response_models,
        task=lambda model: (
            model,
            score_response(
                conversation=example.conversation,
                response=bad_responses[model],
                rubric=true_rubric,
                llm=llm,
                cacher=cacher,
                scoring_model=scoring_model,
                temperature=temperature,
                reasoning_effort=scoring_reasoning_effort,
                samples=judge_samples,
            ).result,
        ),
        max_workers=max_workers,
    )
    bad_true_scores = {}
    for model, result in scored:
        if result is None:
            print(
                f"{'-' * 50}\n bad scoring refusal: {index} {example.prompt_id}: {model}\n{'-' * 50}"
            )
            return None
        bad_true_scores[model] = result
    return bad_responses, bad_true_scores


def main(cfg: RunConfig) -> list[AppendixResult]:

    # oss_eval.jsonl also contains the hard examples. We remove them.
    # A skipped refusal is replaced by the next example in its theme.
    hard_examples = load_dataset(cfg.hard_data_path)
    examples = load_dataset(
        cfg.typical_data_path,
        exclude_prompt_ids={example.prompt_id for example in hard_examples},
        skip_prompt_ids=set(cfg.skip_prompt_ids),
        per_theme=cfg.per_theme,
        sample_seed=cfg.sample_seed,
    )

    llm = LLM(model_reasoning_effort=cfg.model_reasoning_effort)
    cacher = Cacher(cache_dir=cfg.cache_dir)

    # One foreign rubric for all examples. It comes from the hard split, so it is never the rubric of the conversation.
    foreign = next(
        example
        for example in hard_examples
        if example.prompt_id == cfg.foreign_rubric_prompt_id
    )
    foreign_step: StepOutput[Rubric | None] = convert_negative_criteria(
        criteria=[c.text for c in foreign.true_criteria],
        negative_indices=[
            j for j, c in enumerate(foreign.true_criteria) if c.points < 0
        ],
        llm=llm,
        cacher=cacher,
        convert_model=cfg.convert_model,
        temperature=cfg.temperature,
        reasoning_effort=cfg.convert_reasoning_effort,
    )
    assert foreign_step.result is not None, (
        "the foreign rubric's negative criteria could not be restated"
    )
    foreign_rubric = foreign_step.result

    results = []
    for index, example in enumerate(tqdm(examples)):
        rubrics_step = generated_rubrics(
            example=example,
            index=index,
            llm=llm,
            cacher=cacher,
            convert_model=cfg.convert_model,
            convert_reasoning_effort=cfg.convert_reasoning_effort,
            rubric_gen_models=cfg.rubric_gen_models,
            temperature=cfg.temperature,
        )
        if rubrics_step is None:
            continue
        true_rubric, rubrics = rubrics_step

        matches = matching_framings(
            example=example,
            index=index,
            true_rubric=true_rubric,
            rubrics=rubrics,
            llm=llm,
            cacher=cacher,
            match_model=cfg.match_model,
            match_reasoning_effort=cfg.match_reasoning_effort,
            temperature=cfg.temperature,
            max_workers=cfg.max_workers,
        )
        if matches is None:
            continue

        panel = gen_rubric_panel(
            example=example,
            index=index,
            true_rubric=true_rubric,
            rubrics=rubrics,
            llm=llm,
            cacher=cacher,
            response_models=cfg.response_models,
            scoring_model=cfg.scoring_model,
            scoring_reasoning_effort=cfg.scoring_reasoning_effort,
            judge_samples=cfg.judge_samples,
            temperature=cfg.temperature,
            max_workers=cfg.max_workers,
        )
        if panel is None:
            continue
        responses, scores = panel

        bad = foreign_rubric_panel(
            example=example,
            index=index,
            true_rubric=true_rubric,
            foreign_rubric=foreign_rubric,
            llm=llm,
            cacher=cacher,
            response_models=cfg.response_models,
            scoring_model=cfg.scoring_model,
            scoring_reasoning_effort=cfg.scoring_reasoning_effort,
            judge_samples=cfg.judge_samples,
            temperature=cfg.temperature,
            max_workers=cfg.max_workers,
        )
        if bad is None:
            continue
        bad_responses, bad_true_scores = bad

        results.append(
            AppendixResult(
                example=example,
                true_rubric=true_rubric,
                rubrics=rubrics,
                matches=matches,
                responses=responses,
                scores=scores,
                bad_responses=bad_responses,
                bad_true_scores=bad_true_scores,
            )
        )

    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.results_path.write_bytes(pickle.dumps(results))
    print(f"wrote {len(results)} examples to {cfg.results_path}")
    return results


if __name__ == "__main__":
    main(
        cfg=load_config(
            config_path=Path(sys.argv[1])
            if len(sys.argv) > 1
            else Path(__file__).parent / "config.json"
        )
    )
