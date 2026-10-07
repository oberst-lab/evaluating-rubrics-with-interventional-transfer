"""Stage 1: all model calls for the four interventional metrics, for one response model. Pickles one InterventionalResult for each conversation.

    optimization    do(C^g: 0 -> 1), measured on the expert rubric      `generated_rubric_rewrites(edit="revise")`
    rejection       do(C^g: 1 -> 0), measured on the expert rubric      `generated_rubric_rewrites(edit="degrade")`
    degradation     do(C^*: 1 -> 0), measured on the generated rubric   `intervene_on_expert_rubric(edit="degrade")`
    improvement     do(C^*: 0 -> 1), measured on the generated rubric   `intervene_on_expert_rubric(edit="revise")`

Each intervention is one whole-rubric edit. The response model edits its own response.
We keep each edit, also when it does not move the target score. Only a refusal removes an example.
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
from utils.llm import LLM

from common.concurrency import parallel_map
from common.rubrics import generated_rubrics
from common.interventions import intervene_on_expert_rubric
from llm_functions import (
    ScoreResponseOutput,
    generate_response,
    edit_response,
    score_response,
)
from common.results import InterventionalResult


@dataclass
class RunConfig:
    """No field has a default. Each value comes from config.json, so a missing key raises an error."""

    typical_data_path: Path
    hard_data_path: Path
    model_reasoning_effort: dict[
        str, str
    ]  # reasoning effort of each generator and response model; a model that is not in the map uses its default
    cache_dir: Path
    results_dir: Path  # each response model writes <results_dir>/<model>/results.pkl
    per_theme: int  # examples from each HealthBench theme
    sample_seed: int  # the seed of the shuffle in each theme
    response_models: list[
        str
    ]  # the first is the main response model; each model writes its response and does all edits of it
    rubric_gen_models: list[
        str
    ]  # weakest first; evaluate.py calculates "later minus earlier"
    convert_model: str  # writes the negative expert criteria as positive
    convert_reasoning_effort: str
    scoring_model: str  # the judge
    scoring_reasoning_effort: str
    temperature: float | None
    judge_samples: int  # judgements for each score; the score is the majority, so use an odd number
    skip_prompt_ids: list[str]  # examples that a model refused
    max_workers: int  # parallel calls in one example


def load_config(config_path: Path) -> RunConfig:
    """The paths in config.json are relative to the repository root."""
    raw = json.loads(config_path.read_text())
    return RunConfig(
        typical_data_path=REPO_ROOT / raw["paths"]["typical_data_path"],
        hard_data_path=REPO_ROOT / raw["paths"]["hard_data_path"],
        cache_dir=REPO_ROOT / raw["paths"]["cache_dir"],
        results_dir=REPO_ROOT / raw["paths"]["results_dir"],
        **raw["run"],
    )


def score_every_rubric(
    conversation: str,
    response: str,
    by_name: dict[str, Rubric],
    llm: LLM,
    cacher: Cacher,
    scoring_model: str,
    scoring_reasoning_effort: str,
    judge_samples: int,
    temperature: float | None,
    max_workers: int,
) -> dict[str, ScoreResponseOutput] | None:
    """The scores of each rubric on one response. "true" is the expert rubric."""
    steps = parallel_map(
        work=list(by_name),
        task=lambda name: (
            name,
            score_response(
                conversation=conversation,
                response=response,
                rubric=by_name[name],
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
    scores = {}
    for name, result in steps:
        if result is None:
            print(f"{'-' * 50}\n scoring refusal: {name}\n{'-' * 50}")
            return None
        scores[name] = result
    return scores


def generated_rubric_rewrites(
    edit: str,
    name: str,
    example: Example,
    index: int,
    response: str,
    rubrics: dict[str, Rubric],
    scores: dict[str, ScoreResponseOutput],
    true_rubric: Rubric,
    llm: LLM,
    cacher: Cacher,
    response_model: str,
    scoring_model: str,
    scoring_reasoning_effort: str,
    judge_samples: int,
    temperature: float | None,
    max_workers: int,
) -> tuple[dict[str, str], dict[str, ScoreResponseOutput]] | None:
    """For each generator, edit the response for the generated rubric, then score the edit with the expert rubric.

    edit="revise" gives do(C^g: 0 -> 1). edit="degrade" gives do(C^g: 1 -> 0).
    There is one edit for each generator.
    `name` is only for the log lines.
    """
    revise_steps = parallel_map(
        work=list(rubrics),
        task=lambda generator: (
            generator,
            edit_response(
                edit=edit,
                conversation=example.conversation,
                response=response,
                rubric=rubrics[generator],
                scores=scores[generator].scores,
                response_model=response_model,
                llm=llm,
                cacher=cacher,
                temperature=temperature,
            ).result,
        ),
        max_workers=max_workers,
    )

    revised = {}
    for generator, result in revise_steps:
        if result is None:
            print(
                f"{'-' * 50}\n {name} refusal: {index} {example.prompt_id}: against {generator}\n{'-' * 50}"
            )
            return None
        revised[generator] = result

    score_steps = parallel_map(
        work=list(revised),
        task=lambda generator: (
            generator,
            score_response(
                conversation=example.conversation,
                response=revised[generator],
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

    revised_true_scores = {}
    for generator, result in score_steps:
        if result is None:
            print(
                f"{'-' * 50}\n {name} scoring refusal: {index} {example.prompt_id}: against {generator}\n{'-' * 50}"
            )
            return None
        revised_true_scores[generator] = result

    return revised, revised_true_scores


def main(cfg: RunConfig, response_model: str) -> list[InterventionalResult]:

    # oss_eval.jsonl also contains the hard examples. We remove them.
    # A skipped refusal is replaced by the next example in its theme.
    hard_prompt_ids = {
        example.prompt_id for example in load_dataset(cfg.hard_data_path)
    }
    examples = load_dataset(
        cfg.typical_data_path,
        exclude_prompt_ids=hard_prompt_ids,
        skip_prompt_ids=set(cfg.skip_prompt_ids),
        per_theme=cfg.per_theme,
        sample_seed=cfg.sample_seed,
    )

    llm = LLM(model_reasoning_effort=cfg.model_reasoning_effort)
    cacher = Cacher(cache_dir=cfg.cache_dir)

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
        by_name = {"true": true_rubric, **rubrics}

        response_step = generate_response(
            conversation=example.conversation,
            response_model=response_model,
            llm=llm,
            cacher=cacher,
            temperature=cfg.temperature,
        )
        if response_step.result is None:
            print(
                f"{'-' * 50}\n response refusal: {index} {example.prompt_id}: {response_model}\n{'-' * 50}"
            )
            continue
        response = response_step.result

        scores = score_every_rubric(
            conversation=example.conversation,
            response=response,
            by_name=by_name,
            llm=llm,
            cacher=cacher,
            scoring_model=cfg.scoring_model,
            scoring_reasoning_effort=cfg.scoring_reasoning_effort,
            judge_samples=cfg.judge_samples,
            temperature=cfg.temperature,
            max_workers=cfg.max_workers,
        )
        if scores is None:
            continue

        revised = generated_rubric_rewrites(
            edit="revise",
            name="revise",
            example=example,
            index=index,
            response=response,
            rubrics=rubrics,
            scores=scores,
            true_rubric=true_rubric,
            llm=llm,
            cacher=cacher,
            response_model=response_model,
            scoring_model=cfg.scoring_model,
            scoring_reasoning_effort=cfg.scoring_reasoning_effort,
            judge_samples=cfg.judge_samples,
            temperature=cfg.temperature,
            max_workers=cfg.max_workers,
        )
        if revised is None:
            continue
        revised_responses, revised_true_scores = revised

        rejected = generated_rubric_rewrites(
            edit="degrade",
            name="reject",
            example=example,
            index=index,
            response=response,
            rubrics=rubrics,
            scores=scores,
            true_rubric=true_rubric,
            llm=llm,
            cacher=cacher,
            response_model=response_model,
            scoring_model=cfg.scoring_model,
            scoring_reasoning_effort=cfg.scoring_reasoning_effort,
            judge_samples=cfg.judge_samples,
            temperature=cfg.temperature,
            max_workers=cfg.max_workers,
        )
        if rejected is None:
            continue
        rejected_responses, rejected_true_scores = rejected

        # The two interventions on the expert rubric. Each gives one edited response, which all generated rubrics then score.
        perturbed = {}
        for name, edit in [("degraded", "degrade"), ("improved", "revise")]:
            step = intervene_on_expert_rubric(
                edit=edit,
                conversation=example.conversation,
                response=response,
                true_rubric=true_rubric,
                true_scores=scores["true"],
                response_model=response_model,
                llm=llm,
                cacher=cacher,
                scoring_model=cfg.scoring_model,
                scoring_reasoning_effort=cfg.scoring_reasoning_effort,
                temperature=cfg.temperature,
                judge_samples=cfg.judge_samples,
            )
            if step is None:
                print(
                    f"{'-' * 50}\n {name} refusal: {index} {example.prompt_id}\n{'-' * 50}"
                )
                break
            perturbed_response, _ = step  # we do not use the expert scores here
            perturbed_scores = score_every_rubric(
                conversation=example.conversation,
                response=perturbed_response,
                by_name=by_name,
                llm=llm,
                cacher=cacher,
                scoring_model=cfg.scoring_model,
                scoring_reasoning_effort=cfg.scoring_reasoning_effort,
                judge_samples=cfg.judge_samples,
                temperature=cfg.temperature,
                max_workers=cfg.max_workers,
            )
            if perturbed_scores is None:
                break
            perturbed[name] = (perturbed_response, perturbed_scores)
        if len(perturbed) < 2:
            continue

        results.append(
            InterventionalResult(
                example=example,
                true_rubric=true_rubric,
                rubrics=rubrics,
                response=response,
                scores=scores,
                revised_responses=revised_responses,
                revised_true_scores=revised_true_scores,
                rejected_responses=rejected_responses,
                rejected_true_scores=rejected_true_scores,
                degraded_response=perturbed["degraded"][0],
                degraded_scores=perturbed["degraded"][1],
                improved_response=perturbed["improved"][0],
                improved_scores=perturbed["improved"][1],
            )
        )

    results_path = (
        cfg.results_dir
        / response_model.replace("/", "_").replace(":", "_")
        / "results.pkl"
    )
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
    for response_model in cfg.response_models:
        main(cfg=cfg, response_model=response_model)
