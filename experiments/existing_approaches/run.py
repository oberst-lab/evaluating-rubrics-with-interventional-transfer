"""Stage 1: all model calls for the existing approaches. Pickles one ExistingApproachesResult for each conversation.

All four metrics use the same examples and the same generated rubrics.
RubricRAG LLM-matching (Dhole and Agichtein, arXiv 2603.20882): see `rubric_rag_similarities`.
RubricRAG score correlation on physician responses: see `physician_response_scores`.
RubricRAG discrimination between good and bad responses: see `bad_response_scores`.
GenRubric score correlation (Chen et al., arXiv 2608.29856): see `gen_rubric_panel`.
The docstring of each function gives its differences from the paper.
"""

import json
import pickle
import random
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
    ScoreResponseOutput,
    ask_rubric_rag_similarity,
    convert_negative_criteria,
    respond_with_rubric,
    score_response,
)
from common.results import ExistingApproachesResult


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
    scoring_model: str  # the judge
    scoring_reasoning_effort: str
    judge_samples: int  # judgements for each score; the score is the majority, so use an odd number
    response_models: list[
        str
    ]  # GenRubric: the panel, in the order of the score vectors
    similarity_model: (
        str  # RubricRAG: gives each criterion pair a similarity from 0 to 9
    )
    similarity_reasoning_effort: str
    similarity_max_workers: int  # parallel similarity calls in one example
    bad_response_model: str  # RubricRAG: writes the bad response. Their Section 3.4 uses Qwen3-30B-A3B-Instruct-2507
    foreign_rubric_seed: int  # RubricRAG: selects the rubric of a different example for each bad response
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


def rubric_rag_similarities(
    example: Example,
    index: int,
    true_rubric: Rubric,
    rubrics: dict[str, Rubric],
    llm: LLM,
    cacher: Cacher,
    similarity_model: str,
    similarity_reasoning_effort: str,
    temperature: float | None,
    max_workers: int,
) -> dict[str, dict[int, dict[int, int]]] | None:
    """RubricRAG LLM-matching: the similarity of each (generated criterion, expert criterion) pair in one example.

    As in the paper, there is one call for each pair.
    The grid keeps the integers from 0 to 9. evaluate.py divides by 9.
    Differences from the paper:
        1. The similarity model is `similarity_model`. Theirs is Qwen3-4B-Instruct-2507.
        2. The expert rubric has positive criteria only. See generated_rubrics.
        3. The generated rubrics come from our generate_rubric prompt.
    None if a pair has no answer. The caller then removes the example.
    """
    pairs = [
        (generator, generated_index, true_index)
        for generator, rubric in rubrics.items()
        for generated_index in range(len(rubric.criteria))
        for true_index in range(len(true_rubric.criteria))
    ]

    scored = parallel_map(
        work=pairs,
        task=lambda pair: (
            pair,
            ask_rubric_rag_similarity(
                true_criterion=true_rubric.criteria[pair[2]],
                generated_criterion=rubrics[pair[0]].criteria[pair[1]],
                llm=llm,
                cacher=cacher,
                similarity_model=similarity_model,
                temperature=temperature,
                reasoning_effort=similarity_reasoning_effort,
            ).result,
        ),
        max_workers=max_workers,
    )

    grid = {}
    for (generator, generated_index, true_index), score in scored:
        if score is None:
            print(
                f"{'-' * 50}\n similarity refusal: {index} {example.prompt_id}: {generator} G{generated_index} vs T{true_index}\n{'-' * 50}"
            )
            return None
        grid.setdefault(generator, {}).setdefault(generated_index, {})[true_index] = (
            score
        )
    return grid


def physician_response_scores(
    example: Example,
    index: int,
    true_rubric: Rubric,
    rubrics: dict[str, Rubric],
    llm: LLM,
    cacher: Cacher,
    scoring_model: str,
    scoring_reasoning_effort: str,
    judge_samples: int,
    temperature: float | None,
    max_workers: int,
) -> dict[str, ScoreResponseOutput] | None:
    """RubricRAG score correlation: the score of each rubric on the HealthBench physician response.

    evaluate.py calculates the correlation across examples.
    Differences from the paper:
        1. The score is the fraction of satisfied criteria, with no point weights. Our generated rubrics have no points.
        2. The judge is `scoring_model`. Theirs is Qwen3-4B-Instruct-2507.
    The caller calls this function only for examples with a physician response.
    None if the judge refuses. The caller then removes the example.
    """
    by_name = {"true": true_rubric, **rubrics}
    scored = parallel_map(
        work=list(by_name),
        task=lambda rubric_name: (
            rubric_name,
            score_response(
                conversation=example.conversation,
                response=example.ideal_completion,
                rubric=by_name[rubric_name],
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
    for rubric_name, result in scored:
        if result is None:
            print(
                f"{'-' * 50}\n physician scoring refusal: {index} {example.prompt_id}: {rubric_name}\n{'-' * 50}"
            )
            return None
        scores[rubric_name] = result
    return scores


def bad_response_scores(
    example: Example,
    index: int,
    foreign_example: Example,
    true_rubric: Rubric,
    rubrics: dict[str, Rubric],
    llm: LLM,
    cacher: Cacher,
    convert_model: str,
    convert_reasoning_effort: str,
    bad_response_model: str,
    scoring_model: str,
    scoring_reasoning_effort: str,
    judge_samples: int,
    temperature: float | None,
    max_workers: int,
) -> tuple[str, dict[str, ScoreResponseOutput]] | None:
    """RubricRAG discrimination: one bad response to this conversation, and the score of each rubric on it.

    As in their Section 3.4, a model answers the conversation and tries to satisfy the expert rubric of a different example.
    evaluate.py calculates how often a rubric gives the physician response a higher score.
    We make the foreign rubric positive first, as all other rubrics.
    `foreign_example` comes from the hard split. Thus, it is never one of the examples that we score.
    Differences from the paper:
        1. The expert rubric has positive criteria only, as in generated_rubrics.
        2. The score is the fraction of satisfied criteria, with no point weights.
        3. The judge is `scoring_model`. Theirs is Qwen3-4B-Instruct-2507.
    None if a step fails. The caller then removes the example.
    """
    foreign_step: StepOutput[Rubric | None] = convert_negative_criteria(
        criteria=[c.text for c in foreign_example.true_criteria],
        negative_indices=[
            j for j, c in enumerate(foreign_example.true_criteria) if c.points < 0
        ],
        llm=llm,
        cacher=cacher,
        convert_model=convert_model,
        temperature=temperature,
        reasoning_effort=convert_reasoning_effort,
    )
    if foreign_step.result is None:
        print(
            f"{'-' * 50}\n foreign convert refusal: {index} {example.prompt_id}: {foreign_example.prompt_id}\n{'-' * 50}"
        )
        return None

    bad_step: StepOutput[str | None] = respond_with_rubric(
        conversation=example.conversation,
        rubric=foreign_step.result,
        response_model=bad_response_model,
        llm=llm,
        cacher=cacher,
        temperature=temperature,
    )
    if bad_step.result is None:
        print(
            f"{'-' * 50}\n bad response refusal: {index} {example.prompt_id}: {bad_response_model}\n{'-' * 50}"
        )
        return None

    by_name = {"true": true_rubric, **rubrics}
    scored = parallel_map(
        work=list(by_name),
        task=lambda rubric_name: (
            rubric_name,
            score_response(
                conversation=example.conversation,
                response=bad_step.result,
                rubric=by_name[rubric_name],
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
    for rubric_name, result in scored:
        if result is None:
            print(
                f"{'-' * 50}\n bad scoring refusal: {index} {example.prompt_id}: {rubric_name}\n{'-' * 50}"
            )
            return None
        scores[rubric_name] = result
    return bad_step.result, scores


def main(cfg: RunConfig) -> list[ExistingApproachesResult]:

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

    # Each bad response gets a different random foreign rubric, as in RubricRAG.
    # The seed uses the prompt id. Thus, a change to the sample does not change the foreign rubric of other examples.
    foreign_examples = [
        random.Random(f"{cfg.foreign_rubric_seed}:{example.prompt_id}").choice(
            hard_examples
        )
        for example in examples
    ]

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

        similarity = rubric_rag_similarities(
            example=example,
            index=index,
            true_rubric=true_rubric,
            rubrics=rubrics,
            llm=llm,
            cacher=cacher,
            similarity_model=cfg.similarity_model,
            similarity_reasoning_effort=cfg.similarity_reasoning_effort,
            temperature=cfg.temperature,
            max_workers=cfg.similarity_max_workers,
        )
        if similarity is None:
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

        # Some examples have no physician response. We keep them. The physician metrics use only the examples that have one.
        if example.ideal_completion is None:
            physician_scores = None
        else:
            physician_scores = physician_response_scores(
                example=example,
                index=index,
                true_rubric=true_rubric,
                rubrics=rubrics,
                llm=llm,
                cacher=cacher,
                scoring_model=cfg.scoring_model,
                scoring_reasoning_effort=cfg.scoring_reasoning_effort,
                judge_samples=cfg.judge_samples,
                temperature=cfg.temperature,
                max_workers=cfg.max_workers,
            )
            if physician_scores is None:
                continue

        bad = bad_response_scores(
            example=example,
            index=index,
            foreign_example=foreign_examples[index],
            true_rubric=true_rubric,
            rubrics=rubrics,
            llm=llm,
            cacher=cacher,
            convert_model=cfg.convert_model,
            convert_reasoning_effort=cfg.convert_reasoning_effort,
            bad_response_model=cfg.bad_response_model,
            scoring_model=cfg.scoring_model,
            scoring_reasoning_effort=cfg.scoring_reasoning_effort,
            judge_samples=cfg.judge_samples,
            temperature=cfg.temperature,
            max_workers=cfg.max_workers,
        )
        if bad is None:
            continue
        _, bad_scores = bad

        results.append(
            ExistingApproachesResult(
                example=example,
                true_rubric=true_rubric,
                rubrics=rubrics,
                similarity=similarity,
                responses=responses,
                scores=scores,
                physician_scores=physician_scores,
                bad_scores=bad_scores,
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
