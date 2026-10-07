"""The rubric steps that more than one experiment uses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # llm_functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.cacher import Cacher
from utils.data_model import Example, Rubric
from utils.llm import LLM, StepOutput

from common.concurrency import parallel_map
from llm_functions import (
    convert_negative_criteria,
    generate_response,
    generate_rubric,
    score_response,
)


def generated_rubrics(
    example: Example,
    index: int,
    llm: LLM,
    cacher: Cacher,
    convert_model: str,
    convert_reasoning_effort: str,
    rubric_gen_models: list[str],
    temperature: float | None,
) -> tuple[Rubric, dict[str, Rubric]] | None:
    """The expert rubric with positive criteria only, and one generated rubric for each generator.

    The generated rubrics are positive. Thus, we also make the expert rubric positive.
    "Fail all criteria" is not a clear instruction for a negative criterion. This is a second reason.
    None if a model refuses. The caller then removes the example.
    """
    convert_step: StepOutput[Rubric | None] = convert_negative_criteria(
        criteria=[c.text for c in example.true_criteria],
        negative_indices=[
            j for j, c in enumerate(example.true_criteria) if c.points < 0
        ],
        llm=llm,
        cacher=cacher,
        convert_model=convert_model,
        temperature=temperature,
        reasoning_effort=convert_reasoning_effort,
    )
    if convert_step.result is None:
        print(
            f"{'-' * 50}\n convert refusal: {index} {example.prompt_id}: {convert_model}\n{'-' * 50}"
        )
        return None

    rubrics = {}
    for rubric_model in rubric_gen_models:
        rubric_step: StepOutput[Rubric | None] = generate_rubric(
            conversation=example.conversation,
            llm=llm,
            cacher=cacher,
            rubric_gen_model=rubric_model,
            temperature=temperature,
        )
        if rubric_step.result is None:
            print(
                f"{'-' * 50}\n rubric refusal: {index} {example.prompt_id}: {rubric_model}\n{'-' * 50}"
            )
            return None
        rubrics[rubric_model] = rubric_step.result

    return convert_step.result, rubrics


def gen_rubric_panel(
    example: Example,
    index: int,
    true_rubric: Rubric,
    rubrics: dict[str, Rubric],
    llm: LLM,
    cacher: Cacher,
    response_models: list[str],
    scoring_model: str,
    scoring_reasoning_effort: str,
    judge_samples: int,
    temperature: float | None,
    max_workers: int,
) -> tuple[dict[str, str], dict[str, dict[str, object]]] | None:
    """GenRubric score correlation: one response from each panel model, and the score of each rubric on each response.

    evaluate.py calculates the correlation.
    Differences from the paper:
        1. The judge is `scoring_model`.
        2. The panel of models is not the panel of the paper, see config.json.
        3. There is one rubric for each generator. The paper calculates a mean over many generation runs.
        4. We use 105 HealthBench examples. The paper uses 700 queries from four domains.
        5. The judge uses our score_response prompt. The paper does not give its full judge prompt.
    None if a model refuses. The caller then removes the example.
    """
    responses = dict(
        parallel_map(
            work=response_models,
            task=lambda response_model: (
                response_model,
                generate_response(
                    conversation=example.conversation,
                    response_model=response_model,
                    llm=llm,
                    cacher=cacher,
                    temperature=temperature,
                ).result,
            ),
            max_workers=max_workers,
        )
    )
    refused = [model for model, response in responses.items() if response is None]
    if refused:
        print(
            f"{'-' * 50}\n response refusal: {index} {example.prompt_id}: {refused}\n{'-' * 50}"
        )
        return None

    by_name = {"true": true_rubric, **rubrics}
    score_steps = parallel_map(
        work=[
            (rubric_name, response_model)
            for rubric_name in by_name
            for response_model in response_models
        ],
        task=lambda pair: (
            pair,
            score_response(
                conversation=example.conversation,
                response=responses[pair[1]],
                rubric=by_name[pair[0]],
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
    for (rubric_name, response_model), result in score_steps:
        if result is None:
            print(
                f"{'-' * 50}\n scoring refusal: {index} {example.prompt_id}: {rubric_name} on {response_model}\n{'-' * 50}"
            )
            return None
        scores.setdefault(rubric_name, {})[response_model] = result
    return responses, scores
