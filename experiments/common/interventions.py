"""The whole-rubric intervention on the expert rubric.

do(C*: 0 -> 1) and do(C*: 1 -> 0) are one edit each. The response model edits its own response.
We keep each edit, also when it does not move the expert score. That failure is part of the method.
We apply the edit at all start scores.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # llm_functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from utils.cacher import Cacher
from utils.data_model import Rubric
from utils.llm import LLM

from llm_functions import ScoreResponseOutput, edit_response, score_response


def intervene_on_expert_rubric(
    edit: str,
    conversation: str,
    response: str,
    true_rubric: Rubric,
    true_scores: ScoreResponseOutput,
    response_model: str,
    llm: LLM,
    cacher: Cacher,
    scoring_model: str,
    scoring_reasoning_effort: str,
    temperature: float | None,
    judge_samples: int,
) -> tuple[str, ScoreResponseOutput] | None:
    """Edit the response for the expert rubric, then score the edit with the expert rubric.

    edit="revise" gives do(C*: 0 -> 1). edit="degrade" gives do(C*: 1 -> 0).
    Returns the edited response and its expert scores. Returns None if a model refuses.
    """
    step = edit_response(
        edit=edit,
        conversation=conversation,
        response=response,
        rubric=true_rubric,
        scores=true_scores.scores,
        response_model=response_model,
        llm=llm,
        cacher=cacher,
        temperature=temperature,
    )
    if step.result is None:
        return None
    scored = score_response(
        conversation=conversation,
        response=step.result,
        rubric=true_rubric,
        llm=llm,
        cacher=cacher,
        scoring_model=scoring_model,
        temperature=temperature,
        reasoning_effort=scoring_reasoning_effort,
        samples=judge_samples,
    )
    if scored.result is None:
        return None
    return step.result, scored.result
