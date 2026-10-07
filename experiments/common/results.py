"""The records that each run.py pickles and each evaluate.py reads.

Pickle stores the module name of a class. If a run.py defined its class, `python run.py` would store it as `__main__.<class>`, and evaluate.py could not load it.
"""

from dataclasses import dataclass
from utils.data_model import Rubric, Example
from llm_functions import ScoreResponseOutput


@dataclass
class InterventionalResult:
    """One HealthBench conversation, one response, and the four interventions on it.

    The interventions on a generated rubric give one edit for each generator.
    The interventions on the expert rubric give one edit, which all generated rubrics score.
    """

    example: Example
    true_rubric: Rubric  # the expert rubric with positive criteria only
    rubrics: dict[str, Rubric]  # rubric generator -> generated rubric
    response: str
    scores: dict[
        str, ScoreResponseOutput
    ]  # rubric ("true" or a generator) -> scores on the response

    # do(C^g: 0 -> 1), one edit for each generator, scored with the expert rubric
    revised_responses: dict[str, str]  # generator -> edit
    revised_true_scores: dict[
        str, ScoreResponseOutput
    ]  # generator -> expert scores on the edit

    # do(C^g: 1 -> 0), one edit for each generator, scored with the expert rubric
    rejected_responses: dict[str, str]  # generator -> edit
    rejected_true_scores: dict[
        str, ScoreResponseOutput
    ]  # generator -> expert scores on the edit

    # do(C^*: 1 -> 0) and do(C^*: 0 -> 1), one edit each, scored with all rubrics
    degraded_response: str
    degraded_scores: dict[
        str, ScoreResponseOutput
    ]  # rubric -> scores on the degraded edit
    improved_response: str
    improved_scores: dict[
        str, ScoreResponseOutput
    ]  # rubric -> scores on the improved edit


@dataclass
class ExistingApproachesResult:
    """One HealthBench conversation and all the data for the four existing metrics.

    RubricRAG LLM-matching uses `similarity`. GenRubric uses `responses` and `scores`.
    The two RubricRAG physician metrics use `physician_scores` and `bad_scores`.
    In each score map, "true" is the expert rubric.
    """

    example: Example
    true_rubric: Rubric  # the expert rubric with positive criteria only
    rubrics: dict[str, Rubric]  # rubric generator -> generated rubric
    similarity: dict[
        str, dict[int, dict[int, int]]
    ]  # generator -> generated criterion -> expert criterion -> similarity from 0 to 9
    responses: dict[str, str]  # response model -> response text
    scores: dict[
        str, dict[str, ScoreResponseOutput]
    ]  # rubric ("true" or a generator) -> response model -> scores

    # The RubricRAG physician metrics: the physician response and the bad response.
    physician_scores: (
        dict[str, ScoreResponseOutput] | None
    )  # rubric -> scores on the physician response; None if there is no physician response
    bad_scores: dict[str, ScoreResponseOutput]  # rubric -> scores on the bad response


@dataclass
class AppendixResult:
    """One HealthBench conversation and the data for all appendix experiments.

    The matching framings use `matches`. The panel subsets use `scores`.
    The foreign-rubric experiment compares `bad_true_scores` with the expert scores in `scores`.
    """

    example: Example
    true_rubric: Rubric  # the expert rubric with positive criteria only
    rubrics: dict[str, Rubric]  # rubric generator -> generated rubric
    matches: dict[
        str, dict[str, dict[int, list[int]]]
    ]  # framing -> generator -> generated criterion -> the expert criteria it matches
    responses: dict[str, str]  # response model -> response text
    scores: dict[
        str, dict[str, ScoreResponseOutput]
    ]  # rubric ("true" or a generator) -> response model -> scores

    # The RubricRAG bad response of each panel model. All examples use the same foreign rubric.
    bad_responses: dict[str, str]  # panel model -> its answer to the foreign rubric
    bad_true_scores: dict[
        str, ScoreResponseOutput
    ]  # panel model -> expert scores on that answer


@dataclass
class CriteriaResult:
    """The single-criterion interventions of one example, on all rubrics, in one direction.

    "pass" (results_pass.pkl): do(c: 0 -> 1) for each criterion c that R fails.
    "fail" (results_fail.pkl): do(c: 1 -> 0) for each criterion c that R passes.
    R, the rubrics and their scores on R are in the source InterventionalResult at `source_index`.
    """

    prompt_id: str
    source_index: int  # the position in the interventional_metrics pickle

    # do(c^g), scored with the expert rubric: generator -> criterion number -> (edit, expert scores on it)
    on_generated: dict[str, dict[int, tuple[str, ScoreResponseOutput]]]

    # do(c^*), scored with all generated rubrics: expert criterion number -> (edit, generator -> scores on it)
    on_expert: dict[int, tuple[str, dict[str, ScoreResponseOutput]]]


@dataclass
class OwnScores:
    """The edits of one example, scored with the own rubric of each target. score_own.py makes them."""

    prompt_id: str
    source_index: int
    own: dict[
        tuple[str, int], ScoreResponseOutput
    ]  # (target rubric, criterion number) -> scores of that rubric on the edit
