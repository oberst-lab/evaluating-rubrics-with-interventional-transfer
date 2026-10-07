"""The LLM steps that all experiments use.

Each step fills a prompt template, looks in the cache, and returns a StepOutput.
If the cache does not have the answer, the step calls LLM.generate. In this repository, that raises CacheMiss.
Most prompt files are default arguments, prompts/<function>_<role>.txt. Python reads them at import.
ask_llm_map_rubrics and edit_response have more than one prompt. An argument selects the prompt, and the function reads the file.
The judge, matcher and converter steps take reasoning_effort as an argument.
The generator and response model steps get the effort from llm.model_reasoning_effort.
"""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from utils.cacher import Cacher
from utils.data_model import Rubric
from utils.llm import LLM, Tool, StepOutput

PROMPTS = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Schemas + tools
# ---------------------------------------------------------------------------


class ScoreResponseOutput(BaseModel):
    # criterion number -> 0 or 1.
    # A map, so the judge cannot skip a criterion and move the scores that follow it.
    scores: dict[int, Literal[0, 1]]


class IndexedRubric(BaseModel):
    # criterion number -> criterion text.
    # The keys keep the original numbers when the prompt shows only some criteria of a rubric.
    criteria: dict[int, str]


class GeneratedToTrueMap(BaseModel):
    # "G<generated number>" -> a list of "T<expert number>". The match, simulate and implies framings use it.
    # The G and T prefixes stop the model from mixing up the numbers of the two rubrics.
    # Do not add a free-text field. With one, the model wrote the map into the text and returned no map.
    indices: dict[str, list[str]]

    @field_validator("indices")
    @classmethod
    def drop_generated_labels_from_values(
        cls, indices: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Remove G labels from the value lists. The values must be expert criteria only.

        Other bad labels raise ValueError. Pydantic gives it as a ValidationError.
        After this, as_indices can parse each label.
        """
        cleaned, dropped = {}, []
        for gen_label, true_labels in indices.items():
            if not (gen_label.startswith("G") and gen_label[1:].isdigit()):
                raise ValueError(
                    f"{gen_label} is not a generated criterion (G0, G1, ...)"
                )
            kept = [label for label in true_labels if not label.startswith("G")]
            dropped += [
                f"{gen_label}:{label}" for label in true_labels if label.startswith("G")
            ]
            for label in kept:
                if not (label.startswith("T") and label[1:].isdigit()):
                    raise ValueError(
                        f"{label} in row {gen_label} is not a true criterion (T0, T1, ...)"
                    )
            cleaned[gen_label] = kept
        if dropped:
            print(
                f"dropped {len(dropped)} generated labels found in value lists: {dropped}"
            )
        return cleaned

    def as_indices(self) -> dict[int, list[int]]:
        """{generated criterion number: true criterion numbers}, e.g. {"G2": ["T5", "T9"]} -> {2: [5, 9]}, so everything downstream stays plain numbers."""
        return {
            int(gen_label[1:]): [int(label[1:]) for label in true_labels]
            for gen_label, true_labels in self.indices.items()
        }


RUBRIC_TOOL = Tool(
    name="emit_rubric",
    description="Return the generated rubric as structured JSON.",
    schema=Rubric,
)
SCORING_TOOL = Tool(
    name="emit_scores",
    description=(
        "Return a JSON object mapping each criterion's number (as shown in the"
        " rubric) to its 0 or 1 score. Every criterion number must appear."
    ),
    schema=ScoreResponseOutput,
)
CONVERT_TOOL = Tool(
    name="emit_converted_criteria",
    description=(
        "Return a JSON object mapping each criterion's number, as shown, to that criterion"
        " restated so that satisfying it means the response is better."
    ),
    schema=IndexedRubric,
)

# One tool for each framing of ask_llm_map_rubrics.
RUBRIC_MAP_TOOLS = {
    "match_rubric": Tool(
        name="emit_matching",
        description=(
            "Return a JSON object mapping each generated criterion's label (G0, G1, ...) to the"
            " true criteria that measure the same thing (T0, T1, ...). Every generated criterion"
            " label must appear, mapped to an empty list if none of them measures it."
        ),
        schema=GeneratedToTrueMap,
    ),
    "simulate_intervention": Tool(
        name="emit_simulated_intervention",
        description=(
            "Return a JSON object mapping each generated criterion's label (G0, G1, ...) to the"
            " true criteria that move with it (T0, T1, ...). Every generated criterion label must"
            " appear, mapped to an empty list if nothing moves with it."
        ),
        schema=GeneratedToTrueMap,
    ),
    "implies": Tool(
        name="emit_implications",
        description=(
            "Return a JSON object mapping each generated criterion's label (G0, G1, ...) to the"
            " true criteria it logically implies (T0, T1, ...). Every generated criterion label"
            " must appear, mapped to an empty list if it implies none of them."
        ),
        schema=GeneratedToTrueMap,
    ),
}


def keyed(tool: Tool, field: str, keys: list[str], value: dict) -> Tool:
    """Return `tool` with a schema that names each key of its map field as a required property.

    Without key names, Gemini 3.8 Flash makes up keys such as "q13".
    The keys are different for each call. Thus, each call makes its own schema.
    `tool.schema` still does the validation.
    """
    return Tool(
        name=tool.name,
        description=tool.description,
        schema=tool.schema,
        parameters={
            "type": "object",
            "properties": {
                field: {
                    "type": "object",
                    "properties": {key: value for key in keys},
                    "required": keys,
                }
            },
            "required": [field],
        },
    )


# ---------------------------------------------------------------------------
# LLM functions
# ---------------------------------------------------------------------------


def generate_response(
    conversation: str,
    response_model: str,
    llm: LLM,
    cacher: Cacher,
    temperature: float | None,
    response_system_prompt: str = open(PROMPTS / "generate_response_system.txt").read(),
) -> StepOutput[str | None]:
    reasoning_effort = llm.model_reasoning_effort.get(response_model)
    cache_key = dict(
        model=response_model,
        system=response_system_prompt,
        template="{conversation}"
        + ("" if reasoning_effort is None else f"|effort={reasoning_effort}"),
        prompt=conversation,
        temperature=temperature,
    )
    if cacher.has(**cache_key):
        response = cacher.get(**cache_key)
    else:
        response = llm.generate(
            model=response_model,
            prompt=conversation,
            system=response_system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        if not response:
            # The model refused or gave no answer. The caller sees result=None.
            return StepOutput(result=None)
        cacher.set(**cache_key, value=response)
    return StepOutput(result=response)


def respond_with_rubric(
    conversation: str,
    rubric: Rubric,
    response_model: str,
    llm: LLM,
    cacher: Cacher,
    temperature: float | None,
    respond_message_template: str = open(
        PROMPTS / "respond_with_rubric_message.txt"
    ).read(),
    respond_system_prompt: str = open(
        PROMPTS / "respond_with_rubric_system.txt"
    ).read(),
) -> StepOutput[str | None]:
    """Answer the conversation and satisfy all criteria of `rubric`.

    RubricRAG makes its bad responses this way, with the expert rubric of a different example.
    The prompt does not tell where the rubric comes from.
    """
    prompt = respond_message_template.format(
        conversation=conversation, rubric=rubric.format()
    )
    reasoning_effort = llm.model_reasoning_effort.get(response_model)
    cache_key = dict(
        model=response_model,
        system=respond_system_prompt,
        template=respond_message_template
        + ("" if reasoning_effort is None else f"|effort={reasoning_effort}"),
        prompt=prompt,
        temperature=temperature,
    )
    if cacher.has(**cache_key):
        response = cacher.get(**cache_key)
    else:
        response = llm.generate(
            model=response_model,
            prompt=prompt,
            system=respond_system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        if not response:
            # The model refused or gave no answer. The caller sees result=None.
            return StepOutput(result=None)
        cacher.set(**cache_key, value=response)
    return StepOutput(result=response)


def generate_rubric(
    conversation: str,
    llm: LLM,
    cacher: Cacher,
    rubric_gen_model: str,
    temperature: float | None,
    rubric_gen_message_template: str = open(
        PROMPTS / "generate_rubric_message.txt"
    ).read(),
    rubric_gen_system_prompt: str = open(PROMPTS / "generate_rubric_system.txt").read(),
    rubric_tool: Tool = RUBRIC_TOOL,
) -> StepOutput[Rubric | None]:
    prompt = rubric_gen_message_template.format(conversation=conversation)
    reasoning_effort = llm.model_reasoning_effort.get(rubric_gen_model)
    cache_key = dict(
        model=rubric_gen_model,
        system=rubric_gen_system_prompt,
        template=rubric_gen_message_template
        + ("" if reasoning_effort is None else f"|effort={reasoning_effort}"),
        prompt=prompt,
        temperature=temperature,
    )
    if cacher.has(**cache_key):
        rubric = rubric_tool.validate(json.loads(cacher.get(**cache_key)))
    else:
        rubric = llm.generate(
            model=rubric_gen_model,
            prompt=prompt,
            system=rubric_gen_system_prompt,
            temperature=temperature,
            tool=rubric_tool,
            reasoning_effort=reasoning_effort,
        )
        if not rubric:
            # The model refused or gave no answer. The caller sees result=None.
            return StepOutput(result=None)
        cacher.set(**cache_key, value=json.dumps(rubric.model_dump()))
    return StepOutput(result=rubric)


def score_response(
    conversation: str,
    response: str,
    rubric: Rubric,
    llm: LLM,
    cacher: Cacher,
    scoring_model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    samples: int,
    scoring_message_template: str = open(PROMPTS / "score_response_message.txt").read(),
    scoring_system_prompt: str = open(PROMPTS / "score_response_system.txt").read(),
    scoring_tool: Tool = SCORING_TOOL,
) -> StepOutput[ScoreResponseOutput | None]:
    """Score a response. Each criterion gets the majority value of `samples` judgements.

    Two judgements of the same response can be different. The vote decreases this noise.
    The function stops when a majority decides each criterion. The result is the same as with all samples.
    """
    assert samples >= 1 and samples % 2 == 1, (
        f"a majority vote needs an odd number of samples, got {samples}"
    )
    prompt = scoring_message_template.format(
        conversation=conversation, response=response, rubric=rubric.format()
    )
    judgements: list[ScoreResponseOutput] = []
    for sample in range(samples):
        # Sample 0 has no salt. Each other sample adds "|vote<n>" to the cache key.
        drawn = _draw_judgement(
            prompt=prompt,
            rubric=rubric,
            llm=llm,
            cacher=cacher,
            scoring_model=scoring_model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            scoring_message_template=scoring_message_template,
            scoring_system_prompt=scoring_system_prompt,
            scoring_tool=scoring_tool,
            cache_salt="" if sample == 0 else f"|vote{sample}",
        )
        # If one judgement fails, the full score fails. The caller then removes the example.
        if drawn is None:
            return StepOutput(result=None)
        judgements.append(drawn)
        ones = {
            criterion: sum(j.scores[criterion] for j in judgements)
            for criterion in drawn.scores
        }
        if all(
            2 * max(count, len(judgements) - count) > samples for count in ones.values()
        ):
            break
    voted = {
        criterion: (1 if 2 * count > len(judgements) else 0)
        for criterion, count in ones.items()
    }
    return StepOutput(
        result=ScoreResponseOutput(scores=voted),
    )


def _draw_judgement(
    prompt: str,
    rubric: Rubric,
    llm: LLM,
    cacher: Cacher,
    scoring_model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    scoring_message_template: str,
    scoring_system_prompt: str,
    scoring_tool: Tool,
    cache_salt: str,
) -> ScoreResponseOutput | None:
    """Get one judgement of one response from the cache or from the judge.

    Returns None if the judge refuses or does not score all criteria.
    """
    # cache_salt changes the cache key but not the prompt. Thus, each sample has its own cache entry.
    # The salt goes into `template`, which is part of the key. The model does not see `template`.
    cache_key = dict(
        model=scoring_model,
        system=scoring_system_prompt,
        template=scoring_message_template
        + cache_salt
        + (
            "" if reasoning_effort is None else f"|effort={reasoning_effort}"
        ),  # the effort changes the answer, so it is part of the key
        prompt=prompt,
        temperature=temperature,
    )
    expected_keys = set(range(len(rubric.criteria)))
    if cacher.has(**cache_key):
        result = scoring_tool.validate(json.loads(cacher.get(**cache_key)))
        assert set(result.scores) == expected_keys, (
            f"cached judgement scored criteria {sorted(set(result.scores))}"
            f" but the rubric numbers them {sorted(expected_keys)}"
        )
        return result
    # A judge can return a valid map that does not include all criteria. Then we ask again.
    # We do not read a missing criterion as 0.
    # The new request names the missing criteria. A repeat of the same prompt did not help.
    # The note goes into the prompt only. The cache key stays the same.
    missing: list[int] = []
    for attempt in range(8):
        nudge = (
            ""
            if not missing
            else f"\n\nYour previous answer left out criterion {missing}. Score every criterion from 0 to {len(rubric.criteria) - 1}, including that one."
        )
        result = llm.generate(
            model=scoring_model,
            prompt=prompt + nudge,
            system=scoring_system_prompt,
            tool=keyed(
                tool=scoring_tool,
                field="scores",
                keys=[str(i) for i in range(len(rubric.criteria))],
                value={"type": "integer", "enum": [0, 1]},
            ),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        if not result:
            # The model refused or gave no answer. score_response then fails.
            return None
        if set(result.scores) == expected_keys:
            # Write to the cache only after validation. Then the cache cannot keep a bad answer.
            cacher.set(**cache_key, value=json.dumps(result.model_dump()))
            return result
        missing = sorted(expected_keys - set(result.scores))
        invented = sorted(set(result.scores) - expected_keys)
        print(
            f"incomplete judgement from {scoring_model} on attempt {attempt + 1}:"
            f" {len(rubric.criteria)} criteria, missing {missing}, invented {invented}"
        )
    # Eight judgements did not score all criteria. The caller removes the example.
    return None


def ask_llm_map_rubrics(
    framing: Literal["match_rubric", "simulate_intervention", "implies"],
    conversation: str,
    rubric: Rubric,
    true_rubric: Rubric,
    llm: LLM,
    cacher: Cacher,
    match_model: str,
    temperature: float | None,
    reasoning_effort: str | None,
) -> StepOutput[GeneratedToTrueMap | None]:
    """Show the LLM the two full rubrics. For each generated criterion, ask which expert criteria go with it.

    `framing` selects the question and the prompt files prompts/ask_llm_<framing>_<role>.txt.
    "match_rubric" asks which expert criteria measure the same thing.
    "simulate_intervention" asks which expert criteria change when an edit satisfies the generated criterion.
    "implies" asks which expert criteria each response that satisfies the generated criterion also satisfies. This is the most strict framing.
    There is one call for each pair of rubrics.
    """
    message_template = open(PROMPTS / f"ask_llm_{framing}_message.txt").read()
    system_prompt = open(PROMPTS / f"ask_llm_{framing}_system.txt").read()
    tool = RUBRIC_MAP_TOOLS[framing]
    prompt = message_template.format(
        conversation=conversation,
        true_rubric=true_rubric.format(prefix="T"),
        generated_rubric=rubric.format(prefix="G"),
    )
    cache_key = dict(
        model=match_model,
        system=system_prompt,
        template=message_template
        + ("" if reasoning_effort is None else f"|effort={reasoning_effort}"),
        prompt=prompt,
        temperature=temperature,
    )
    cached = cacher.has(**cache_key)
    if cached:
        result = tool.validate(json.loads(cacher.get(**cache_key)))
    else:
        result = llm.generate(
            model=match_model,
            prompt=prompt,
            system=system_prompt,
            temperature=temperature,
            tool=keyed(
                tool=tool,
                field="indices",
                keys=[f"G{i}" for i in range(len(rubric.criteria))],
                value={
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [f"T{j}" for j in range(len(true_rubric.criteria))],
                    },
                },
            ),
            reasoning_effort=reasoning_effort,
        )
        if not result:
            # The model refused or gave no answer. The caller sees result=None.
            return StepOutput(result=None)
    # A missing row means "no match". Then precision uses the full rubric size.
    # Remove rows for criteria that do not exist.
    expected = [f"G{i}" for i in range(len(rubric.criteria))]
    missing = [key for key in expected if key not in result.indices]
    invented = [key for key in result.indices if key not in set(expected)]
    if missing or invented:
        print(
            f"{framing}: {len(missing)} generated criteria unanswered (read as no match) {missing},"
            f" {len(invented)} rows for criteria that do not exist (dropped) {invented}"
        )
        result.indices = {key: result.indices.get(key, []) for key in expected}
    # Remove expert labels that do not exist, usually one past the last criterion. They cannot increase the score.
    # The validator cannot do this, because it does not know the rubric length.
    valid = {f"T{i}" for i in range(len(true_rubric.criteria))}
    dropped = [
        label
        for labels in result.indices.values()
        for label in labels
        if label not in valid
    ]
    if dropped:
        print(
            f"{framing}: dropped {len(dropped)} citations of true criteria that do not exist: {sorted(set(dropped))}"
            f" (the true rubric has {len(true_rubric.criteria)})"
        )
        result.indices = {
            key: [label for label in labels if label in valid]
            for key, labels in result.indices.items()
        }
    # Write to the cache only after validation. Then the cache cannot keep a bad answer.
    if not cached:
        cacher.set(**cache_key, value=json.dumps(result.model_dump()))
    return StepOutput(result=result)


def convert_negative_criteria(
    criteria: list[str],
    negative_indices: list[int],
    llm: LLM,
    cacher: Cacher,
    convert_model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    convert_message_template: str = open(
        PROMPTS / "convert_negative_criteria_to_positive_message.txt"
    ).read(),
    convert_system_prompt: str = open(
        PROMPTS / "convert_negative_criteria_to_positive_system.txt"
    ).read(),
    convert_tool: Tool = CONVERT_TOOL,
) -> StepOutput[Rubric | None]:
    """Write the negative criteria of a rubric again, so that to satisfy them is good.

    The prompt shows only the negative criteria, in one call.
    The function returns the full rubric, with each new text in the place of its old criterion.
    The prompt keeps the original numbers. Rubric.format() would start the numbers again at 0.
    """
    negative = {i: criteria[i] for i in negative_indices}
    if not negative:
        return StepOutput(result=Rubric(criteria=criteria))

    numbered = "\n".join(f"{i}. {negative[i]}" for i in sorted(negative))
    prompt = convert_message_template.format(criteria=numbered)
    cache_key = dict(
        model=convert_model,
        system=convert_system_prompt,
        template=convert_message_template
        + ("" if reasoning_effort is None else f"|effort={reasoning_effort}"),
        prompt=prompt,
        temperature=temperature,
    )
    cached = cacher.has(**cache_key)
    if cached:
        result = convert_tool.validate(json.loads(cacher.get(**cache_key)))
    else:
        result = llm.generate(
            model=convert_model,
            prompt=prompt,
            system=convert_system_prompt,
            temperature=temperature,
            tool=keyed(
                tool=convert_tool,
                field="criteria",
                keys=[str(i) for i in sorted(negative)],
                value={"type": "string"},
            ),
            reasoning_effort=reasoning_effort,
        )
        if not result:
            # The model refused or gave no answer. The caller sees result=None.
            return StepOutput(result=None)
    # The texts can change. The numbers must not change.
    numbers_sent = sorted(negative.keys())
    numbers_returned = sorted(result.criteria.keys())
    assert numbers_returned == numbers_sent, (
        f"asked to convert criteria numbered {numbers_sent} but got back {numbers_returned}"
    )

    converted = list(criteria)
    for i, rewritten in result.criteria.items():
        # A criterion can come back with no change. Some HealthBench criteria have negative points but positive text (for example, f05491d8).
        converted[i] = rewritten
    # Write to the cache only after validation. Then the cache cannot keep a bad answer.
    if not cached:
        cacher.set(**cache_key, value=json.dumps(result.model_dump()))
    return StepOutput(
        result=Rubric(criteria=converted),
    )


def ask_rubric_rag_similarity(
    true_criterion: str,
    generated_criterion: str,
    llm: LLM,
    cacher: Cacher,
    similarity_model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    similarity_message_template: str = open(
        PROMPTS / "rubric_rag_llm_matching_message.txt"
    ).read(),
    similarity_system_prompt: str = open(
        PROMPTS / "rubric_rag_llm_matching_system.txt"
    ).read(),
) -> StepOutput[int | None]:
    """The RubricRAG similarity call. It gives an integer from 0 to 9 for one expert criterion and one generated criterion.

    The prompt is Figure 3 of Dhole and Agichtein (arXiv 2603.20882), with no changes.
    As in their paper, there is one call for each pair of criteria, and the prompt does not show the conversation.
    No tool because their prompt asks for a number only.
    """
    prompt = similarity_message_template.format(
        ref_text=true_criterion, gen_text=generated_criterion
    )
    cache_key = dict(
        model=similarity_model,
        system=similarity_system_prompt,
        template=similarity_message_template
        + ("" if reasoning_effort is None else f"|effort={reasoning_effort}"),
        prompt=prompt,
        temperature=temperature,
    )
    if cacher.has(**cache_key):
        return StepOutput(
            result=int(cacher.get(**cache_key)),
        )
    # Try two times. A reply with no digit usually has extra text, and a second call usually fixes it.
    for _ in range(2):
        reply = llm.generate(
            model=similarity_model,
            prompt=prompt,
            system=similarity_system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        if not reply:
            continue
        # The regex reads replies such as "Similarity score: 7". Use the last digit, so "(0..9)" in the text cannot be the answer.
        digits = re.findall(r"\d", reply)
        if digits:
            score = int(digits[-1])
            cacher.set(**cache_key, value=str(score))
            return StepOutput(result=score)
        print(f"similarity: no digit in reply {reply!r}")
    # Two calls failed. The caller sees result=None.
    return StepOutput(result=None)


def edit_response(
    edit: Literal["revise", "degrade"],
    conversation: str,
    response: str,
    rubric: Rubric,
    scores: dict[int, int],
    response_model: str,  # the model that wrote the response edits its own work
    llm: LLM,
    cacher: Cacher,
    temperature: float | None,
) -> StepOutput[str | None]:
    """Make the smallest edit to `response` that satisfies ("revise") or fails ("degrade") all criteria of `rubric`.

    `edit` selects the prompt files prompts/<edit>_response_<role>.txt.
    `scores` gives the current 0 or 1 of each criterion. The prompt shows the criteria that `scores` contains.
    The degrade prompt tells the model to keep the response fluent, on topic and of approximately the same length.
    Without these limits, the edit can give nonsense, and all rubrics score 0 on nonsense.
    """
    message_template = open(PROMPTS / f"{edit}_response_message.txt").read()
    system_prompt = open(PROMPTS / f"{edit}_response_system.txt").read()
    satisfied = [i for i in sorted(scores) if scores[i] == 1]
    unsatisfied = [i for i in sorted(scores) if scores[i] == 0]

    def block(indices: list[int]) -> str:
        if not indices:
            return "(none)"
        return "\n".join(f"{i}. {rubric.criteria[i]}" for i in indices)

    prompt = message_template.format(
        conversation=conversation,
        response=response,
        satisfied=block(satisfied),
        unsatisfied=block(unsatisfied),
    )
    reasoning_effort = llm.model_reasoning_effort.get(response_model)
    cache_key = dict(
        model=response_model,
        system=system_prompt,
        template=message_template
        + ("" if reasoning_effort is None else f"|effort={reasoning_effort}"),
        prompt=prompt,
        temperature=temperature,
    )
    if cacher.has(**cache_key):
        return StepOutput(result=cacher.get(**cache_key))
    edited = llm.generate(
        model=response_model,
        prompt=prompt,
        system=system_prompt,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    if not edited:
        # The model refused or gave no answer. The caller sees result=None.
        return StepOutput(result=None)
    cacher.set(**cache_key, value=edited)
    return StepOutput(result=edited)
