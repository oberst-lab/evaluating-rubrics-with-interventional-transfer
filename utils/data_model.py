"""Typed data model for HealthBench examples and criteria."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel


class Rubric(BaseModel):
    """A rubric as an ordered list of criterion texts.

    Numbers start at 0. Only this class numbers criteria, so all prompts use the same numbers.
    """

    criteria: list[str]

    def format(self, prefix: str = "") -> str:
        """Number each criterion. With prefix="T", the numbers are T0, T1, and so on.

        Use a prefix when one prompt shows two rubrics. Without it, the model can mix up the numbers of the two rubrics.
        """
        return "\n".join(f"{prefix}{i}. {c}" for i, c in enumerate(self.criteria))


@dataclass(frozen=True)
class Criterion:
    """A single rubric criterion."""

    text: str
    points: float
    level: str  # "example" or "cluster"


@dataclass
class Example:
    """One HealthBench encounter/prompt."""

    prompt_id: str
    prompt: list[dict]
    theme: str
    criteria: list[Criterion] = field(default_factory=list)
    ideal_completion: str | None = (
        None  # the physician response from HealthBench; not all examples have one
    )

    @property
    def conversation(self) -> str:
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in self.prompt)

    @property
    def true_criteria(self) -> list[Criterion]:
        """The expert rubric of this example: the criteria with level "example"."""
        return [c for c in self.criteria if c.level == "example"]


def _get_tag_value(tags: list[str], prefix: str, default: str = "") -> str:
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix) :]
    return default


def _parse_criteria(rubrics: list[dict]) -> list[Criterion]:
    result = []
    for r in rubrics:
        level = _get_tag_value(r.get("tags", []), "level:", default="unknown")
        result.append(Criterion(text=r["criterion"], points=r["points"], level=level))
    return result


def load_dataset(
    path: str | Path,
    exclude_prompt_ids: set[str] | None = None,
    skip_prompt_ids: set[str] | None = None,
    per_theme: int | None = None,
    sample_seed: int | None = None,
) -> list[Example]:
    """Load Examples from a JSONL file.

    The function keeps only examples that have one or more example-level criteria.
    exclude_prompt_ids: remove these examples before the sample. We remove the hard split here.
    skip_prompt_ids: step over these examples after the shuffle. The next example in the theme takes the place. We skip refusals here.
    per_theme: the number of random examples to take from each theme.
    sample_seed: the seed of the shuffle. You must give it with per_theme.
    A larger per_theme contains all the examples of a smaller one.
    """

    examples = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            criteria = _parse_criteria(raw.get("rubrics", []))
            # The theme is a `theme:` tag.
            theme = _get_tag_value(raw.get("example_tags", []), "theme:")
            examples.append(
                Example(
                    prompt_id=raw["prompt_id"],
                    prompt=raw["prompt"],
                    theme=theme,
                    criteria=criteria,
                    ideal_completion=(raw.get("ideal_completions_data") or {}).get(
                        "ideal_completion"
                    ),
                )
            )

    examples = [ex for ex in examples if ex.true_criteria]

    if exclude_prompt_ids is not None:
        examples = [ex for ex in examples if ex.prompt_id not in exclude_prompt_ids]

    if per_theme is not None:
        assert sample_seed is not None, "per_theme draws at random, so it needs a seed"
        skip = skip_prompt_ids or set()
        by_theme = defaultdict(list)
        for ex in examples:
            assert ex.theme, (
                f"{ex.prompt_id} has no theme, so it cannot be balanced by one"
            )
            by_theme[ex.theme].append(ex)
        examples = []
        for theme in sorted(by_theme):
            # Shuffle the full theme before the skip. Then a skip does not move other examples.
            order = random.Random(sample_seed).sample(
                by_theme[theme], k=len(by_theme[theme])
            )
            kept = [ex for ex in order if ex.prompt_id not in skip][:per_theme]
            assert len(kept) == per_theme, (
                f"theme {theme} has {len(kept)} examples after skipping, need {per_theme}"
            )
            examples.extend(kept)
    else:
        assert skip_prompt_ids is None, (
            "skip_prompt_ids is read only when per_theme selects"
        )

    return examples
