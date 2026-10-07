"""The place where a pipeline step calls a model.

Each step in llm_functions.py looks for its answer in the response cache first.
The cache has all the answers that the paper uses. Thus, the reproduction calls no model.
If a step does not find its answer, it calls LLM.generate. Here, generate raises CacheMiss.
To make new calls, write generate() for a provider.
"""

from dataclasses import dataclass
from typing import Generic, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class CacheMiss(RuntimeError):
    """A step asked for an answer the response cache does not hold."""


@dataclass
class StepOutput(Generic[T]):
    result: T


@dataclass
class Tool:
    name: str
    description: str
    schema: Type[BaseModel]
    # The JSON schema to send to the model. None sends the schema of `schema`.
    # Use it to name each key of a map. Without key names, Gemini 3.8 Flash makes up keys such as "q13".
    # Validation always uses `schema`.
    parameters: dict | None = None

    def validate(self, raw: dict) -> BaseModel:
        return self.schema.model_validate(raw)


class LLM:
    """The interface to the models."""

    def __init__(self, model_reasoning_effort: dict[str, str]):
        # The reasoning effort of each generator and response model, by model id. A model that is not in the map uses its default.
        # The effort is part of the cache key.
        self.model_reasoning_effort = model_reasoning_effort

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        tool: Tool | None = None,
        temperature: float | None = None,
        *,
        reasoning_effort: str | None,
    ) -> BaseModel | str | None:
        raise CacheMiss(
            f"the cache has no response from {model} for this prompt. The published cache is complete. Thus, a prompt, a config value or the cache directory is different from the published run."
        )
