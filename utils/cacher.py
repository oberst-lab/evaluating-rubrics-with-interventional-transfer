import hashlib
from pathlib import Path


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Cacher:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def _response_path(
        self,
        model: str,
        system: str,
        template: str,
        prompt: str,
        temperature: float | None,
    ) -> Path:
        model_safe = model.replace("/", "_").replace(":", "_")
        return (
            self.cache_dir
            / model_safe
            / _hash(system)
            / _hash(template)
            / f"{_hash(prompt)}_temp_{temperature}.txt"
        )

    def has(
        self,
        model: str,
        system: str,
        template: str,
        prompt: str,
        temperature: float | None,
    ) -> bool:
        return self._response_path(
            model=model,
            system=system,
            template=template,
            prompt=prompt,
            temperature=temperature,
        ).exists()

    def get(
        self,
        model: str,
        system: str,
        template: str,
        prompt: str,
        temperature: float | None,
    ) -> str:
        return self._response_path(
            model=model,
            system=system,
            template=template,
            prompt=prompt,
            temperature=temperature,
        ).read_text()

    def set(
        self,
        model: str,
        system: str,
        template: str,
        prompt: str,
        temperature: float | None,
        value: str,
    ) -> None:
        response_path = self._response_path(
            model=model,
            system=system,
            template=template,
            prompt=prompt,
            temperature=temperature,
        )
        response_path.parent.mkdir(parents=True, exist_ok=True)

        system_txt = response_path.parent.parent / "system.txt"
        if not system_txt.exists():
            system_txt.write_text(system)

        message_txt = response_path.parent / "message.txt"
        if not message_txt.exists():
            message_txt.write_text(template)

        response_path.write_text(value)
