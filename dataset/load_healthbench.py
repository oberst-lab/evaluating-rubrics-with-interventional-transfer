"""Download the two HealthBench splits into healthbench_data/.

  oss_eval  5,000 examples. The experiments take their sample from this split.
  hard      1,000 examples. We remove them from the sample. They give the foreign rubrics.

Usage, from the repository root:
  python dataset/load_healthbench.py
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))  # utils

from utils.config import DATA_DIR

FILES = {
    "oss_eval": "https://huggingface.co/datasets/openai/healthbench/resolve/main/2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "https://huggingface.co/datasets/openai/healthbench/resolve/main/hard_2025-05-08-21-00-10.jsonl",
}


def download(name: str, url: str, dest: Path) -> Path:
    path = dest / f"{name}.jsonl"
    if path.exists():
        print(f"  {name}: already downloaded")
        return path
    print(f"  {name}: downloading...")
    response = requests.get(url)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"  {name}: {path.stat().st_size / 1e6:.1f} MB")
    return path


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in FILES.items():
        download(name=name, url=url, dest=DATA_DIR)
