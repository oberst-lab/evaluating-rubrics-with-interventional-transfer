"""Score each edit from run.py with the rubric of its target criterion. Pickles one OwnScores for each example.

run.py scores each edit only with the other rubric. The own rubric tells two things:
    1. Did the target criterion change? This is the manipulation check.
    2. Which other criteria of the same rubric changed? This is the redundancy.
The output is results_<direction>_own.pkl, next to the run.py pickle.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/: llm_functions, common
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # utils

from tqdm import tqdm

from utils.cacher import Cacher
from utils.llm import LLM

from common.concurrency import parallel_map
from llm_functions import score_response
from common.results import CriteriaResult, InterventionalResult, OwnScores
from run import RunConfig, load_config


def main(cfg: RunConfig, direction: str) -> list[OwnScores]:
    sources: list[InterventionalResult] = pickle.loads(
        cfg.source_results_path.read_bytes()
    )
    interventions: list[CriteriaResult] = pickle.loads(
        (cfg.results_dir / f"results_{direction}.pkl").read_bytes()
    )

    llm = LLM(model_reasoning_effort=cfg.model_reasoning_effort)
    cacher = Cacher(cache_dir=cfg.cache_dir)

    out = []
    for record in tqdm(interventions):
        source = sources[record.source_index]
        rubrics = {"true": source.true_rubric, **source.rubrics}
        rewrites = {("true", i): text for i, (text, _) in record.on_expert.items()}
        rewrites |= {
            (g, i): text
            for g, by_index in record.on_generated.items()
            for i, (text, _) in by_index.items()
        }

        def score(target: tuple[str, int]):
            return target, score_response(
                conversation=source.example.conversation,
                response=rewrites[target],
                rubric=rubrics[target[0]],
                llm=llm,
                cacher=cacher,
                scoring_model=cfg.scoring_model,
                temperature=cfg.temperature,
                reasoning_effort=cfg.scoring_reasoning_effort,
                samples=cfg.judge_samples,
            ).result

        own = {}
        for target, result in parallel_map(
            work=list(rewrites), task=score, max_workers=cfg.max_workers
        ):
            if result is None:
                print(
                    f"{'-' * 50}\n scoring refusal: {record.source_index} {record.prompt_id}: {target}\n{'-' * 50}"
                )
                continue
            own[target] = result
        out.append(
            OwnScores(
                prompt_id=record.prompt_id, source_index=record.source_index, own=own
            )
        )

    path = cfg.results_dir / f"results_{direction}_own.pkl"
    path.write_bytes(pickle.dumps(out))
    print(f"wrote {len(out)} examples to {path}")
    return out


if __name__ == "__main__":
    cfg = load_config(
        config_path=Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "config.json"
    )
    for direction in cfg.directions:
        main(cfg=cfg, direction=direction)
