"""Run run.py, then evaluate.py. They exchange data through the pickle in config.json.

figures.py is a separate step.
"""

import sys
from pathlib import Path

import run
import evaluate

if __name__ == "__main__":
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "config.json"
    )
    run.main(cfg=run.load_config(config_path=config_path))
    evaluate.main(cfg=evaluate.load_config(config_path=config_path))
