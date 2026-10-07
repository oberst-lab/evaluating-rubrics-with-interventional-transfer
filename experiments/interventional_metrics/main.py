"""
run.py:         creates the data by making API calls
evaluate.py:    reads the pickle written by run.py
figures.py:     reads the json written by evaluate.py to create the output latex

This file runs run.py, then evaluate.py, for each response model in config.json.
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
    run_cfg = run.load_config(config_path=config_path)
    evaluate_cfg = evaluate.load_config(config_path=config_path)
    for response_model in run_cfg.response_models:
        run.main(cfg=run_cfg, response_model=response_model)
        evaluate.main(cfg=evaluate_cfg, response_model=response_model)
