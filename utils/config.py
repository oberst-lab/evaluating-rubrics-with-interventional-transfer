import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "healthbench_data"
# All figure and table generators write into this folder.
PAPER_INPUTS = Path(os.environ.get("PAPER_INPUTS", REPO_ROOT / "out" / "inputs"))
