"""Single source of truth for filesystem paths.

Override the data root with the SCHOLAR_DATA env var (e.g. on a rented GPU box);
code never hard-codes absolute paths elsewhere.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SCHOLAR_DATA", REPO_ROOT / "data"))

# configs (checked into the repo, not gitignored)
CONFIG_DIR = REPO_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "base.yaml"

# raw inputs (symlinked from the Desktop downloads; gitignored)
RAW_DIR = DATA_DIR / "raw"
TRAIN_JSONL = RAW_DIR / "train.jsonl"
DEV_JSONL = RAW_DIR / "dev.jsonl"
TEST_JSONL = RAW_DIR / "test.jsonl"
ID2PAPER = RAW_DIR / "id2paper.json"
CORPUS_ZIP = RAW_DIR / "cs_paper_2nd.zip"

# built artifacts (generated locally; gitignored)
CORPUS_DIR = DATA_DIR / "corpus"
PAPERS_JSONL = CORPUS_DIR / "papers.jsonl"          # subset corpus: {paper_id, title, abstract}
# cached namelist of CORPUS_ZIP (one filename per line). Lets a box that has the built
# index but NOT the 2.3GB zip still compute retrievable_gold_ids() — see data/retrievable.py.
ZIP_NAMELIST_CACHE = CORPUS_DIR / "zip_namelist.txt"

# run outputs (gitignored)
OUTPUTS_DIR = REPO_ROOT / "outputs"
TRAJ_DIR = OUTPUTS_DIR / "trajectories"
CKPT_DIR = OUTPUTS_DIR / "checkpoints"
LOG_DIR = OUTPUTS_DIR / "logs"
