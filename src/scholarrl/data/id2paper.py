"""Load the id2paper.json mapping (arxiv_id -> title, ~58MB, 569k entries).

Cached at module level so the 58MB file is read at most once per process.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

from ..paths import ID2PAPER

try:
    import orjson as _json

    def _loads(b: bytes):
        return _json.loads(b)

    _READ_MODE = "rb"
except ImportError:  # fall back to stdlib json
    import json as _json

    def _loads(b):
        return _json.loads(b)

    _READ_MODE = "r"


@lru_cache(maxsize=1)
def load_id2paper() -> Dict[str, str]:
    """arxiv_id -> title. Keys are normalized-style ids (e.g. '2404.00001')."""
    with open(ID2PAPER, _READ_MODE) as f:
        return _loads(f.read())
