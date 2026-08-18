"""Which gold papers are end-to-end retrievable (present as a real file in the zip).

This is the one place in `data/` that touches the zip — but only the namelist (filenames),
never the content. Reading paper content is the corpus module's job.

Cached: the namelist is read once per process.
"""
from __future__ import annotations

import zipfile
from functools import lru_cache
from typing import Set

from ..paths import CORPUS_ZIP
from .queries import all_gold_ids
from .resolve import resolve_gold


@lru_cache(maxsize=1)
def zip_filenames() -> frozenset:
    """All filenames inside cs_paper_2nd.zip (namelist only, no extraction)."""
    with zipfile.ZipFile(CORPUS_ZIP) as zf:
        return frozenset(zf.namelist())


@lru_cache(maxsize=1)
def retrievable_gold_ids() -> frozenset:
    """Gold arxiv ids whose paper resolves to a real file in the zip (end-to-end).

    Ids are NORMALIZED (version suffix stripped) so this set is directly comparable
    to the normalized ids used by reward / corpus. Different versions of the same
    paper (e.g. '2202.07565v1' and '2202.07565') collapse to one entry.
    """
    names = zip_filenames()
    out: Set[str] = set()
    for a in all_gold_ids():
        res = resolve_gold(a)
        if res.filename and res.filename in names:
            out.add(res.norm_id)
    return frozenset(out)
