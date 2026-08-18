"""Load AutoScholarQuery records from the train/dev/test jsonl files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from ..paths import TRAIN_JSONL, DEV_JSONL, TEST_JSONL

_SPLIT_PATHS = {"train": TRAIN_JSONL, "dev": DEV_JSONL, "test": TEST_JSONL}


@dataclass
class QueryRecord:
    """One AutoScholarQuery example."""
    qid: str
    question: str
    answer_titles: List[str] = field(default_factory=list)   # gold paper titles
    answer_ids: List[str] = field(default_factory=list)       # gold arxiv ids (reward key)
    published_time: str = ""                                  # source paper's publish date

    @classmethod
    def from_json(cls, obj: dict) -> "QueryRecord":
        return cls(
            qid=obj.get("qid", ""),
            question=obj.get("question", ""),
            answer_titles=obj.get("answer", []) or [],
            answer_ids=obj.get("answer_arxiv_id", []) or [],
            published_time=(obj.get("source_meta") or {}).get("published_time", ""),
        )

    def availability(self, retrievable: set) -> str:
        """'full' | 'partial' | 'none' | 'empty' given the set of retrievable gold ids.

        `retrievable` holds normalized ids, so answer_ids are normalized before the
        membership test (a versioned answer_id must still match its normalized entry).
        """
        from .normalize import norm_arxiv_id
        if not self.answer_ids:
            return "empty"
        hit = sum(1 for a in self.answer_ids if norm_arxiv_id(a) in retrievable)
        if hit == 0:
            return "none"
        if hit == len(self.answer_ids):
            return "full"
        return "partial"


def load_queries(split: str, only_retrievable: bool = False) -> List[QueryRecord]:
    """Load one split: 'train' | 'dev' | 'test'.

    only_retrievable=True drops queries whose answers are ALL missing from the corpus
    ('none' / 'empty') — they can never score, so they are noise for training.
    'partial' queries are kept (some answers are reachable). Intended for the train split;
    for dev/test prefer reporting both full-set and satisfiable-subset metrics instead.
    """
    if split not in _SPLIT_PATHS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(_SPLIT_PATHS)}")
    path = _SPLIT_PATHS[split]
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(QueryRecord.from_json(json.loads(line)))

    if only_retrievable:
        from .retrievable import retrievable_gold_ids
        r = retrievable_gold_ids()
        records = [rec for rec in records if rec.availability(r) in ("full", "partial")]
    return records


def all_gold_ids() -> set:
    """Union of gold arxiv ids across all three splits (the gold pool for the corpus)."""
    ids = set()
    for split in ("train", "dev", "test"):
        for r in load_queries(split):
            ids.update(r.answer_ids)
    return ids


# Section-header phrases that leaked into id2paper/zip as fake titles.
_JUNK_TITLES = {
    "introduction", "abstract", "references", "related work", "background",
    "conclusion", "conclusions", "methodology", "methods", "experiments", "results",
}


def _looks_like_title(title: str) -> bool:
    """A usable human title: non-empty, >3 chars, and not a bare section header."""
    if not title:
        return False
    t = title.strip()
    if len(t) <= 3:
        return False
    import re
    canon = re.sub(r"\s+", " ", re.sub(r"^[\d\.\s]+", "", t.lower())).strip()
    return canon not in _JUNK_TITLES


def gold_id_to_title() -> dict:
    """Map normalized gold id -> a good human title from AutoScholarQuery answer_titles.

    Used to backfill corpus papers whose zip-extracted title is junk ('1 Introduction').
    answer_ids and answer_titles are paired lists; we keep the first usable title per id.
    """
    from .normalize import norm_arxiv_id
    out: dict = {}
    for split in ("train", "dev", "test"):
        for r in load_queries(split):
            for aid, title in zip(r.answer_ids, r.answer_titles):
                nid = norm_arxiv_id(aid)
                if nid and nid not in out and _looks_like_title(title):
                    out[nid] = title.strip()
    return out
