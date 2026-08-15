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
        """'full' | 'partial' | 'none' | 'empty' given the set of retrievable gold ids."""
        if not self.answer_ids:
            return "empty"
        hit = sum(1 for a in self.answer_ids if a in retrievable)
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
