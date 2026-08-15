"""Turn cs_paper_2nd.zip into a usable subset corpus.

Public API:
    build_corpus(distractor_ratio=5, seed=42) -> counts dict
        writes data/corpus/papers.jsonl  {paper_id, title, abstract}

- gold papers keyed by arxiv_id (reward-matchable); distractors keyed by zip filename.
- SUBSET: every retrievable gold + distractor_ratio * |gold| random distractors.
"""
from .build import build_corpus

__all__ = ["build_corpus"]
