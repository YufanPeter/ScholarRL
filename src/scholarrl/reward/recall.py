"""Task reward for Scholar-R1 (rule-based / RLVR, aligned with Search-R1's outcome reward).

Search-R1 scores QA answers with EM / F1 over gold strings. Our task is retrieval,
so the verifiable signal is Recall@K / F1 of the agent's selected paper ids against
the gold answer ids. Only task-reward logic lives here; format scoring is separate.

Two rules matter for a clean signal:
  1. ids are normalized on BOTH sides before comparison (a versioned selection like
     '2006.01043v2' must still match gold '2006.01043').
  2. gold is restricted to the RETRIEVABLE subset — papers actually present in the
     corpus. Gold that can never be found would otherwise depress the reward through
     no fault of the agent.
"""

from __future__ import annotations

from typing import Iterable, Set

from scholarrl.data.normalize import norm_arxiv_id
from scholarrl.data.retrievable import retrievable_gold_ids


def _norm_set(ids: Iterable[str]) -> Set[str]:
    """Normalize arxiv ids (strip version / category) and drop empties."""
    return {n for n in (norm_arxiv_id(x) for x in ids if x) if n}


def recall_at_k(selected: Iterable[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold ids found within the first k selected ids. Denominator = |gold|.

    Order matters: only the first k selections count (the agent should put its best
    picks first). Duplicate selections collapse. Empty gold -> 0.0.
    """
    gold_set = _norm_set(gold)
    if not gold_set:
        return 0.0
    # keep order, take first k, then normalize
    topk = _norm_set(list(selected)[:k])
    hit = len(topk & gold_set)
    return hit / len(gold_set)


def f1(selected: Iterable[str], gold: Iterable[str]) -> float:
    """Harmonic mean of precision and recall over the FULL selected set.

    Precision penalizes selecting irrelevant papers; recall rewards finding gold.
    Any empty side, or zero hits, -> 0.0.
    """
    sel_set = _norm_set(selected)
    gold_set = _norm_set(gold)
    if not sel_set or not gold_set:
        return 0.0
    hit = len(sel_set & gold_set)
    if hit == 0:
        return 0.0
    precision = hit / len(sel_set)
    recall = hit / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def task_reward(
    selected: Iterable[str],
    gold: Iterable[str],
    metric: str = "recall",
    k: int = 20,
) -> float:
    """Single entry point for the env / trainer.

    Restricts gold to the retrievable subset (papers actually in the corpus) so the
    signal reflects what the agent can control, then dispatches on `metric`.
    """
    retrievable = retrievable_gold_ids()  # frozenset of already-normalized ids
    gold_ret = [g for g in gold if norm_arxiv_id(g) in retrievable]

    if metric == "recall":
        return recall_at_k(selected, gold_ret, k)
    if metric == "f1":
        return f1(selected, gold_ret)
    raise ValueError(f"unknown reward metric: {metric!r} (expected 'recall' or 'f1')")
