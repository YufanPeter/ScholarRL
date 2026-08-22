"""Measure the CANDIDATE-POOL CEILING: what fraction of gold can the agent even see?

Motivation. The dev-50 3B baseline scored task_reward 0.083, but BM25 only ever surfaced
16.4% of the gold papers across a whole episode. An agent cannot select a paper it never
saw, so 0.164 was the hard ceiling and 0.083 was already half of it. Before training,
the ceiling has to be raised — otherwise every sampled trajectory in a GRPO group scores
~0, the group advantage collapses to 0, and no gradient flows.

Two questions, two tables:

  Table A (raw question, no agent) — how much does top_k alone buy?
      The zero-rewrite baseline: feed the untouched question to BM25 at increasing k.

  Table B (replay the agent's own queries) — how much does budget shape buy?
      Replays the <search> strings the model actually emitted in a baseline run, so the
      grid holds the policy fixed and varies only (n_searches x top_k). That is exactly
      the knob `max_search_turns` / `retriever.top_k` control.

  Table C (oracle: search the gold TITLE) — what is BM25 itself capable of?
      Separates "BM25 can never find this paper" from "the query was bad". The gap
      between B and C is the headroom RL can actually recover by learning to rewrite;
      whatever C leaves on the table is a retriever limit, not a policy limit.

Ceiling metric = |pool ∩ retrievable_gold| / |retrievable_gold|, averaged over queries.
Gold is restricted to the retrievable subset so the number is comparable to task_reward
(see reward/recall.py, which applies the same restriction).

Run:
    python -m scripts.ceiling --baseline baseline_3b_dev50.jsonl
    python -m scripts.ceiling --split dev --n 50          # Table A only
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set

from scholarrl.data import load_queries, retrievable_gold_ids
from scholarrl.data.normalize import norm_arxiv_id
from scholarrl.retriever import BM25Retriever

K_GRID = (5, 10, 20, 50)
N_SEARCH_GRID = (1, 2, 3, 6)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S)


class _PoolCache:
    """Search each distinct query string once at max(K_GRID); smaller k is a prefix."""

    def __init__(self, retriever: BM25Retriever, max_k: int):
        self._retriever = retriever
        self._max_k = max_k
        self._cache: Dict[str, List[str]] = {}

    def top(self, query: str, k: int) -> List[str]:
        hits = self._cache.get(query)
        if hits is None:
            hits = [norm_arxiv_id(pid) for pid, _ in self._retriever.search(query, k=self._max_k)]
            self._cache[query] = hits
        return hits[:k]

    def pool(self, queries: Sequence[str], k: int) -> Set[str]:
        out: Set[str] = set()
        for q in queries:
            out.update(self.top(q, k))
        return out


def _coverage(pool: Set[str], gold: Set[str]) -> float:
    return len(pool & gold) / len(gold) if gold else 0.0


def _report_row(label: str, covs: List[float]) -> str:
    n = len(covs)
    mean = sum(covs) / n if n else 0.0
    hit_any = sum(1 for c in covs if c > 0)
    return f"  {label:<22} ceiling {mean:6.1%}   >=1 gold in pool {hit_any:>3}/{n}"


def _load_cases(baseline: Path | None, split: str, n: int) -> List[dict]:
    """Each case: {'question': str, 'gold': set, 'searches': [str]} with gold retrievable."""
    retr = retrievable_gold_ids()
    cases: List[dict] = []

    if baseline is not None:
        with open(baseline, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        for r in records:
            if "query_id" not in r:  # skip the _meta header line
                continue
            gold = {norm_arxiv_id(g) for g in r["gold"] if norm_arxiv_id(g) in retr}
            if not gold:
                continue
            searches = [m.group(1).strip()
                        for a in r.get("actions", [])
                        for m in [_SEARCH_RE.search(a)] if m and m.group(1).strip()]
            cases.append({"question": r["question"], "gold": gold, "searches": searches})
        return cases

    for q in load_queries(split)[: n * 3]:
        gold = {norm_arxiv_id(a) for a in q.answer_ids if norm_arxiv_id(a) in retr}
        if gold:
            cases.append({"question": q.question, "gold": gold, "searches": []})
        if len(cases) >= n:
            break
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 candidate-pool ceiling under different budgets")
    ap.add_argument("--baseline", type=str, default=None,
                    help="baseline jsonl with trajectories; enables Table B (query replay)")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n", type=int, default=50, help="queries to evaluate when no --baseline")
    args = ap.parse_args()

    cases = _load_cases(Path(args.baseline) if args.baseline else None, args.split, args.n)
    if not cases:
        print("no answerable queries found (all gold unretrievable?)")
        return

    retriever = BM25Retriever.load()
    cache = _PoolCache(retriever, max_k=max(K_GRID))
    print(f"corpus: {len(retriever)} papers | queries evaluated: {len(cases)}\n")

    print("Table A — raw question, 1 search (zero-rewrite baseline)")
    for k in K_GRID:
        covs = [_coverage(cache.pool([c["question"]], k), c["gold"]) for c in cases]
        print(_report_row(f"1 search x top_{k}", covs))

    replayed = [c for c in cases if c["searches"]]
    if not replayed:
        print("\n(no --baseline trajectories: skipping Table B)")
        return

    used = [len(c["searches"]) for c in replayed]
    max_used = max(used)
    print(f"\nTable B — replay of the agent's own rewrites "
          f"({len(replayed)} queries, avg {sum(used)/len(used):.1f} searches emitted)")
    for n_search in N_SEARCH_GRID:
        # rows beyond what the model actually emitted just repeat the last row
        note = "  (capped by emitted count)" if n_search > max_used else ""
        for k in K_GRID:
            covs = [_coverage(cache.pool(c["searches"][:n_search], k), c["gold"]) for c in replayed]
            print(_report_row(f"{n_search} searches x top_{k}", covs) + note)
        print()

    print("Table C — ORACLE: one search per gold paper, using its exact title as the query")
    for k in K_GRID:
        covs = []
        for c in replayed:
            titles = [t for t in (retriever.get(g).get("title", "").strip() for g in c["gold"]) if t]
            covs.append(_coverage(cache.pool(titles, k), c["gold"]) if titles else 0.0)
        print(_report_row(f"oracle title x top_{k}", covs))

    print("\nread: the current run is '3 searches x top_5'. Across a row = the top_k knob;")
    print("down a column = the search-count knob. Table C is the retriever's own limit —")
    print("the B-to-C gap is the headroom RL can recover by learning to rewrite queries.")


if __name__ == "__main__":
    main()
