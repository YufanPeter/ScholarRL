"""Eyeball what BM25 retrieves for a query — a manual inspection tool.

Companion to eval_retriever.py: that one gives the *number* (mean recall@k),
this one shows the *cases* (which gold hit or missed, and what came back instead),
so you can see WHY a query is hard and what the agent needs to learn.

Run:
  python -m scripts.search_demo                        # a few sample dev queries
  python -m scripts.search_demo "your free-text query" # your own query

In dev-sample mode it shows the gold answers and marks which top-k hits are gold,
so you can see how well the RAW question does (the baseline the agent must beat).
"""
from __future__ import annotations

import sys

from scholarrl.data import load_queries, retrievable_gold_ids
from scholarrl.retriever import BM25Retriever

K = 5


def show(r, question, gold_ids=None, gold_titles=None):
    print("=" * 90)
    print(f"QUERY: {question}")
    if gold_ids is not None:
        retr = retrievable_gold_ids()
        print(f"GOLD ({len(gold_ids)} answers, * = in corpus):")
        for gid, gt in zip(gold_ids, gold_titles or gold_ids):
            mark = "*" if gid in retr else " "
            print(f"   [{mark}] {gid:16} {gt[:65]}")
    print(f"\nBM25 top-{K}:")
    goldset = set(gold_ids or [])
    for rank, (pid, score) in enumerate(r.search(question, k=K), 1):
        hit = "<== GOLD" if pid in goldset else ""
        title = r.get(pid).get("title", "")[:60]
        print(f"  {rank:2}. {score:6.2f}  {pid:16} {title} {hit}")
    print()


def main() -> None:
    r = BM25Retriever.load()

    if len(sys.argv) > 1:
        show(r, " ".join(sys.argv[1:]))
        return

    # a few dev queries with varied difficulty
    dev = load_queries("dev")
    for q in dev[:4]:
        show(r, q.question, q.answer_ids, q.answer_titles)


if __name__ == "__main__":
    main()
