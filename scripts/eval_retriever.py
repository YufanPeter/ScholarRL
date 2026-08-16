"""Sanity-check the BM25 retriever: gold recall@k using the RAW question as query.

Run:  python -m scripts.eval_retriever [k] [n_queries]

This does NOT involve any agent or training. It answers two questions:
  - is the retriever wired up correctly? (raw questions should find *some* gold)
  - is the task the right difficulty? (if raw questions already find ALL gold, the task
    is too easy and leaves nothing for the agent to learn via query rewriting)

recall@k here = fraction of a query's retrievable gold ids that appear in the top-k.
Only retrievable gold (papers actually in the corpus) counts toward the denominator.
"""
from __future__ import annotations

import sys

from src.scholarrl.data import load_queries, retrievable_gold_ids
from src.scholarrl.retriever import BM25Retriever


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    r = BM25Retriever.load()
    retr = retrievable_gold_ids()
    queries = load_queries("dev")[:n]

    recalls = []
    hit_any = 0
    for q in queries:
        gold = [a for a in q.answer_ids if a in retr]
        if not gold:
            continue
        found = {pid for pid, _ in r.search(q.question, k=k)}
        hits = sum(1 for g in gold if g in found)
        recalls.append(hits / len(gold))
        if hits > 0:
            hit_any += 1

    n_eval = len(recalls)
    avg = sum(recalls) / n_eval if n_eval else 0.0
    perfect = sum(1 for x in recalls if x == 1.0)
    print(f"BM25 zero-rewrite baseline on {n_eval} dev queries (k={k}):")
    print(f"  mean gold recall@{k}   : {avg:.3f}")
    print(f"  queries w/ >=1 gold hit: {hit_any}/{n_eval}  ({100*hit_any/n_eval:.0f}%)")
    print(f"  queries w/ ALL gold    : {perfect}/{n_eval}  ({100*perfect/n_eval:.0f}%)")
    print("\ninterpretation: high mean recall -> task too easy (little to learn);")
    print("                low/moderate -> room for the agent to improve via rewriting.")


if __name__ == "__main__":
    main()
