"""Build the BM25 index over data/corpus/papers.jsonl.

Run:  python -m scripts.build_index

Writes the index to data/corpus/bm25_index/ and prints stats.
"""
from __future__ import annotations

import time

from scholarrl.retriever import BM25Retriever, INDEX_DIR


def main() -> None:
    t0 = time.time()
    r = BM25Retriever.build_index()
    print(f"\nindexed {len(r)} papers in {time.time() - t0:.1f}s")
    print(f"  -> {INDEX_DIR}")
    # quick smoke query
    hits = r.search("retrieval augmented generation for question answering", k=3)
    print("\nsmoke query 'retrieval augmented generation for question answering':")
    for pid, score in hits:
        title = r.get(pid).get("title", "")[:70]
        print(f"  {score:6.2f}  {pid:16}  {title}")


if __name__ == "__main__":
    main()
