"""Build the subset retrieval corpus.

Run:  python -m scripts.build_corpus [distractor_ratio]

Writes data/corpus/papers.jsonl and prints counts.
"""
from __future__ import annotations

import sys

from src.scholarrl.corpus import build_corpus
from src.scholarrl.paths import PAPERS_JSONL


def main() -> None:
    ratio = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"building subset corpus (distractor_ratio={ratio}) ...")
    counts = build_corpus(distractor_ratio=ratio)
    print(f"  gold papers    : {counts['gold']}")
    print(f"  distractors    : {counts['distractors']}")
    print(f"  total          : {counts['total']}")
    print(f"  skipped (empty): {counts['skipped']}")
    print(f"  -> {PAPERS_JSONL}")


if __name__ == "__main__":
    main()
