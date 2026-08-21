"""Build the subset retrieval corpus.

Config-driven: the `corpus:` section of configs/base.yaml is the source of truth.
CLI flags override the yaml so you can sweep difficulty without editing the file.

Run:  python -m scripts.build_corpus
      python -m scripts.build_corpus --distractor-ratio 10
      python -m scripts.build_corpus --config configs/base.yaml --gold-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scholarrl.config import load_config
from scholarrl.corpus import build_corpus
from scholarrl.paths import BASE_CONFIG, PAPERS_JSONL


def main() -> None:
    p = argparse.ArgumentParser(description="Build the subset retrieval corpus.")
    p.add_argument("--config", type=Path, default=BASE_CONFIG,
                   help="path to yaml config (default: configs/base.yaml)")
    p.add_argument("--distractor-ratio", type=int, default=None,
                   help="override corpus.distractor_ratio from config")
    p.add_argument("--gold-only", action="store_true",
                   help="override: drop random distractors (gold-only corpus)")
    p.add_argument("--seed", type=int, default=None,
                   help="override top-level seed from config")
    args = p.parse_args()

    cfg = load_config(args.config)

    # yaml is the default; CLI flags override when present.
    ratio = args.distractor_ratio if args.distractor_ratio is not None else cfg.corpus.distractor_ratio
    include = False if args.gold_only else cfg.corpus.include_random_distractors
    seed = args.seed if args.seed is not None else cfg.seed

    print(f"building subset corpus (distractor_ratio={ratio}, "
          f"include_random_distractors={include}, seed={seed}) ...")

    # hard_negative_topk is a Phase-2 knob (needs a trained retriever); surface it
    # instead of silently ignoring the config value.
    if cfg.corpus.hard_negative_topk:
        print(f"  note: corpus.hard_negative_topk={cfg.corpus.hard_negative_topk} "
              f"is reserved for Phase 2 (hard negatives) and not applied yet")

    counts = build_corpus(
        distractor_ratio=ratio,
        include_random_distractors=include,
        seed=seed,
    )
    print(f"  gold papers    : {counts['gold']}")
    print(f"  distractors    : {counts['distractors']}")
    print(f"  total          : {counts['total']}")
    print(f"  skipped (empty): {counts['skipped']}")
    print(f"  backfilled title: {counts.get('backfilled_titles', 0)}")
    print(f"  -> {PAPERS_JSONL}")


if __name__ == "__main__":
    main()
