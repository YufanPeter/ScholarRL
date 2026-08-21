"""Analyze baseline results from run_baseline.py output.

Usage:
    python -m scripts.analyze_baseline outputs/baseline_3b_dev.jsonl
    python -m scripts.analyze_baseline outputs/baseline_3b_dev.jsonl --show-failures 5

Reads a JSONL file (first line = metadata, rest = trajectories) and prints:
- Configuration used
- Recall@K distribution (mean, median, min, max, std)
- Hit rate (% of queries with ≥1 gold found)
- Budget usage (avg retrieval turns, % budget-capped)
- Failure analysis (queries with recall=0)
- Optional: show sample failure cases with their questions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def analyze(path: Path, show_failures: int = 0) -> None:
    """Load and analyze baseline results."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    if not lines:
        print("Error: empty file")
        return

    # First line is metadata (if present)
    if lines[0].get("_meta"):
        meta = lines[0]
        trajs = lines[1:]
    else:
        print("Warning: no metadata found (old format?)")
        meta = {}
        trajs = lines

    if not trajs:
        print("Error: no trajectories found")
        return

    # Extract fields
    task_rewards = [t["task_reward"] for t in trajs]
    rewards = [t["reward"] for t in trajs]
    format_rewards = [t["format_reward"] for t in trajs]
    retrieval_turns = [t["retrieval_turns"] for t in trajs]
    steps = [t["steps"] for t in trajs]

    # Display metadata
    if meta:
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print(f"Split:        {meta.get('split', '?')}")
        print(f"Policy:       {meta.get('policy', '?')}")
        print(f"Model:        {meta.get('model', '?')}")
        print(f"Temperature:  {meta.get('temperature', '?')}")
        print(f"Seed:         {meta.get('seed', '?')}")
        print(f"N queries:    {meta.get('n_queries', len(trajs))}")
        if "config" in meta:
            cfg = meta["config"]
            print(f"\nEnv config:")
            print(f"  max_retrieval_turns: {cfg.get('max_retrieval_turns', '?')}")
            print(f"  max_steps:           {cfg.get('max_steps', '?')}")
            print(f"  top_k:               {cfg.get('top_k', '?')}")
            print(f"  reward_metric:       {cfg.get('reward_metric', '?')}")
            print(f"  reward_k:            {cfg.get('reward_k', '?')}")
            print(f"  lambda_fmt:          {cfg.get('lambda_fmt', '?')}")
            print(f"  distractor_ratio:    {cfg.get('distractor_ratio', '?')}")

    # Compute statistics
    def stats(vals: List[float]) -> dict:
        import math
        n = len(vals)
        mean = sum(vals) / n
        sorted_vals = sorted(vals)
        median = sorted_vals[n // 2]
        variance = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(variance)
        return {
            "mean": mean,
            "median": median,
            "min": min(vals),
            "max": max(vals),
            "std": std,
        }

    task_stats = stats(task_rewards)
    reward_stats = stats(rewards)
    format_stats = stats(format_rewards)

    # Hit rate
    hit_any = sum(1 for r in task_rewards if r > 0)
    hit_rate = 100 * hit_any / len(task_rewards)

    # Budget usage
    avg_retrieval_turns = sum(retrieval_turns) / len(retrieval_turns)
    avg_steps = sum(steps) / len(steps)
    if meta and "config" in meta:
        max_ret = meta["config"].get("max_retrieval_turns", float("inf"))
        budget_capped = sum(1 for r in retrieval_turns if r >= max_ret)
        budget_capped_pct = 100 * budget_capped / len(retrieval_turns)
    else:
        budget_capped = budget_capped_pct = None

    # Print results
    print("\n" + "=" * 80)
    print("TASK REWARD (Recall@K)")
    print("=" * 80)
    print(f"Mean:     {task_stats['mean']:.4f}")
    print(f"Median:   {task_stats['median']:.4f}")
    print(f"Std dev:  {task_stats['std']:.4f}")
    print(f"Min:      {task_stats['min']:.4f}")
    print(f"Max:      {task_stats['max']:.4f}")
    print(f"Hit rate (≥1 gold): {hit_any}/{len(task_rewards)} ({hit_rate:.1f}%)")

    print("\n" + "=" * 80)
    print("TOTAL REWARD (task + λ·format)")
    print("=" * 80)
    print(f"Mean:     {reward_stats['mean']:.4f}")
    print(f"Median:   {reward_stats['median']:.4f}")
    print(f"Std dev:  {reward_stats['std']:.4f}")
    print(f"Min:      {reward_stats['min']:.4f}")
    print(f"Max:      {reward_stats['max']:.4f}")

    print("\n" + "=" * 80)
    print("FORMAT REWARD")
    print("=" * 80)
    print(f"Mean:     {format_stats['mean']:+.4f}")
    print(f"Median:   {format_stats['median']:+.4f}")
    print(f"Std dev:  {format_stats['std']:.4f}")
    print(f"Min:      {format_stats['min']:+.4f}")
    print(f"Max:      {format_stats['max']:+.4f}")

    print("\n" + "=" * 80)
    print("BUDGET USAGE")
    print("=" * 80)
    print(f"Avg retrieval turns: {avg_retrieval_turns:.2f}")
    print(f"Avg total steps:     {avg_steps:.2f}")
    if budget_capped is not None:
        print(f"Budget-capped:       {budget_capped}/{len(retrieval_turns)} ({budget_capped_pct:.1f}%)")

    # Failure analysis
    failures = [t for t in trajs if t["task_reward"] == 0]
    print("\n" + "=" * 80)
    print(f"FAILURES (task_reward = 0): {len(failures)}/{len(trajs)} ({100*len(failures)/len(trajs):.1f}%)")
    print("=" * 80)
    if failures and show_failures > 0:
        print(f"\nShowing {min(show_failures, len(failures))} sample failure cases:\n")
        for i, fail in enumerate(failures[:show_failures], 1):
            print(f"{i}. Query ID: {fail['query_id']}")
            print(f"   Question: {fail['question'][:100]}...")
            print(f"   Gold:     {fail.get('gold', [])}")
            print(f"   Selected: {fail.get('selected', [])}")
            print(f"   Reason:   {fail.get('reason', '?')}")
            print(f"   Retrieval turns: {fail.get('retrieval_turns', '?')}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze baseline evaluation results")
    parser.add_argument("jsonl", type=Path, help="Path to baseline output JSONL")
    parser.add_argument("--show-failures", type=int, default=0, metavar="N",
                        help="Show N sample failure cases (default: 0)")
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"Error: file not found: {args.jsonl}")
        sys.exit(1)

    analyze(args.jsonl, show_failures=args.show_failures)


if __name__ == "__main__":
    main()
