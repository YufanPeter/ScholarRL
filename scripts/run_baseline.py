"""Run baseline evaluation with StubPolicy (no model) or HFPolicy (real model).

Usage:
    python -m scripts.run_baseline --policy stub --split dev --num_queries 100
    python -m scripts.run_baseline --policy hf --split dev --num_queries 50
"""
import argparse
import json
from pathlib import Path

from scholarrl.config import load_config, build_env
from scholarrl.data.queries import load_queries
from scholarrl.retriever import BM25Retriever
from scholarrl.rollout import StubPolicy, HFPolicy, run_episode


def _set_seed(seed: int) -> None:
    """Seed python / numpy / torch RNGs (torch only if it's installed)."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Baseline agent evaluation")
    parser.add_argument("--policy", default="stub", choices=["stub", "hf"])
    parser.add_argument("--config", type=str, default=None, help="path to a config yaml (default: configs/base.yaml)")
    parser.add_argument("--model", default=None, help="model name for hf policy (default: config model.name)")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--num_queries", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--output", type=str, default=None, help="Output jsonl path")
    parser.add_argument("--save-trajectories", action="store_true", default=True,
                        help="Save full trajectories (messages/actions); default True. "
                             "Use --no-save-trajectories for lightweight summary-only mode.")
    parser.add_argument("--no-save-trajectories", dest="save_trajectories", action="store_false",
                        help="Skip saving full trajectories (summary only)")
    # Baseline eval should be reproducible: greedy decoding (temperature=0) by default,
    # plus a fixed seed. Pass --temperature >0 to sample instead.
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="hf sampling temperature; 0 = greedy/deterministic (default)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: config seed)")
    parser.add_argument("--keep-unanswerable", action="store_true",
                        help="keep queries whose gold has no retrievable paper (recall forced to 0); "
                             "default skips them so the score reflects what the agent can control")
    args = parser.parse_args()

    # Config is the source of truth for env budget, retriever top_k, reward, model,
    # and seed; CLI flags override it when given.
    cfg = load_config(Path(args.config)) if args.config else load_config()
    seed = args.seed if args.seed is not None else cfg.seed
    model_name = args.model or cfg.model.name

    # Fix all RNGs before building the policy so a run is reproducible.
    _set_seed(seed)

    # Load components
    retriever = BM25Retriever.load()
    if args.policy == "hf":
        print(f"Loading model {model_name} (temperature={args.temperature}) ...")
        policy = HFPolicy(model_name=model_name, temperature=args.temperature)
    else:
        policy = StubPolicy()
    env = build_env(retriever, cfg)   # env budget / top_k / reward all from config
    queries = load_queries(args.split)

    # Drop queries with no retrievable gold: their recall is forced to 0 no matter
    # what the agent does, so keeping them just deflates the score. Filter before the
    # --num_queries slice so the limit counts answerable queries.
    if not args.keep_unanswerable:
        from scholarrl.data.retrievable import retrievable_gold_ids
        from scholarrl.data.normalize import norm_arxiv_id
        ret = retrievable_gold_ids()
        n_before = len(queries)
        queries = [q for q in queries
                   if any(norm_arxiv_id(g) in ret for g in q.answer_ids)]
        n_skipped = n_before - len(queries)
        if n_skipped:
            print(f"Skipped {n_skipped} unanswerable queries (no retrievable gold); "
                  f"{len(queries)} remain. Use --keep-unanswerable to include them.")

    if args.num_queries is not None:
        queries = queries[:args.num_queries]

    if not queries:
        print(f"No queries to run for split={args.split} (num_queries={args.num_queries}).")
        return

    print(f"Running baseline on {len(queries)} queries from {args.split}...")

    # Run episodes
    trajectories = []
    total_reward = 0.0
    total_task_reward = 0.0

    for i, record in enumerate(queries):
        traj = run_episode(env, policy, record)
        trajectories.append(traj)
        total_reward += traj.reward
        total_task_reward += traj.task_reward

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} completed")

    # Report
    avg_reward = total_reward / len(queries)
    avg_task = total_task_reward / len(queries)

    print(f"\nBaseline results ({args.split}):")
    print(f"  Average reward:      {avg_reward:.4f}")
    print(f"  Average task reward: {avg_task:.4f}")
    print(f"  Episodes:            {len(trajectories)}")

    # Save trajectories and summary
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build metadata
        metadata = {
            "_meta": True,
            "split": args.split,
            "policy": args.policy,
            "model": model_name,
            "temperature": args.temperature,
            "seed": seed,
            "n_queries": len(queries),
            "avg_reward": avg_reward,
            "avg_task_reward": avg_task,
            "config": {
                "max_search_turns": cfg.env.max_search_turns,
                "max_read_turns": cfg.env.max_read_turns,
                "max_steps": cfg.env.max_steps,
                "top_k": cfg.retriever.top_k,
                "reward_metric": cfg.reward.metric,
                "reward_k": cfg.reward.k,
                "lambda_fmt": cfg.reward.lambda_fmt,
                "distractor_ratio": cfg.corpus.distractor_ratio,
            },
        }

        # Write JSONL: first line = metadata, subsequent lines = trajectories
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for traj in trajectories:
                obj = traj.to_dict(include_messages=args.save_trajectories)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"\nTrajectories saved to {output_path}")

        # Write human-readable summary
        summary_path = output_path.with_suffix(".summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"=== Baseline Evaluation Summary ===\n\n")
            f.write(f"Split:       {args.split}\n")
            f.write(f"Policy:      {args.policy}\n")
            f.write(f"Model:       {model_name}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Seed:        {seed}\n\n")

            f.write(f"--- Config ---\n")
            f.write(f"max_search_turns:    {cfg.env.max_search_turns}\n")
            f.write(f"max_read_turns:      {cfg.env.max_read_turns}\n")
            f.write(f"max_steps:           {cfg.env.max_steps}\n")
            f.write(f"top_k:               {cfg.retriever.top_k}\n")
            f.write(f"reward_metric:       {cfg.reward.metric}\n")
            f.write(f"reward_k:            {cfg.reward.k}\n")
            f.write(f"lambda_fmt:          {cfg.reward.lambda_fmt}\n")
            f.write(f"distractor_ratio:    {cfg.corpus.distractor_ratio}\n\n")

            f.write(f"--- Results ---\n")
            f.write(f"Queries evaluated:   {len(trajectories)}\n")
            f.write(f"Average reward:      {avg_reward:.4f}\n")
            f.write(f"Average task reward: {avg_task:.4f}\n")

            # Basic stats
            task_rewards = [t.task_reward for t in trajectories]
            hit_any = sum(1 for r in task_rewards if r > 0)
            search_turns = [t.search_turns for t in trajectories]
            read_turns = [t.read_turns for t in trajectories]
            search_capped = sum(1 for s in search_turns if s >= cfg.env.max_search_turns)
            read_capped = sum(1 for s in read_turns if s >= cfg.env.max_read_turns)

            f.write(f"\n--- Statistics ---\n")
            f.write(f"Recall@{cfg.reward.k} (task reward):\n")
            f.write(f"  Mean:   {avg_task:.4f}\n")
            f.write(f"  Median: {sorted(task_rewards)[len(task_rewards)//2]:.4f}\n")
            f.write(f"  Min:    {min(task_rewards):.4f}\n")
            f.write(f"  Max:    {max(task_rewards):.4f}\n")
            f.write(f"Hit rate (≥1 gold): {hit_any}/{len(task_rewards)} ({100*hit_any/len(task_rewards):.1f}%)\n")
            n = len(trajectories)
            f.write(f"\nSearch turns:\n")
            f.write(f"  Mean:   {sum(search_turns)/n:.1f}\n")
            f.write(f"  Budget-capped: {search_capped}/{n} ({100*search_capped/n:.1f}%)\n")
            f.write(f"Read turns:\n")
            f.write(f"  Mean:   {sum(read_turns)/n:.1f}\n")
            f.write(f"  Budget-capped: {read_capped}/{n} ({100*read_capped/n:.1f}%)\n")

        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
