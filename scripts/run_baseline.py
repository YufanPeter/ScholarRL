"""Run baseline evaluation with StubPolicy (no model) or HFPolicy (real model).

Usage:
    python -m scripts.run_baseline --policy stub --split dev --num_queries 100
    python -m scripts.run_baseline --policy hf --split dev --num_queries 50
"""
import argparse
import json
from pathlib import Path

from scholarrl.data.queries import load_queries
from scholarrl.env import SearchEnv
from scholarrl.retriever import BM25Retriever
from scholarrl.rollout import StubPolicy, HFPolicy, run_episode


def main():
    parser = argparse.ArgumentParser(description="Baseline agent evaluation")
    parser.add_argument("--policy", default="stub", choices=["stub", "hf"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="model name for hf policy")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--num_queries", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--output", type=str, default=None, help="Output jsonl path")
    args = parser.parse_args()

    # Load components
    retriever = BM25Retriever.load()
    if args.policy == "hf":
        print(f"Loading model {args.model} ...")
        policy = HFPolicy(model_name=args.model)
    else:
        policy = StubPolicy()
    env = SearchEnv(retriever)
    queries = load_queries(args.split)

    if args.num_queries:
        queries = queries[:args.num_queries]

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

    # Save trajectories
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                obj = {
                    "query_id": traj.query_id,
                    "question": traj.question,
                    "reward": traj.reward,
                    "task_reward": traj.task_reward,
                    "format_reward": traj.format_reward,
                    "selected": traj.selected,
                    "gold": traj.gold,
                    "retrieval_turns": traj.retrieval_turns,
                    "steps": traj.steps,
                    "reason": traj.reason,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"\nTrajectories saved to {output_path}")


if __name__ == "__main__":
    main()
