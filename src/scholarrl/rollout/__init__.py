"""Rollout loop: query -> (policy acts, env responds)* -> reward + logged trajectory.

For each query, sample G trajectories (GRPO group). Log per step:
state, action, generated tokens, retrieved-token mask, final reward.
Trajectories -> outputs/trajectories/*.jsonl (reused by eval, GRPO, failure mining).

This is the closed loop; Phase 0 acceptance = this runs end-to-end, no gradient step yet.
"""
