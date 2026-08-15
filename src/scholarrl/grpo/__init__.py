"""GRPO in two stages (see plan Phase 2).

Stage A - minimal GRPO from scratch (the learning core, single A100):
    group-relative advantage = (r_i - mean(r_group)) / (std(r_group) + eps)
    policy-gradient loss with retrieved-token masking
    KL to reference (LoRA adapter disabled = base weights)

Stage B - port the same env/reward into veRL's GRPO recipe for the scaled run.
Watch for GRPO collapse; keep the BEST dev checkpoint, not the last.
"""
