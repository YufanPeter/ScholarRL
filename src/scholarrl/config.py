"""Load configs/base.yaml into typed dataclasses — the single source of truth for
hyperparameters (env budget, retriever top_k, reward weights, model, GRPO).

Why this exists: env/reward params used to live only as SearchEnv constructor
defaults, with base.yaml as unwired documentation that silently drifted. Now the
yaml is authoritative: scripts call load_config() and build_env() so changing a
value in one place actually takes effect everywhere.

Fields mirror the yaml sections. Unknown keys raise, so a typo in the yaml fails
loudly instead of being silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

from scholarrl.paths import BASE_CONFIG

T = TypeVar("T")


@dataclass
class CorpusConfig:
    distractor_ratio: int = 5
    hard_negative_topk: int = 20
    include_random_distractors: bool = True


@dataclass
class RetrieverConfig:
    kind: str = "bm25"          # bm25 (V1) | dense (V2)
    top_k: int = 5              # results returned per <search>


@dataclass
class EnvConfig:
    max_retrieval_turns: int = 8    # budget for expensive actions (search + read)
    max_steps: int = 20             # hard cap on total steps
    dedup_queries: bool = True
    abstract_max_words: int = 256


@dataclass
class RewardConfig:
    metric: str = "recall"          # recall | f1
    k: int = 20
    lambda_fmt: float = 0.1
    alpha_search_cost: float = 0.0  # Phase 3 (off for now)
    beta_token_cost: float = 0.0


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_rank: int = 32


@dataclass
class GRPOConfig:
    group_size: int = 8
    kl_coef: float = 0.001
    lr: float = 1.0e-6
    train_subset: int = 1000


@dataclass
class Config:
    corpus: CorpusConfig
    retriever: RetrieverConfig
    env: EnvConfig
    reward: RewardConfig
    model: ModelConfig
    grpo: GRPOConfig
    seed: int = 42


def _build(cls: Type[T], data: Dict[str, Any], section: str) -> T:
    """Instantiate a section dataclass from a dict, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in config section '{section}'; "
            f"expected any of {sorted(known)}"
        )
    return cls(**data)


def load_config(path: Path = BASE_CONFIG) -> Config:
    """Load and validate a YAML config. Missing sections fall back to defaults."""
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    known_sections = {f.name for f in fields(Config)}
    unknown = set(raw) - known_sections
    if unknown:
        raise ValueError(
            f"unknown top-level section(s) {sorted(unknown)} in {path}; "
            f"expected any of {sorted(known_sections)}"
        )

    return Config(
        corpus=_build(CorpusConfig, raw.get("corpus", {}), "corpus"),
        retriever=_build(RetrieverConfig, raw.get("retriever", {}), "retriever"),
        env=_build(EnvConfig, raw.get("env", {}), "env"),
        reward=_build(RewardConfig, raw.get("reward", {}), "reward"),
        model=_build(ModelConfig, raw.get("model", {}), "model"),
        grpo=_build(GRPOConfig, raw.get("grpo", {}), "grpo"),
        seed=raw.get("seed", 42),
    )


def build_env(retriever, config: Config | None = None):
    """Construct a SearchEnv from config. Pulls env budget, retriever top_k, and
    reward settings so all three yaml sections drive one env instance.

    Imported here (not at module top) to avoid a config <-> env import cycle.
    """
    from scholarrl.env import SearchEnv

    cfg = config or load_config()
    return SearchEnv(
        retriever,
        max_retrieval_turns=cfg.env.max_retrieval_turns,
        max_steps=cfg.env.max_steps,
        top_k=cfg.retriever.top_k,
        abstract_max_words=cfg.env.abstract_max_words,
        dedup_queries=cfg.env.dedup_queries,
        metric=cfg.reward.metric,
        reward_k=cfg.reward.k,
        lambda_fmt=cfg.reward.lambda_fmt,
    )
