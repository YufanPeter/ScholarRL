"""Tests for the config loader: base.yaml parses, validates, and drives the env."""
import textwrap

import pytest

from scholarrl.config import load_config, build_env, Config
from scholarrl.paths import BASE_CONFIG


def test_base_config_loads():
    """The checked-in base.yaml parses into a Config with expected sections."""
    cfg = load_config(BASE_CONFIG)
    assert isinstance(cfg, Config)
    # values we tuned should be reflected
    assert cfg.retriever.top_k == 10
    assert cfg.env.max_search_turns == 6
    assert cfg.env.max_read_turns == 10
    assert cfg.reward.k == 20
    assert cfg.model.name.startswith("Qwen")
    assert cfg.seed == 42


def test_missing_section_uses_defaults(tmp_path):
    """A yaml with only some sections fills the rest from dataclass defaults."""
    p = tmp_path / "partial.yaml"
    p.write_text("env:\n  max_steps: 15\n")
    cfg = load_config(p)
    assert cfg.env.max_steps == 15
    assert cfg.env.max_search_turns == 6      # default
    assert cfg.env.max_read_turns == 10       # default
    assert cfg.reward.metric == "recall"      # whole section defaulted


def test_unknown_key_raises(tmp_path):
    """A typo'd key fails loudly instead of being silently ignored."""
    p = tmp_path / "typo.yaml"
    p.write_text("env:\n  max_retrieval_turnz: 8\n")   # typo
    with pytest.raises(ValueError, match="unknown key"):
        load_config(p)


def test_unknown_section_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("nonsense:\n  x: 1\n")
    with pytest.raises(ValueError, match="unknown top-level section"):
        load_config(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_build_env_applies_config(tmp_path):
    """build_env wires config values into the SearchEnv (no retriever needed)."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        retriever:
          top_k: 7
        env:
          max_search_turns: 9
          max_read_turns: 4
          max_steps: 12
        reward:
          metric: f1
          k: 15
          lambda_fmt: 0.25
    """))
    cfg = load_config(p)
    env = build_env(retriever=None, config=cfg)
    assert env.top_k == 7
    assert env.max_search_turns == 9
    assert env.max_read_turns == 4
    assert env.max_steps == 12
    assert env.metric == "f1"
    assert env.reward_k == 15
    assert env.lambda_fmt == 0.25
