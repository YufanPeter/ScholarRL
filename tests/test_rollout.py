"""Test rollout with StubPolicy (no model needed)."""
import pytest

from scholarrl.data.queries import load_queries
from scholarrl.rollout import StubPolicy, run_episode
from scholarrl.retriever import BM25Retriever, INDEX_DIR
from scholarrl.env import SearchEnv


@pytest.fixture(scope="module")
def retriever():
    if not INDEX_DIR.exists():
        pytest.skip("BM25 index not built")
    return BM25Retriever.load()


@pytest.fixture
def policy():
    return StubPolicy()


@pytest.fixture
def record():
    return load_queries("dev")[0]


def test_stub_policy_completes_episode(retriever, policy, record):
    env = SearchEnv(retriever)
    traj = run_episode(env, policy, record)
    assert traj.steps > 0
    assert traj.reason in ["finish", "max_steps"]


def test_trajectory_has_reward_fields(retriever, policy, record):
    env = SearchEnv(retriever)
    traj = run_episode(env, policy, record)
    assert hasattr(traj, "reward")
    assert hasattr(traj, "task_reward")
    assert hasattr(traj, "format_reward")
    assert traj.reward == pytest.approx(traj.task_reward + 0.1 * traj.format_reward)


def test_trajectory_records_actions_and_observations(retriever, policy, record):
    env = SearchEnv(retriever)
    traj = run_episode(env, policy, record)
    assert len(traj.actions) > 0
    assert len(traj.observations) > 0
    assert len(traj.messages) > 0


def test_messages_start_with_system_then_user(retriever, policy, record):
    env = SearchEnv(retriever)
    traj = run_episode(env, policy, record)
    roles = [msg["role"] for msg in traj.messages]
    # [system, user, assistant, user, assistant, ...]
    assert roles[0] == "system"
    assert roles[1] == "user"
    # after the system message, user/assistant strictly alternate
    for i in range(1, len(roles)):
        expected = "user" if i % 2 == 1 else "assistant"
        assert roles[i] == expected


def test_system_prompt_comes_from_env(retriever, policy, record):
    env = SearchEnv(retriever)
    traj = run_episode(env, policy, record)
    assert traj.messages[0]["content"] == env.system_prompt()
    # the per-query user turn is just the question, not the rules
    assert traj.messages[1]["content"].startswith("Question:")


def test_system_prompt_override(retriever, policy, record):
    env = SearchEnv(retriever)
    traj = run_episode(env, policy, record, system_prompt="CUSTOM RULES")
    assert traj.messages[0]["content"] == "CUSTOM RULES"


def test_trajectory_captures_metadata(retriever, policy, record):
    env = SearchEnv(retriever)
    traj = run_episode(env, policy, record)
    assert traj.query_id == record.qid
    assert traj.question == record.question
    assert len(traj.gold) > 0
    assert traj.retrieval_turns >= 0
