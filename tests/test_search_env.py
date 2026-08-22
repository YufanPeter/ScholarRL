"""SearchEnv integration: the full agent<->env loop over the real BM25 index.

These tests need the built corpus + index (data/corpus/bm25_index). If it's missing
they skip rather than fail, so the pure-logic suite still runs on a fresh checkout.
"""
import pytest

from scholarrl.data.queries import load_queries

bm25 = pytest.importorskip("scholarrl.retriever")
from scholarrl.retriever import BM25Retriever, INDEX_DIR
from scholarrl.env import SearchEnv


@pytest.fixture(scope="module")
def retriever():
    if not INDEX_DIR.exists():
        pytest.skip("BM25 index not built (run scripts.build_index)")
    return BM25Retriever.load()


@pytest.fixture
def rec(retriever):
    return load_queries("dev")[0]


def test_reset_shows_question(retriever, rec):
    env = SearchEnv(retriever)
    obs = env.reset(rec)
    assert rec.question[:20] in obs


def test_search_returns_titles_only(retriever, rec):
    env = SearchEnv(retriever)
    env.reset(rec)
    obs, done, _ = env.step("<search>vision language 3D scene</search>")
    assert "Search results" in obs and not done
    assert "Abstract:" not in obs  # search must not leak abstracts


def test_select_requires_read(retriever, rec):
    env = SearchEnv(retriever)
    env.reset(rec)
    env.step("<search>vision language 3D scene</search>")
    pid = next(iter(env.state.seen_ids))
    obs, _, _ = env.step(f"<select>{pid}</select>")
    assert "rejected" in obs and pid not in env.state.selected


def test_read_then_select_commits(retriever, rec):
    env = SearchEnv(retriever)
    env.reset(rec)
    env.step("<search>vision language 3D scene</search>")
    pid = next(iter(env.state.seen_ids))
    obs, _, _ = env.step(f"<read>{pid}</read>")
    assert "Abstract:" in obs and pid in env.state.read_ids
    env.step(f"<select>{pid}</select>")
    assert pid in env.state.selected


def test_finish_composes_reward(retriever, rec):
    env = SearchEnv(retriever, lambda_fmt=0.1)
    env.reset(rec)
    _, done, info = env.step("<finish/>")
    assert done
    assert info["reward"] == pytest.approx(info["task_reward"] + 0.1 * info["format_reward"])


def test_step_after_done_is_safe(retriever, rec):
    env = SearchEnv(retriever)
    env.reset(rec)
    env.step("<finish/>")
    obs, done, _ = env.step("<search>x</search>")
    assert done and "already finished" in obs


def test_dedup_query(retriever, rec):
    env = SearchEnv(retriever, dedup_queries=True)
    env.reset(rec)
    env.step("<search>graph neural networks</search>")
    obs, _, _ = env.step("<search>graph neural networks</search>")
    assert "already searched" in obs


def test_read_unseen_id_is_gated(retriever, rec):
    env = SearchEnv(retriever)
    env.reset(rec)
    obs, _, _ = env.step("<read>9999.99999</read>")
    assert "not in your search" in obs.lower()


# --- turn budget: search and read hold SEPARATE budgets; select/finish are free ---

def test_search_budget_gates_but_does_not_end_episode(retriever, rec):
    env = SearchEnv(retriever, max_search_turns=1, max_read_turns=5, max_steps=20)
    env.reset(rec)
    env.step("<search>vision language 3D scene</search>")   # search_turn 1 -> budget full
    pid = next(iter(env.state.seen_ids))
    assert env.state.search_turns == 1 and not env.state.done
    # a further search is refused but the episode continues
    obs, done, _ = env.step("<search>another angle</search>")
    assert not done and "search budget exhausted" in obs
    assert env.state.search_turns == 1   # refused action did not increment
    # reading still works: it draws on its own budget
    obs, done, _ = env.step(f"<read>{pid}</read>")
    assert not done and "Abstract:" in obs and env.state.read_turns == 1
    env.step(f"<select>{pid}</select>")
    assert pid in env.state.selected
    _, done, info = env.step("<finish/>")
    assert done and info["reason"] == "finish"


def test_read_budget_does_not_consume_search_budget(retriever, rec):
    # exhausting reads must leave searching untouched (the old shared budget did not)
    env = SearchEnv(retriever, max_search_turns=3, max_read_turns=1, max_steps=20)
    env.reset(rec)
    env.step("<search>vision language 3D scene</search>")
    ids = list(env.state.seen_ids)[:2]
    env.step(f"<read>{ids[0]}</read>")                       # read budget now full
    obs, done, _ = env.step(f"<read>{ids[1]}</read>")
    assert not done and "read budget exhausted" in obs
    obs, done, _ = env.step("<search>a different angle entirely</search>")
    assert not done and "Search results" in obs
    assert env.state.search_turns == 2 and env.state.read_turns == 1


def test_select_and_finish_do_not_consume_retrieval_budget(retriever, rec):
    # a full multi-step answer must fit even with a tight retrieval budget
    env = SearchEnv(retriever, max_search_turns=1, max_read_turns=2, max_steps=20)
    env.reset(rec)
    env.step("<search>vision language 3D scene</search>")    # search 1
    ids = list(env.state.seen_ids)[:2]
    env.step(f"<read>{ids[0]}</read>")                       # read 1
    env.step(f"<read>{ids[1]}</read>")                       # read 2 (both budgets full)
    env.step(f"<select>{ids[0]},{ids[1]}</select>")
    assert set(ids) <= set(env.state.selected)
    _, done, info = env.step("<finish/>")
    assert done and info["reason"] == "finish"
    assert info["search_turns"] == 1 and info["read_turns"] == 2


def test_consecutive_invalid_ends_episode(retriever, rec):
    # a model that stops emitting actions repeats prose until max_steps; cut it short
    env = SearchEnv(retriever, max_consecutive_invalid=2, max_steps=20)
    env.reset(rec)
    _, done, _ = env.step("I think none of these papers are relevant.")
    assert not done
    _, done, info = env.step("I think none of these papers are relevant.")
    assert done and info["reason"] == "invalid_loop"


def test_valid_action_resets_invalid_streak(retriever, rec):
    env = SearchEnv(retriever, max_consecutive_invalid=2, max_steps=20)
    env.reset(rec)
    env.step("some prose")
    env.step("<search>vision language 3D scene</search>")   # streak reset
    _, done, _ = env.step("some prose again")
    assert not done and env.state.consecutive_invalid == 1


def test_max_steps_hard_cap(retriever, rec):
    # cheap actions (empty selects) must not loop forever
    env = SearchEnv(retriever, max_search_turns=99, max_read_turns=99, max_steps=3)
    env.reset(rec)
    env.step("<select></select>")
    env.step("<select></select>")
    _, done, info = env.step("<select></select>")
    assert done and info["reason"] == "max_steps"
