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


# --- turn budget (method C): only search/read count; select/finish are free ---

def test_retrieval_budget_gates_but_does_not_end_episode(retriever, rec):
    env = SearchEnv(retriever, max_retrieval_turns=2, max_steps=20)
    env.reset(rec)
    env.step("<search>vision language 3D scene</search>")   # retrieval_turn 1
    pid = next(iter(env.state.seen_ids))
    env.step(f"<read>{pid}</read>")                          # retrieval_turn 2 -> budget full
    assert env.state.retrieval_turns == 2 and not env.state.done
    # a further search/read is refused but the episode continues
    obs, done, _ = env.step("<search>another angle</search>")
    assert not done and "budget exhausted" in obs
    assert env.state.retrieval_turns == 2   # refused action did not increment
    # the paper already read can still be selected, then finished
    env.step(f"<select>{pid}</select>")
    assert pid in env.state.selected
    _, done, info = env.step("<finish/>")
    assert done and info["reason"] == "finish"


def test_select_and_finish_do_not_consume_retrieval_budget(retriever, rec):
    # a full multi-step answer must fit even with a tight retrieval budget
    env = SearchEnv(retriever, max_retrieval_turns=3, max_steps=20)
    env.reset(rec)
    env.step("<search>vision language 3D scene</search>")    # 1
    ids = list(env.state.seen_ids)[:2]
    env.step(f"<read>{ids[0]}</read>")                       # 2
    env.step(f"<read>{ids[1]}</read>")                       # 3  (budget now full)
    # selecting both should still work because select is free, and env terminates on the
    # next retrieval action or finish — verify select committed before termination
    env.step(f"<select>{ids[0]},{ids[1]}</select>")
    assert set(ids) <= set(env.state.selected)
    _, done, info = env.step("<finish/>")
    assert done and info["reason"] == "finish"


def test_max_steps_hard_cap(retriever, rec):
    # cheap actions (empty selects) must not loop forever
    env = SearchEnv(retriever, max_retrieval_turns=99, max_steps=3)
    env.reset(rec)
    env.step("<select></select>")
    env.step("<select></select>")
    _, done, info = env.step("<select></select>")
    assert done and info["reason"] == "max_steps"
