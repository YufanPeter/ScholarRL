"""Task reward: recall@k and f1 over normalized ids, restricted to retrievable gold."""
import pytest
from scholarrl.reward.recall import recall_at_k, f1, task_reward

GOLD = ["2210.05663", "2302.07241", "2211.15654"]


def test_recall_full_and_none():
    assert recall_at_k(GOLD, GOLD, 20) == 1.0
    assert recall_at_k([], GOLD, 20) == 0.0


def test_recall_partial():
    assert recall_at_k(["2210.05663"], GOLD, 20) == pytest.approx(1 / 3)
    assert recall_at_k(["2210.05663", "2302.07241"], GOLD, 20) == pytest.approx(2 / 3)


def test_recall_version_suffix_matches_both_sides():
    assert recall_at_k(["2210.05663v2"], GOLD, 20) == pytest.approx(1 / 3)
    assert recall_at_k(["2210.05663"], ["2210.05663v5"], 20) == 1.0


def test_recall_empty_gold_is_zero_not_crash():
    assert recall_at_k(["x"], [], 20) == 0.0


def test_recall_topk_cutoff_and_dedup():
    assert recall_at_k(["a", "b", "2210.05663"], GOLD, 2) == 0.0
    assert recall_at_k(["a", "b", "2210.05663"], GOLD, 3) == pytest.approx(1 / 3)
    assert recall_at_k(["2210.05663", "2210.05663"], GOLD, 20) == pytest.approx(1 / 3)


def test_f1_precision_is_over_selected():
    # 1 correct pick out of 1 selected: precision=1, recall=1/3 -> f1=0.5
    assert f1(["2210.05663"], GOLD) == pytest.approx(0.5)


def test_f1_penalizes_overselection():
    # 1 hit out of 4 selected: precision=1/4, recall=1/3
    p, r = 1 / 4, 1 / 3
    assert f1(["2210.05663", "x", "y", "z"], GOLD) == pytest.approx(2 * p * r / (p + r))


def test_f1_zero_and_empty():
    assert f1(["x", "y"], GOLD) == 0.0
    assert f1([], GOLD) == 0.0
    assert f1(GOLD, []) == 0.0


def test_task_reward_dispatch_and_bad_metric():
    assert task_reward(GOLD, GOLD, metric="recall", k=20) == 1.0
    assert task_reward(GOLD, GOLD, metric="f1") == 1.0
    with pytest.raises(ValueError):
        task_reward(GOLD, GOLD, metric="bogus")


def test_task_reward_restricts_to_retrievable():
    # a fake gold id not in the corpus is dropped from the denominator
    from scholarrl.data.retrievable import retrievable_gold_ids
    real = list(retrievable_gold_ids())[:2]
    fake = "9999.99999"
    assert task_reward(real, real + [fake], metric="recall", k=20) == 1.0
