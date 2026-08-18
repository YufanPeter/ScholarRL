"""Format (shaping) reward: scores EFFECTIVE actions, not merely well-formed tags."""
from scholarrl.env.actions import Action, SEARCH, READ, SELECT, FINISH, INVALID
from scholarrl.reward.format import format_reward


def A(kind, effective=True, ids=None):
    return Action(kind=kind, paper_ids=ids or [], effective=effective)


IDEAL = [A(SEARCH), A(READ), A(SELECT, ids=["x"]), A(FINISH)]


def test_ideal_trajectory_scores_high():
    assert format_reward(IDEAL) > 0.7


def test_empty_trajectory_floor():
    assert format_reward([]) == -1.0


def test_all_invalid_floor():
    assert format_reward([A(INVALID, effective=False)] * 3) <= -0.9


def test_never_select_not_positive():
    traj = [A(SEARCH), A(READ), A(FINISH)]
    assert format_reward(traj) <= 0.0


def test_selected_but_no_finish_positive_below_ideal():
    traj = [A(SEARCH), A(SELECT, ids=["x"])]
    s = format_reward(traj)
    assert 0.0 < s < format_reward(IDEAL)


def test_bounds_within_unit_interval():
    for traj in [IDEAL, [], [A(INVALID, effective=False)] * 3,
                 [A(SEARCH), A(READ), A(FINISH)]]:
        assert -1.0 <= format_reward(traj) <= 1.0


# --- the gaming loophole these fixes close ---

def test_inert_valid_actions_do_not_earn_reward():
    # duplicate/failed searches parse as valid SEARCH but env marks them ineffective;
    # a trajectory of only-inert actions must not score positive.
    inert = [A(SEARCH, effective=False), A(READ, effective=False),
             A(SEARCH, effective=False), A(FINISH)]
    assert format_reward(inert) <= 0.0


def test_select_rejected_by_gate_is_not_a_commit():
    # select whose ids were all rejected (never read) -> effective=False -> not a real commit
    traj = [A(SEARCH), A(SELECT, effective=False, ids=["x"]), A(FINISH)]
    assert format_reward(traj) <= 0.0


def test_effective_select_beats_inert_select():
    good = [A(SEARCH), A(READ), A(SELECT, effective=True, ids=["x"]), A(FINISH)]
    bad = [A(SEARCH), A(READ), A(SELECT, effective=False, ids=["x"]), A(FINISH)]
    assert format_reward(good) > format_reward(bad)
