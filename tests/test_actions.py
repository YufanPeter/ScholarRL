"""Action parsing: the contract the env and rollout depend on."""
from scholarrl.env.actions import parse_action, SEARCH, READ, SELECT, FINISH, INVALID


def test_search_parsed():
    a = parse_action("<search>graph neural networks</search>")
    assert a.kind == SEARCH and a.query == "graph neural networks"


def test_read_parsed():
    a = parse_action("<read>2210.05663</read>")
    assert a.kind == READ and a.paper_id == "2210.05663"


def test_select_single_and_multi():
    assert parse_action("<select>2210.05663</select>").paper_ids == ["2210.05663"]
    a = parse_action("<select>2210.05663, 2302.07241</select>")
    assert a.kind == SELECT and a.paper_ids == ["2210.05663", "2302.07241"]


def test_finish_parsed():
    assert parse_action("<finish/>").kind == FINISH


def test_garbage_is_invalid():
    assert parse_action("no tags at all").kind == INVALID
    assert parse_action("").kind == INVALID


def test_first_action_wins_when_multiple():
    # a rollout stops at the first action; parser should reflect the first tag
    a = parse_action("<search>a</search><read>1</read><finish/>")
    assert a.kind == SEARCH


def test_effective_defaults_false():
    # env sets this after execution; parser must not presume effectiveness
    assert parse_action("<search>x</search>").effective is False
