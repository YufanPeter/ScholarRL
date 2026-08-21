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


# --- tolerance for the code-like id syntax small models fall back to ---------
# A 3B model routinely wraps ids as id="...", p_id=..., quotes them, or glues a
# stray 'p' onto an arxiv id. These must de-wrap to the bare id, or the select
# gate silently rejects an otherwise-correct answer (the 0.0095 baseline bug).

def test_select_strips_kwarg_prefix():
    assert parse_action('<select>id=2210.05663</select>').paper_ids == ["2210.05663"]
    assert parse_action('<select>paper_id=2302.07241</select>').paper_ids == ["2302.07241"]
    assert parse_action('<select>p_id=1709.10441</select>').paper_ids == ["1709.10441"]


def test_select_strips_quotes():
    assert parse_action('<select>"2210.05663"</select>').paper_ids == ["2210.05663"]
    assert parse_action("<select>'2210.05663'</select>").paper_ids == ["2210.05663"]
    assert parse_action('<select>p_id="1709.10441"</select>').paper_ids == ["1709.10441"]


def test_select_strips_stray_p_prefix():
    assert parse_action("<select>p1805.12152</select>").paper_ids == ["1805.12152"]


def test_select_mixed_wrappers_multi():
    a = parse_action('<select>id="2210.05663", p2302.07241, 2211.15654</select>')
    assert a.paper_ids == ["2210.05663", "2302.07241", "2211.15654"]


def test_kebab_id_untouched():
    # distractor ids are all-letters; the stray-'p' rule only fires before a digit,
    # so a kebab id that legitimately starts with 'p' must not be truncated.
    pid = "paparazziadeepdiveintothecapabilities"
    assert parse_action(f"<select>{pid}</select>").paper_ids == [pid]


def test_read_strips_wrappers():
    assert parse_action('<read>id="2210.05663"</read>').paper_id == "2210.05663"
    assert parse_action("<read>p1805.12152</read>").paper_id == "1805.12152"
