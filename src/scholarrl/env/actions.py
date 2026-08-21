"""Action parsing for the Scholar-R1 retrieval environment.

Design (aligned with Search-R1 / veRL rollout):
  During rollout, the closing tags in STOP_TOKENS are used as generation stop
  sequences, so each model turn emits exactly ONE action (Search-R1 stops decoding
  at `</search>` / `</answer>`; we do the same for our four actions). Parsing
  therefore normally sees a single tag.

  As a defensive fallback (offline eval, or malformed output where a stop token
  did not fire), we pick the FIRST-appearing action by text position, respecting
  the model's left-to-right generation order. `parse_action` never raises: any
  unrecognized turn becomes INVALID and is handled by the env / format reward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- action kinds ---
SEARCH = "search"
READ = "read"
SELECT = "select"
FINISH = "finish"
INVALID = "invalid"

# Closing tags that rollout passes to the generator as stop sequences.
# Ordering here is irrelevant; the generator stops at whichever fires first.
STOP_TOKENS = ["</search>", "</read>", "</select>", "<finish/>"]

# --- regexes (DOTALL so bodies can span newlines; IGNORECASE for robustness) ---
_FLAGS = re.DOTALL | re.IGNORECASE
_SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", _FLAGS)
_READ_RE = re.compile(r"<read>\s*(.*?)\s*</read>", _FLAGS)
_SELECT_RE = re.compile(r"<select>\s*(.*?)\s*</select>", _FLAGS)
# finish is self-closing; accept both <finish/> and <finish></finish>.
_FINISH_RE = re.compile(r"<finish\s*/>|<finish>\s*</finish>", _FLAGS)


@dataclass
class Action:
    kind: str
    query: Optional[str] = None
    paper_id: Optional[str] = None
    paper_ids: List[str] = field(default_factory=list)
    raw: str = ""
    # Set by the env after execution: did this action actually do something useful?
    # A tag can be well-formed (kind != INVALID) yet no-op — a duplicate/empty search,
    # a gated read, a select whose ids were all rejected. Format reward scores on this,
    # so a model can't farm shaping reward by emitting valid-but-inert actions.
    effective: bool = False


# A single id token wrapped in the code-like syntax small models fall back to:
#   id="1709.10441"  paper_id=1810.12281  p_id='...'  "quoted"  p2010.01736
# We strip these wrappers so a well-intentioned-but-malformed select still resolves
# against the read/corpus ids, instead of being silently rejected by the select gate.
_KWARG_RE = re.compile(r"^(?:paper_?id|p_id|id)\s*=\s*", re.IGNORECASE)
_STRAY_P_RE = re.compile(r"^p(?=\d{4}\.\d)")  # stray 'p' glued to an arxiv id: p2010.01736


def _clean_id(tok: str) -> str:
    """Strip the kwarg / quote / stray-'p' wrappers off one <select> token.

    Order matters: drop the `id=` prefix, then surrounding quotes, then a stray 'p'
    that the model sometimes prepends to an arxiv id. Kebab-case corpus ids (all
    letters) are left untouched — the stray-'p' rule only fires before a digit.
    """
    tok = tok.strip()
    tok = _KWARG_RE.sub("", tok)          # id="..."  -> "..."
    tok = tok.strip().strip("\"'")         # "1709.10441" -> 1709.10441
    tok = _STRAY_P_RE.sub("", tok)         # p2010.01736  -> 2010.01736
    return tok.strip()


def _split_ids(text: str) -> List[str]:
    """Split a <select> body into paper ids. Accepts commas / whitespace / newlines.

    Each token is de-wrapped (see _clean_id) so code-like output such as
    'id="2211.15654", p2302.07241' still yields ['2211.15654', '2302.07241'].
    """
    return [c for c in (_clean_id(t) for t in re.split(r"[,\s]+", text)) if c]


def parse_action(text: str) -> Action:
    """Parse one model turn into a single Action.

    Picks the FIRST-appearing recognized tag by position (left-to-right
    generation order). Returns INVALID (never raises) when no tag is present.
    """
    if not text:
        return Action(kind=INVALID, raw=text or "")

    # Collect (position, kind, match) for every recognized tag present.
    candidates: List[Tuple[int, str, re.Match]] = []
    for regex, kind in (
        (_SEARCH_RE, SEARCH),
        (_READ_RE, READ),
        (_SELECT_RE, SELECT),
        (_FINISH_RE, FINISH),
    ):
        m = regex.search(text)
        if m:
            candidates.append((m.start(), kind, m))

    if not candidates:
        return Action(kind=INVALID, raw=text)

    # First-appearing action wins.
    _, kind, match = min(candidates, key=lambda c: c[0])

    if kind == SEARCH:
        query = match.group(1).strip()
        # An empty <search></search> is not actionable.
        return Action(kind=SEARCH, query=query, raw=text) if query else Action(kind=INVALID, raw=text)
    if kind == READ:
        # de-wrap the same code-like syntax we tolerate in <select> (id=, quotes, stray p)
        return Action(kind=READ, paper_id=_clean_id(match.group(1)), raw=text)
    if kind == SELECT:
        return Action(kind=SELECT, paper_ids=_split_ids(match.group(1)), raw=text)
    return Action(kind=FINISH, raw=text)  # FINISH
