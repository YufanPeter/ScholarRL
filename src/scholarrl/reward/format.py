"""Format reward: a dense shaping signal that stabilizes small models (aligned with Search-R1).

The task reward (recall/f1) is SPARSE — early in training a 3B model selects nothing
correct, so every trajectory scores 0 and GRPO gets no gradient (all advantages equal).
The format reward gives a small, dense signal for *acting effectively* — emitting actions
that actually do something, committing a real answer, and finishing cleanly — so the model
first learns the protocol, then learns to be accurate. It is weighted small (lambda_fmt=0.1):
it guides, it does not dominate the task reward.

Scoring is on EFFECTIVENESS, not mere tag validity: the env sets Action.effective=False for
valid-but-inert actions (duplicate/empty/failed search, gated read, all-rejected select), so
a model cannot farm shaping reward by emitting syntactically valid actions that do nothing.

Input: the list of parsed Actions for one trajectory, after the env has run them (so
Action.effective is populated). Output: a score in [-1, 1].
"""

from __future__ import annotations

from typing import List

from scholarrl.env.actions import Action, SELECT, FINISH


def format_reward(actions: List[Action]) -> float:
    """Score the shape of one trajectory's action sequence, in [-1, 1].

    Rewards (encourage the protocol):
      + effective actions (that actually did something), as a fraction of all turns
      + at least one effective SELECT (the agent committed a real answer)
      + a clean FINISH to terminate

    Penalties (discourage degenerate behavior):
      - nothing effective at all (all actions invalid or inert)
      - never committing a paper (empty answer set)
    """
    if not actions:
        return -1.0  # produced nothing usable

    n = len(actions)
    # EFFECTIVE fraction, not merely well-formed. The env sets action.effective=False for
    # valid-but-inert actions (duplicate/empty/failed search, gated read, all-rejected
    # select). Scoring effectiveness closes the loophole where a model farms shaping reward
    # by emitting syntactically valid actions that do nothing.
    effective = sum(1 for a in actions if a.effective)
    # a real commit = a select that actually added a paper to the answer set
    has_select = any(a.kind == SELECT and a.effective for a in actions)
    has_finish = any(a.kind == FINISH for a in actions)
    all_inert = effective == 0

    score = 0.0
    # (1) fraction of EFFECTIVE turns -> up to +0.6
    score += 0.6 * (effective / n)
    # (2) actually committed an answer -> +0.2, else -0.8. Strong enough to cancel the
    #     effective + finish bonuses: a trajectory that never commits a paper has done
    #     nothing toward the retrieval goal and must not earn positive shaping reward.
    score += 0.2 if has_select else -0.8
    # (3) terminated cleanly -> +0.2
    score += 0.2 if has_finish else 0.0
    # (4) nothing effective at all (all invalid or all inert) -> hard floor penalty
    if all_inert:
        score -= 0.5

    # clamp into [-1, 1]
    return max(-1.0, min(1.0, score))
