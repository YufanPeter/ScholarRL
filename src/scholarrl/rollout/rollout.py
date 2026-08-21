"""Rollout logic: run trajectories and record them.

Trajectory stores structured messages (role + content), not tokens.
Tokenization and mask generation happen in training data collator (Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from scholarrl.data.queries import QueryRecord
from scholarrl.env import SearchEnv
from scholarrl.rollout.policy import Policy


@dataclass
class Trajectory:
    """One complete episode trajectory.

    messages: conversation history in chat format ({"role": ..., "content": ...})
    reward fields: from env termination info
    """
    query_id: str
    question: str
    messages: List[dict] = field(default_factory=list)  # {"role": "system"|"user"|"assistant", "content": str}
    actions: List[str] = field(default_factory=list)     # raw model outputs per turn
    observations: List[str] = field(default_factory=list)  # env responses per turn
    reward: float = 0.0
    task_reward: float = 0.0
    format_reward: float = 0.0
    selected: List[str] = field(default_factory=list)
    gold: List[str] = field(default_factory=list)
    retrieval_turns: int = 0
    steps: int = 0
    reason: str = ""

    def to_dict(self, include_messages: bool = True) -> dict:
        """Serialize to a dict for JSON output.

        include_messages: if False, omit messages/actions/observations (lightweight mode).
        """
        obj = {
            "query_id": self.query_id,
            "question": self.question,
            "reward": self.reward,
            "task_reward": self.task_reward,
            "format_reward": self.format_reward,
            "selected": self.selected,
            "gold": self.gold,
            "retrieval_turns": self.retrieval_turns,
            "steps": self.steps,
            "reason": self.reason,
        }
        if include_messages:
            obj["messages"] = self.messages
            obj["actions"] = self.actions
            obj["observations"] = self.observations
        return obj


def run_episode(
    env: SearchEnv,
    policy: Policy,
    record: QueryRecord,
    system_prompt: str | None = None,
) -> Trajectory:
    """Run one episode and return trajectory.

    Messages are stored in chat format (role + content):
      [system] action protocol (from env, or `system_prompt` override)
      [user]   the question, then alternating assistant/user turns.
    Tokenization and mask generation happen in training collator.
    """
    traj = Trajectory(
        query_id=record.qid,
        question=record.question,
        gold=list(record.answer_ids),
    )

    # Initial observation from env (the per-query question)
    obs = env.reset(record)

    # System message = env's action protocol, unless the caller overrides it.
    sys_msg = system_prompt if system_prompt is not None else env.system_prompt()
    traj.messages.append({"role": "system", "content": sys_msg})

    # First user turn = the question
    traj.messages.append({"role": "user", "content": obs})

    done = False
    while not done:
        # Policy sees the full history and generates the next action
        action_text = policy.generate(traj.messages)
        traj.actions.append(action_text)
        traj.messages.append({"role": "assistant", "content": action_text})

        # Env responds
        obs, done, info = env.step(action_text)
        traj.observations.append(obs)

        if not done:
            # Continue conversation
            traj.messages.append({"role": "user", "content": obs})
        else:
            # Terminal step: record reward
            traj.reward = info.get("reward", 0.0)
            traj.task_reward = info.get("task_reward", 0.0)
            traj.format_reward = info.get("format_reward", 0.0)
            traj.selected = info.get("selected", [])
            traj.retrieval_turns = info.get("retrieval_turns", 0)
            traj.steps = info.get("steps", 0)
            traj.reason = info.get("reason", "")

    return traj
