"""The search environment: a Gym-style loop that ties actions + retriever + reward together.

Interaction (aligned with Search-R1's generate -> retrieve -> observe loop, extended to a
retrieval task with an explicit select step):

    obs = env.reset(record)          # pose a question
    while not done:
        text = model.generate(obs)   # one action per turn (rollout uses STOP_TOKENS)
        obs, done, info = env.step(text)

Design (Style A):
  - <search> returns TITLES only (not abstracts) — the agent must <read> to see detail.
  - <select> requires the paper to have been <read> first — forces a real judgment,
    not title-only guessing.
  - reward is computed once at termination: task_reward (recall/f1) + lambda_fmt * format.

The env NEVER raises: any malformed / illegal turn becomes an observation and the episode
continues, because a training-time model emits all kinds of junk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scholarrl.data.queries import QueryRecord
from scholarrl.env.actions import (
    Action, parse_action,
    SEARCH, READ, SELECT, FINISH,
)
from scholarrl.retriever import BM25Retriever
# reward.* is imported lazily inside _terminate() to avoid a circular import:
# reward.format imports env.actions, and env.__init__ imports this module.


@dataclass
class EnvState:
    """Mutable state for one episode."""
    question: str = ""
    gold: List[str] = field(default_factory=list)      # gold arxiv ids for this query
    selected: List[str] = field(default_factory=list)  # accumulated answer set (order kept)
    read_ids: set = field(default_factory=set)          # papers the agent has read (select gate)
    seen_ids: set = field(default_factory=set)          # candidates surfaced by search (read gate)
    queries: set = field(default_factory=set)           # past search queries (dedup)
    actions: List[Action] = field(default_factory=list)  # every parsed action (for format reward)
    retrieval_turns: int = 0   # count of EXPENSIVE actions (search + read) — the real budget
    steps: int = 0             # count of ALL steps — hard cap against infinite loops
    done: bool = False


class SearchEnv:
    """One agent episode over the BM25 corpus."""

    def __init__(
        self,
        retriever: BM25Retriever,
        max_retrieval_turns: int = 6,
        max_steps: int = 20,
        top_k: int = 3,
        abstract_max_words: int = 256,
        dedup_queries: bool = True,
        metric: str = "recall",
        reward_k: int = 20,
        lambda_fmt: float = 0.1,
    ):
        self.retriever = retriever
        # Budget only the EXPENSIVE actions (search + read drive generation/retrieval cost).
        # select/finish are cheap CPU ops and must not compete with retrieval for the budget,
        # otherwise a multi-paper answer (avg 2.6 gold/query, read-before-select) can't fit.
        self.max_retrieval_turns = max_retrieval_turns
        self.max_steps = max_steps          # hard cap so cheap actions can't loop forever
        self.top_k = top_k
        self.abstract_max_words = abstract_max_words
        self.dedup_queries = dedup_queries
        self.metric = metric
        self.reward_k = reward_k
        self.lambda_fmt = lambda_fmt
        self.state = EnvState()

    # --- lifecycle ---------------------------------------------------------

    def reset(self, record: QueryRecord) -> str:
        """Start a new episode from a query record. Returns the initial observation."""
        self.state = EnvState(question=record.question, gold=list(record.answer_ids))
        return self._prompt()

    def step(self, text: str) -> Tuple[str, bool, dict]:
        """Advance one turn given the model's generated text.

        Returns (observation, done, info). info carries reward on the terminal step.
        """
        if self.state.done:
            return "[episode already finished]", True, {}

        self.state.steps += 1
        action = parse_action(text)
        self.state.actions.append(action)

        if action.kind == SEARCH or action.kind == READ:
            # Retrieval budget GATES expensive actions but does not end the episode:
            # the agent must still be able to select what it already read and then finish.
            if self.state.retrieval_turns >= self.max_retrieval_turns:
                obs = (f"[{action.kind}] retrieval budget exhausted "
                       f"({self.max_retrieval_turns}); you may still <select> and <finish>.")
            elif action.kind == SEARCH:
                self.state.retrieval_turns += 1
                obs, action.effective = self._do_search(action.query)
            else:
                self.state.retrieval_turns += 1
                obs, action.effective = self._do_read(action.paper_id)
        elif action.kind == SELECT:
            obs, action.effective = self._do_select(action.paper_ids)
        elif action.kind == FINISH:
            action.effective = True
            return self._terminate("finish")
        else:  # INVALID
            obs = ("[invalid action] use exactly one of: <search>q</search>, "
                   "<read>id</read>, <select>id,...</select>, <finish/>")

        # hard step cap: only the total-step limit ends a non-finished episode, so cheap
        # actions can't loop forever, but the agent always gets to commit its answer.
        if self.state.steps >= self.max_steps:
            term_obs, done, info = self._terminate("max_steps")
            return f"{obs}\n{term_obs}", done, info

        return obs, False, {}

    # --- action handlers ---------------------------------------------------

    def _do_search(self, query: Optional[str]) -> Tuple[str, bool]:
        query = (query or "").strip()
        if not query:
            return "[search] empty query ignored.", False
        key = query.lower()
        if self.dedup_queries and key in self.state.queries:
            return f"[search] '{query}' already searched; try a different query.", False
        self.state.queries.add(key)

        hits = self.retriever.search(query, k=self.top_k)
        if not hits:
            return f"[search] no results for '{query}'.", False
        lines = [f"Search results for '{query}' (titles only; <read> id for the abstract):"]
        for pid, _score in hits:
            self.state.seen_ids.add(pid)
            title = (self.retriever.get(pid).get("title") or "").strip() or "(no title)"
            lines.append(f"  [{pid}] {title}")
        return "\n".join(lines), True

    def _do_read(self, paper_id: Optional[str]) -> Tuple[str, bool]:
        pid = (paper_id or "").strip()
        if not pid:
            return "[read] no paper id given.", False
        # gate: only read papers surfaced by a prior search
        if pid not in self.state.seen_ids:
            return f"[read] '{pid}' is not in your search results; <search> first.", False
        meta = self.retriever.get(pid)
        if not meta:
            return f"[read] '{pid}' not found in corpus.", False
        already_read = pid in self.state.read_ids
        self.state.read_ids.add(pid)
        title = (meta.get("title") or "").strip() or "(no title)"
        abstract = self._truncate(meta.get("abstract") or "")
        obs = f"[read] {pid}\nTitle: {title}\nAbstract: {abstract or '(no abstract available)'}"
        # re-reading the same paper is a no-op for shaping purposes
        return obs, (not already_read)

    def _do_select(self, paper_ids: List[str]) -> Tuple[str, bool]:
        if not paper_ids:
            return "[select] no paper id given.", False
        accepted, rejected = [], []
        for pid in paper_ids:
            pid = pid.strip()
            if not pid:
                continue
            # Style A gate: must have read the paper before selecting it
            if pid not in self.state.read_ids:
                rejected.append(pid)
                continue
            if pid not in self.state.selected:
                self.state.selected.append(pid)
            accepted.append(pid)
        parts = []
        if accepted:
            parts.append(f"[select] added: {', '.join(accepted)}")
        if rejected:
            parts.append(f"[select] rejected (must <read> before <select>): {', '.join(rejected)}")
        parts.append(f"current answer set: {self.state.selected or '[]'}")
        # effective only if at least one id was actually accepted into the answer set
        return "\n".join(parts), bool(accepted)

    # --- termination & reward ---------------------------------------------

    def _terminate(self, reason: str) -> Tuple[str, bool, dict]:
        # lazy import breaks the env <-> reward import cycle (see top-of-file note)
        from scholarrl.reward.recall import task_reward
        from scholarrl.reward.format import format_reward

        self.state.done = True
        task = task_reward(self.state.selected, self.state.gold,
                           metric=self.metric, k=self.reward_k)
        fmt = format_reward(self.state.actions)
        total = task + self.lambda_fmt * fmt
        info = {
            "reward": total,
            "task_reward": task,
            "format_reward": fmt,
            "selected": list(self.state.selected),
            "gold": list(self.state.gold),
            "retrieval_turns": self.state.retrieval_turns,
            "steps": self.state.steps,
            "reason": reason,
        }
        obs = (f"[done: {reason}] selected {len(self.state.selected)} papers | "
               f"task={task:.3f} format={fmt:+.3f} total={total:.3f}")
        return obs, True, info

    # --- helpers -----------------------------------------------------------

    def _prompt(self) -> str:
        return (
            "You are a research assistant finding papers that answer a question.\n"
            "Actions (emit exactly one per turn):\n"
            "  <search>query</search>   search the corpus (returns titles)\n"
            "  <read>paper_id</read>    read one paper's abstract\n"
            "  <select>id,...</select>  add papers to your answer (must <read> first)\n"
            "  <finish/>                submit your answer set\n"
            f"You may search/read up to {self.max_retrieval_turns} times; "
            "selecting and finishing are free.\n\n"
            f"Question: {self.state.question}"
        )

    def _truncate(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.abstract_max_words:
            return text.strip()
        return " ".join(words[: self.abstract_max_words]) + " ..."
