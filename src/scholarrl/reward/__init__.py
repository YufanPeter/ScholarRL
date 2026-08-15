"""Reward = task quality + format bonus (- cost terms, added in Phase 3).

- task reward: Recall@K / F1 of selected paper_ids vs gold answer_arxiv_id (rule-based, RLVR).
- format reward (lambda_fmt): small bonus for well-formed action tags; stabilizes 1.5B.
- Phase 3 cost-aware: - alpha * (#search calls) - beta * (token cost).
"""
