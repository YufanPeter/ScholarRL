"""Evaluator: Recall@K, Precision@K, NDCG@K, avg #search calls, avg tokens.

Used for the zero-shot baseline (Phase 0) and for dev-set checkpoint selection during RL.
Never evaluate on test.jsonl except for the final report.
"""
