# Scholar-R1

A from-scratch, RL-trained academic search agent for AI/CS literature search.
Goal: learn the full Agentic-RL pipeline (environment → rollout → reward → GRPO → veRL)
on a single A100. Not SOTA — a complete, reproducible loop.

See `../Scholar-R1_Project_Plan.md` for the full plan.

## Layout

```
src/scholarrl/
  paths.py       central path config (override root with $SCHOLAR_DATA)
  data/          load AutoScholarQuery + id2paper; answer-coverage report
  corpus/        cs_paper_2nd.zip -> subset corpus (gold + distractors) -> papers.jsonl
  retriever/     pluggable Retriever interface; BM25 (V1), dense (V2)
  env/           search environment + action parsing
  reward/        Recall@K/F1 + format bonus (+ cost terms in Phase 3)
  rollout/       query -> actions -> reward + logged trajectory (the closed loop)
  eval/          Recall/Precision/NDCG@K, search-call & token stats
  grpo/          Stage A minimal GRPO from scratch; Stage B veRL port
configs/         base.yaml (starting hyperparameters)
scripts/         entrypoints
data/            gitignored; raw/ symlinks the Desktop downloads
outputs/         gitignored; trajectories / checkpoints / logs
```

## Data

Raw files live on the Desktop and are symlinked into `data/raw/` (all gitignored):
`train.jsonl` (33,551), `dev.jsonl` (1,000), `test.jsonl` (1,000),
`id2paper.json` (arxiv_id → title), `cs_paper_2nd.zip` (555,262 papers, ~2.3 GB,
each a JSON with title/abstract/sections; corpus starts at arxiv 2404).

On another machine, set `SCHOLAR_DATA` to wherever the data lives — code paths are relative.

## Install

Editable install puts `scholarrl` on the path so scripts and tests import it the same way
(no `PYTHONPATH` hacks). Set `SCHOLAR_DATA` if the data lives elsewhere.

```
pip install -e .
python -m pytest                 # run the test suite
python -m scripts.eval_retriever # BM25 zero-rewrite baseline
```

## Status

Phase 0 (close the loop) is implemented and tested:

- `data/` — query loading, id/title normalization, gold resolution, retrievable set.
- `corpus/` — subset corpus built (gold + distractors) → `data/corpus/papers.jsonl`.
- `retriever/` — BM25 behind a pluggable interface; index built.
- `env/` — action parsing + `SearchEnv` (search/read/select/finish, Style A: read-before-select).
- `reward/` — Recall@K / F1 (task) + effectiveness-based format shaping.
- `rollout/` — `run_episode` + pluggable `Policy` (`StubPolicy` for tests, `HFPolicy` for real models); records chat-format trajectories (system rules + question + turns) for GRPO.
- tests — 44 passing (`tests/`); env tests skip if the index isn't built.

**BM25 zero-rewrite baseline (dev, k=20): mean gold recall ≈ 0.25**, ~38% of queries hit
≥1 gold — moderate, so there's real room for the agent to improve via query rewriting.

Turn budget uses **method C**: `max_retrieval_turns` bounds the expensive actions
(search + read); `select`/`finish` are free, with a `max_steps` hard cap against loops.

Next: zero-shot baseline with `HFPolicy` (Qwen2.5-3B on the server) → `grpo/`.
