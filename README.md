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

## Status

Scaffold only — no implementation yet. Next: answer-coverage report, then build the
subset corpus, then BM25 + env + reward → zero-shot baseline (closes the loop).
