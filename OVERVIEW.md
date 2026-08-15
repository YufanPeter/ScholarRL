# Scholar-R1 — Overview

A from-scratch, RL-trained **academic search agent** for AI/CS literature search,
built to learn the full Agentic-RL pipeline on a **single A100**. Not SOTA — a complete,
reproducible loop.

## The core idea

BM25 is a **fixed tool**, not the agent. The agent is a policy (an LLM) that learns to
**drive** the tool well: rewrite the query, search again from a new angle, read abstracts,
decide which papers to select, and decide when to stop. RL trains those decisions.

```
research query
   -> agent: <think> ... </think>
   -> agent: <search> rewritten keywords </search>   --BM25 tool-->  <observation> top-k papers </observation>
   -> agent: <search> another angle </search>         --BM25 tool-->  <observation> ... </observation>
   -> agent: <read> paper_id </read>                  ----------->   <observation> abstract </observation>
   -> agent: <select> id, id, ... </select>
   -> agent: <finish/>
   -> reward = Recall@K(selected vs gold)  +  format bonus  ( - search/token cost, Phase 3 )
```

One run of this = a **trajectory**. RL samples several trajectories per query, compares them,
and nudges the policy toward the higher-reward behaviors. The BM25 tool never changes; the
policy does.

## Data (already downloaded, in `data/raw/`)

- **AutoScholarQuery**: `train.jsonl` (33,551), `dev.jsonl` / `test.jsonl` (1,000 each).
  Each = `question` + gold `answer_arxiv_id` (avg 2.6 answers/query).
- **Corpus**: `cs_paper_2nd.zip` (569,432 CS papers, 2007-2024, each with title+abstract+sections)
  and `id2paper.json` (arxiv_id -> title).
- **Verified**: 92.0% of gold answers are end-to-end retrievable from the corpus
  (id -> title -> normalized filename). Offline BM25 plan is sound; no live API.

## Model & hardware

- **Main model: Qwen2.5-3B-Instruct** (strong enough for multi-turn agentic RL).
- **Debug: Qwen2.5-0.5B-Instruct** (smoke-test the pipeline fast).
- **Must use LoRA + vLLM rollout** (no full fine-tune). LoRA also gives the KL reference for
  free: disable the adapter = base weights, no second model copy in memory.
- **A100 80GB**: comfortable (group size 8). **A100 40GB**: works with LoRA, group size 4-5,
  3-4 turns, truncated abstracts.
- Memory driver is **sequence length x group size**, not the base model — keep trajectories
  short and abstracts truncated.

## Roadmap

**Phase 0 — close the loop** (query in -> reward out, before any training)
1. `data/` — load queries, id/title normalization, gold resolution (verified logic).
2. `corpus/` — build subset corpus: gold papers + distractors -> `data/corpus/papers.jsonl`.
3. `retriever/` — BM25 behind a pluggable `Retriever` interface (dense is a drop-in later).
4. `env/` + `reward/` — action parsing (search/read/select/finish), Recall@K + format bonus.
5. `rollout/` + `eval/` — zero-shot baseline: Recall/Precision/NDCG@K, avg search calls.
   *Loop closes here — only the policy-update step is missing.*

**Phase 2 — RL, in two stages**
- **Stage A**: minimal GRPO from scratch (group-relative advantage, retrieved-token masking,
  KL) on 0.5B/3B — the learning core. Read TRL's `GRPOTrainer` + DeepSeekMath for the mechanics.
- **Stage B**: port the same env/reward into **veRL** (reference: Search-R1), train 3B on 1-2K
  samples. Watch for GRPO collapse; keep the best dev checkpoint, not the last.

**Phase 3 — cost-aware search RL** (the highlight)
Compare zero-shot vs vanilla GRPO vs cost-aware GRPO: can the agent keep retrieval quality
while using fewer searches?

**Optional ablations / future**: retriever strength (BM25 vs dense), reward shaping,
failure-driven resampling, a deep-research layer. Not on the critical path.

## Why BM25 (not sparse+dense+rerank) for now

A **weak** retriever leaves more for the agent to learn (query rewriting, multi-turn) — a
stronger retriever would shrink the decision space and the learning signal. BM25 is also
zero-GPU, sub-second to index, and keeps all VRAM for the policy. Dense/rerank become an
*ablation* later via the pluggable interface, not a dependency now.

## Status

Data explored and verified. Next: write `data/` (Phase 0, Step 1) and produce the coverage
report + subset corpus. See `../Scholar-R1_Project_Plan.md` for the detailed plan.
