# Enhanced Baseline Output Format

## Summary

`run_baseline.py` now saves **three files** for each run:

1. **`output.jsonl`** — Full trajectories with metadata (default mode)
2. **`output.summary.txt`** — Human-readable statistics
3. Analysis via **`analyze_baseline.py`** — Detailed breakdown

## What Changed

### Before (old format)
```bash
python -m scripts.run_baseline --output baseline.jsonl
# Output: one JSONL with basic fields (query_id, reward, selected, gold)
# Problems:
#   - No hyperparameters saved (can't reproduce later)
#   - No messages/actions (can't debug failures)
#   - Terminal output gets lost
```

### After (new format)
```bash
python -m scripts.run_baseline --output baseline.jsonl
# Generates:
#   baseline.jsonl         (metadata + full trajectories)
#   baseline.summary.txt   (human-readable stats)
# Plus: analyze_baseline.py for detailed breakdown
```

## Output Files

### 1. JSONL Format

**First line: metadata**
```json
{
  "_meta": true,
  "split": "dev",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "temperature": 0.0,
  "seed": 42,
  "n_queries": 969,
  "avg_reward": 0.2470,
  "avg_task_reward": 0.2470,
  "config": {
    "max_retrieval_turns": 8,
    "top_k": 5,
    "reward_k": 20,
    "lambda_fmt": 0.1,
    "distractor_ratio": 10
  }
}
```

**Subsequent lines: trajectories (full mode, default)**
```json
{
  "query_id": "...",
  "question": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<search>...</search>"},
    {"role": "user", "content": "[search results]..."}
  ],
  "actions": ["<search>...</search>", "<read>...</read>", ...],
  "observations": ["[search results]...", "[read]...", ...],
  "reward": 0.25,
  "task_reward": 0.25,
  "format_reward": 0.0,
  "selected": ["paper1", "paper2"],
  "gold": ["gold1", "gold2"],
  "retrieval_turns": 5,
  "steps": 7,
  "reason": "finish"
}
```

**Lightweight mode (--no-save-trajectories)**
```json
{
  "query_id": "...",
  "question": "...",
  "reward": 0.25,
  "task_reward": 0.25,
  "format_reward": 0.0,
  "selected": ["paper1", "paper2"],
  "gold": ["gold1", "gold2"],
  "retrieval_turns": 5,
  "steps": 7,
  "reason": "finish"
}
```
(No `messages`/`actions`/`observations` — saves ~90% disk space)

### 2. Summary File (.summary.txt)

```
=== Baseline Evaluation Summary ===

Split:       dev
Policy:      hf
Model:       Qwen/Qwen2.5-3B-Instruct
Temperature: 0.0
Seed:        42

--- Config ---
max_retrieval_turns: 8
max_steps:           20
top_k:               5
reward_metric:       recall
reward_k:            20
lambda_fmt:          0.1
distractor_ratio:    10

--- Results ---
Queries evaluated:   969
Average reward:      0.2470
Average task reward: 0.2470

--- Statistics ---
Recall@20 (task reward):
  Mean:   0.2470
  Median: 0.2000
  Min:    0.0000
  Max:    1.0000
Hit rate (≥1 gold): 380/969 (39.2%)

Retrieval turns:
  Mean:   6.5
  Budget-capped: 123/969 (12.7%)
```

### 3. Analyze Script

```bash
python -m scripts.analyze_baseline outputs/baseline.jsonl --show-failures 5
```

**Output:**
```
================================================================================
CONFIGURATION
================================================================================
Split:        dev
Model:        Qwen/Qwen2.5-3B-Instruct
Temperature:  0.0
Seed:         42
...

================================================================================
TASK REWARD (Recall@K)
================================================================================
Mean:     0.2470
Median:   0.2000
Std dev:  0.2815
Min:      0.0000
Max:      1.0000
Hit rate (≥1 gold): 380/969 (39.2%)

================================================================================
TOTAL REWARD (task + λ·format)
================================================================================
Mean:     0.2550
Median:   0.2100
...

================================================================================
BUDGET USAGE
================================================================================
Avg retrieval turns: 6.50
Avg total steps:     8.20
Budget-capped:       123/969 (12.7%)

================================================================================
FAILURES (task_reward = 0): 589/969 (60.8%)
================================================================================

Showing 5 sample failure cases:

1. Query ID: AutoScholarQuery_dev_42
   Question: Could you list papers about neural network pruning...
   Gold:     ['2104.08378', '2203.15556']
   Selected: []
   Reason:   finish
   Retrieval turns: 8 (budget capped)
   ...
```

## Usage Examples

### Default (full trajectories)
```bash
python -m scripts.run_baseline \
  --policy hf \
  --model Qwen/Qwen2.5-3B-Instruct \
  --split dev \
  --output outputs/baseline_3b_dev.jsonl
```
**Generates:**
- `outputs/baseline_3b_dev.jsonl` (~2 MB for 969 queries)
- `outputs/baseline_3b_dev.summary.txt` (~1 KB)

### Lightweight (summary only)
```bash
python -m scripts.run_baseline \
  --policy hf \
  --split dev \
  --output outputs/baseline_lite.jsonl \
  --no-save-trajectories
```
**Generates:**
- `outputs/baseline_lite.jsonl` (~200 KB for 969 queries, 10x smaller)
- `outputs/baseline_lite.summary.txt`

### Analyze results
```bash
# Basic statistics
python -m scripts.analyze_baseline outputs/baseline_3b_dev.jsonl

# With failure case study
python -m scripts.analyze_baseline outputs/baseline_3b_dev.jsonl --show-failures 10
```

## Why These Changes?

### For Baseline Evaluation
✓ **Reproducibility**: Metadata captures exact config/seed/model  
✓ **Analysis**: Summary file gives quick stats without parsing JSONL  
✓ **Debugging**: Can see what agent actually did (messages/actions)

### For RL Training
✗ **GRPO doesn't need saved trajectories** (on-policy, real-time rollout)  
✓ **But useful for**:
  - Case study (why did this query fail?)
  - Offline RL (if you switch to that later)
  - Behavioral cloning (use good trajectories as demonstrations)
  - Paper writing (qualitative examples)

### Disk Space
- **Full mode**: ~2 KB/trajectory (969 queries = ~2 MB)
- **Lite mode**: ~200 bytes/trajectory (969 queries = ~200 KB)
- On server: disk is cheap, flexibility is valuable → use full mode

## Migration Guide

**Old code that reads baseline output:**
```python
# Before: assumed every line was a trajectory
with open("baseline.jsonl") as f:
    trajs = [json.loads(line) for line in f]
```

**New code (backward compatible):**
```python
with open("baseline.jsonl") as f:
    lines = [json.loads(line) for line in f]

# Skip metadata if present
if lines[0].get("_meta"):
    meta = lines[0]
    trajs = lines[1:]
else:
    trajs = lines  # old format
```

## Files Modified
- `src/scholarrl/rollout/rollout.py` — added `Trajectory.to_dict()`
- `scripts/run_baseline.py` — metadata + summary + `--save-trajectories` flag
- `scripts/analyze_baseline.py` — new script for detailed analysis
