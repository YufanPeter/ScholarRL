#!/bin/bash
# ScholarRL setup — two modes:
#
#   bash server_setup.sh pack     run on the LOCAL machine: package the minimal data set
#   bash server_setup.sh          run on the SERVER: install, verify data, smoke-test
#
# Why "minimal": the 2.5GB cs_paper_2nd.zip is only needed to BUILD the corpus and to
# list its filenames. Both are already done locally — the built index carries the paper
# text (papers_meta.jsonl) and data/corpus/zip_namelist.txt replaces the zip's namelist
# (see src/scholarrl/data/retrievable.py). Shipping the prebuilt artifacts instead of
# the raw zip is ~500MB rather than ~3.2GB, and skips a re-index on the server.
#
# Ship the zip too only if you intend to REBUILD the corpus there (changing
# distractor_ratio, or Phase 2 hard negatives).

set -euo pipefail

ARCHIVE="scholar_data.tar.gz"
REPO_URL="https://github.com/YufanPeter/ScholarRL.git"

# Honour the same override the code uses (src/scholarrl/paths.py).
DATA_DIR="${SCHOLAR_DATA:-data}"

# Minimal set needed to RUN (train / eval / rollout). Paths are relative to DATA_DIR.
RUNTIME_FILES=(
    "raw/train.jsonl"
    "raw/dev.jsonl"
    "raw/test.jsonl"
    "raw/id2paper.json"              # resolve_gold(): arxiv id -> title
    "corpus/zip_namelist.txt"        # stands in for the 2.5GB zip
    "corpus/bm25_index/papers_meta.jsonl"
    "corpus/bm25_index/params.index.json"
    "corpus/bm25_index/vocab.index.json"
    "corpus/bm25_index/data.csc.index.npy"
    "corpus/bm25_index/indices.csc.index.npy"
    "corpus/bm25_index/indptr.csc.index.npy"
)

ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠ $*"; }
die()  { echo "  ✗ $*" >&2; exit 1; }
step() { echo; echo "[$1] $2"; }

# --- pack mode (local machine) --------------------------------------------------

pack() {
    echo "======================================"
    echo "Packing minimal data set -> $ARCHIVE"
    echo "======================================"
    [ -d "$DATA_DIR" ] || die "no data dir at '$DATA_DIR' (run from the repo root, or set SCHOLAR_DATA)"

    local missing=0
    for rel in "${RUNTIME_FILES[@]}"; do
        if [ -f "$DATA_DIR/$rel" ]; then
            ok "$rel"
        else
            warn "MISSING $rel"
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo
        echo "Build the missing artifacts first:"
        echo "  python -m scripts.build_corpus && python -m scripts.build_index"
        die "cannot pack an incomplete data set"
    fi

    # -h follows symlinks: data/raw/* are symlinks to the Desktop downloads.
    tar -czhf "$ARCHIVE" -C "$DATA_DIR" "${RUNTIME_FILES[@]}"

    echo
    ok "created $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
    echo
    echo "Next — copy it over and unpack into the repo's data/ dir:"
    echo "  SERVER=your_user@your_host"
    echo "  rsync -avzP $ARCHIVE \$SERVER:~/"
    echo "  ssh \$SERVER 'mkdir -p ~/ScholarRL/data && tar -xzf ~/$ARCHIVE -C ~/ScholarRL/data'"
    echo "  ssh \$SERVER 'cd ~/ScholarRL && bash server_setup.sh'"
}

# --- setup mode (server) --------------------------------------------------------

setup() {
    echo "======================================"
    echo "ScholarRL Server Setup"
    echo "======================================"

    step 1/6 "Repository"
    if [ -f "pyproject.toml" ] && [ -d "src/scholarrl" ]; then
        ok "already inside the repo ($(pwd))"
    elif [ -d "ScholarRL" ]; then
        cd ScholarRL && git pull --ff-only && ok "pulled latest"
    else
        git clone "$REPO_URL" && cd ScholarRL && ok "cloned"
    fi

    step 2/6 "Python environment"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    # venv activation trips `set -u` on older virtualenv scripts
    set +u; source venv/bin/activate; set -u
    ok "$(python --version) at $(command -v python)"

    step 3/6 "Dependencies"
    pip install --quiet --upgrade pip
    pip install --quiet -e .
    ok "scholarrl installed (editable)"
    if python -c "import torch" 2>/dev/null; then
        python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
else:
    print("  ⚠ no CUDA visible — a 3B rollout on CPU is unusably slow")
PY
    else
        warn "torch not installed; install it before running --policy hf"
    fi

    step 4/6 "Data"
    mkdir -p "$DATA_DIR/raw" "$DATA_DIR/corpus" outputs/trajectories outputs/checkpoints outputs/logs
    local missing=()
    for rel in "${RUNTIME_FILES[@]}"; do
        [ -f "$DATA_DIR/$rel" ] || missing+=("$rel")
    done
    if [ ${#missing[@]} -eq 0 ]; then
        ok "all runtime files present in '$DATA_DIR' ($(du -sh "$DATA_DIR" | cut -f1))"
    else
        echo "  missing ${#missing[@]} file(s) under '$DATA_DIR':"
        printf '    - %s\n' "${missing[@]}"
        echo
        echo "  Pack them on your local machine and copy them over:"
        echo "    bash server_setup.sh pack"
        echo "  Or, if the raw zip IS here, build them now:"
        echo "    python -m scripts.build_corpus && python -m scripts.build_index"
        die "data incomplete"
    fi

    step 5/6 "Tests"
    python -m pytest -q

    step 6/6 "Retrieval sanity check"
    # End-to-end proof that index + id2paper + the namelist stand-in all line up.
    # Expect mean gold recall ~0.25; a hard failure here means a missing/stale artifact.
    python -m scripts.eval_retriever 20 200

    cat <<'EOF'

======================================
Setup complete
======================================

Smoke-test the model path first (0.5B is ~1GB and takes minutes, not tens of minutes):

  python -m scripts.run_baseline --policy hf --model Qwen/Qwen2.5-0.5B-Instruct \
    --split dev --num_queries 3 --output /tmp/smoke.jsonl

Then the real 3B baseline under the split search/read budget (detach: SSH drops kill it):

  nohup python -m scripts.run_baseline --policy hf --split dev --num_queries 50 \
    --output outputs/baseline_3b_dev50_v2.jsonl > outputs/logs/baseline_v2.log 2>&1 &
  tail -f outputs/logs/baseline_v2.log

Then inspect — analyze_baseline for behaviour, ceiling for how much gold was reachable:

  python -m scripts.analyze_baseline outputs/baseline_3b_dev50_v2.jsonl --show-failures 5
  python -m scripts.ceiling --baseline outputs/baseline_3b_dev50_v2.jsonl

outputs/ is gitignored, so pull results back explicitly:

  rsync -avzP your_user@your_host:~/ScholarRL/outputs/baseline_3b_dev50_v2.jsonl ./outputs/
EOF
}

case "${1:-setup}" in
    pack)  pack ;;
    setup) setup ;;
    *)     die "usage: bash server_setup.sh [pack|setup]" ;;
esac
