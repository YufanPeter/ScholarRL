#!/bin/bash
# ScholarRL 服务器初始化脚本
# 用法：ssh 到服务器后运行 bash server_setup.sh

set -e  # 遇到错误就停止

echo "======================================"
echo "ScholarRL Server Setup"
echo "======================================"

# 1. Clone 仓库（HTTPS，不需要配置 SSH key）
echo ""
echo "[1/6] Cloning repository..."
if [ ! -d "ScholarRL" ]; then
    git clone https://github.com/YufanPeter/ScholarRL.git
    echo "✓ Repository cloned"
else
    echo "✓ Repository already exists, pulling latest..."
    cd ScholarRL && git pull && cd ..
fi

cd ScholarRL

# 2. 创建虚拟环境（可选，建议用）
echo ""
echo "[2/6] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
source venv/bin/activate
echo "✓ Virtual environment activated"

# 3. 安装依赖
echo ""
echo "[3/6] Installing dependencies..."
pip install --upgrade pip
pip install -e .
echo "✓ Dependencies installed"

# 4. 准备数据目录
echo ""
echo "[4/6] Setting up data directories..."
mkdir -p data/raw data/corpus outputs/trajectories outputs/checkpoints outputs/logs

# 检查数据文件是否存在
if [ ! -f "data/raw/train.jsonl" ]; then
    echo "⚠ WARNING: Data files not found in data/raw/"
    echo "You need to upload:"
    echo "  - train.jsonl, dev.jsonl, test.jsonl"
    echo "  - id2paper.json"
    echo "  - cs_paper_2nd.zip"
    echo ""
    echo "Option 1: Use scp from local machine:"
    echo "  scp ~/Desktop/data/raw/* server:/path/to/ScholarRL/data/raw/"
    echo ""
    echo "Option 2: Set SCHOLAR_DATA env var to point to existing data:"
    echo "  export SCHOLAR_DATA=/path/to/existing/data"
    echo ""
    read -p "Press Enter to continue (will skip data-dependent steps)..."
else
    echo "✓ Data files found"
fi

# 5. 构建 corpus（如果数据存在）
echo ""
echo "[5/6] Building corpus and BM25 index..."
if [ -f "data/raw/train.jsonl" ]; then
    echo "Building corpus (distractor_ratio=10)..."
    python -m scripts.build_corpus

    echo "Building BM25 index..."
    python -m scripts.build_index

    echo "✓ Corpus and index built"
else
    echo "⚠ Skipped (data not found)"
fi

# 6. 运行测试（验证安装）
echo ""
echo "[6/6] Running tests..."
python -m pytest tests/test_config.py tests/test_actions.py -v
echo "✓ Tests passed"

# 完成
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. If data files are missing, upload them to data/raw/"
echo "   (train.jsonl, dev.jsonl, test.jsonl, id2paper.json, cs_paper_2nd.zip)"
echo ""
echo "2. Run baseline evaluation:"
echo "   python -m scripts.run_baseline \\"
echo "     --policy hf --model Qwen/Qwen2.5-3B-Instruct \\"
echo "     --split dev --num_queries 50 \\"
echo "     --output outputs/baseline_3b_dev50.jsonl"
echo ""
echo "3. Analyze results:"
echo "   cat outputs/baseline_3b_dev50.summary.txt"
echo "   python -m scripts.analyze_baseline outputs/baseline_3b_dev50.jsonl --show-failures 5"
echo ""
echo "4. Full dev evaluation (all 969 queries):"
echo "   python -m scripts.run_baseline \\"
echo "     --policy hf --model Qwen/Qwen2.5-3B-Instruct \\"
echo "     --split dev \\"
echo "     --output outputs/baseline_3b_dev_full.jsonl"
echo ""
