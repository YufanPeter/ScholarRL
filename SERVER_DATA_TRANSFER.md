# 服务器数据传输指南

## 方案 1：用 scp 传输数据（推荐）

### 从本地 Mac 传输到服务器

```bash
# 替换 your_username 和 server_address
SERVER="your_username@server_address"

# 1. 传输所有数据文件（一次性）
scp ~/Desktop/data/raw/train.jsonl \
    ~/Desktop/data/raw/dev.jsonl \
    ~/Desktop/data/raw/test.jsonl \
    ~/Desktop/data/raw/id2paper.json \
    ~/Desktop/data/raw/cs_paper_2nd.zip \
    $SERVER:~/ScholarRL/data/raw/

# 2. 验证传输
ssh $SERVER "ls -lh ~/ScholarRL/data/raw/"
```

### 压缩后传输（如果网络慢）

```bash
# 本地先打包
cd ~/Desktop/data
tar -czf raw.tar.gz raw/

# 传输压缩包（~2.3 GB → ~1.5 GB）
scp raw.tar.gz $SERVER:~/ScholarRL/data/

# 服务器上解压
ssh $SERVER "cd ~/ScholarRL/data && tar -xzf raw.tar.gz && rm raw.tar.gz"
```

---

## 方案 2：用 rsync 增量同步（适合多次传输）

```bash
# 第一次传输（完整）
rsync -avz --progress ~/Desktop/data/raw/ $SERVER:~/ScholarRL/data/raw/

# 后续只传输改动的文件（增量）
rsync -avz --progress ~/Desktop/data/raw/ $SERVER:~/ScholarRL/data/raw/
```

**优点**：
- 断点续传
- 增量更新（只传改动的部分）
- 显示进度条

---

## 方案 3：服务器直接下载（如果数据在云端）

如果你的数据在 Google Drive / Dropbox / 其他云盘：

```bash
# SSH 到服务器
ssh $SERVER

cd ~/ScholarRL/data/raw

# 用 wget 或 curl 下载（需要分享链接）
wget "https://your-shared-link/train.jsonl"
wget "https://your-shared-link/dev.jsonl"
# ... 其他文件
```

---

## 方案 4：用 SCHOLAR_DATA 环境变量（数据已在服务器上）

如果数据已经在服务器的其他位置（比如共享存储）：

```bash
# 不复制数据，用环境变量指向
ssh $SERVER

# 在 ~/.bashrc 或 ~/.zshrc 加一行
echo 'export SCHOLAR_DATA=/path/to/existing/data' >> ~/.bashrc
source ~/.bashrc

# ScholarRL 会自动读取那个位置的数据
cd ~/ScholarRL
python -m scripts.build_corpus  # 会读取 $SCHOLAR_DATA/raw/
```

---

## 数据文件清单（确认完整性）

传输后在服务器上验证：

```bash
ssh $SERVER
cd ~/ScholarRL/data/raw
ls -lh

# 应该看到：
# train.jsonl        (~40 MB)
# dev.jsonl          (~1.2 MB)
# test.jsonl         (~1.2 MB)
# id2paper.json      (~100 MB)
# cs_paper_2nd.zip   (~2.3 GB)

# 验证行数
wc -l *.jsonl
# train.jsonl:  33551
# dev.jsonl:    1000
# test.jsonl:   1000
```

---

## 完整服务器设置流程（从零开始）

### 步骤 1：SSH 登录
```bash
ssh your_username@server_address
```

### 步骤 2：Clone 仓库
```bash
git clone https://github.com/YufanPeter/ScholarRL.git
cd ScholarRL
```

### 步骤 3：传输数据（从本地 Mac）
```bash
# 在本地 Mac 的另一个终端运行
scp ~/Desktop/data/raw/* your_username@server_address:~/ScholarRL/data/raw/
```

### 步骤 4：运行自动设置脚本
```bash
# 回到服务器终端
bash server_setup.sh
```

### 步骤 5：测试 baseline
```bash
# 激活虚拟环境（如果用了）
source venv/bin/activate

# 跑 50 条测试
python -m scripts.run_baseline \
  --policy hf \
  --model Qwen/Qwen2.5-3B-Instruct \
  --split dev \
  --num_queries 50 \
  --output outputs/baseline_3b_dev50.jsonl
```

---

## 常见问题

### Q: scp 传输很慢怎么办？
A: 用压缩 + rsync：
```bash
tar -czf data.tar.gz ~/Desktop/data/raw/
rsync -avz --progress data.tar.gz $SERVER:~/
ssh $SERVER "cd ~ && tar -xzf data.tar.gz && rm data.tar.gz"
```

### Q: 服务器网络不允许 SSH（只能用 HTTP/HTTPS）？
A: 把数据上传到云盘（Google Drive），生成分享链接，服务器上用 `wget` 下载。

### Q: 服务器上已有数据，不想重复存储？
A: 用软链接：
```bash
ln -s /existing/data/path ~/ScholarRL/data/raw
```
或者设置 `SCHOLAR_DATA` 环境变量。

### Q: 需要 push 代码到 GitHub（需要 SSH key）？
A: 参考前面的"方法 2"配置 SSH key，或者用 HTTPS + Personal Access Token。
