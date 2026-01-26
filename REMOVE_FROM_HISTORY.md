# 从 Git 历史中移除大文件

## 问题
即使文件已经从当前跟踪中移除，它们仍然在 git 历史中。GitHub 会检查整个历史，拒绝超过 100MB 的文件。

## 解决方案

### 方案 1: 使用 git filter-repo (推荐)

```bash
cd /home/qian/bias-mitigation-with-rl

# 安装 git-filter-repo (如果还没安装)
# pip install git-filter-repo

# 从整个历史中移除 data 目录
git filter-repo --path data --invert-paths --force

# 或者只移除特定的大文件
git filter-repo --path data/kk/instruct/jppl/train.parquet --invert-paths --force
```

### 方案 2: 使用 git filter-branch (如果没有 git-filter-repo)

```bash
cd /home/qian/bias-mitigation-with-rl

# 从整个历史中移除 data 目录
git filter-branch --force --index-filter \
  "git rm -rf --cached --ignore-unmatch data" \
  --prune-empty --tag-name-filter cat -- --all

# 清理引用
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### 方案 3: 创建新分支（如果历史不重要）

如果 git 历史不重要，可以创建一个新的干净分支：

```bash
cd /home/qian/bias-mitigation-with-rl

# 创建一个新的孤儿分支（没有历史）
git checkout --orphan new-main

# 添加所有文件（除了 data）
git add .
git commit -m "Initial commit without large data files"

# 强制推送到远程
git push -f origin new-main:main
```

### 方案 4: 使用 BFG Repo-Cleaner (最快)

```bash
# 下载 BFG
# wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# 克隆一个裸仓库
cd /tmp
git clone --mirror /home/qian/bias-mitigation-with-rl bias-mitigation-with-rl.git

# 删除大文件
java -jar bfg-1.14.0.jar --delete-folders data bias-mitigation-with-rl.git

# 清理
cd bias-mitigation-with-rl.git
git reflog expire --expire=now --all && git gc --prune=now --aggressive

# 推送
git push
```

## 推荐步骤（使用 git filter-repo）

```bash
cd /home/qian/bias-mitigation-with-rl

# 1. 备份（重要！）
cd ..
cp -r bias-mitigation-with-rl bias-mitigation-with-rl-backup

# 2. 安装 git-filter-repo
pip install git-filter-repo

# 3. 从历史中移除 data 目录
git filter-repo --path data --invert-paths --force

# 4. 强制推送（会重写历史）
git push origin --force --all
```

## 警告

⚠️ **重写 git 历史会改变所有 commit 的 hash**，如果其他人已经克隆了仓库，他们需要重新克隆。

如果这是个人仓库或新仓库，可以安全地使用 `--force` 推送。

