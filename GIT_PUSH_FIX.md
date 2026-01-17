# Git Push 超时问题解决方案

## 问题原因
- 推送数据量过大（359.12 MiB）
- HTTP 408 超时：服务器在传输过程中断开连接
- Git 默认的 HTTP 超时时间太短

## 解决方案

### 方案 1: 增加 Git HTTP 超时和缓冲区（推荐）

```bash
# 增加 HTTP 超时时间（秒）
git config --global http.postBuffer 524288000  # 500MB
git config --global http.timeout 600  # 10分钟超时
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300

# 然后重新推送
git push origin
```

### 方案 2: 使用 SSH 而不是 HTTPS

如果当前使用 HTTPS，可以切换到 SSH（通常更稳定）：

```bash
# 查看当前 remote URL
git remote -v

# 如果使用 HTTPS，切换到 SSH
git remote set-url origin git@github.com:username/repo.git

# 然后推送
git push origin
```

### 方案 3: 分批推送（如果数据太大）

```bash
# 先推送最近的几个 commit
git push origin HEAD~5:main
git push origin HEAD~3:main
git push origin HEAD~1:main
git push origin
```

### 方案 4: 检查并移除大文件

```bash
# 检查仓库中的大文件
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 | \
  tail -20

# 如果发现大文件，应该：
# 1. 添加到 .gitignore
# 2. 从 git 历史中移除（如果已经提交）
git rm --cached large_file.parquet
git commit -m "Remove large file"
```

### 方案 5: 使用 Git LFS 管理大文件

如果数据文件必须包含在仓库中，使用 Git LFS：

```bash
# 安装 git-lfs
# 然后
git lfs install
git lfs track "*.parquet"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## 快速修复（推荐先试这个）

```bash
cd /home/qian/bias-mitigation-with-rl

# 设置更大的缓冲区和超时
git config http.postBuffer 524288000
git config http.timeout 600

# 重新推送
git push origin
```

