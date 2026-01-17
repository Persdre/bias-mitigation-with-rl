# 从 Git 中移除数据文件的步骤

数据文件太大，会导致推送超时。按照以下步骤移除数据文件但保留本地文件：

## 步骤 1: 从 Git 跟踪中移除数据文件（保留本地文件）

```bash
cd /home/qian/bias-mitigation-with-rl

# 从 git 中移除 data 目录下的所有文件（但保留本地文件）
git rm -r --cached data/

# 检查状态
git status
```

## 步骤 2: 提交更改

```bash
# 提交 .gitignore 的更改和数据文件的移除
git add .gitignore
git commit -m "Remove large data files from git tracking, add to .gitignore"
```

## 步骤 3: 可选 - 创建 data/.gitkeep 保留目录结构

```bash
# 创建 .gitkeep 文件以保留 data 目录结构
touch data/.gitkeep
git add data/.gitkeep
git commit -m "Add .gitkeep to preserve data directory structure"
```

## 步骤 4: 重新推送（现在应该快很多）

```bash
# 设置更大的缓冲区（如果还没设置）
git config http.postBuffer 524288000
git config http.timeout 600

# 推送
git push origin
```

## 验证

推送后，数据文件将：
- ✅ 保留在本地文件系统中
- ✅ 不再被 git 跟踪
- ✅ 不会上传到远程仓库
- ✅ 其他用户克隆仓库时不会下载数据文件

## 如果用户需要数据文件

在 README 中说明：
- 数据文件太大，不包含在仓库中
- 用户需要自己准备数据或从指定位置下载
- 或者使用数据预处理脚本生成数据

