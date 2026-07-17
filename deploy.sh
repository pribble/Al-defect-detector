#!/bin/bash
# deploy.sh — 将本仓库同步到树莓派
# 用法: ./deploy.sh [目标路径]
# 默认: root@172.16.68.111:/opt/HaoYao/

set -e

# 仓库根目录
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_PI="172.16.68.111"
DEFAULT_PATH="/opt/HaoYao"

# 参数: deploy.sh <user@host> <目标路径>
PI="${1:-root@$DEFAULT_PI}"
DEST="${2:-$DEFAULT_PATH}"

echo "===== 部署 HaoYao 到 $PI:$DEST ====="

# 检查 rsync 是否可用
if ! command -v rsync &>/dev/null; then
    echo "错误: 需要安装 rsync"
    echo "  sudo apt install rsync"
    exit 1
fi

# 干跑一次，看看会传什么
echo "--- 预览变更 ---"
rsync -avzc --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.sw?' \
    --exclude='defect.db' \
    --exclude='files/' \
    --exclude='original_files/' \
    --exclude='detect_files/' \
    --dry-run \
    "$REPO_DIR/" "$PI:$DEST/"

echo ""
echo "以上是将会同步的文件列表"
read -p "确认同步? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "已取消"
    exit 0
fi

# 正式同步
echo "--- 同步中 ---"
rsync -avz --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.sw?' \
    --exclude='defect.db' \
    --exclude='files/' \
    --exclude='original_files/' \
    --exclude='detect_files/' \
    "$REPO_DIR/" "$PI:$DEST/"

echo ""
echo "===== 同步完成 ====="
echo ""
echo "下一步（在树莓派上）："
echo "  systemctl restart detect-api.service api.service"
echo "  journalctl -u detect-api.service -n 20 --no-pager"
