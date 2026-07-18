#!/bin/bash
set -euo pipefail

HOST="root@172.16.68.110"
SSH_OPTS="-o ConnectTimeout=5 -o ControlMaster=auto -o ControlPath=/tmp/ssh-ctrl -o ControlPersist=300"

cd paddle_frame
echo "=== 同步到 $HOST:/opt/paddle_frame ==="

echo "[1/3] 获取远端文件列表..."
ssh $SSH_OPTS "$HOST" "cd /opt/paddle_frame && find . -type f 2>/dev/null | sort" > /tmp/up.r || true

echo "[2/3] 删除远端多余文件..."
find . -type f | sort > /tmp/up.l
n=$(comm -23 /tmp/up.r /tmp/up.l | tee /tmp/up.d | wc -l)
if [ "$n" -gt 0 ]; then
  echo "      删除 $n 个文件"
  ssh $SSH_OPTS "$HOST" "cd /opt/paddle_frame && xargs -r rm" < /tmp/up.d
else
  echo "      无多余文件"
fi

echo "[3/3] 推送变化文件..."
[ -f .last_time ] && find . -type f -not -path './data/*' -not -path './images/*' -newer .last_time > /tmp/up.n \
                  || find . -type f -not -path './data/*' -not -path './images/*' > /tmp/up.n
n=$(wc -l < /tmp/up.n)
if [ "$n" -gt 0 ]; then
  echo "      上传 $n 个文件"
  tar cf - -T /tmp/up.n | ssh $SSH_OPTS "$HOST" "tar xf - -C /opt/paddle_frame -m"
else
  echo "      无变化文件"
fi

touch .last_time
echo "=== 完成 ==="
