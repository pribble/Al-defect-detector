#!/bin/bash
set -euo pipefail

HOST="root@172.16.68.110"
SSH_OPTS="-o ConnectTimeout=5 -o ControlMaster=auto -o ControlPath=/tmp/ssh-ctrl -o ControlPersist=300"

cd "$(dirname "$0")/deploy"
echo "=== 同步到 $HOST:/opt/paddle_frame ==="

echo "推送变化文件..."
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
