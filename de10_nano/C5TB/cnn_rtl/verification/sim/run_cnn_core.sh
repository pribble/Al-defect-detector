#!/bin/bash
# run_cnn_core.sh — cnn_core.v 参数驱动对拍回归（verilator）
# 用法（在 cnn_rtl/ 根目录执行）：bash verification/sim/run_cnn_core.sh [层数]
set -e
cd "$(dirname "$0")/../.."   # 回到 cnn_rtl/

N=${1:-24}
echo "== [1/3] 生成向量（$N 层）=="
python3 tools/gen_cnn_core_vectors.py $N

echo "== [2/3] verilator 编译 =="
rm -rf /tmp/cnn_core_vlt
verilator --binary --timing --public-flat-rw \
    -Wno-lint -Wno-fatal \
    --Mdir /tmp/cnn_core_vlt -o sim_core \
    src/cnn_core.v \
    verification/tb/tb_cnn_core.v

echo "== [3/3] 逐层仿真 =="
cd verification
fail=0
for d in vec_core/layer_*; do
    ln -sf "$(basename "$d")/cfg.hex"    vec_core/cfg.hex
    ln -sf "$(basename "$d")/in.hex"     vec_core/in.hex
    ln -sf "$(basename "$d")/w.hex"      vec_core/w.hex
    ln -sf "$(basename "$d")/expect.hex" vec_core/expect.hex
    out=$(/tmp/cnn_core_vlt/sim_core 2>&1 | grep -E "PASS|FAIL|TIMEOUT" | tail -1)
    echo "$d: $out"
    echo "$out" | grep -q "PASS" || fail=1
done
if [ $fail -eq 0 ]; then echo "ALL LAYERS PASS"; else echo "SOME LAYERS FAILED"; fi
