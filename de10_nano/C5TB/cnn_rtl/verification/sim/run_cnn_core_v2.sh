#!/bin/bash
# cnn_core_v2 回归：生成 N 层随机向量 → verilator 对拍（需 verilator + python3）
# 用法: bash run_cnn_core_v2.sh [层数] [seed]
set -e
cd "$(dirname "$0")/../.."
N=${1:-16}
SEED=${2:-7}
echo "== 生成 ${N} 层随机向量 (seed=${SEED}) =="
rm -rf verification/v2
python3 tools/gen_cnn_core_v2_vectors.py "$N" "$SEED"
rm -rf /tmp/cnn_core_v2_vlt
verilator --binary --timing --public-flat-rw -Wno-lint -Wno-fatal \
    --Mdir /tmp/cnn_core_v2_vlt -o sim_v2 \
    src/cnn_core_v2.v src/mac8x8_dsp.v src/mac8x8_lut.v verification/tb/tb_cnn_core_v2.v
cd verification
pass=0; fail=0
for d in v2/layer_*; do
    rm -f vec_core_v2 && ln -s "$(basename "$d")" vec_core_v2
    out=$(timeout 300 /tmp/cnn_core_v2_vlt/sim_v2 2>&1 | grep -E "PASS|FAIL|TIMEOUT" | head -1)
    if echo "$out" | grep -q PASS; then pass=$((pass+1)); else fail=$((fail+1)); echo "$d: $out"; fi
done
echo "== pass=$pass fail=$fail =="
[ "$fail" -eq 0 ] || exit 1
