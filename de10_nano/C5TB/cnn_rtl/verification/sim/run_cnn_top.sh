#!/bin/bash
# cnn_top 顶层回归：生成 N 层随机向量 → verilator 对拍（需 verilator + python3）
# 用法: bash run_cnn_top.sh [层数] [seed]
set -e
cd "$(dirname "$0")/../.."
N=${1:-6}
SEED=${2:-7}
echo "== 生成 ${N} 层随机向量 (seed=${SEED}) =="
rm -rf verification/vct
python3 tools/gen_cnn_top_vectors.py "$N" "$SEED"
rm -rf /tmp/cnn_top_vlt
verilator --binary --timing --public-flat-rw -Wno-lint -Wno-fatal -DSIMULATION \
    --Mdir /tmp/cnn_top_vlt -o sim_top \
    src/cnn_top.v src/cnn_top_core.v src/cnn_core_v2.v src/mac8x8_dsp.v src/mac8x8_lut.v \
    verification/tb/tb_cnn_top.v
cd verification
pass=0; fail=0
for d in vct/layer_*; do
    rm -f vec_cnn_top && ln -s "vct/$(basename "$d")" vec_cnn_top
    [ -e vec_cnn_top/param.hex ] || { echo "$d: 软链断链"; fail=$((fail+1)); continue; }
    out=$(timeout 600 /tmp/cnn_top_vlt/sim_top 2>&1 | grep -E "PASS|FAIL|TIMEOUT" | head -1)
    if echo "$out" | grep -q PASS; then pass=$((pass+1)); else fail=$((fail+1)); echo "$d: $out"; fi
done
echo "== pass=$pass fail=$fail =="
[ "$fail" -eq 0 ] || exit 1
