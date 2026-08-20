#!/bin/bash
# cnn_top 顶层回归（iverilog 版）：与 verilator 版同构，额外支持
# --float-expect：期望输出用 cpu_ref float 公式生成（RTL 仍用定点 scale），
# 用于量化 "RTL 定点实现 vs CPU float 参考" 的逐层偏差。
# 用法: bash run_cnn_top_iv.sh [层数] [seed] [--float-expect]
set -e
cd "$(dirname "$0")/../.."
N=${1:-6}
SEED=${2:-7}
FLOAT=""
BOX=""
for a in "$@"; do
    [ "$a" = "--float-expect" ] && FLOAT="--float-expect"
    [ "$a" = "--boxhead" ] && BOX="--boxhead"
done
echo "== 生成 ${N} 层随机向量 (seed=${SEED}${FLOAT:+ $FLOAT}${BOX:+ $BOX}) =="
rm -rf verification/vct
python3 tools/gen_cnn_top_vectors.py "$N" "$SEED" $FLOAT $BOX
OUT=/tmp/cnn_top_iv.out
iverilog -g2012 -DSIMULATION -s tb_cnn_top -o $OUT \
    src/cnn_top_core.v src/cnn_core.v src/requant_store.v \
    src/mac8x8_dsp.v src/mac8x8_lut.v \
    verification/tb/tb_cnn_top.v
cd verification
pass=0; fail=0
for d in vct/layer_*; do
    rm -f vec_cnn_top && ln -s "vct/$(basename "$d")" vec_cnn_top
    [ -e vec_cnn_top/param.hex ] || { echo "$d: 软链断链"; fail=$((fail+1)); continue; }
    out=$(timeout 600 $OUT 2>&1 | grep -E "PASS|FAIL|TIMEOUT" | head -1)
    if echo "$out" | grep -q PASS; then pass=$((pass+1)); else fail=$((fail+1)); echo "$d: $out"; fi
done
echo "== pass=$pass fail=$fail =="
[ "$fail" -eq 0 ] || exit 1
