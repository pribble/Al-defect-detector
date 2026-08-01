#!/bin/bash
# run_conv_s8.sh — conv_layer_s8 对拍回归（verilator 编译型仿真，秒级）
# 用法（在 cnn_rtl/ 根目录执行）：
#   bash verification/sim/run_conv_s8.sh
# 前置：
#   1. python3 tools/gen_tb_vectors.py   （生成 verification/vec/*.hex）
#   2. 已安装 verilator（sudo apt install verilator）+ numpy
set -e
cd "$(dirname "$0")/../.."   # 回到 cnn_rtl/

echo "== [1/3] 生成测试向量 =="
python3 tools/gen_tb_vectors.py

echo "== [2/3] verilator 编译 =="
rm -rf /tmp/conv_s8_vlt
verilator --binary --timing --public-flat-rw \
    -Wno-lint -Wno-fatal \
    --Mdir /tmp/conv_s8_vlt -o sim_conv \
    src/conv_layer_s8.v \
    verification/tb/tb_conv_layer_s8.v

echo "== [3/3] 仿真 =="
cd verification/tb
/tmp/conv_s8_vlt/sim_conv
