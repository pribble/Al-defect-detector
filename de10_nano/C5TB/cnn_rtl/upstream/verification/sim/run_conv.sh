#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="verification/sim/build/conv"
RESULT_DIR="verification/results/directed/conv"

RTL_FILE="hardware/rtl/layers/conv_layer.vhd"
TB_PACKAGE="verification/tb/conv/conv_tb_pkg.vhd"

mkdir -p "$RESULT_DIR"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Analyzing conv_layer..."
ghdl -a \
    --std=08 \
    --workdir="$BUILD_DIR" \
    "$RTL_FILE"

echo "Analyzing shared testbench package..."
ghdl -a \
    --std=08 \
    --workdir="$BUILD_DIR" \
    "$TB_PACKAGE"


run_testbench() {
    local tb_file="$1"
    local tb_entity="$2"

    echo
    echo "Analyzing $tb_entity..."

    ghdl -a \
        --std=08 \
        --workdir="$BUILD_DIR" \
        "$tb_file"

    echo "Elaborating $tb_entity..."

    ghdl -e \
        --std=08 \
        --workdir="$BUILD_DIR" \
        "$tb_entity"

    echo "Running $tb_entity..."

    ghdl -r \
        --std=08 \
        --workdir="$BUILD_DIR" \
        "$tb_entity" \
        --assert-level=error \
        --wave="$RESULT_DIR/$tb_entity.ghw"
}


run_testbench \
    "verification/tb/conv/tb_conv_layer_initial_line_fill.vhd" \
    "tb_conv_layer_initial_line_fill"

run_testbench \
    "verification/tb/conv/tb_conv_layer_prime_k_line.vhd" \
    "tb_conv_layer_prime_k_line"

run_testbench \
    "verification/tb/conv/tb_conv_layer_weight_filling.vhd" \
    "tb_conv_layer_weight_filling"

run_testbench \
    "verification/tb/conv/tb_conv_layer_prime_weight_concurrent.vhd" \
    "tb_conv_layer_prime_weight_concurrent"

run_testbench \
    "verification/tb/conv/tb_conv_layer_calculation.vhd" \
    "tb_conv_layer_calculation"

run_testbench \
    "verification/tb/conv/tb_conv_layer_horizontal_sliding.vhd" \
    "tb_conv_layer_horizontal_sliding"

run_testbench \
    "verification/tb/conv/tb_conv_layer_line_rotation.vhd" \
    "tb_conv_layer_line_rotation"

run_testbench \
    "verification/tb/conv/tb_conv_layer_output_backpressure.vhd" \
    "tb_conv_layer_output_backpressure"

run_testbench \
    "verification/tb/conv/tb_conv_layer_output_groups.vhd" \
    "tb_conv_layer_output_groups"

run_testbench \
    "verification/tb/conv/tb_conv_layer_requantization.vhd"\
    "tb_conv_layer_requantization"

echo
echo "All convolution testbenches passed."