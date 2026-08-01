#!/usr/bin/env bash
set -euo pipefail

WORKDIR=${WORKDIR:-verification/sim/work_conv_traversal}

STD=(--std=08)
WORK=(--work=work --workdir="$WORKDIR")

RTL=hardware/rtl/layers/conv_layer.vhd
TB=verification/tb/conv/tb_conv_layer_traversal.vhd

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

for file in "$RTL" "$TB"; do
    if [ ! -f "$file" ]; then
        echo \
            "run_conv_traversal.sh: missing $file" \
            >&2

        exit 1
    fi
done

echo "Analyzing conv_layer..."
ghdl -a \
    "${STD[@]}" \
    "${WORK[@]}" \
    "$RTL"

echo "Analyzing traversal matrix testbench..."
ghdl -a \
    "${STD[@]}" \
    "${WORK[@]}" \
    "$TB"

echo "Elaborating tb_conv_layer_traversal..."
ghdl -e \
    "${STD[@]}" \
    "${WORK[@]}" \
    tb_conv_layer_traversal

run_case() {
    local name=$1
    shift

    echo
    echo \
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Traversal case: $name"
    echo \
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    ghdl -r \
        "${STD[@]}" \
        "${WORK[@]}" \
        tb_conv_layer_traversal \
        "-gG_NAME=$name" \
        "$@" \
        --assert-level=error
}

# Padding + stride 2 + two output groups.
run_case padding_stride2_groups \
    -gG_C_IN=1 \
    -gG_C_OUT=4 \
    -gG_W_IN=5 \
    -gG_H_IN=5 \
    -gG_C_PAR=2 \
    -gG_KERNEL=3 \
    -gG_PADDING=1 \
    -gG_STRIDE=2

# Two input channels, including weight reloading between channels.
# Gaps are inserted into both activation and weight streams.
run_case multichannel_with_source_gaps \
    -gG_C_IN=2 \
    -gG_C_OUT=4 \
    -gG_W_IN=5 \
    -gG_H_IN=5 \
    -gG_C_PAR=2 \
    -gG_KERNEL=3 \
    -gG_PADDING=1 \
    -gG_STRIDE=2 \
    -gG_INPUT_GAP_PERIOD=5 \
    -gG_WEIGHT_GAP_PERIOD=7

# Exercises vertical_advance_remaining values greater than one.
run_case padding_stride3_groups \
    -gG_C_IN=1 \
    -gG_C_OUT=4 \
    -gG_W_IN=7 \
    -gG_H_IN=7 \
    -gG_C_PAR=2 \
    -gG_KERNEL=3 \
    -gG_PADDING=1 \
    -gG_STRIDE=3

# Input row 5 is unused mathematically and must still be consumed.
run_case trailing_row_drain \
    -gG_C_IN=1 \
    -gG_C_OUT=2 \
    -gG_W_IN=4 \
    -gG_H_IN=6 \
    -gG_C_PAR=2 \
    -gG_KERNEL=3 \
    -gG_PADDING=0 \
    -gG_STRIDE=2

# Combined traversal, source gaps and output backpressure.
run_case combined_backpressure \
    -gG_C_IN=1 \
    -gG_C_OUT=4 \
    -gG_W_IN=5 \
    -gG_H_IN=5 \
    -gG_C_PAR=2 \
    -gG_KERNEL=3 \
    -gG_PADDING=1 \
    -gG_STRIDE=2 \
    -gG_INPUT_GAP_PERIOD=6 \
    -gG_WEIGHT_GAP_PERIOD=5 \
    -gG_STALL_PERIOD=4

echo
echo "PASS: all traversal matrix cases completed."