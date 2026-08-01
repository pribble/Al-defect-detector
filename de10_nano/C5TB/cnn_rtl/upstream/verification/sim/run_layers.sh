#!/usr/bin/env bash

set -euo pipefail


WORKDIR=${WORKDIR:-verification/sim/work_layers}

STD=(--std=08)
WORK=(--work=work --workdir="$WORKDIR")

VECTORS_DIR=${VECTORS_DIR:-verification/vectors/default}
RESULTS_DIR=${RESULTS_DIR:-verification/results/default}
RAW_ACC_DIR=${RAW_ACC_DIR:-"$RESULTS_DIR/raw_acc"}

FPGAQPARMS_JSON=${FPGAQPARMS_JSON:-ml/outputs/run0/fpgaqparms.json}

PAR_MACS_DEFAULT=${PAR_MACS_DEFAULT:-64}

STALL_PERIOD=${STALL_PERIOD:-7}
PROGRESS_STEP=${PROGRESS_STEP:-10}
TIMEOUT_CYCLES=${TIMEOUT_CYCLES:-0}

# Maximum size, in KB, of an individual object GHDL may allocate
# on the simulation stack.
MAX_STACK_ALLOC_KB=${MAX_STACK_ALLOC_KB:-65536}

FORCE_GOLDEN=${FORCE_GOLDEN:-0}
WAVE=${WAVE:-0}


with_trailing_slash() {
    case "$1" in
        */)
            printf "%s" "$1"
            ;;
        *)
            printf "%s/" "$1"
            ;;
    esac
}


VECTORS_G=$(with_trailing_slash "$VECTORS_DIR")

mkdir -p "$WORKDIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RAW_ACC_DIR"

# The vector-driven VHDL testbench creates several large constant arrays.
# Raise the process stack limit before launching GHDL.
if ! ulimit -s unlimited 2>/dev/null; then
    echo \
        "run_layers.sh: warning: could not set unlimited stack size" \
        >&2
fi

selected=("$@")


for required_file in \
    hardware/rtl/layers/conv_layer.vhd \
    verification/tb/layers/tb_conv_layer.vhd
do
    if [ ! -f "$required_file" ]; then
        echo \
            "run_layers.sh: missing $required_file" \
            >&2

        exit 1
    fi
done


if [ ! -f "$FPGAQPARMS_JSON" ]; then
    echo \
        "run_layers.sh: missing metadata JSON: $FPGAQPARMS_JSON" \
        >&2

    exit 1
fi


MANIFEST=$(mktemp "$WORKDIR/conv_layers_manifest.XXXXXX")

trap 'rm -f "$MANIFEST"' EXIT


echo "Preparing layer metadata and raw accumulator goldens..."


python - \
    "$VECTORS_DIR" \
    "$FPGAQPARMS_JSON" \
    "$PAR_MACS_DEFAULT" \
    "$RAW_ACC_DIR" \
    "$FORCE_GOLDEN" \
    "${selected[@]}" \
    > "$MANIFEST" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

from ml.src.models.alexnet64gray import AlexNet64Gray


vectors_dir = Path(sys.argv[1])
metadata_path = Path(sys.argv[2])
par_default = int(sys.argv[3])
raw_acc_dir = Path(sys.argv[4])
force_golden = bool(int(sys.argv[5]))
selected = set(sys.argv[6:])


def die(message: str) -> None:
    raise SystemExit(
        f"run_layers.sh: {message}"
    )


def pair(
    value,
    label: str,
) -> tuple[int, int]:

    if isinstance(value, tuple):
        if len(value) != 2:
            die(
                f"unsupported {label}: {value}"
            )

        return int(value[0]), int(value[1])

    return int(value), int(value)


def conv_out_size(
    size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:

    return (
        (
            size
            + 2 * padding
            - dilation * (kernel - 1)
            - 1
        )
        // stride
    ) + 1


def choose_parallelism(
    c_out: int,
) -> int:

    limit = min(
        par_default,
        c_out,
    )

    for candidate in range(
        limit,
        0,
        -1,
    ):
        if c_out % candidate == 0:
            return candidate

    return 1


def require_file(
    path: Path,
) -> None:

    if not path.is_file():
        die(
            f"missing vector file: {path}"
        )


def require_size(
    path: Path,
    expected_bytes: int,
) -> None:

    require_file(path)

    actual_bytes = path.stat().st_size

    if actual_bytes != expected_bytes:
        die(
            f"{path}: expected "
            f"{expected_bytes} bytes, got "
            f"{actual_bytes}"
        )


def generate_raw_accumulators(
    *,
    prefix: str,
    c_in: int,
    c_out: int,
    h_in: int,
    w_in: int,
    kernel: int,
    padding: int,
    stride: int,
    input_path: Path,
    weight_path: Path,
    output_path: Path,
) -> None:

    h_out = conv_out_size(
        h_in,
        kernel,
        stride,
        padding,
        1,
    )

    w_out = conv_out_size(
        w_in,
        kernel,
        stride,
        padding,
        1,
    )

    if h_out <= 0 or w_out <= 0:
        die(
            f"{prefix}: invalid convolution "
            f"output {h_out}x{w_out}"
        )

    activations_flat = np.fromfile(
        input_path,
        dtype=np.uint8,
    )

    expected_input_count = (
        h_in
        * w_in
        * c_in
    )

    if activations_flat.size != expected_input_count:
        die(
            f"{input_path}: expected "
            f"{expected_input_count} bytes, got "
            f"{activations_flat.size}"
        )

    weights_flat = np.fromfile(
        weight_path,
        dtype=np.int8,
    )

    expected_weight_count = (
        c_out
        * c_in
        * kernel
        * kernel
    )

    if weights_flat.size != expected_weight_count:
        die(
            f"{weight_path}: expected "
            f"{expected_weight_count} bytes, got "
            f"{weights_flat.size}"
        )

    activations = activations_flat.reshape(
        h_in,
        w_in,
        c_in,
    ).astype(np.int64)

    weights = weights_flat.reshape(
        c_out,
        c_in,
        kernel,
        kernel,
    ).astype(np.int64)

    padded = np.pad(
        activations,
        (
            (padding, padding),
            (padding, padding),
            (0, 0),
        ),
        mode="constant",
        constant_values=0,
    )

    accumulators = np.zeros(
        (
            h_out,
            w_out,
            c_out,
        ),
        dtype=np.int64,
    )

    for kernel_row in range(kernel):
        row_stop = (
            kernel_row
            + stride * h_out
        )

        for kernel_column in range(kernel):
            column_stop = (
                kernel_column
                + stride * w_out
            )

            activation_patch = padded[
                kernel_row:row_stop:stride,
                kernel_column:column_stop:stride,
                :,
            ]

            if activation_patch.shape != (
                h_out,
                w_out,
                c_in,
            ):
                die(
                    f"{prefix}: internal activation "
                    f"patch shape error: "
                    f"{activation_patch.shape}"
                )

            weight_slice = weights[
                :,
                :,
                kernel_row,
                kernel_column,
            ]

            accumulators += np.tensordot(
                activation_patch,
                weight_slice,
                axes=([2], [1]),
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    accumulators.astype("<i4").tofile(
        output_path
    )


if not metadata_path.is_file():
    die(
        f"missing metadata JSON: "
        f"{metadata_path}"
    )


raw_acc_dir.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = json.loads(
    metadata_path.read_text()
)


if "layers" not in metadata:
    die(
        f"{metadata_path} does not "
        f"contain a layers array"
    )


layer_metadata = {
    entry["name"]: entry
    for entry in metadata["layers"]
}


model = AlexNet64Gray()


h = int(
    metadata.get(
        "mnist64_cfg",
        {},
    ).get(
        "image_size",
        64,
    )
)

w = h

matched: set[str] = set()


for child_name, module in model.features.named_children():

    layer_name = f"features.{child_name}"
    prefix = layer_name.replace(".", "_")

    if isinstance(module, nn.Conv2d):

        if layer_name not in layer_metadata:
            die(
                f"{layer_name} exists in the model "
                f"but is missing from {metadata_path}"
            )

        entry = layer_metadata[layer_name]

        if entry.get("type") != "Conv2d":
            die(
                f"{layer_name}: metadata type is "
                f"{entry.get('type')}, expected Conv2d"
            )

        c_out, c_in, kh_meta, kw_meta = [
            int(value)
            for value in entry["weight_shape"]
        ]

        kh, kw = pair(
            module.kernel_size,
            f"{layer_name} kernel_size",
        )

        sh, sw = pair(
            module.stride,
            f"{layer_name} stride",
        )

        ph, pw = pair(
            module.padding,
            f"{layer_name} padding",
        )

        dh, dw = pair(
            module.dilation,
            f"{layer_name} dilation",
        )

        if kh != kw:
            die(
                f"{layer_name}: non-square "
                f"kernels are unsupported"
            )

        if sh != sw:
            die(
                f"{layer_name}: unequal horizontal "
                f"and vertical stride is unsupported"
            )

        if ph != pw:
            die(
                f"{layer_name}: unequal horizontal "
                f"and vertical padding is unsupported"
            )

        if dh != 1 or dw != 1:
            die(
                f"{layer_name}: dilation must be 1"
            )

        if kh != kh_meta or kw != kw_meta:
            die(
                f"{layer_name}: model kernel "
                f"{kh}x{kw} does not match metadata "
                f"{kh_meta}x{kw_meta}"
            )

        h_out = conv_out_size(
            h,
            kh,
            sh,
            ph,
            dh,
        )

        w_out = conv_out_size(
            w,
            kw,
            sw,
            pw,
            dw,
        )

        input_path = (
            vectors_dir
            / f"{prefix}_in.bin"
        )

        weight_path = (
            vectors_dir
            / f"{prefix}_weights.bin"
        )

        bias_path = (
            vectors_dir
            / f"{prefix}_biases.bin"
        )

        rq_m_path = (
            vectors_dir
            / f"{prefix}_requant_m.bin"
        )

        rq_r_path = (
            vectors_dir
            / f"{prefix}_requant_r.bin"
        )

        output_path = (
            vectors_dir
            / f"{prefix}_out.bin"
        )

        raw_path = (
            raw_acc_dir
            / f"{prefix}_raw_acc.bin"
        )

        require_size(
            input_path,
            h * w * c_in,
        )

        require_size(
            weight_path,
            c_out * c_in * kh * kw,
        )

        require_size(
            bias_path,
            c_out * 4,
        )

        require_size(
            rq_m_path,
            c_out * 4,
        )

        require_size(
            rq_r_path,
            c_out,
        )

        require_size(
            output_path,
            h_out * w_out * c_out,
        )

        if not selected or prefix in selected:

            matched.add(prefix)

            if (
                force_golden
                or not raw_path.is_file()
            ):
                print(
                    f"Generating {raw_path}",
                    file=sys.stderr,
                )

                generate_raw_accumulators(
                    prefix=prefix,
                    c_in=c_in,
                    c_out=c_out,
                    h_in=h,
                    w_in=w,
                    kernel=kh,
                    padding=ph,
                    stride=sh,
                    input_path=input_path,
                    weight_path=weight_path,
                    output_path=raw_path,
                )

            require_size(
                raw_path,
                h_out * w_out * c_out * 4,
            )

            parallelism = choose_parallelism(
                c_out
            )

            print(
                prefix,
                c_in,
                c_out,
                h,
                w,
                kh,
                ph,
                sh,
                parallelism,
                h_out,
                w_out,
                raw_path,
            )

        h = h_out
        w = w_out

    elif isinstance(module, nn.MaxPool2d):

        kh, kw = pair(
            module.kernel_size,
            f"{layer_name} kernel_size",
        )

        sh, sw = pair(
            module.stride,
            f"{layer_name} stride",
        )

        ph, pw = pair(
            module.padding,
            f"{layer_name} padding",
        )

        dh, dw = pair(
            module.dilation,
            f"{layer_name} dilation",
        )

        h = conv_out_size(
            h,
            kh,
            sh,
            ph,
            dh,
        )

        w = conv_out_size(
            w,
            kw,
            sw,
            pw,
            dw,
        )

    elif isinstance(module, nn.ReLU):
        pass

    else:
        die(
            f"unsupported feature module "
            f"{layer_name}: "
            f"{module.__class__.__name__}"
        )


if selected:
    missing = selected - matched

    if missing:
        die(
            "selected layer prefixes "
            "were not found: "
            + ", ".join(sorted(missing))
        )
PY


if [ ! -s "$MANIFEST" ]; then
    if [ ${#selected[@]} -eq 0 ]; then
        echo \
            "run_layers.sh: no convolution layers were discovered" \
            >&2
    else
        echo \
            "run_layers.sh: no selected layers matched: ${selected[*]}" \
            >&2
    fi

    exit 1
fi


echo
echo "Analyzing conv_layer..."

ghdl -a \
    "${STD[@]}" \
    "${WORK[@]}" \
    hardware/rtl/layers/conv_layer.vhd


echo "Analyzing vector-driven testbench..."

ghdl -a \
    "${STD[@]}" \
    "${WORK[@]}" \
    verification/tb/layers/tb_conv_layer.vhd


echo "Elaborating tb_conv_layer..."

ghdl -e \
    "${STD[@]}" \
    "${WORK[@]}" \
    tb_conv_layer


pass=0
fail=0
ran=0


while read -r \
    prefix \
    c_in \
    c_out \
    h_in \
    w_in \
    kernel \
    padding \
    stride \
    c_par \
    h_out \
    w_out \
    raw_file
do
    ran=$((ran + 1))

    groups=$((c_out / c_par))

    output_transfers=$((h_out * w_out * groups))

    kernel_size=$((kernel * kernel))

    activation_cycles=$((h_in * w_in * c_in))

    mac_cycles=$((output_transfers * c_in * kernel_size))

    if [ "$c_in" -eq 1 ]; then
        if [ "$groups" -eq 1 ]; then
            weight_batches=1
        else
            weight_batches=$((h_out * groups))
        fi
    else
        weight_batches=$((h_out * groups * w_out * c_in))
    fi

    weight_cycles=$((weight_batches * c_par * kernel_size))

    parameter_cycles=$((c_out * 3))

    estimated_cycles=$((
        activation_cycles
        + mac_cycles
        + weight_cycles
        + parameter_cycles
        + output_transfers * 32
        + 20000
    ))

    if [ "$TIMEOUT_CYCLES" -gt 0 ]; then
        layer_timeout_cycles=$TIMEOUT_CYCLES
    else
        layer_timeout_cycles=$((estimated_cycles * 2))
    fi

    wave_file="$RESULTS_DIR/${prefix}.ghw"

    echo
    echo \
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo "Layer: $prefix"

    echo \
        "Input: Cin=$c_in HxW=${h_in}x${w_in}"

    echo \
        "Output: Cout=$c_out HxW=${h_out}x${w_out}"

    echo \
        "Kernel=$kernel Padding=$padding " \
        "Stride=$stride Cpar=$c_par Groups=$groups"

    echo \
        "Weight batches: $weight_batches"

    echo \
        "Watchdog: $layer_timeout_cycles cycles"

    echo \
        "GHDL max stack allocation: ${MAX_STACK_ALLOC_KB} KB"

    echo \
        "Raw golden: $raw_file"

    echo \
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    run_command=(
        ghdl -r
        "${STD[@]}"
        "${WORK[@]}"
        tb_conv_layer

        "--max-stack-alloc=$MAX_STACK_ALLOC_KB"

        "-gG_PREFIX=$prefix"

        "-gG_C_IN=$c_in"
        "-gG_C_OUT=$c_out"
        "-gG_H_IN=$h_in"
        "-gG_W_IN=$w_in"
        "-gG_KERNEL=$kernel"
        "-gG_PADDING=$padding"
        "-gG_STRIDE=$stride"
        "-gG_C_PAR=$c_par"

        "-gG_VECS=$VECTORS_G"
        "-gG_RAW_FILE=$raw_file"

        "-gG_STALL_PERIOD=$STALL_PERIOD"
        "-gG_PROGRESS_STEP=$PROGRESS_STEP"

        "-gG_TIMEOUT_CYCLES=$layer_timeout_cycles"

        --assert-level=error
    )

    if [ "$WAVE" -eq 1 ]; then
        run_command+=(
            "--wave=$wave_file"
        )
    fi

    if "${run_command[@]}"; then
        echo "PASS: $prefix"

        pass=$((pass + 1))
    else
        echo "FAIL: $prefix"

        fail=$((fail + 1))
    fi

done < "$MANIFEST"


echo
echo \
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Convolution real-vector results"
echo "  Ran:    $ran layers"
echo "  Passed: $pass"
echo "  Failed: $fail"

echo \
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


if [ "$ran" -eq 0 ]; then
    exit 1
fi


[ "$fail" -eq 0 ]