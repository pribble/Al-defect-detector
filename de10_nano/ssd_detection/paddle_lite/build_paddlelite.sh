#!/bin/bash
# Paddle-Lite 增量编译 (避免每次 clean build)
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
BUILD=$ROOT/build
OUT_LIB="$ROOT/../lib/libpaddle_light_api_shared.so"

if [ ! -f "$BUILD/CMakeCache.txt" ] || [ ! -f "$BUILD/api/paddle_use_kernels.h" ]; then
    echo "=== [2/3a] cmake configure ==="
    mkdir -p "$BUILD"
    cd "$BUILD"
    cmake "$ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DINTEL_FPGA_SDK_ROOT=$(cd "$ROOT"/../ && pwd)

    # 用最小 kernel 集替换自动生成列表 (SSD MobileNet FPGA 只需这些)
    cat > "$BUILD/api/paddle_use_kernels.h" << 'KERNELS'
#pragma once
#include "paddle_lite_factory_helper.h"
USE_LITE_KERNEL(subgraph, kIntelFPGA, kInt8, kNCHW, def);
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);
USE_LITE_KERNEL(prior_box, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(reshape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(concat, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(relu, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kARM, kFloat, kNCHW, int32);
USE_LITE_KERNEL(slice, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(softmax, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(calib, kARM, kInt8, kNCHW, fp32_to_int8);
USE_LITE_KERNEL(calib, kARM, kInt8, kNCHW, int8_to_fp32);
USE_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, fp32_out);
USE_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kARM, kInt8, kNCHW, fp32_out);
USE_LITE_KERNEL(transpose, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(transpose2, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(layout, kARM, kFloat, kNCHW, nchw2nhwc);
USE_LITE_KERNEL(layout, kARM, kFloat, kNCHW, nhwc2nchw);
USE_LITE_KERNEL(layout, kARM, kInt8, kNCHW, int8_nchw2nhwc);
USE_LITE_KERNEL(layout, kARM, kInt8, kNCHW, int8_nhwc2nchw);
KERNELS
    echo "--- kernel list trimmed to 25 entries (was 120) ---"

    echo "=== [2/3b] make ==="
    make -j$(nproc) publish_inference
    cp "$BUILD/inference_lite/lib/libpaddle_light_api_shared.so" "$ROOT/../lib/"
else
    echo "=== [2/3a] make only (incremental) ==="
    cd "$BUILD"
    make -j$(nproc) publish_inference
    cp "$BUILD/inference_lite/lib/libpaddle_light_api_shared.so" "$ROOT/../lib/"
fi

echo "=== [2/3a] Done: $OUT_LIB ==="
ls -lh "$OUT_LIB"
