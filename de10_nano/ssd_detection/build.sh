#!/bin/bash
# 一键构建 nna 项目 (交叉编译: x86_64 → armv7hf)
set -e
export PATH=/opt/software/cmake-3.10.3-Linux-x86_64/bin:/opt/software/gcc-linaro-5.4.1-2017.05-x86_64_arm-linux-gnueabihf/bin:$PATH
ROOT=$(cd "$(dirname "$0")" && pwd)

echo "=== [1/3] intelfpga_sdk → libvnna.so ==="
arm-linux-gnueabihf-g++ \
  -std=c++11 -fopenmp -fPIC -shared \
  -o "$ROOT/lib/libvnna.so" "$ROOT/intelfpga.cc" \
  -I"$ROOT/include" -DARCH_ABI_ARM32 \
  -march=armv7-a -mfloat-abi=hard -mfpu=neon

echo "=== [2/3] Paddle-Lite ==="
cd "$ROOT/paddle_lite"
bash build_paddlelite.sh
cp "$ROOT/lib/libpaddle_light_api_shared.so" "$ROOT/deploy/paddlelite_lib/"

echo "=== [3/3] ssd_detection ==="
cd "$ROOT"
mkdir -p build
cd build
# 旧 CMakeCache 可能残留其他源码树路径（如 /opt/HAOYAO/nna），
# 导致增量 make 编译的不是本仓库源码。始终校验 cache 源码根：
CACHE_ROOT=""
if [ -f CMakeCache.txt ]; then
    CACHE_ROOT=$(grep -E '^CMAKE_HOME_DIRECTORY:INTERNAL=' CMakeCache.txt | cut -d= -f2)
fi
if [ ! -f Makefile ] || [ "$CACHE_ROOT" != "$ROOT" ]; then
    echo "--- cmake configure (cache root: ${CACHE_ROOT:-none} -> $ROOT) ---"
    rm -rf ./*
    cmake -DCMAKE_BUILD_TYPE=Release ..
fi
make -j4 ssd_detection
cp ssd_detection "$ROOT/deploy/"
cp "$ROOT/lib/libvnna.so" "$ROOT/deploy/paddlelite_lib/"
