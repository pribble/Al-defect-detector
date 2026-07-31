#!/bin/bash
# 编译 cmadrv 内核模块
# 依赖: pre-built kernel source tree at KSRC_DIR (配置在 Makefile 中)
set -e
export PATH=/opt/software/gcc-linaro-5.4.1-2017.05-x86_64_arm-linux-gnueabihf/bin:$PATH

cd "$(dirname "$0")"
make all
echo "=== 编译完成: $(ls -lh cmadrv.ko 2>/dev/null | awk '{print $5, $NF}') ==="
mv cmadrv.ko ../ssd_detection/deploy/
make clean
