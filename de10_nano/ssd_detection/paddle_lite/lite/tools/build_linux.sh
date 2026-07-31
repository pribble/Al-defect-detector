#!/bin/bash
set -e

#####################################################################################################
# Paddle-Lite 构建脚本 — Cyclone V (armv7hf + intel_fpga)
#####################################################################################################

readonly NUM_PROC=6
readonly workspace=$(dirname $(readlink -f "$0"))/../../

function build {
    local build_dir=$workspace/build

    if [ -d $build_dir ]; then
        rm -rf $build_dir
    fi
    mkdir -p $build_dir
    cd $build_dir

    cmake $workspace \
        -DCMAKE_BUILD_TYPE=Release \
        -DINTEL_FPGA_SDK_ROOT=$(cd $workspace/../ && pwd)

    make publish_inference -j$NUM_PROC
    cd - > /dev/null
}

build
