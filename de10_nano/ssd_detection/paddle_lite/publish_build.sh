#!/bin/bash
set -e
cd "$(dirname "$0")"
bash lite/tools/build_linux.sh

cp build/inference_lite/lib/libpaddle_light_api_shared.so ../lib/
