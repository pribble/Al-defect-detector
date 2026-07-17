#!/bin/bash

export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:/opt/MVS/lib/aarch64:/opt/MVS/lib/aarch64:
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export CRYPTOGRAPHY_ALLOW_OPENSSL_102=1
cd /opt/HaoYao/GrabImage/
python3 api.py
