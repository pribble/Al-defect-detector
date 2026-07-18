# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPGA-accelerated SSD MobileNet V1 object detection system running on a **Cyclone V SoC FPGA** (ARM Cortex-A9 HPS + FPGA fabric). Uses Paddle-Lite inference engine with FPGA-accelerated convolution via a custom Intel FPGA CNN accelerator.

Serves as an HTTP inference server at port 8080, called by a Raspberry Pi defect detection system (HaoYao, external repo) that POSTs images and receives JSON detection results.

## Architecture

```
Host (x86_64, cross-compile)
  └─ upload.sh ──scp──> Target (Cyclone V, armv7hf, /opt/paddle_frame)
                         └─ run.sh ──> systemd detect.service ──> HTTP :8080

Raspberry Pi (HaoYao, 172.16.68.111)
  └─ POST /predict ──> FPGA board (172.16.68.110:8080)
                        └─ Response JSON: {"len":N, "result":[{"class_name","loc","score","prediction_time"}]}
```

### Component Dependency Chain

```
cmadrv (kernel module)
  └─ CMA phys-contiguous memory allocator + DMA memcpy via /dev/cmadrv0
  └─ loaded by: insmod cmadrv.ko
  └─ kernel source tree at KSRC_DIR (configured in cmadrv/build/Makefile)

intelfpga_sdk (libvnna.so)
  └─ user-space library controlling FPGA CNN accelerator
  └─ opens /dev/cmadrv0 + /dev/mem for MMIO
  └─ weight/data reorganization, FPGA subgraph execution

Paddle-Lite (libpaddle_full_api_shared.so)
  └─ Baidu lightweight inference engine, cross-compiled with intel_fpga backend

ssd_detection (main executable)
  └─ links: libpaddle_full_api_shared.so + libvnna.so + OpenCV
  └─ loads ssd_mobilenet_v1_opt.nb via Paddle-Lite MobileConfig
  └─ two modes: image file processing OR HTTP REST API (port 8080)
```

## Directory Structure

| Directory | Purpose | Build Output |
|-----------|---------|-------------|
| `cmadrv/` | Linux kernel module (CMA allocator + DMA) | `cmadrv.ko` |
| `intelfpga_sdk/` | FPGA accelerator user-space SDK | `libvnna.so` |
| `Paddle-Lite/` | Paddle-Lite inference engine (fork) | `libpaddle_full_api_shared.so` |
| `ssd_detection/` | Main detection application | `ssd_detection` binary |
| `paddle_frame/` | Deployment staging directory | (artifacts copied here, then scp'd) |
| `tool/` | Build helper scripts | — |
| `diff/` | Quick re-deploy directory | (binary-only copy for rapid updates) |

## Build Commands

All cross-compilation from x86_64 host for armv7hf target.

**Prerequisites** (set by `tool/exportPATH.sh`):
- CMake 3.10.3 at `/opt/software/cmake-3.10.3-Linux-x86_64/bin/cmake`
- GCC Linaro 5.4.1 at `/opt/software/gcc-linaro-5.4.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc`

### Full build (all components):
```bash
source ./tool/exportPATH.sh
bash all_build.sh
```
This runs: intelfpga_sdk → Paddle-Lite → ssd_detection in sequence.

### Build individual components:

**FPGA SDK:**
```bash
cd intelfpga_sdk/lib && bash build.sh
# Produces: intelfpga_sdk/lib/libvnna.so
# Also copies to: ssd_detection/Paddlelite/lib/
```

**Paddle-Lite:**
```bash
cd Paddle-Lite && bash publish_build.sh armv7hf
# Produces: Paddle-Lite/build.lite.linux.armv7hf.gcc/.../libpaddle_full_api_shared.so
```

**SSD Detection app:**
```bash
source ./tool/exportPATH.sh
cd ssd_detection/ssd_detection_src && bash build.sh
# Produces: ssd_detection/ssd_detection_src/build/ssd_detection
```
The build script accepts optional arguments: `bash build.sh <armv7hf|armv8> [camera] [aiep]`
- Default: armv7hf (Cyclone V)
- armv8 target is also supported (for aarch64 platforms)
- Build variants exist at `build.sh.debug` and `build.sh.release`

**Kernel module (cmadrv):**
```bash
cd cmadrv/build && make
# Requires kernel source tree at KSRC_DIR (configured in cmadrv/build/Makefile)
```

### Switch between debug/release:
```bash
bash tool/switch2debug.sh   # copies .debug build configs over release
bash tool/switch2release.sh # copies .release build configs over debug
```

### Deploy to target board:
```bash
bash upload.sh
# Syncs paddle_frame/ to root@172.16.68.110:/opt/paddle_frame
# Uses tar+ssh, skips data/ and images/ directories.
# Maintains .last_time for incremental sync.
```

### Quick re-deploy (binary only):
```bash
cp ssd_detection/ssd_detection_src/build/ssd_detection paddle_frame/
bash upload.sh
# Copies just the binary to paddle_frame/, then uploads everything
```

## On-Device Runtime

**Start service:**
```bash
systemctl start detect.service
```

**Check status:**
```bash
systemctl status detect.service
```

**Service file:** `/etc/systemd/system/detect.service`
- `WorkingDirectory=/opt/paddle_frame` is required (relative paths in run.sh)
- ExecStart: runs `run.sh`

**Run `run.sh`** (on device):
```bash
insmod cmadrv.ko              # load kernel module
export LD_LIBRARY_PATH=./paddlelite_lib:$LD_LIBRARY_PATH+
./ssd_detection config.txt     # start HTTP server on port 8080
```

**Quick test:**
```bash
cd paddle_frame
./ssd_detection config.txt data   # process images in data/ directory
```

## HTTP API

### `/predict` endpoint

Form POST:
```bash
curl -F image_file=@test.jpg http://172.16.68.110:8080/predict
```

**Response JSON format** (aligned with HaoYao GrabImage Consumer expectations):
```json
{
  "len": 1,
  "result": [
    {
      "class_name": "ca_shang",
      "loc": [100, 200, 300, 400],
      "score": 0.95,
      "prediction_time": 45.23
    }
  ]
}
```

- `class_name` — from `label_list` file: `ca_shang`, `zang_wu`, `zhe_zhou`, `zhen_kong`
- `loc` — bounding box `[xmin, ymin, xmax, ymax]`
- `score` — confidence (0.0–1.0, threshold >0.45 hardcoded)
- `prediction_time` — inference time in milliseconds
- Empty detection: `{"len":0,"result":[]}`

## Config (`config.txt`)

```
model_file ssd_mobilenet_v1_opt.nb    # Paddle-Lite Naive Buffer model
label_path ./label_list                # one class name per line
num_threads 2                          # inference threads
mean 127.5,127.5,127.5                 # channel-wise mean
std 127.502231,127.502231,127.502231   # channel-wise std (1/255 ≈ 0.00392)
```

## Label Classes

`label_list` defines class index → name mapping (index 0 = first line):

```
ca_shang    → scratch
zang_wu     → dirt/stain
zhe_zhou    → wrinkle
zhen_kong   → pinhole
```

## Key Technical Details

- **Model**: SSD MobileNet V1, quantized int8, Paddle-Lite Naive Buffer format (`ssd_mobilenet_v1_opt.nb`)
- **Input**: 300×300 RGB images, preprocessed with NEON-optimized mean/scale (NCHW layout)
- **Output**: flat buffer of 6-element tuples [class_id, confidence, xmin, ymin, xmax, ymax]
- **Confidence threshold**: 0.45 (hardcoded in `ssd_detection.cc`)
- **FPGA conv tile**: INPUT_CHANNEL_TILE=8, OUTPUT_CHANNEL_TILE=8, OUTPUT_ROW_TILE=5 (for ARM32)
- **Data reorganization**: channel-interleaved layout for FPGA; configurable between ARM (NEON/polling) or FPGA reorg
- **Kernel module interface**: IOCTL-based (CMA_CMD_MGET/CMA_CMD_FREE/CMA_CMD_MCPY) + mmap
- **Dependencies**: OpenCV 3.1.0 (armv7hf, cross-compiled at `ssd_detection/ocv3.1.0/`)
