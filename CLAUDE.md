# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HaoYao** — Automated aluminum sheet defect detection and sorting system. Captures images via Hikvision industrial camera (MVS SDK), detects defects via FPGA inference server (SSD MobileNet V1 + Paddle Lite + custom CNN accelerator on a DE10-Nano Cyclone V SoC), and controls a 5-DOF Yahboom robotic arm with pneumatic suction gripper for OK/NG sorting. Runs on Debian 12 Bookworm (64-bit) on a Raspberry Pi 5.

Repository is split by hardware platform:

- `raspberryPi5/` — all code that runs on the Raspberry Pi 5 (GrabImage detection server, ArmControl, frontend, MVS bindings, deploy/restart scripts)
- `de10_nano/` — all code for the DE10-Nano FPGA board: inference server (`ssd_detection/`, detailed docs in its own `CLAUDE.md`), Quartus SoC project (`C5TB/`), CMA/DMA kernel module (`kernel/`)

### Data Flow

```
Hikvision Camera ──frame_queue──> raspberryPi5/GrabImage (:7777) ──POST /predict──> DE10-Nano (172.16.68.110:8080)
                                        │                                    └── de10_nano/ssd_detection
                                        ├── SQLite (defect.db)                    + C5TB FPGA accelerator
                                        ├── POST /grab ──> raspberryPi5/ArmControl (:8899) ──> /dev/XIPAN (5-DOF arm)
                                        └── MJPEG stream ──> nginx (:80/8080) ──> Browser
```

### Hardware

| Device | Interface | Purpose |
|--------|-----------|---------|
| Raspberry Pi 5 (Debian 12) | — | Main controller, runs both services |
| Hikvision GigE/USB camera | Ethernet/USB | Image capture via MVS SDK |
| DE10-Nano (Cyclone V SoC) | Ethernet :8080 | Paddle Lite inference + CNN FPGA accelerator (de10_nano/) |
| 5-DOF Yahboom arm | `/dev/XIPAN` → ttyUSB0 (9600 baud) | OK/NG pick-and-place sorting |
| Buzzer alarm | `/dev/BAOJING` | Defect alert |

### Network

| Interface | Address | Purpose |
|-----------|---------|---------|
| `eth0` | `172.16.68.111/24` | Industrial network (camera, FPGA, arm) |
| `wlan0` | `192.168.1.11/24` | Management/home network (gateway 192.168.1.1) |

| Service | Address | Port |
|---------|---------|------|
| GrabImage (Flask) | 172.16.68.111 | 7777 |
| ArmControl (Flask) | 172.16.68.111 | 8899 |
| nginx (frontend) | 172.16.68.111 | 80, 8080 |
| FPGA inference | 172.16.68.110 | 8080 |

## Component Architecture

### raspberryPi5/GrabImage/ (detection server, Flask :7777)

```
raspberryPi5/GrabImage/
├── api.py        # Flask app, Consumer thread (SSIM + FPGA inference + sorting trigger)
├── routes.py     # HTTP API routes (config, stats, history, video stream)
├── camera.py     # Hikvision MVS SDK bindings, Producer thread
├── database.py   # SQLite wrapper (defect.db)
├── shared.py     # Singleton state module (shared between api.py and routes.py)
├── config.ini    # Timing/speed + defect_name Chinese mapping
├── run.sh        # Sets MVS lib paths, starts api.py
├── detect-api.service  # systemd unit
├── yuanshi.jpg   # Reference background image for SSIM comparison
├── Font/platech.ttf    # Chinese font for defect label overlay
└── templates/index.html # Simple MJPEG stream viewer
```

### raspberryPi5/ArmControl/ (arm controller, Flask :8899)

```
raspberryPi5/ArmControl/
├── api.py        # Flask app, grab_task sequence, GrabTaskConsumer queue
├── api.service   # systemd unit
└── run.sh        # Starts api.py
```

### raspberryPi5/frontend/ (SPA)

```
raspberryPi5/frontend/
├── index.html    # Self-contained app (Vue 2 + Element UI + ECharts), no build step
├── config.json   # API endpoints: ArmControl :8899, GrabImage :7777
└── vendor/       # Local libraries (Vue 2, Element UI, ECharts, Axios)
```

### de10_nano/ (DE10-Nano FPGA board)

```
de10_nano/
├── ssd_detection/   # C++ inference server (cross-compiled, armv7hf): ssd_detection.cc,
│                    #   intelfpga.cc (FPGA SDK), radically-simplified Paddle-Lite, deploy/ staging
├── C5TB/            # Quartus SoC project (C5TB_top top-level)
├── kernel/          # cmadrv.ko — CMA allocator + DMA engine kernel module source
└── .gitignore       # Ignore rules for build outputs (db/, output_files/, etc.)
```

- `de10_nano/C5TB/` — Quartus project for the FPGA fabric side: `C5TB_top.v` (SoC top level: HPS DDR3, Ethernet, SDMMC, …), `soc_system.qsys/.sopcinfo` (QSys system), `ip/cnn_top*` (custom CNN accelerator IP), `tools/` (generated `soc_system.rbf` bitstream + device tree `.dtb/.dts`, plus `sof_to_rbf`/`gen_dtb` scripts). Build via Quartus; the `.rbf` configures the FPGA at boot and the `.dtb` goes to the HPS boot partition. Only the source files are committed — `db/`, `output_files/`, `ip/` generated outputs etc. are gitignored.
- `de10_nano/kernel/` — `cmadrv.c` Linux kernel module (CMA physical-memory allocator + DMA memcpy engine, char device `/dev/cmadrv0`). `kernel/build.sh` cross-compiles it and copies `cmadrv.ko` into `ssd_detection/deploy/`. Requires `KSRC_DIR` (default `/opt/software/linux-4.9.78`) pointing at a cross-compiled kernel tree.

### raspberryPi5/tools/logger.py

Shared logging utility — writes to `/var/logs/{name}.log` with auto-truncation at 10 MB. Both services import it:

```python
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from logger import setup_log
logger = setup_log('detect', 'detect_server.log')  # or ('arm', 'api_server.log')
```

## Key Architecture Decisions

### Threading Model

Three concurrent threads across the two services:

1. **Producer** (`camera.py`) — daemon thread that reads Hikvision camera frames via MVS SDK, pushes to `frame_queue` (unbounded Queue), and updates `stream_image_ref[0]` for video feed. Auto-reconnects on `CAMERA_NEED_RESTART` error code (2147483655). Frame size: 3072×2048 downsampled to 768×512.

2. **Consumer** (`api.py` Consumer thread) — dequeues frames, processes 1 in 10 (`FRAME_SKIP_COUNT`). Performs SSIM trigger detection → FPGA inference → triggers arm/alarm via ThreadPoolExecutor.

3. **GrabTaskConsumer** (`ArmControl/api.py`) — serializes pick-and-place tasks from a Queue to prevent concurrent arm movement. This is a single-threaded consumer pattern, not a thread pool.

Additional: `ThreadPoolExecutor` in `shared.py` fires async HTTP calls (`trigger_grab`) and alarm triggers. No await/futures pattern — fire-and-forget via `submit()`.

### shared.py Singleton Pattern

`api.py` and `routes.py` must share mutable runtime state (config, logger, queue, image references). A plain import would create two separate module instances when `api.py` runs as `__main__` but `routes.py` is imported. Solution: `shared.py` holds all shared objects:

```python
from shared import config, logger, frame_queue, stream_image_ref, thread_pool, GRAB_SPEED, GRAB_DELAY
```

`stream_image_ref` is `list[np.ndarray]` — the list wrapper avoids Python scoping issues; both modules hold a reference to the same list and access `[0]` for the latest frame.

### SSIM Trigger Logic (Consumer._process_sampling_frame)

1. Every 10 frames, resize to 64×48, Gaussian blur (kernel=21), binary threshold (100), dilate (4 iterations)
2. Compute SSIM vs `yuanshi.jpg` (reference background image)
3. Track SSIM sliding window (size 9) — if stable (std < 0.01, mean < 0.8, all non-zero), auto-update reference
4. Trigger inference when: `diff_curr > 0 > diff_prev2 AND diff_prev < 0 AND white_ratio > 0.1 AND current_ssim < 0.9`
5. FPGA returns classification → `zheng_chang` triggers OK grab, defects trigger NG grab + alarm

### Database API (database.py)

Thread-safe SQLite wrapper for `defect.db` with a mutex lock. Prefer the unified `query()` method:

```python
database.query(columns, source, condition)   # Build + execute SELECT, return all rows
database.insert_data(uid, path, name, pred_time, score)  # INSERT with UNIQUE constraint
database.select_day_data(offset_start, offset_end)  # Count by day offset
```

The `query()` method replaces the old `select_instructions()` + `select_data()` pair — use `query()` for all new code. The old functions remain for backward compatibility.

Table: `defect_list(id INTEGER PK, uuid CHAR, path CHAR(64), name CHAR, prediction_time CHAR, score CHAR, CreatedTime TIMESTAMP DEFAULT datetime('now','localtime'), UNIQUE(uuid, path, name))`

### ArmControl Serial Task Execution

The `/grab` endpoint puts requests on a `Queue`, and `GrabTaskConsumer` (daemon thread) pops and executes them one at a time via `grab_task()`. This prevents concurrent arm movement. The sequence is: HOME → PICKUP → suction_on → PICKUP_LIFT → OK_ABOVE/NG_ABOVE → OK_PLACE/NG_PLACE → suction_off → lift → HOME.

**Crucial note:** All arm angles in `api.py` (HOME, PICKUP, PICKUP_LIFT, OK_ABOVE, OK_PLACE, NG_ABOVE, NG_PLACE) are **placeholder values** that must be field-calibrated for the specific physical arm.

### Inter-Service Communication

| From | To | Method | Endpoint | Payload |
|------|----|--------|----------|---------|
| GrabImage Consumer | FPGA | HTTP POST | `172.16.68.110:8080/predict` | `multipart/form-data; image_file=<90000 raw grayscale bytes>` |
| GrabImage Consumer | ArmControl | HTTP POST | `172.16.68.111:8899/grab` | `{"flags": "OK"/"NG", "speed": int, "time": int}` |
| Frontend SPA | ArmControl | HTTP GET/POST | `172.16.68.111:8899/*` | JSON |
| Frontend SPA | GrabImage | HTTP GET | `172.16.68.111:7777/*` | JSON / base64 images |
| Browser | GrabImage | HTTP GET | `172.16.68.111:7777/img` | MJPEG stream |

FPGA response format:
```json
{"len": N, "result": [{"class_name": "ca_shang", "loc": [x1,y1,x2,y2], "score": 0.95, "prediction_time": 45.2}], "action": "NG"}
```
Empty detection: `{"len": 0, "result": []}`

## Configuration (config.ini)

```ini
[Configuration]
time = 4              # Conveyor delay (seconds) — aluminum sheet transit from camera to pickup point
speed = 1300         # Servo movement speed (milliseconds)
grab_position = 29    # Unused in current code
release_position = 17 # Unused in current code

[defect_name]
ca_shang = 划痕       # Scratch
zhen_kong = 针孔       # Pinhole
zang_wu = 脏污         # Dirt/stain
zhe_zhou = 褶皱        # Wrinkle
zheng_chang = 正常     # Normal (no defect)
```

## Deployment

```bash
# From dev machine: push Pi code to Pi (dry-run first, then prompts for confirmation)
cd raspberryPi5 && bash deploy.sh [user@pi-ip] [/opt/HaoYao]

# On Pi: restart both services
cd raspberryPi5 && bash restart.sh

# Individual service control
systemctl start|stop|restart|status detect-api.service   # GrabImage :7777
systemctl start|stop|restart|status api.service           # ArmControl :8899
systemctl status nginx.service                            # Frontend :80/:8080

# Logs
journalctl -u detect-api.service -n 50 --no-pager -l
journalctl -u api.service -n 50 --no-pager -l
tail -f /var/logs/detect_server.log    # GrabImage debug log
tail -f /var/logs/api_server.log       # ArmControl debug log
```

### deploy.sh Notes

Excludes from rsync: `.git/`, `__pycache__/`, `*.pyc`, `defect.db`, `files/`, `original_files/`, `detect_files/`, `fpga/`, `deploy.sh`, `CLAUDE.md`, `README.md`, `GrabImage/yuanshi.jpg`. The `de10_nano/` directory is not deployed to the Pi — its code runs on the separate FPGA board.

### GrabImage run.sh

Sets MVS SDK library paths and starts `api.py`:
```bash
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export CRYPTOGRAPHY_ALLOW_OPENSSL_102=1
```

## FPGA Build & Deploy (de10_nano/)

```bash
# Cross-compile SSD inference server (x86_64 host → armv7hf target)
cd de10_nano/ssd_detection && bash build.sh   # 3 stages: intelfpga_sdk → Paddle-Lite → ssd_detection

# Incrementally sync to FPGA board (172.16.68.110)
cd de10_nano/ssd_detection && bash upload.sh

# Rebuild kernel module only (auto-copies cmadrv.ko into ssd_detection/deploy/)
cd de10_nano/kernel && bash build.sh

# On FPGA board: start/stop inference service
cd /opt/paddle_frame
insmod cmadrv.ko          # CMA allocator + DMA kernel module
export LD_LIBRARY_PATH=./paddlelite_lib:$LD_LIBRARY_PATH
./ssd_detection config.txt   # HTTP server mode (port 8080)
```

**FPGA-side key notes:**
- **Model has 355 instructions in the main block**: calib → subgraph → 353 post-processing ops (prior_box, slice, elementwise_*, concat, softmax, multiclass_nms3). `PaddlePredictor::Run()` loops ALL instructions — do not hardcode to only [calib, subgraph]. The subgraph internally has 47 conv/depthwise_conv ops (FPGA-accelerated).
- **Call chain**: `KernelBase::Launch()` → PrepareForRun (one-time) → ReInitWhenNeeded → WorkSpace::AllocReset → Run(). Prefer it over calling PrepareForRun+Run directly. Subgraph kernel: BuildInstructions + BuildDeviceProgram (bridge conv ops to FPGA); falls back to ARM kernels if BuildDeviceProgram fails — keep the ARM conv2d/depthwise_conv2d kernels registered.
- **Model I/O names**: inputs `im_shape_0` (feed 0), `image_0` (feed 1), `scale_0` (feed 2); output `save_infer_model/scale_0.tmp_1` (fetch 0), produced by multiclass_nms3, not by a subgraph tensor.
- **Confidence threshold**: per-class, space-separated in `deploy/label_list` (~0.45).
- **CMA allocation failure on target**: reload the kernel module — `rmmod cmadrv && insmod cmadrv.ko`.
- **Debug tip**: add `fprintf(stderr, ...)` in `PaddlePredictor::Run()` to see instruction count/types, or in `SubgraphCompute::BuildDeviceProgram()` to see bridged ops. Output tensor dims are empty if post-processing instructions are skipped.

## Image Inference Pipeline

Current optimized flow (300×300 grayscale raw pixels, no JPEG):

```
Pi: 768×512 gray → cv2.resize(300×300) → .tobytes() → POST
FPGA: Mat(300×300, CV_8UC1) → preprocessImgGray (1-ch NEON → 3-ch NCHW tensor) → model
```

Detection coordinates (in 300×300 space from FPGA) are scaled back to 768×512 for result image annotation via `_draw_defect_box`.

## Frontend

Self-contained single-page application at `raspberryPi5/frontend/index.html`. No build step — libraries (Vue 2, Element UI, ECharts, Axios) are served locally from `frontend/vendor/`.

Two URLs serving the same app:
- `http://172.16.68.111` (nginx :80)
- `http://172.16.68.111:8080` (nginx :8080)

Configured via `raspberryPi5/frontend/config.json`:
```json
{"ip": "http://172.16.68.111:8899", "url": "http://172.16.68.111:7777"}
```

Features: live MJPEG stream, statistics (pie + 7-day trend lines), manual arm control, conveyor calibration, history browser with defect images.

Live MJPEG video stream at `http://172.16.68.111:7777/img`. Flask-rendered Bootstrap page at `http://172.16.68.111:7777/`.

## MVS SDK Python Bindings

Located at `raspberryPi5/MvImport/` — hand-authored ctypes wrappers for Hikvision's MVS SDK. Files:
- `MvCameraControl_class.py` — Main camera control class
- `CameraParams_header.py` / `CameraParams_const.py` — Parameter constants
- `MvErrorDefine_const.py` — Error code definitions
- `PixelType_header.py` / `PixelType_const.py` — Pixel format definitions

Key error code: `CAMERA_NEED_RESTART = 2147483655` (0x7FFF0007) — camera needs re-initialization.

## Known Bugs (Not Fixed)

Keep these in mind — they are documented but unresolved:

1. **`_generate_frames()` yields `None` on encode failure** — `_encode_frame()` returns `None` on exception; `_generate_frames()` does not check for this before yielding, causing a `TypeError` in the MJPEG stream response.
2. **`_log_device_info()` GigE model name includes null bytes** — The `if p != 0` filter exists for GigE now (was fixed), but should be verified.
3. **`cleanup()` races with `database.create_database()`** — The cleanup thread starts in `if __name__ == "__main__":` before `Consumer.run()` calls `database.create_database()`. On first run, cleanup tries to query a table that doesn't exist yet.
4. **File deletion race in `get_detect_pic` / `get_history`** — A file can be deleted by `cleanup()` between the database query (which returns the path) and the file read (which encodes it to base64). Results in a file-not-found error served to the frontend.
5. **`/dev/BAOJING` hardware type unconfirmed** — The code opens it as a serial port (`serial.Serial()` with 9600 baud), but CLAUDE.md documentation suggests it may be an input event device (`/dev/input/event0`). If it's an event device, serial writes will have no effect.
