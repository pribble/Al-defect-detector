# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HaoYao** — Automated aluminum sheet defect detection and sorting system for a Raspberry Pi 5 deployment. Captures images via Hikvision industrial camera (MVS SDK), detects defects via FPGA inference server (SSD MobileNet V1 + Paddle Lite + FPGA acceleration), and controls a 5-DOF Yahboom robotic arm with pneumatic suction gripper for OK/NG sorting. Runs on Debian 12 Bookworm (64-bit) on a Raspberry Pi 5.

### Data Flow

```
Hikvision Camera ──frame_queue──> GrabImage (:7777) ──POST /predict──> FPGA Board (172.16.68.110:8080)
                                       │                                    └── SSD MobileNet V1
                                       ├── SQLite (defect.db)                    + Paddle Lite + FPGA
                                       ├── POST /grab ──> ArmControl (:8899) ──> /dev/XIPAN (5-DOF arm)
                                       └── MJPEG stream ──> nginx (:80/8080) ──> Browser
```

### Hardware

| Device | Interface | Purpose |
|--------|-----------|---------|
| Raspberry Pi 5 (Debian 12) | — | Main controller, runs both services |
| Hikvision GigE/USB camera | Ethernet/USB | Image capture via MVS SDK |
| FPGA compute board (172.16.68.110) | Ethernet :8080 | Paddle Lite inference + SSD MobileNet V1 |
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

### GrabImage/ (detection server, Flask :7777)

```
GrabImage/
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

### ArmControl/ (arm controller, Flask :8899)

```
ArmControl/
├── api.py        # Flask app, grab_task sequence, GrabTaskConsumer queue
├── api.service   # systemd unit
└── run.sh        # Starts api.py
```

### frontend/ (SPA)

```
frontend/
├── index.html    # Self-contained app (Vue 2 + Element UI + ECharts), no build step
├── config.json   # API endpoints: ArmControl :8899, GrabImage :7777
└── vendor/       # Local libraries (Vue 2, Element UI, ECharts, Axios)
```

### fpga/ (inference server — see fpga/CLAUDE.md for full details)

```
fpga/
├── ssd_detection/        # C++ source (ssd_detection.cc, CMakeLists.txt)
├── paddle_frame/         # Deployment staging directory
├── upload.sh             # Syncs to 172.16.68.110:/opt/paddle_frame
├── detect_build.sh       # Build pipeline (cmake → copy to paddle_frame/)
└── tool/                 # Build helpers
```

### tools/logger.py

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
| GrabImage Consumer | FPGA | HTTP POST | `172.16.68.110:8080/predict` | `multipart/form-data; image_file=<JPEG bytes>` |
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
# From dev machine: push code to Pi (dry-run first, then prompts for confirmation)
bash deploy.sh [user@pi-ip] [/opt/HaoYao]

# On Pi: restart both services
bash restart.sh

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

Excludes from rsync: `.git/`, `__pycache__/`, `*.pyc`, `defect.db`, `files/`, `original_files/`, `detect_files/`, `fpga/`, `deploy.sh`, `CLAUDE.md`, `README.md`. The `fpga/` directory is not deployed to the Pi — its code runs on the separate FPGA board.

### GrabImage run.sh

Sets MVS SDK library paths and starts `api.py`:
```bash
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export CRYPTOGRAPHY_ALLOW_OPENSSL_102=1
```

## Common Commands

### Deploy & Restart

```bash
# Push Pi code (dry-run first, then confirms)
bash deploy.sh [user@pi-ip] [/opt/HaoYao]

# Restart both services on Pi
bash restart.sh

# Service control on Pi
systemctl start|stop|restart|status detect-api.service   # GrabImage :7777
systemctl start|stop|restart|status api.service           # ArmControl :8899
systemctl status nginx.service                            # Frontend :80/:8080
```

### FPGA Build & Deploy

```bash
# Cross-compile SSD detection binary (x86_64 host → armv7hf target)
cd fpga && bash detect_build.sh

# Incrementally sync to FPGA board (172.16.68.110)
cd fpga && bash upload.sh

# On FPGA board: start/stop inference service
systemctl start|stop|status detect.service
```

### Logs

```bash
journalctl -u detect-api.service -n 50 --no-pager -l   # Pi detect service
journalctl -u api.service -n 50 --no-pager -l           # Pi arm service
tail -f /var/logs/detect_server.log                     # GrabImage debug log
tail -f /var/logs/api_server.log                        # ArmControl debug log
```

### Tests & Lint

No test framework or linter is configured in this repository.

### Image Inference Pipeline

Current optimized flow (300×300 grayscale raw pixels, no JPEG):

```
Pi: 768×512 gray → cv2.resize(300×300) → .tobytes() → POST
FPGA: Mat(300×300, CV_8UC1) → preprocessImgGray (1-ch NEON → 3-ch NCHW tensor) → model
```

Detection coordinates (in 300×300 space from FPGA) are scaled back to 768×512 for result image annotation via `_draw_defect_box`.

### FPGA Inference Server (cross-reference)

The `fpga/` directory is a separate C++ cross-compiled project (Cyclone V SoC FPGA, Paddle-Lite, SSD MobileNet V1). See `fpga/CLAUDE.md` for full build/deploy instructions, HTTP API reference, and FPGA architecture details. Key points:

- Build: cross-compile on x86_64 → deploy to `172.16.68.110:/opt/paddle_frame`
- HTTP `/predict`: POST multipart with field `image_file` (raw 300×300 grayscale bytes)
- Response: `{"len":N, "action":"OK"/"NG", "result":[{"class_name","loc","score","prediction_time"}]}`
- Confidence threshold: 0.45 (hardcoded)
- Label classes: ca_shang (scratch), zang_wu (dirt), zhe_zhou (wrinkle), zhen_kong (pinhole)

## Frontend

Self-contained single-page application at `frontend/index.html`. No build step — libraries (Vue 2, Element UI, ECharts, Axios) are served locally from `frontend/vendor/`.

Two URLs serving the same app:
- `http://172.16.68.111` (nginx :80)
- `http://172.16.68.111:8080` (nginx :8080)

Configured via `frontend/config.json`:
```json
{"ip": "http://172.16.68.111:8899", "url": "http://172.16.68.111:7777"}
```

Features: live MJPEG stream, statistics (pie + 7-day trend lines), manual arm control, conveyor calibration, history browser with defect images.

Live MJPEG video stream at `http://172.16.68.111:7777/img`. Flask-rendered Bootstrap page at `http://172.16.68.111:7777/`.

## MVS SDK Python Bindings

Located at `MvImport/` — hand-authored ctypes wrappers for Hikvision's MVS SDK. Files:
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
