# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HaoYao** — Automated aluminum sheet defect detection and sorting system. Runs on Raspberry Pi 5 (Debian 12, Python 3.11.2). Captures images via Hikvision industrial camera (MVS SDK), performs AI inference on an FPGA board (Paddle Lite + FPGA), and controls a 5-DOF Yahboom robotic arm with pneumatic suction gripper for OK/NG sorting.

### Hardware

- Raspberry Pi 5 Model B Rev 1.1 (Debian 12 Bookworm / 64GB SD card, 35GB used)
- Hikvision GigE/USB industrial camera (MVS SDK installed at `/opt/MVS/`)
- FPGA compute board — Paddle Lite inference server at `172.16.68.110:8080/predict`
- 5-DOF Yahboom robot arm + pneumatic gripper (`/dev/XIPAN` → `ttyUSB0`, 9600 baud)
- Buzzer alarm (`/dev/BAOJING` → `input/event0`, **not a serial device**)

### Network

| Interface | Address | Purpose |
|-----------|---------|---------|
| `eth0` | `172.16.68.111/24` | Industrial network (camera, FPGA, arm) |
| `wlan0` | `192.168.1.11/24` | Management/home network (gateway `192.168.1.1`) |
| `docker0` | `172.17.0.1/16` | Docker bridge |

| Service | Address | Port |
|---------|---------|------|
| GrabImage (detection) | 172.16.68.111 | 7777 |
| ArmControl (robot arm) | 172.16.68.111 | 8899 |
| FPGA inference | 172.16.68.110 | 8080 |
| nginx (frontend) | 172.16.68.111 | 80 |
| Jupyter | 172.16.68.111 | (enabled) |
| NoMachine (remote desktop) | 172.16.68.111 | (enabled) |
| motion (camera) | 172.16.68.111 | (enabled) |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Raspberry Pi 5                                                  │
│  Debian 12 / Python 3.11.2 / 2 NICs                              │
│                                                                  │
│  ┌────────────────────┐      ┌────────────────────┐             │
│  │  GrabImage/         │      │  ArmControl/        │             │
│  │  Flask :7777        │      │  Flask :8899        │             │
│  │  systemd: detect-api│      │  systemd: api       │             │
│  │                     │      │                     │             │
│  │  Producer ──────────┼─────→│  /grab POST         │──→ ttyUSB0 │
│  │  (Hikvision camera) │      │  Queue-based async  │   /dev/XIPAN│
│  │  Consumer           │      │                     │  5-DOF Arm  │
│  │  (FPGA inference)   │      │  /use_arm           │             │
│  │       │             │      │  /get_arm           │             │
│  │       ▼             │      └────────────────────┘             │
│  │  FPGA 172.16.68.110 │                                         │
│  │  :8080/predict      │                                         │
│  └────────────────────┘                                          │
│       │                                                          │
│       ▼                                                          │
│  SQLite defect.db + Alarm (/dev/BAOJING→input/event0)            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Docker: yahboomtechnology/ros-melodic:dofbot            │    │
│  │  Separate system — Yahboom DoFBot arm demo               │    │
│  │  Started manually by user `pi`, NOT part of HaoYao       │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Components

### GrabImage/ — Defect detection server (Flask :7777)

| File | Role |
|------|------|
| `api.py` | Main web server. Producer thread reads camera via MVS SDK, Consumer thread detects defects via FPGA inference, stores results in SQLite. |
| `database.py` | SQLite wrapper with thread-safe locks. Table: `defect_list(uuid, path, name, prediction_time, score, CreatedTime)` |
| `config.ini` | `[Configuration]` for timing/speed; `[defect_name]` mapping pinyin → Chinese labels |
| `logger.py` | Logging to `/var/logs/detect_server.log`, 10MB truncation |
| `GrabImage.py` | Standalone camera enumeration utility. **Not used in production.** |
| `grab_image.py` | Standalone grab + OpenCV display with keypress save. **Not used in production.** |

Key runtime flows in `api.py`:
- `Producer.run()` — reads camera frames via MVS SDK, puts resized frames in `q1` Queue, updates `stream_image` for video feed
- `Consumer.run()` — processes 1 in 10 frames: SSIM trigger logic (compare with `yuanshi.jpg`) → POST to FPGA server → parse results → save images → trigger arm via HTTP / alarm via `/dev/BAOJING`
- `compare_image()` — SSIM between current frame and reference (`yuanshi.jpg`, committed in repo)
- `DeleteImg` — cleanup thread: keeps 10 most recent images, deletes rest (runs **once** at import time)
- Flask routes: `/` (UI), `/img` (video stream), `/get_history`, `/get_detect_pic`, `/get_num`, `/get_statistics`, `/get_seven_days_num`, `/get_conf`, `/change_conf`

### ArmControl/ — Robotic arm controller (Flask :8899)

| File | Role |
|------|------|
| `api.py` | Flask server. `/use_arm` for single servo move + grip, `/grab` for full pick-and-place sequence (queued async via `Queue` + Consumer thread). |
| `logger.py` | Logging to `/var/logs/api_server.log` |
| `run.sh` | Sets MVS library paths, starts api.py |
| `dist/` | Pre-built Vue.js SPA (Element UI), served via nginx |
| `api.service` | systemd unit file |

### MvImport/ — Hikvision MVS Python SDK

Camera control wrapper. Used only by GrabImage's Producer thread and the standalone utilities.

## Deployment on Raspberry Pi

### Code vs Data separation

All code and data live under the project directory (`BASE_DIR = GrabImage/`):
- `files/` — defect detection result images
- `original_files/` — original captured images before detection
- `detect_files/` — annotated defect images
- `Font/platech.ttf` — font for Chinese label rendering
- `config.ini` — configuration
- `defect.db` — SQLite database

**Note:** Earlier copies exist at `/opt/MVS/Samples/aarch64/Python/GrabImage/` (MVS SDK sample directory) and `/home/jetson/Desktop/` (original ArmControl deployment). These are **remnants** — the canonical location is `/opt/HaoYao/`.

### Systemd services

```bash
# Detection server (port 7777)
systemctl status detect-api.service
journalctl -u detect-api.service -n 50 --no-pager

# Arm controller (port 8899)
systemctl status api.service
journalctl -u api.service -n 50 --no-pager

# Frontend
systemctl status nginx.service

# Other enabled services on this Pi: docker, jupyter, motion, nxserver (NoMachine)
```

### Dependencies (all pre-installed on the Pi)

```bash
# Python packages
Flask==2.2.2
flask-cors==5.0.1
numpy==1.24.2
pillow==11.2.1
pyserial==3.5
opencv-python     # (installed, version unknown)
scikit-image      # (for compare_ssim)
# Arm library
Arm_Lib           # Yahboom robotic arm library
# Camera SDK
MvImport          # Hikvision MVS SDK (in-repo wrapper)
```

### Devices

| Device | Symlink target | Type |
|--------|---------------|------|
| `/dev/XIPAN` | `ttyUSB0` | Serial port (arm control) |
| `/dev/BAOJING` | `input/event0` | Input event device (alarm) |

**`/dev/BAOJING` is NOT a serial port** — it's a Linux input event device. The code in `api.py` opens it with `serial.Serial()` and writes raw bytes, which likely does not work as intended. The exception handler in `bao_jing_async()` silently catches the failure.

## Frontend

- **Main dashboard**: `ArmControl/dist/` — Vue.js SPA, built with Element UI, served by nginx
- **Live video**: `GrabImage/templates/index.html` — simple Bootstrap page embedding `/img` stream
- Frontend config at `ArmControl/dist/config.json` points to `172.16.68.111:8899` (ArmControl) and `172.16.68.111:7777` (GrabImage)

## ROS on This Pi (separate system, not part of this repo)

The Pi has a Docker container `yahboomtechnology/ros-melodic:dofbot` with:
- ROS Melodic + Yahboom DoFBot arm kinematics (`kinemarics_arm` node)
- `YahboomArm.pyc` Flask app (started manually from within the container)
- `arm_garbage_identify` ROS package with TensorFlow garbage classification (`garbage.h5`)
- This is a **separate system** from HaoYao. It was left over from an earlier project and is not used by the defect detection pipeline.

Old stopped containers from 16 months ago: `ros-melodic:1.2`, `ros2-foxy:2.0.1`.

## Known Issues (verified bugs)

1. **`requests.post` missing timeout** — Consumer blocks forever if FPGA server hangs (`GrabImage/api.py:504`). Also affects arm control POST (`GrabImage/api.py:225`). Fix: add `timeout=30`.
2. **`initMvCamera()` error checks commented out** — Camera init silently fails (`GrabImage/api.py:158-186`). Fix: restore return value checks.
3. **Camera reconnect calls `sys.exit()`** — If re-init fails, entire process dies (`GrabImage/api.py:427-431`). Fix: retry loop instead of exit.
4. **`/dev/BAOJING` is `input/event0`, not a serial port** — `serial.Serial()` opens it but writes likely have no effect (`GrabImage/api.py:32, 87-96`). Alarm may never actually sound. Fix: investigate the actual alarm hardware, use correct interface.
5. **`hua_shang` vs `ca_shang` in SQL** — Statistics query uses `hua_shang` but model returns `ca_shang` (`GrabImage/api.py:393`). `ca_shang` defects never counted.
6. **`q1.queue.clear()` on video feed** — Every `/img` request discards pending processing frames (`GrabImage/api.py:616`).
7. **`debug=True` in production** — Flask debug mode exposes info (`GrabImage/api.py:645`).
8. **Hardcoded IPs** — `172.16.68.111` and `172.16.68.110` hardcoded (`GrabImage/api.py:223, 500`).

## Missing from this repository

- `requirements.txt` — dependencies not consolidated
- `nginx.conf` — mentioned in ArmControl README but not in repo
- Frontend Vue.js source code — only pre-built `dist/`
- FPGA inference server code — runs on separate board, not in this repo
- ROS / Docker files — separate system, not part of this project
