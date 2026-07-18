# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HaoYao** — Automated aluminum sheet defect detection and sorting system. Runs on Raspberry Pi 5 (Debian 12, Python 3.11.2). Captures images via Hikvision industrial camera (MVS SDK), performs AI inference on an FPGA board (Paddle Lite + FPGA), and controls a 5-DOF Yahboom robotic arm with pneumatic suction gripper for OK/NG sorting.

### Hardware

- Raspberry Pi 5 Model B Rev 1.1 (Debian 12 Bookworm / 64GB SD card)
- Hikvision GigE/USB industrial camera (MVS SDK installed at `/opt/MVS/`)
- FPGA compute board — Paddle Lite inference server at `172.16.68.110:8080/predict`
- 5-DOF Yahboom robot arm + pneumatic gripper (`/dev/XIPAN` → `ttyUSB0`, 9600 baud)
- Buzzer alarm (`/dev/BAOJING` → `input/event0`, **not a serial device**)

### Network

| Interface | Address | Purpose |
|-----------|---------|---------|
| `eth0` | `172.16.68.111/24` | Industrial network (camera, FPGA, arm) |
| `wlan0` | `192.168.1.11/24` | Management/home network (gateway `192.168.1.1`) |

| Service | Address | Port |
|---------|---------|------|
| GrabImage (detection) | 172.16.68.111 | 7777 |
| ArmControl (robot arm) | 172.16.68.111 | 8899 |
| nginx (frontend) | 172.16.68.111 | 80, 8080 |
| FPGA inference | 172.16.68.110 | 8080 |

## Module Structure

```
HaoYao/
├── GrabImage/              # Defect detection server (Flask :7777)
│   ├── api.py              # Routes, SSIM trigger, Consumer thread
│   ├── camera.py           # Camera init (init_camera) + Producer thread
│   ├── database.py         # SQLite wrapper (defect.db)
│   ├── config.ini          # Timing/speed + defect_name mapping
│   ├── run.sh              # Sets MVS lib paths, starts api.py
│   ├── detect-api.service  # systemd unit
│   └── templates/index.html  # Simple video stream page (Flask-rendered)
├── ArmControl/             # Robot arm controller (Flask :8899)
│   ├── api.py              # Routes, pump control, grab_task sequence
│   ├── run.sh              # Starts api.py
│   ├── api.service         # systemd unit
│   └── dist/               # Pre-built Vue.js SPA (Element UI), served by nginx
└── tools/
    └── logger.py           # Shared logging to /var/logs/
```

## Key Architecture Decisions

### Threading Model

- **Producer** (`camera.py`) — reads Hikvision camera frames, pushes to `frame_queue`, writes `stream_image_ref[0]` for video feed
- **Consumer** (`api.py`) — dequeues 1 in 10 frames, performs SSIM trigger detection, calls FPGA inference, triggers arm/alarm
- **GrabTaskConsumer** (`ArmControl/api.py`) — serializes pick-and-place tasks from a Queue, prevents concurrent arm movement
- **ThreadPoolExecutor** — used for async HTTP calls (`trigger_grab`) and alarm
- `stream_image_ref` is a `list[np.ndarray]` — passed by reference across modules to avoid `global`

### Camera Module (`camera.py`)

```python
from camera import init_camera, Producer, CAMERA_NEED_RESTART
```

- `init_camera()` — retry-loop that enumerates devices, opens first one, starts grabbing
- `Producer(frame_queue, stream_image_ref)` — daemon thread, auto-reconnects on `CAMERA_NEED_RESTART` error

### Database (`database.py`)

Thread-safe SQLite wrapper. Key functions:

```python
database.query(columns, source, condition)  # Build + execute SELECT, return rows
database.select_day_data(offset_start, offset_end)  # Count by day
database.insert_data(uid, path, name, pred_time, score)  # Insert detection record
```

The `query()` function replaces the old `select_instructions()` + `select_data()` pair — just call `query()` directly.

### Detection Trigger Logic (Consumer._process_sampling_frame)

1. Every 10 frames, resize to 64×48, Gaussian blur, threshold, dilate
2. Compute SSIM vs `yuanshi.jpg` (reference background image)
3. Track SSIM sliding window (size 9) — if stable (std < 0.01, mean < 0.8), auto-update reference
4. Trigger inference when: `diff_curr > 0 > diff_prev2 and diff_prev < 0 and white_ratio > 0.1`
5. FPGA returns classification → "zheng_chang" triggers OK grab, defects trigger NG grab + alarm

## Deployment

```bash
# From dev machine: push code to Pi
bash deploy.sh [user@pi-ip] [/opt/HaoYao]

# On Pi: restart services
bash restart.sh

# Systemd
systemctl status detect-api.service   # Flask :7777
systemctl status api.service          # Flask :8899
systemctl status nginx.service        # Frontend :80 / :8080
```

### Frontend

Two paths to the same UI:
- `http://172.16.68.111` (nginx 80) — configured in `/etc/nginx/sites-enabled/default`
- `http://172.16.68.111:8080` (nginx 8080) — configured inline in `/etc/nginx/nginx.conf`

Both serve `ArmControl/dist/` (Vue.js SPA). Vue source not in this repo — only pre-built dist.

Live video stream at `http://172.16.68.111:7777` (Flask-rendered Bootstrap page).

## Known Bugs (Not Fixed)

1. `_generate_frames()` yields `None` on encode failure → TypeError in MJPEG stream
2. `_log_device_info()` GigE model name includes null bytes (missing `if p != 0` filter)
3. `cleanup()` races with `database.create_database()` — may crash on first run
4. `get_detect_pic` / `get_history` — file could be deleted between DB query and read
5. `/dev/BAOJING` hardware type unconfirmed (code treats it as serial, CLAUDE.md says input event device)
