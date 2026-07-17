# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HaoYao** — Automated aluminum sheet defect detection and sorting system. Runs on Raspberry Pi 5 (Debian 12, Python 3.11.2). Captures images via Hikvision industrial camera (MVS SDK), performs AI inference on an FPGA board (Paddle Lite + FPGA), and controls a 5-DOF robotic arm with pneumatic suction gripper for OK/NG sorting.

### Hardware

- Raspberry Pi 5 Model B Rev 1.1 (Debian 12 Bookworm)
- Hikvision GigE/USB industrial camera (MVS SDK)
- FPGA compute board (inference server at 172.16.68.110:8080/predict)
- 5-DOF robot arm + pneumatic gripper (serial: /dev/XIPAN, 9600 baud)
- Buzzer alarm (serial: /dev/BAOJING, 9600 baud)

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  Raspberry Pi 5 (Debian 12, Python 3.11.2)            │
│                                                        │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  GrabImage/   │    │  ArmControl/ │                  │
│  │  Flask :7777  │    │  Flask :8899 │                  │
│  │               │    │              │                  │
│  │  Producer ───→│───→│  /grab POST  │──→ 5-DOF Arm    │
│  │  (camera)     │    │  Queue-based │    /dev/XIPAN    │
│  │  Consumer     │    │  async exec  │                  │
│  │  (detection)  │    │              │                  │
│  │       │       │    │  /use_arm    │──→ Servo angles  │
│  │       ▼       │    │  /get_arm    │                  │
│  │  FPGA Server  │    │              │                  │
│  │  172.16.68.110│    └──────────────┘                  │
│  │  :8080/predict│                                      │
│  └──────────────┘                                       │
│       │                                                 │
│       ▼                                                 │
│  SQLite (defect.db) + Alarm (/dev/BAOJING)              │
└────────────────────────────────────────────────────────┘
```

### GrabImage/ — Defect detection server (Flask :7777)

| File | Role |
|------|------|
| `api.py` | Main web server (Flask). Producer thread reads camera, Consumer thread detects defects via FPGA inference, stores results in SQLite. |
| `database.py` | SQLite wrapper with thread-safe locks. Table: `defect_list(uuid, path, name, prediction_time, score, CreatedTime)` |
| `config.ini` | `[Configuration]` for timing/speed; `[defect_name]` mapping pinyin → Chinese |
| `logger.py` | Logging setup to `/var/logs/detect_server.log`, with 10MB rotation (truncate first half) |
| `GrabImage.py` | Standalone utility: enumerate camera, grab frames, display. **Not used in production.** |
| `grab_image.py` | Standalone utility: grab + OpenCV display with keypress save. **Not used in production.** |

Key runtime flows in `api.py`:
- `Producer.run()` — reads camera frames continuously, puts them in `q1` Queue
- `Consumer.run()` — processes 1 in 10 frames: SSIM trigger logic → POST to FPGA server → parse results → save images → trigger arm/alarm
- `compare_image()` — SSIM between current frame and reference (`yuanshi.jpg`)
- `delete_img` — cleanup thread: keeps 10 most recent images, deletes rest (runs once at startup)
- Flask routes: `/` (UI), `/img` (video stream), `/get_history`, `/get_detect_pic`, `/get_num`, `/get_statistics`, `/get_seven_days_num`, `/get_conf`, `/change_conf`

### ArmControl/ — Robotic arm controller (Flask :8899)

| File | Role |
|------|------|
| `api.py` | Flask server. `/use_arm` POST for single servo move + grip, `/grab` POST for full pick-and-place sequence (queued). Consumer thread processes queue. |
| `logger.py` | Logging setup to `/var/logs/api_server.log` |
| `run.sh` | Sets MVS library paths, starts api.py |

Hardcoded IPs: GrabImage POSTs to `172.16.68.111:8899/grab` (arm). ArmControl is accessed by the frontend at the same IP.

### MvImport/ — Hikvision MVS Python SDK

Camera control wrapper. Used by GrabImage's Producer thread and the standalone utilities.

### Frontend

- `ArmControl/dist/` — Vue.js SPA (Element UI), served via nginx.
- `GrabImage/templates/index.html` — Simple Bootstrap page with live video stream from `/img`.

## Network

| Service | Address | Port |
|---------|---------|------|
| GrabImage (detection) | 172.16.68.110 | 7777 |
| ArmControl (robot arm) | 172.16.68.111 | 8899 |
| FPGA inference | 172.16.68.110 | 8080 |

## Service Management

```bash
# Detection server
systemctl status detect-api.service
journalctl -u detect-api.service -n 50 --no-pager

# Arm controller
systemctl status api.service
journalctl -u api.service -n 50 --no-pager

# Camera & serial devices
ls -la /dev/BAOJING /dev/XIPAN
```

## Known Issues (verified)

These are confirmed bugs found during code review. Prioritize fixing in this order:

1. **`requests.post` missing timeout** — Consumer thread blocks forever if FPGA server hangs (`api.py:504`). Also affects arm control POST (`api.py:225`). Add `timeout=30`.
2. **`initMvCamera()` error checks commented out** — Camera init silently fails (`api.py:158-186`). Add return value checks.
3. **Camera reconnect calls `sys.exit()`** — If re-init fails, entire process dies (`api.py:427-431`). Replace with retry loop.
4. **Alarm serial never initialized** — `bao_jing_async()` first call always fails (`api.py:32, 87-96`). Uncomment or initialize at module level.
5. **`hua_shang` vs `ca_shang` in SQL** — Statistics query uses wrong defect name (`api.py:393`), `ca_shang` defects not counted.
6. **`q1.queue.clear()` on video feed** — Every `/img` request clears pending frames (`api.py:616`).
7. **`debug=True` in production** — `api.py:645` exposes debug info.
8. **Hardcoded IPs in `api.py:223, 500`** — Should be in config.ini.

## Deployment Paths (Raspberry Pi)

| Component | Path |
|-----------|------|
| Project root | `/opt/HaoYao/` |
| GrabImage | `/opt/HaoYao/GrabImage/` |
| ArmControl | `/opt/HaoYao/ArmControl/` |
| MVS SDK | `/opt/MVS/` |
| Logs | `/var/logs/detect_server.log`, `/var/logs/api_server.log` |

