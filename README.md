# HaoYao

Automated aluminum sheet defect detection and sorting system.

Captures images via Hikvision industrial camera (MVS SDK), detects defects via FPGA inference server (SSD MobileNet V1, Paddle Lite + FPGA acceleration), and controls a 5-DOF Yahboom robotic arm with pneumatic suction gripper for OK/NG sorting.

## Architecture

```
Hikvision Camera ──frame_queue──> GrabImage (:7777)
                                       │
                                       ├── POST /predict ──> FPGA Board (172.16.68.110:8080)
                                       │                         └── SSD MobileNet V1
                                       │                             + Paddle Lite + FPGA
                                       ├── POST /grab ──> ArmControl (:8899) ──> /dev/XIPAN
                                       ├── SQLite (defect.db)
                                       └── GET /img (MJPEG) <── Browser (Vue.js SPA) <── nginx (:8080) <── frontend/
```

### Inference Pipeline

```
Pi: 768×512 gray → cv2.resize(300×300) → .tobytes() raw pixels → POST
FPGA: Mat(300×300, CV_8UC1) → 1-ch NEON→3-ch NCHW tensor → SSD MobileNet V1
```

No JPEG encode/decode overhead. No BGR conversion. Raw grayscale pixels sent directly.

## Hardware

| Device | Interface | Purpose |
|--------|-----------|---------|
| Raspberry Pi 5 (Debian 12) | — | Main controller, runs GrabImage + ArmControl |
| Hikvision GigE/USB camera | Ethernet / USB | Image capture (3072×2048 → 768×512) |
| FPGA compute board (172.16.68.110) | Ethernet :8080 | Paddle Lite inference (Cyclone V SoC) |
| 5-DOF Yahboom arm | `/dev/XIPAN` (ttyUSB0, 9600 baud) | OK/NG pick-and-place sorting |
| Buzzer alarm | `/dev/BAOJING` | Defect alert |

## Services

| Service | Port | Description |
|---------|------|-------------|
| GrabImage/ | 7777 | Camera capture + SSIM trigger + FPGA inference client + SQLite logging |
| ArmControl/ | 8899 | 5-DOF robotic arm sorting (queue-based async, serial task execution) |
| nginx | 80 / 8080 | Serves frontend SPA (frontend/) |

## Quick Start

```bash
# Detection server
systemctl start detect-api.service

# Arm controller
systemctl start api.service

# Check logs
journalctl -u detect-api.service -n 50 --no-pager
```

## Deploy

```bash
# 在 x86_64 主机上：推送代码到树莓派
bash deploy.sh [user@pi-ip] [/opt/HaoYao]

# 在树莓派上：重启服务使新代码生效
bash /opt/HaoYao/restart.sh
```

## FPGA Inference Server

See [fpga/CLAUDE.md](fpga/CLAUDE.md) for full FPGA build/deploy instructions. The inference server runs on a separate Cyclone V SoC board at `172.16.68.110:8080/predict`.

```bash
# Cross-compile (from x86_64 host)
cd fpga && bash detect_build.sh

# Deploy to FPGA board
cd fpga && bash upload.sh
```

## Frontend

Self-contained single-page application (Vue 2 + Element UI + ECharts) at `frontend/index.html`. No build step needed — libraries are served locally from `frontend/vendor/`. Configured in `config.json`:

```json
{"ip": "http://172.16.68.111:8899", "url": "http://172.16.68.111:7777", "title": "工业缺陷检测管理平台", "version": "v4.0"}
```

Served by nginx on ports 80 and 8080. Features:

- Live MJPEG video stream (`http://172.16.68.111:7777/img`)
- Real-time statistics (pie chart by defect type, 7-day trend lines)
- Manual arm control (servo angles, pump on/off)
- Conveyor speed calibration mode
- History browser with annotated defect images

## Dependencies

- Python 3.11+, Flask, OpenCV, numpy, pillow, pyserial, scikit-image, flask-cors
- Hikvision MVS SDK at `/opt/MVS/`
- FPGA inference server at `172.16.68.110:8080`
