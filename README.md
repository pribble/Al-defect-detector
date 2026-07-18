# HaoYao

Automated aluminum sheet defect detection and sorting system.

Captures images via Hikvision industrial camera, detects defects via FPGA inference server (SSD MobileNet V1, Paddle Lite), and controls a 5-DOF robotic arm with pneumatic gripper for OK/NG sorting.

## Architecture

```
Hikvision Camera  ──>  GrabImage (:7777)  ──POST──>  FPGA Board (:8080/predict)
                            │                              └── SSD MobileNet V1
                            ├── SQLite (defect.db)               + FPGA acceleration
                            └── ArmControl (:8899) ──> /dev/XIPAN (5-DOF arm)
```

## Components

| Service | Port | Description |
|---------|------|-------------|
| `GrabImage/` | 7777 | Camera capture + SSIM trigger logic + FPGA inference client + SQLite logging |
| `ArmControl/` | 8899 | 5-DOF robotic arm sorting (pick-and-place sequence, queue-based async) |
| `MvImport/` | — | Hikvision MVS SDK Python wrapper |

## Hardware

| Device | Interface | Purpose |
|--------|-----------|---------|
| Hikvision GigE/USB camera | Ethernet/USB | Image capture |
| FPGA compute board (172.16.68.110) | Ethernet :8080 | Paddle Lite inference |
| 5-DOF Yahboom arm | `/dev/XIPAN` (ttyUSB0) | OK/NG sorting |
| Buzzer alarm | `/dev/BAOJING` | Defect alert |

## Quick Start

```bash
# Detection server
systemctl start detect-api.service

# Arm controller
systemctl start api.service

# Check logs
journalctl -u detect-api.service -n 50 --no-pager
```

## Dependencies

- Python 3.11+, Flask, OpenCV, numpy, pillow, pyserial, scikit-image
- Hikvision MVS SDK at `/opt/MVS/`
- FPGA inference server at `172.16.68.110:8080`
