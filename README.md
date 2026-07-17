# Al-defect-detector

**HaoYao** — Automated aluminum sheet defect detection and sorting system.

Captures images via Hikvision industrial camera, performs AI inference with Paddle Lite + FPGA, and controls a 5-DOF robotic arm with suction gripper for OK/NG sorting. Built with Flask, OpenCV, and serial control.

## Components

| Directory | Description |
|-----------|-------------|
| `GrabImage/` | Hikvision camera capture, defect detection, alarm trigger. Flask web UI, SSIM-based trigger logic, SQLite records. |
| `ArmControl/` | 5-DOF robotic arm sorting controller with pneumatic suction pump. Queue-based async execution. |
| `MvImport/` | Hikvision MVS SDK Python wrapper (CameraParams, MvCameraControl). |

Inference server (separate board): Paddle Lite + FPGA accelerated SSD detection.

## Hardware

- Hikvision GigE / USB industrial camera
- 5-DOF robot arm + pneumatic suction gripper
- Buzzer alarm (serial controlled, /dev/BAOJING)
- Raspberry Pi / Jetson (device side)
- FPGA accelerated compute board (inference server)
# Al-defect-detector
