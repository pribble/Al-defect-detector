# raspberryPi5/ — 树莓派 5 主控制器平台

> 导航：上一级 [根 README](../README.md) · 子模块 [GrabImage/](GrabImage/README.md)、[ArmControl/](ArmControl/README.md) · 相关 [de10_nano/](../de10_nano/README.md)（FPGA 推理板）

树莓派 5（Debian 12 Bookworm, aarch64）是整个系统的**主控制器**：采集 Hikvision
工业相机图像（MVS SDK）、做 SSIM 触发判定、调用 FPGA 推理（172.16.68.110:8080）、
按结果触发机械臂抓取，并把过程数据写入 SQLite。前端为浏览器访问的单页应用。

## 服务总览

| 服务 | 目录 | 端口 | 说明 |
|------|------|------|------|
| GrabImage（Flask） | `GrabImage/` | 7777 | 相机采集 + SSIM 触发 + FPGA 推理客户端 + SQLite 记录 + MJPEG 视频流 |
| ArmControl（Flask） | `ArmControl/` | 8899 | 5-DOF 机械臂 OK/NG 分拣（队列化串行执行，串口 9600） |
| nginx | `frontend/` | 80 / 8080 | 托管前端 SPA |
| systemd | — | — | `detect-api.service`（GrabImage）、`api.service`（ArmControl） |

### 网络

| 接口 | 地址 | 用途 |
|------|------|------|
| `eth0` | 172.16.68.111/24 | 工业网：相机、FPGA（172.16.68.110:8080）、机械臂 |
| `wlan0` | 192.168.1.11/24 | 管理网（网关 192.168.1.1） |

## 目录结构

```
raspberryPi5/
├── GrabImage/    # 检测服务（线程模型、SSIM 触发等详见其 README）
├── ArmControl/   # 机械臂控制服务（详见其 README）
├── frontend/     # 前端 SPA（Vue 2 + Element UI + ECharts，无构建步骤）
├── MvImport/     # Hikvision MVS SDK 的 ctypes 绑定（手写）
├── tools/        # logger.py —— 共享日志工具
├── deploy.sh     # rsync 部署脚本（dev 机 → Pi）
└── restart.sh    # 重启两个服务并查看状态
```

## 部署（开发机 → Pi）

```bash
cd raspberryPi5 && bash deploy.sh [user@pi-ip] [/opt/HaoYao]
```

- `deploy.sh` 使用 `rsync -avzc --delete`（先 `--dry-run` 预览，确认后正式同步）。
- 排除项：`.git/`、`__pycache__/`、`*.pyc`、`defect.db`、`files/`、
  `original_files/`、`detect_files/`、`fpga/`、`deploy.sh`、`CLAUDE.md`、
  `README.md`、`GrabImage/yuanshi.jpg`。
- **所有子目录的 README.md/CLAUDE.md 都不会被部署**（rsync exclude 按 basename
  匹配任意层级）；`de10_nano/` 不在此部署范围（代码在 FPGA 板）。

在 Pi 上生效：

```bash
cd raspberryPi5 && bash restart.sh        # 等价于：
systemctl restart detect-api.service api.service
systemctl status detect-api.service api.service --no-pager -l
```

## 日志

| 日志 | 说明 |
|------|------|
| `/var/logs/detect_server.log` | GrabImage 调试日志（tools/logger.py 写入，10MB 自动截半） |
| `/var/logs/api_server.log` | ArmControl 调试日志 |
| `journalctl -u detect-api.service -n 50 --no-pager -l` | 服务日志 |
| `journalctl -u api.service -n 50 --no-pager -l` | 服务日志 |

`tools/logger.py` 的 `setup_log(name, log_file)` 返回 logger，两服务共用：

```python
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from logger import setup_log
logger = setup_log('detect', 'detect_server.log')   # 或 ('arm', 'api_server.log')
```

## 前端（frontend/）

自包含单页应用，**无构建步骤**：库全部本地化在 `vendor/`
（vue@2.7.14、element-ui@2.15.14、echarts@5.5.0、axios@1.7.0）。由 nginx 同时托管
于 `http://172.16.68.111`（:80）与 `http://172.16.68.111:8080`。

`frontend/config.json` 配置后端地址：

```json
{"ip": "http://172.16.68.111:8899", "url": "http://172.16.68.111:7777", "title": "工业缺陷检测管理平台", "version": "v4.0"}
```

功能：实时 MJPEG 画面、检测统计（缺陷分布饼图 + 当月趋势）、手动机械臂控制、
传送带速度标定、历史记录与缺陷图像浏览。视频流源：
`http://172.16.68.111:7777/img`。

## MVS SDK 绑定（MvImport/）

Hikvision MVS SDK 的**手写 ctypes 包装**（非官方 python 包）：

| 文件 | 内容 |
|------|------|
| `MvCameraControl_class.py` | 相机控制主类（枚举/打开/取流/关闭等） |
| `CameraParams_header.py` / `CameraParams_const.py` | 相机参数结构体与常量 |
| `MvErrorDefine_const.py` | 错误码定义（含 `CAMERA_NEED_RESTART = 2147483655`） |
| `PixelType_header.py` / `PixelType_const.py` | 像素格式定义 |

运行需 MVS SDK 库（见 GrabImage 的 `run.sh`：`LD_LIBRARY_PATH=/opt/MVS/lib/aarch64`）。
