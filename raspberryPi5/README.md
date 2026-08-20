# raspberryPi5/ — 树莓派 5 主控制器平台

> 导航：上一级 [README.md](../README.md) · 相关 [main/](main/README.md)、[de10_nano/](../de10_nano/README.md)

树莓派 5（Debian 12 Bookworm, aarch64）是整个系统的**主控制器**：采集 Hikvision
工业相机图像（MVS SDK）、做前景触发判定（背景差分）、调用 FPGA 推理（172.16.68.110:8080）、
按结果触发机械臂抓取，并把过程数据写入 SQLite。前端为浏览器访问的单页应用。

## 服务总览

| 服务 | 目录 | 端口 | 说明 |
|------|------|------|------|
| main（Flask） | `main/` | 8080 | 检测 + 机械臂分拣 + 前端合一：相机采集 + 前景触发 + FPGA 推理客户端 + SQLite 记录 + MJPEG 视频流 + 静态托管前端 SPA + 5-DOF 机械臂 OK/NG 分拣（队列化串行执行，串口 9600） |
| systemd | — | — | `detect-api.service`（唯一单元；原 `api.service` 已随服务合并废弃） |

> 原 GrabImage（:7777 检测）与 ArmControl（:8899 机械臂）已合并为 `main/`
> 单进程；检测结果触发机械臂改为本地入队（`arm_control.enqueue_grab`），不再
> HTTP 自调用；前端改由 Flask 同端口静态托管，不再依赖 nginx。详见
> [main/README.md](main/README.md)。

### 网络

| 接口 | 地址 | 用途 |
|------|------|------|
| `eth0` | 172.16.68.111/24 | 工业网：相机、FPGA（172.16.68.110:8080）、机械臂 |
| `wlan0` | 192.168.1.11/24 | 管理网（网关 192.168.1.1） |

## 目录结构

```
raspberryPi5/
├── main/         # 主服务（检测 + 机械臂，详见其 README）
├── frontend/     # 前端 SPA（Vue 2 + Element UI + ECharts，无构建步骤，由 Flask 静态托管）
├── MvImport/     # Hikvision MVS SDK 的 ctypes 绑定（手写）
├── deploy.sh     # rsync 部署脚本（dev 机 → Pi）
└── restart.sh    # 重启主服务并查看状态
```

## 部署（开发机 → Pi）

```bash
cd raspberryPi5 && bash deploy.sh [user@pi-ip] [/opt/HaoYao]
```

- `deploy.sh` 使用 `rsync -avzc --delete`（先 `--dry-run` 预览，确认后正式同步）。
- 排除项：`.git/`、`__pycache__/`、`*.pyc`、`defect.db`、`files/`、
  `original_files/`、`detect_files/`、`fpga/`、`deploy.sh`、`CLAUDE.md`、
  `README.md`。
- **所有子目录的 README.md/CLAUDE.md 都不会被部署**（rsync exclude 按 basename
  匹配任意层级）；`de10_nano/` 不在此部署范围（代码在 FPGA 板）。

在 Pi 上生效：

```bash
cd raspberryPi5 && bash restart.sh        # 等价于：
systemctl restart detect-api.service
systemctl status detect-api.service --no-pager -l
```

## 日志

| 日志 | 说明 |
|------|------|
| `/var/logs/server.log` | 主服务调试日志（main/logger.py 写入，10MB 自动截半；合并后统一日志） |

> 合并前的 `/var/logs/detect_server.log`（检测）与 `/var/logs/api_server.log`
> （机械臂）已废弃，统一为 `server.log`。

`main/logger.py` 的 `setup_log(name, log_file)` 返回 logger，同服务多处调用（api.py /
camera.py / arm_control.py）复用同一实例：

```python
from logger import setup_log
logger = setup_log('main', 'server.log')
```

## 前端（frontend/）

自包含单页应用，**无构建步骤**：库全部本地化在 `vendor/`
（vue@2.7.14、element-ui@2.15.14、echarts@5.5.0、axios@1.7.0）。由 Flask 在
`http://172.16.68.111:8080` 同端口静态托管，与检测/机械臂 API、视频流同源。

`frontend/config.json` 的 `ip`/`url` 为空字符串（同源相对路径），由前端拼成
`/get_conf`、`/img_disc` 等请求：

```json
{"ip": "", "url": "", "title": "工业缺陷检测管理平台", "version": "v5.0"}
```

功能：实时 MJPEG 画面、检测统计（缺陷分布饼图 + 当月趋势）、手动机械臂控制、
传送带速度标定、历史记录与缺陷图像浏览。视频流源：
`http://172.16.68.111:8080/img_disc`（带圆检测叠加：绿圆 + 黄裁剪框）。机械臂
API（`/use_arm`、`/get_arm`）与检测 API 同端口。

## MVS SDK 绑定（MvImport/）

Hikvision MVS SDK 的**手写 ctypes 包装**（非官方 python 包）：

| 文件 | 内容 |
|------|------|
| `MvCameraControl_class.py` | 相机控制主类（枚举/打开/取流/关闭等） |
| `CameraParams_header.py` / `CameraParams_const.py` | 相机参数结构体与常量 |
| `MvErrorDefine_const.py` | 错误码定义（含 `CAMERA_NEED_RESTART = 2147483655`） |
| `PixelType_header.py` / `PixelType_const.py` | 像素格式定义 |

运行需 MVS SDK 库（见 main 的 `run.sh`：`LD_LIBRARY_PATH=/opt/MVS/lib/aarch64`）。
