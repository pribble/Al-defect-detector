# HaoYao — 铝片表面缺陷自动检测与分拣系统

Automated aluminum sheet defect detection and sorting system.

系统通过 Hikvision 工业相机（MVS SDK）采集图像，交由 FPGA 推理服务器
（SSD MobileNet V1，Paddle Lite + 自研 CNN 加速器）检测缺陷，再由 5-DOF
Yahboom 机械臂 + 气动吸盘按 OK/NG 结果分拣。主控制器为树莓派 5
（Debian 12 Bookworm, aarch64），推理节点为 DE10-Nano（Cyclone V SoC）。

## 系统数据流

```
Hikvision Camera ──frame_queue──> main (:7777, 树莓派)
                                       │
                                       ├── POST /predict ──> DE10-Nano FPGA (172.16.68.110:8080)
                                       │                         └── ssd_detection（Paddle Lite + CNN 加速器）
                                       ├── 本地入队 enqueue_grab ──> /dev/XIPAN（5-DOF 机械臂）
                                       ├── SQLite (defect.db)
                                       └── MJPEG /img ──> nginx (:80/:8080) ──> 浏览器前端
```

### 推理管线（300×300 raw，无 JPEG）

```
Pi: 768×512 gray → cv2.resize(300×300) → .tobytes() → POST /predict
FPGA: Mat(300×300, CV_8UC1) → NEON 1ch→3ch NCHW tensor → SSD MobileNet V1（卷积卸载 FPGA）
```

## 硬件

| 设备 | 接口 | 用途 |
|------|------|------|
| 树莓派 5（Debian 12） | — | 主控制器：main（检测 + 机械臂，单服务） |
| Hikvision 工业相机 | 以太网 / USB | 图像采集（3072×2048 → 768×512） |
| DE10-Nano（Cyclone V SoC） | 以太网 :8080 | Paddle Lite 推理 + CNN FPGA 加速器（172.16.68.110） |
| 5-DOF Yahboom 机械臂 | `/dev/XIPAN`（ttyUSB0，9600 baud） | OK/NG 吸取分拣 |
| 蜂鸣器报警 | `/dev/BAOJING` | 缺陷告警 |

## 网络

| 接口 | 地址 | 用途 |
|------|------|------|
| `eth0` | 172.16.68.111/24 | 工业网（相机、FPGA、机械臂） |
| `wlan0` | 192.168.1.11/24 | 管理网（网关 192.168.1.1） |

| 服务 | 地址 | 端口 |
|------|------|------|
| main（Flask，检测 + 机械臂） | 172.16.68.111 | 7777 |
| nginx（前端） | 172.16.68.111 | 80 / 8080 |
| FPGA 推理 | 172.16.68.110 | 8080 |

## 快速开始

### 树莓派服务（:7777）

```bash
# 主服务（检测 + 机械臂，合并自原 GrabImage/ArmControl）
systemctl start detect-api.service

# 日志
journalctl -u detect-api.service -n 50 --no-pager -l
tail -f /var/logs/server.log    # 主服务调试日志
```

### FPGA 推理板（172.16.68.110 :8080）

```bash
# 在 FPGA 板部署目录（/opt/paddle_frame）
cd /opt/paddle_frame
insmod cmadrv.ko                       # CMA + DMA 内核驱动
./run.sh                               # 启动 HTTP 推理服务（监听 :8080）
```

> 推理板由开发机交叉编译后部署，完整构建/运行说明见
> [de10_nano/README.md](de10_nano/README.md) 与 [ssd_detection/README.md](de10_nano/ssd_detection/README.md)。

## 部署

### 树莓派代码

```bash
# 开发机：推送 raspberryPi5 代码到树莓派（dry-run 预览后确认）
cd raspberryPi5 && bash deploy.sh [user@pi-ip] [/opt/HaoYao]

# 树莓派上：重启服务使新代码生效（合并后仅一个单元）
cd raspberryPi5 && bash restart.sh
```

### FPGA 推理板

```bash
# 开发机（x86_64）：交叉编译内核驱动与推理服务
cd de10_nano/kernel && bash build.sh           # 编译 cmadrv.ko 并拷入 deploy/
cd de10_nano/ssd_detection && bash build.sh    # 3 阶段：libvnna → Paddle-Lite → 二进制

# 增量同步到 FPGA 板（root@172.16.68.110:/opt/paddle_frame）
cd de10_nano/ssd_detection && bash upload.sh
```

> FPGA 侧全流程（含 Quartus 硬件工程 C5TB 的构建/烧录）见
> [de10_nano/README.md](de10_nano/README.md)。

## 文档地图

本仓库文档按平台/模块分层组织，每份文档聚焦本层内容并互相链接：

```
README.md（本文档）
├── raspberryPi5/README.md            # 树莓派平台：服务、网络、部署、日志、前端、MVS 绑定
│   └── main/README.md                # 主服务（检测 + 机械臂）：线程模型、SSIM 触发、动作序列、角度标定、已知问题
└── de10_nano/README.md               # FPGA 平台：三模块职责、构建/部署流程、.gitignore
    ├── ssd_detection/README.md       # 推理服务：模型 355 指令、调用链、I/O 名称、构建/调试
    ├── C5TB/README.md                # Quartus 工程：顶层/QSys 结构、rbf/dtb 生成、源与生成物
    └── kernel/README.md              # cmadrv 驱动：CMA 分配 + DMA、设备接口
```

## 前端

自包含单页应用（Vue 2 + Element UI + ECharts），无构建步骤，库本地化于
`frontend/vendor/`。地址 `http://172.16.68.111`（:80 与 :8080），后端地址配置在
`frontend/config.json`。功能：实时 MJPEG 画面（`/img`）、检测统计、手动机械臂
控制、传送带速度标定、历史缺陷图浏览。

## 依赖

- Python 3.11+、Flask、OpenCV、numpy、pillow、pyserial、scikit-image、flask-cors
- Hikvision MVS SDK（`/opt/MVS/`，`raspberryPi5/MvImport/` 为 ctypes 绑定）
- FPGA 推理服务器 `172.16.68.110:8080`
