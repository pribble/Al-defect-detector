# AGENTS.md — AI 协作说明

本文件帮助 AI agent 在 HaoYao 仓库中快速定位信息、正确执行任务。保持简单——
**细节一律看对应 README，不要在本文件重复**。

## 项目是什么

铝片表面缺陷自动检测与分拣系统。两个硬件平台：

| 平台 | 目录 | 角色 |
|------|------|------|
| 树莓派 5（Debian 12） | `raspberryPi5/` | 主控制器：相机采集（MVS SDK）、SSIM 触发、FPGA 推理客户端、机械臂分拣、前端 |
| DE10-Nano（Cyclone V SoC） | `de10_nano/` | 推理节点：Paddle Lite + 自研 CNN 加速器，HTTP :8080 |

## 文档地图（重要）

```
README.md                    总览：数据流、硬件/网络、快速开始、部署、文档地图
├── raspberryPi5/README.md   树莓派平台：服务、部署、日志、前端、MVS 绑定
│   ├── GrabImage/README.md  检测服务：线程模型、SSIM 触发、数据库 API、已知问题
│   └── ArmControl/README.md 机械臂服务：动作序列、串行队列、角度标定
└── de10_nano/README.md      FPGA 平台：三模块职责、构建/部署流程
    ├── ssd_detection/README.md  推理服务：355 指令、调用链、I/O 名称、调试
    ├── C5TB/README.md            Quartus 工程：构建/烧录、rbf/dtb、源与生成物
    └── kernel/README.md          cmadrv 驱动：CMA+DMA、设备接口
```

规则：`raspberryPi5/`、`de10_nano/` 内每个子目录的 README 才是权威细节来源。
写文档/改文档时保持此分层，不要向高层文档堆底层细节。

## 常用命令速查

```bash
# 树莓派侧部署（开发机 → Pi）
cd raspberryPi5 && bash deploy.sh [user@pi-ip] [/opt/HaoYao]   # 先 dry-run 预览
cd raspberryPi5 && bash restart.sh                             # Pi 上重启服务

# FPGA 侧构建（开发机，x86_64 交叉编译）
cd de10_nano/kernel && bash build.sh           # cmadrv.ko（KSRC_DIR 默认 /opt/software/linux-4.9.78）
cd de10_nano/ssd_detection && bash build.sh    # 3 阶段：libvnna → Paddle-Lite → 二进制

# FPGA 侧部署
cd de10_nano/ssd_detection && bash upload.sh   # 增量同步 → root@172.16.68.110:/opt/paddle_frame

# 日志
tail -f /var/logs/detect_server.log   # GrabImage
tail -f /var/logs/api_server.log      # ArmControl
journalctl -u detect-api.service -n 50 --no-pager -l
journalctl -u api.service -n 50 --no-pager -l
```

## 工作约定（容易踩坑的点）

- **部署 exclude**：`deploy.sh` 的 rsync 排除 `README.md`/`CLAUDE.md`（任意层级）、
  `defect.db`、`files/`、`original_files/`、`detect_files/`、`GrabImage/yuanshi.jpg`
  等——文档改动不影响部署，但新增运行时文件需注意排除规则。
- **config.ini 键**：`[Configuration]` 下为 `time`/`speed`/`camera_distance`，
  `[defect_name]` 为中文映射。改键要同步 `shared.py`/`routes.py` 的读取处。
- **机械臂角度是占位值**（ArmControl/api.py 的 HOME/PICKUP 等），必须现场标定，
  不要当作真实参数引用。
- **`/dev/BAOJING` 硬件类型未确认**：代码按串口打开（9600 8N1），若实际是
  input event 设备则写入无效。
- **FPGA 推理服务**：模型主 block 共 355 条指令，`PaddlePredictor::Run()` 循环
  执行全部指令，不要只跑 `[calib, subgraph]`；CMA 分配失败先
  `rmmod cmadrv && insmod cmadrv.ko`。
- **SSIM 触发**是帧级状态机（IDLE→TRACKING→COOLDOWN），已废弃旧的导数判定写法。

## 协作偏好

- 回复用简体中文；代码、路径、命令保持原文。
- 走轻量流程：需要分析直接回答；落地任务自行维护简洁清单，不做繁重的逐步骤
  签核；只有大范围/不可逆改动才先出方案再动手。
- 修改代码前先读对应模块的 README（上文地图），避免凭旧文档或记忆猜测。
