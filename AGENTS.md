# AGENTS.md — AI 协作说明

本文件帮助 AI agent 在 HaoYao 仓库中快速定位信息、正确执行任务。保持简单——
**细节一律看对应 README，不要在本文件重复**。

## 项目是什么

铝片表面缺陷自动检测与分拣系统。两个硬件平台：

| 平台 | 目录 | 角色 |
|------|------|------|
| 树莓派 5（Debian 12） | `raspberryPi5/` | 主控制器：相机采集（MVS SDK）、前景触发（背景差分）、FPGA 推理客户端、机械臂分拣、前端（单端口 :8080） |
| DE10-Nano（Cyclone V SoC） | `de10_nano/` | 推理节点：Paddle Lite + 自研 CNN 加速器，HTTP :8080 |

## 文档地图（重要）

```
README.md                    总览：数据流、硬件/网络、快速开始、部署、文档地图
├── raspberryPi5/README.md   树莓派平台：服务、部署、日志、前端、MVS 绑定
│   └── main/README.md       主服务（检测+机械臂）：线程模型、触发状态机、动作序列、角度标定
└── de10_nano/README.md      FPGA 平台：三模块职责、构建/部署流程
    ├── ssd_detection/README.md  推理服务：355 指令、调用链、I/O 名称、调试
    ├── C5TB/README.md            Quartus 工程：构建/烧录、rbf/dtb、源与生成物
    └── kernel/README.md          cmadrv 驱动：CMA+DMA、设备接口
```

规则：`raspberryPi5/`、`de10_nano/` 内每个子目录的 README 才是权威细节来源。
写文档/改文档时保持此分层，不要向高层文档堆底层细节。

## 命令边界（重要）

当前开发环境是**个人电脑/开发机**，目标设备（树莓派 172.16.68.111、FPGA 板
172.16.68.110）需要 ssh 密码访问。**agent 不执行任何需要设备或密码的命令**，
包括但不限于：

- `deploy.sh`、`upload.sh`（rsync/ssh 到设备）
- `systemctl`、`insmod`、`./run.sh`（设备上的服务/驱动操作）
- 任何直接 ssh/scp 到 172.16.68.111 / 172.16.68.110 的命令

agent 的工作边界：**本机工作区内的代码阅读、修改、文档编写、git 操作、本地校验
（如 Python/grep 检查）**。部署到设备一律由用户手动执行。

## 人工部署速查（供用户参考，agent 不执行）

```bash
# 树莓派侧部署（开发机 → Pi）
cd raspberryPi5 && bash deploy.sh [user@pi-ip] [/opt/HaoYao]   # 先 dry-run 预览
cd raspberryPi5 && bash restart.sh                             # Pi 上重启服务

# FPGA 侧构建（开发机，x86_64 交叉编译）
cd de10_nano/kernel && bash build.sh           # cmadrv.ko（KSRC_DIR 默认 /opt/software/linux-4.9.78）
cd de10_nano/ssd_detection && bash build.sh    # 3 阶段：libvnna → Paddle-Lite → 二进制

# FPGA 侧部署
cd de10_nano/ssd_detection && bash upload.sh   # 增量同步 → root@172.16.68.110:/opt/paddle_frame

# 日志（设备上）
tail -f /var/logs/server.log              # 主服务（合并后唯一日志）
journalctl -u detect-api.service -n 50 --no-pager -l
```

## 工作约定（容易踩坑的点）

- **部署 exclude**：`deploy.sh` 的 rsync 排除 `README.md`/`CLAUDE.md`（任意层级）、
  `defect.db`、`files/`、`original_files/`、`detect_files/` 等——文档改动不影响
  部署，但新增运行时文件需注意排除规则。
- **配置已内联到 Python**（不再有 config.ini）：检测/圆识别参数在 `api.py` 顶部
  常量区，运行时可调项（`GRAB_DELAY`/`CAMERA_DISTANCE`）与缺陷中文映射
  `defect_name` 在 `shared.py`。改参数直接改源码后重启服务。
- **机械臂角度是占位值**（main/arm_control.py 的 HOME/PICKUP 等），必须现场标定，
  不要当作真实参数引用。
- **`/dev/BAOJING` 硬件类型未确认**：代码按串口打开（9600 8N1），若实际是
  input event 设备则写入无效。
- **FPGA 推理服务**：模型主 block 共 355 条指令，`PaddlePredictor::Run()` 循环
  执行全部指令，不要只跑 `[calib, subgraph]`；CMA 分配失败先
  `rmmod cmadrv && insmod cmadrv.ko`。
- **触发**是帧级状态机（前景差分 + 基线 z-score，IDLE→TRACKING→COOLDOWN），
  已废弃旧的导数判定写法与 SSIM 硬处理回退。

## 协作偏好

- 回复用简体中文；代码、路径、命令保持原文。
- 走轻量流程：需要分析直接回答；落地任务自行维护简洁清单，不做繁重的逐步骤
  签核；只有大范围/不可逆改动才先出方案再动手。
- **不操作真实设备**：不执行需要 ssh 密码/设备访问的命令（见"命令边界"），
  部署、重启服务、加载驱动等一律留给用户手动完成。
- 修改代码前先读对应模块的 README（上文地图），避免凭旧文档或记忆猜测。
