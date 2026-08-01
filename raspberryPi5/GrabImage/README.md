# GrabImage/ — 检测服务（Flask :7777）

> 导航：上一级 [raspberryPi5/](../README.md) · 相关 [ArmControl/](../ArmControl/README.md)、[ssd_detection/](../../de10_nano/ssd_detection/README.md)

缺陷检测主服务：Hikvision 工业相机实时采集（MVS SDK）→ SSIM 触发判定 → FPGA 推理
（172.16.68.110:8080）→ 按结果触发机械臂分拣（ArmControl :8899）与蜂鸣器报警
（/dev/BAOJING）→ 结果与图片写入 SQLite。

## 目录结构

| 文件 | 说明 |
|------|------|
| `api.py` | Flask 入口 + Consumer 检测线程 + 报警/抓取触发 + cleanup 清理 |
| `camera.py` | MVS SDK 相机封装：Producer 采集线程、FrameBuffer 单槽帧缓冲 |
| `routes.py` | HTTP API（Blueprint）：配置、历史、统计、标定、视频流 |
| `shared.py` | 共享可变状态（api.py 与 routes.py 之间传引用） |
| `database.py` | SQLite 线程安全封装（defect.db） |
| `config.ini` | 传送带参数 + 缺陷中文名映射 |
| `run.sh` | 设置 MVS 库路径后启动 api.py |
| `detect-api.service` | systemd 单元 |
| `yuanshi.jpg` | SSIM 参考背景图（自动更新） |
| `Font/platech.ttf` | 缺陷标注用中文字体 |
| `templates/index.html` | 简单 MJPEG 页面 |

运行时生成目录：`files/`（缺陷图片）、`original_files/`（原图）、`detect_files/`
（标注图）——均在部署时排除、不入库。

## 线程模型

| 线程 | 职责 |
|------|------|
| **Producer**（camera.py） | 采集相机帧 → 裁 4:3 → 缩小 → 写 `frame_queue`（**FrameBuffer 单槽缓冲**，丢旧帧不阻塞）并更新 `stream_image_ref[0]`；遇 `CAMERA_NEED_RESTART`(2147483655) 自动重开相机 |
| **Consumer**（api.py） | 读帧 → SSIM 触发状态机 → 命中后跑推理管线；异常记日志继续 |
| **ThreadPoolExecutor**（shared.py） | fire-and-forget 异步执行 `trigger_grab` / `trigger_alarm`（不等待结果） |
| **cleanup** | 保留最新 `SAVE_IMAGE_NUM=10` 张缺陷图，删除其余 |

相机初始化要点（camera.py）：手动固定曝光（`ExposureAuto=0`、`GainAuto=0`、
`ExposureTime=6000us`、`Gain=0dB`）——**避免自动曝光导致 SSIM 误触发**；宽高比裁到
4:3 以内再 1/4 下采样（最终约 768×512 灰度）。

## shared.py 单例模式

`api.py` 以 `__main__` 运行而 `routes.py` 被 import，直接互相 import 会产生两份模块
实例。解法：`shared.py` 持有全部共享状态，两模块都从这里取：

```python
from shared import config, logger, stream_image_ref, thread_pool, GRAB_SPEED, GRAB_DELAY
```

- `stream_image_ref = [np.ndarray]`：list 包装绕开 Python 作用域，`[0]` 为最新帧。
- `config`/`defect_name`/`GRAB_SPEED`/`GRAB_DELAY`：`/change_conf` 与
  `/calibration` 路由可更新，Consumer 侧即时可见。
- `calibration_active`/`calibration_samples`：传送带速度标定共享状态。

## SSIM 触发逻辑（状态机）

Consumer 每帧处理（`_process_sampling_frame`）：resize 64×48 → GaussianBlur(21)
→ 二值阈值 100 → dilate(4 次) → 计算 `white_ratio`；与参考图 `yuanshi.jpg`
（内存缓存）做 SSIM（`compare_image`）。SSIM 滑动窗口（9 帧）稳定
（std<0.01、mean<0.8、全非零）时自动用当前帧更新参考图。

**触发状态机**（`_state`：0=IDLE → 1=TRACKING → 2=COOLDOWN）：

```
IDLE:     current_ssim < 0.8 && white_ratio > 0.1  连续 3 帧 → TRACKING
TRACKING: 收集帧并维护中间帧（单数帧 append、双数帧 pop 头部再 append）
          SSIM 回升 ≥ 0.8 && white_ratio < 0.1 连续 3 帧 → 取中间帧跑推理 → COOLDOWN
COOLDOWN: 5 帧冷却，防止同一铝片重复触发
```

> 注意：这是当前实现（帧级状态机 + 中间帧选取）。早期版本的
> diff_curr/diff_prev 导数判定已废弃，文档以此状态机为准。

## 推理管线（_run_inference_pipeline）

```
中间帧 → resize(300×300) → tobytes()（灰度 raw，无 JPEG）→ POST FPGA /predict
  → 响应 {"len", "action", "result":[{class_name, loc[x1,y1,x2,y2], score, prediction_time}]}
  → len>0 且 action=NG：报警 + NG 抓取 + 画框标注（_draw_defect_box）写库
  → 否则：OK 抓取 + 正常结果写库
```

- 坐标从 300×300 缩放回实际帧尺寸（`scale_x = w/300`，`scale_y = h/300`）。
- 标注用 PIL + `Font/platech.ttf` 绘制中文缺陷名（config.ini 的 defect_name 映射）。
- 库记录：`insert_data(uid, 文件路径, class_name, prediction_time, score)`，同时写
  `detect.jpg`（最新标注图，供 `/get_detect_pic`）。
- 请求超时 30s（`INFERENCE_TIMEOUT`）。

## 报警与抓取触发

| 触发 | 实现 |
|------|------|
| `trigger_alarm` | 打开 `/dev/BAOJING`（`serial.Serial` 9600 8N1），写命令 `7E FF 06 03 00 00 01 EF`；异常时置 None 下次重连 |
| `trigger_grab(flags)` | `POST http://172.16.68.111:8899/grab`，body `{"flags": "OK"/"NG", "time": GRAB_DELAY}` |

两者均经 `shared.thread_pool.submit()` 异步执行（NG 时先报警后 0.1s 再抓取）。

## 数据库 API（database.py）

线程安全（`threading.Lock`），单连接 `defect.db`。优先使用统一 `query()`：

| 函数 | 说明 |
|------|------|
| `create_database()` | 建 `defect_list` 表（if not exists） |
| `insert_data(uid, path, name, prediction_time, score)` | INSERT（`insert or ignore`，UNIQUE(uuid,path,name)） |
| `query(columns, source, condition, params=())` | 通用 SELECT，返回行列表（新代码统一用它） |
| `query_value(...)` | 聚合快捷（`count()`/`sum()` 等，无结果 None） |
| `select_day_data(offset_start, offset_end)` | 按天偏移计数（"+0"=今天，"-1"=昨天） |

表结构：`defect_list(id INTEGER PK, uuid CHAR, path CHAR(64), name CHAR,
prediction_time CHAR, score CHAR, CreatedTime TIMESTAMP DEFAULT
datetime('now','localtime'), UNIQUE(uuid, path, name))`。

## HTTP API（routes.py）

| 路由 | 说明 |
|------|------|
| `GET /get_conf` `POST /change_conf` | 读/改 config.ini（time/speed/camera_distance），改后同步 shared 缓存 |
| `GET /get_history` | 最新 4 张历史检测图（base64） |
| `GET /get_original_pic` `GET /get_detect_pic` | 最近一次原图 / 缺陷标注图 |
| `GET /get_num` `/get_this_month_num` `/get_today_num` `/get_seven_days_num` `/get_seven_days_by_type` `/get_statistics` | 统计（按类型/当月/当日/7 天/总体） |
| `POST /calibration` `GET /calibration_status` | 传送带速度标定（`camera_distance/速度` 建议 delay） |
| `GET /img` | MJPEG 实时流（前端视频源） |
| `GET /debug_mask` | SSIM 调试流（二值掩码 + SSIM/white_ratio） |
| `GET /` | Bootstrap 简单页面 |

## 配置（config.ini）

```ini
[Configuration]
time = 4              # 传送带延迟（秒）—— 相机到抓取点的传输时间，即 GRAB_DELAY
speed = 1300          # 伺服移动速度（毫秒）—— 即 GRAB_SPEED
camera_distance = 200 # 相机到抓取点的距离（mm），标定模式计算建议 delay 用

[defect_name]
ca_shang = 划痕       # Scratch
zhen_kong = 针孔       # Pinhole
zang_wu = 脏污         # Dirt/stain
zhe_zhou = 褶皱        # Wrinkle
zheng_chang = 正常     # Normal（无缺陷）
```

> 说明：`gaussian_kernel` 可从 config 读（默认 21）；`grab_position`/
> `release_position` 旧键已移除。

## 运行

```bash
# 手动
cd /opt/HaoYao/GrabImage && bash run.sh
# run.sh：export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64; MVCAM_COMMON_RUNENV=/opt/MVS/lib;
#         CRYPTOGRAPHY_ALLOW_OPENSSL_102=1; python3 api.py

# systemd（开机自启，Restart=always）
systemctl restart detect-api.service
journalctl -u detect-api.service -n 50 --no-pager -l
tail -f /var/logs/detect_server.log
```

## 已知问题（当前代码状态）

1. **`/dev/BAOJING` 硬件类型未确认** —— 代码按串口打开（`serial.Serial` 9600 8N1）。
   若实际是 `/dev/input/event0` 输入事件设备，串口写入将无效果。**未修复**。
2. 历史/统计路由中若 `defect_list` 表未建（首次启动竞态）会查询报错——现已在
   `__main__` 中先 `create_database()` 再启动 cleanup/Consumer，**已修复**。
3. `_image_to_base64` 对文件已被 cleanup 删除的情况捕获异常返回空串（前端得到空
   img）——**已缓解**（不再 500，但仍有空图）。
4. GigE 设备名解析已按 `p != 0` 过滤 null 字节（camera.py `_log_device_info`）。
