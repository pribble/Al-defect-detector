# main/ — 主服务（Flask :8080，检测 + 机械臂分拣）

> 导航：上一级 [raspberryPi5/](../README.md) · 相关 [ssd_detection/](../../de10_nano/ssd_detection/README.md)

树莓派上的**唯一后端服务**（单进程单端口 :8080）：Hikvision 工业相机实时采集
（MVS SDK）→ 前景触发判定（背景差分）→ FPGA 推理（172.16.68.110:8080）→ 按结果
**本地触发**机械臂分拣任务与蜂鸣器报警（/dev/BAOJING）→ 结果与图片写入 SQLite；
前端 SPA（`../frontend/`）由本服务同端口静态托管。

> 说明：检测结果触发机械臂不再走 HTTP 自调用，而是 `api.py` 直接调
> `arm_control.enqueue_grab(flags, delay)` 直接触发；前端 `config.json` 的 `ip`/`url`
> 为空字符串（同源相对路径）；配置不再用 config.ini，全部内联到 Python 源码。

## 目录结构

| 文件 | 说明 |
|------|------|
| `api.py` | Flask 入口 + Consumer 检测线程 + 报警/抓取触发 + cleanup 清理 |
| `arm_control.py` | 机械臂控制（Blueprint 'arm'）：动作序列、互斥锁串行、/use_arm /get_arm |
| `camera.py` | MVS SDK 相机封装：Producer 采集线程、FrameBuffer 单槽帧缓冲 |
| `disc_detect.py` | 圆形铝片识别：推理前定位圆心/半径（mask/hough 两方法），供智能裁剪 |
| `routes.py` | 检测类 HTTP API（Blueprint）：配置、历史、统计、标定、视频流、主页 |
| `shared.py` | 共享可变状态（api.py 与 routes.py 之间传引用） |
| `database.py` | SQLite 线程安全封装（defect.db） |
| `logger.py` | 共享日志工具（原 `tools/logger.py`，移入本目录） |
| `run.sh` | 设置 MVS 库路径后启动 api.py |
| `detect-api.service` | systemd 单元（唯一服务） |
| `Font/platech.ttf` | 缺陷标注用中文字体 |

运行时生成目录：`files/`（缺陷图片）、`original_files/`（原图）、`detect_files/`
（标注图）——均在部署时排除、不入库。

## 机械臂硬件接口（arm_control.py）

| 设备 | 接口 | 说明 |
|------|------|------|
| 机械臂 | `/dev/XIPAN`（ttyUSB0，9600 baud） | `Arm_Device`（Arm_Lib）写舵机角度 |
| 气泵继电器 | `/dev/XIPAN`（同一串口，9600） | Modbus-RTU 指令控制吸气/释放 |

气泵指令：

```python
GRIP_ON  = bytes.fromhex('A0 01 01 A2')   # 继电器吸合 → 吸气
GRIP_OFF = bytes.fromhex('A0 01 00 A1')   # 继电器断开 → 释放
```

吸盘时序（秒）：`SUCTION_HOLD=0.6`（吸取保持）、`RELEASE_HOLD=0.4`（释放保持）、
`LIFT_PAUSE=0.3`、`POST_RELEASE=0.3`。

### 舵机角度预设（占位值 — 必须现场标定！）

`arm_control.py` 顶部的角度常量是 **placeholder**，与实际物理机械臂的尺寸/安装
位置无关，**上线前必须逐点标定**：

| 常量 | 用途 | 格式 |
|------|------|------|
| `HOME` | 初始位（竖直收起，不遮挡相机、不干涉传送带） | `[基座, 肩, 肘, 腕, 末端]` 5 个角度（度） |
| `PICKUP` | 吸取位：正前方传送带面 200mm | `[90, 30, 62, 0, 90]` ← 需标定 |
| `PICKUP_LIFT` | 吸取后抬高，留旋转空间 | 需标定 |
| `OK_ABOVE` / `OK_PLACE` | OK 区上方 / OK 区放置位（基座右转） | 需标定 |
| `NG_ABOVE` / `NG_PLACE` | NG 区上方 / NG 区放置位（基座左转） | 需标定 |

`arm_move(angles, move_time)`：5 个舵机顺序写角度（ID5 末端舵机较慢，延迟与
1.2 倍时间补偿），随后 sleep `move_time/1000` 等待到位。

### grab_task 动作序列（单次分拣）

```
0. 等待传送延迟 time 秒（铝片从相机走到吸取点；time 即 shared.GRAB_DELAY）
1. HOME → PICKUP（各 250ms）
2. 吸盘吸取（suction_on + 0.15s 稳定）
3. 抬起 PICKUP_LIFT（250ms）
4. 平移：OK → OK_ABOVE → OK_PLACE / NG → NG_ABOVE → NG_PLACE
5. 吸盘释放（suction_off + POST_RELEASE）
6. 抬起离开放置区（250ms + LIFT_PAUSE，避免复位碰撞工件）
7. 复位 HOME（500ms）
```

`_write_pump` 对串口写入异常自动重连（重新实例化 `serial.Serial`）。

### 分拣触发（互斥锁串行，不缓存）

- `enqueue_grab(flags, delay)`（api.py 检测结果直接调用）直接执行 `grab_task()`，
  用 `_grab_lock`（`threading.Lock`）保证同一时刻只有一次分拣动作。
- 忙时（锁已被占用）新信号直接**丢弃并记 warning**——不缓存、不排队：铝片已越过
  吸取点时再补抓没有意义。
- 效果：机械臂**一次只动一个**，动作不交叠，且不会积压过期的抓取信号。

## 线程模型

| 线程 | 职责 |
|------|------|
| **Producer**（camera.py） | 采集相机帧 → 裁 4:3 → 缩小 → 写 `frame_queue`（**FrameBuffer 单槽缓冲**，丢旧帧不阻塞）并更新 `stream_image_ref[0]`；遇 `CAMERA_NEED_RESTART`(2147483655) 自动重开相机 |
| **Consumer**（api.py） | 读帧 → 前景触发状态机 → 命中后跑推理管线；异常记日志继续 |
| **ThreadPoolExecutor**（shared.py） | fire-and-forget 异步执行 `trigger_grab`（直接跑机械臂分拣动作序列）/ `trigger_alarm`（不等待结果） |
| **cleanup** | 保留最新 `SAVE_IMAGE_NUM=10` 张缺陷图，删除其余 |

相机初始化要点（camera.py）：手动固定曝光（`ExposureAuto=0`、`GainAuto=0`、
`ExposureTime=6000us`、`Gain=0dB`）——**避免自动曝光导致误触发**；宽高比裁到
4:3 以内再 1/4 下采样（最终约 768×512 灰度）。

## shared.py 单例模式

`api.py` 以 `__main__` 运行而 `routes.py` 被 import，直接互相 import 会产生两份模块
实例。解法：`shared.py` 持有全部共享状态，两模块都从这里取：

```python
from shared import logger, stream_image_ref, thread_pool, GRAB_DELAY, CAMERA_DISTANCE
```

- `stream_image_ref = [np.ndarray]`：list 包装绕开 Python 作用域，`[0]` 为最新帧。
- `GRAB_DELAY` / `CAMERA_DISTANCE`：`/change_conf` 可更新，Consumer 侧即时可见。
- `defect_name`：缺陷中文映射（dict）。
- `calibration_active`/`calibration_samples`：传送带速度标定共享状态。
- `disc_method`/`disc_cfg`/`disc_margin_ratio`/`debug_background`：供 `/img_disc`
  视频流对实时帧定圆叠加。

## 触发逻辑（状态机）

Consumer 每帧处理（`_process_sampling_frame`）：resize 64×48 → EMA 背景差分 →
Otsu 二值化 → 形态学 → 连通域过滤 → 计算 `fg_ratio`（前景占比）。空皮带基线
（滚动窗口）的 mean/std 用于 z-score 判定前景；暖机期（`BASELINE_INIT_FRAMES=30`
帧）内无条件更新背景。

**触发状态机**（`_state`：0=IDLE → 1=TRACKING → 2=COOLDOWN）：

```
IDLE:     fg_ratio > 基线mean + trigger_k×std 连续 3 帧 → TRACKING
TRACKING: 每帧对 64×48 前景掩码拟合圆, 打分维护【最佳帧】
          前景回落（< 基线）连续 3 帧 → 取最佳帧跑推理 → COOLDOWN
          [超时保护] 超过 tracking_timeout_frames 帧仍不回落 → 取最佳帧强制推理 (防卡死)
COOLDOWN: 0.5 秒冷却，防止同一铝片重复触发
```

> 说明：推理帧的选帧依据为**两阶段排序键**——每帧对 64×48 前景掩码拟合圆：
> 圆完整度 `in_frac ≥ center_complete(0.9)` 的"完整帧"档内，选圆心最接近画面
> 水平中心的帧（原长方形图中铝片横向最中间）；无完整帧时按完整度打分
> `S = r × in_frac` 兜底。fg_ratio 会被入口反光撑大导致选帧错位，圆打分只由
> 外边界决定、抗反光。另有**软降级门槛** `min_radius_ratio_frame(0.18)`：检出圆
> 半径相对帧短边占比低于此值的帧降为档 0（仍按完整度互相比较、不淘汰，排在任何
> 达标帧之后），过滤"铝片刚进入/偏远"的过小圆帧，保证超时强制推理时仍有相对
> 最好的候选。

## 推理管线（_run_inference_pipeline）

```
最佳帧 → 圆识别+智能裁剪 → resize(300×300) → tobytes()（灰度 raw，无 JPEG）
  → POST FPGA /predict
  → 响应 {"len", "action", "result":[{class_name, loc[x1,y1,x2,y2], score, prediction_time}]}
  → action=NONE（无任何检测框）：按正常(OK)处理——OK 抓取 + 正常存图写库 + 前端
    历史展示（不报警；取舍：模型漏检的缺陷也会被放行，宁丢缺陷不漏片）
  → action=NG（检出缺陷类）：报警 + NG 抓取 + 画框标注（_draw_defect_box）写库
  → action=OK（检出 zheng_chang）：OK 抓取 + 正常结果写库
   （正常类也会产生框，len>0 不再代表缺陷，一律以 action 为准分派）
```

### 圆识别 + 智能裁剪（默认开启）

整帧 768×512（3:2）直接 `resize(300,300)` 是**各向异性压缩**（水平 2.56×、垂直 1.71×），
缺陷形状会被挤压。因此推理前先定位铝片圆心：

1. `find_disc_robust()`（disc_detect.py）在**选中帧**上定圆，返回 `(cx, cy, r)`，
   带**回退链**（`method_fallback=1` 时）：主方法失败自动换下一个，全失败才回退整帧：
   - `method=mask`（默认）：优先**背景差分路径**——与触发链路同源的 EMA 背景
     `absdiff → 高斯模糊 → Otsu → 形态学 → 最大外轮廓 → minEnclosingCircle`，
     光照变化/皮带反光下最稳；无背景时走原始帧 Otsu。反射高光在圆内部不影响外
     边界；圆形度过滤剔除机械臂等非圆亮斑。
   - `method=hough`：`Canny + HoughCircles` 圆弧投票，边缘残缺也能定圆；参数
     （`hough_param1/param2`）需现场调。
2. 以 `(cx, cy)` 为中心裁**正方形**：边长 `side = 2×(r + r×margin_ratio)`（默认留
   半径 10% 的黑边，不完全切边，减少背景干扰），越界自动钳制到帧内。
3. 裁剪图等比缩放到 300×300 送 FPGA —— 内容不失真、铝片居中。
4. **圆完整度门槛**（`min_completeness=0.7`）：检出的圆在画面内占比低于 70% 时
   回退整帧（铝片未完整进入视野，硬裁会切掉铝片）；**出界的圆不再直接拒绝**——
   黑边吸收小越界，选帧阶段已保证取到完整度最高的帧。
5. **未检出圆/裁剪过小时回退整帧缩放**（记 warning，不影响流程）。`_smart_crop`
   整体有 try/except 保护——定圆代码出意外也不会中断推理/存图链路。

坐标映射：`_draw_defect_box` 按 `self._crop_box (x0, y0, side)` 把推理坐标从
300×300 空间先映射回裁剪图、再平移到全帧；未裁剪时保持整帧等比缩放。
`draw_overlay=1` 时在标注图上画绿色圆 + 黄色裁剪框。

**保存的记录图同样裁剪**：裁剪成功时，`files/`、`original.jpg`、`detect.jpg`
保存的都是以铝片为中心的方形图（先在全帧上画框/圆，再按 `_record_box` 裁出，
缺陷框坐标天然正确）——即"提取的图片中铝片在中间"；未裁剪时保持整帧。

- 坐标从 300×300 缩放回实际帧尺寸（`scale_x = w/300`，`scale_y = h/300`）。
- 标注用 PIL + `Font/platech.ttf` 绘制中文缺陷名（`shared.defect_name` 映射）。
- 库记录：`insert_data(uid, 文件路径, class_name, prediction_time, score)`，同时写
  `detect.jpg`（最新标注图，供 `/get_detect_pic`）。
- 请求超时 30s（`INFERENCE_TIMEOUT`）。

## 报警与抓取触发

| 触发 | 实现 |
|------|------|
| `trigger_alarm` | 打开 `/dev/BAOJING`（`serial.Serial` 9600 8N1），写命令 `7E FF 06 03 00 00 01 EF`；异常时置 None 下次重连 |
| `trigger_grab(flags)` | **本地触发**：`arm_control.enqueue_grab(flags, float(shared.GRAB_DELAY))`（互斥锁串行，忙时丢弃） |

两者均经 `shared.thread_pool.submit()` 异步执行（NG 时先报警后 0.1s 再抓取）。

## 数据库 API（database.py）

线程安全（`threading.Lock`），单连接 `defect.db`。优先使用统一 `query()`：

| 函数 | 说明 |
|------|------|
| `create_database()` | 建 `defect_list` 表（if not exists） |
| `insert_data(uid, path, name, prediction_time, score)` | INSERT（`insert or ignore`，UNIQUE(uuid,path,name)） |
| `query(columns, source, condition, params=())` | 通用 SELECT，返回行列表（新代码统一用它） |
| `query_value(...)` | 聚合快捷（`count()`/`sum()` 等，无结果 None） |

表结构：`defect_list(id INTEGER PK, uuid CHAR, path CHAR(64), name CHAR,
prediction_time CHAR, score CHAR, CreatedTime TIMESTAMP DEFAULT
datetime('now','localtime'), UNIQUE(uuid, path, name))`。

## HTTP API（:8080）

检测类路由（routes.py，Blueprint 'main'）：

| 路由 | 说明 |
|------|------|
| `GET /get_conf` `POST /change_conf` | 读/改运行时可调配置（time/camera_distance），改后同步 shared 缓存 |
| `GET /get_history` | 分页历史检测图（base64） |
| `GET /get_original_pic` `GET /get_detect_pic` | 最近一次原图 / 缺陷标注图 |
| `GET /get_this_month_num` `/get_today_num` `/get_num_by_range` `/get_seven_days_by_type` `/get_statistics` | 统计（当月/当日/按范围/7 天趋势/总体） |
| `POST /calibration` `GET /calibration_status` | 传送带速度标定（`camera_distance/速度` 建议 delay） |
| `GET /img_disc` | MJPEG 实时流（带圆检测叠加：绿圆 + 黄裁剪框，无文字；前端主监控页视频源） |
| `GET /` | 前端监控 SPA（`frontend/index.html`） |

机械臂类路由（arm_control.py，Blueprint 'arm'，与检测类同端口）：

| 路由 | 说明 |
|------|------|
| `POST /use_arm` | 单步手动控制：`{"id": <1-5>, "angle": <度>}` 或 `{"switch": "true"/"false"}` 气泵 |
| `GET /get_arm?id=<1-5>` | 读取指定舵机当前角度 |

CORS：api.py 统一 `CORS(app, supports_credentials=True)`，覆盖全部路由。

## 配置（Python 内联）

不再使用 config.ini——Python 是解释型语言，配置直接写在源码常量里（改动后重启
服务生效）：

- **检测/圆识别参数**：`api.py` 顶部常量区（`BINARIZE_MODE`、`OTSU_MIN_THRESHOLD`、
  `TRIGGER_K`、`BASELINE_INIT_FRAMES`、`TRACKING_TIMEOUT_FRAMES`、
  `LIGHT_CHANGE_THRESHOLD`、`DISC_ENABLED`、`DISC_METHOD`、`DISC_CFG` 等）。
- **运行时可调项**（`/get_conf`/`/change_conf` 读写，无需重启）：`shared.py` 的
  `GRAB_DELAY`（传送带延迟秒，相机到抓取点）、`CAMERA_DISTANCE`（相机到抓取点 mm）。
- **缺陷中文映射**：`shared.py` 的 `defect_name` dict（ca_shang=划痕、
  zhen_kong=针孔、zang_wu=脏污、zhe_zhou=褶皱、zheng_chang=正常）。

## 运行

```bash
# 手动
cd /opt/HaoYao/main && bash run.sh
# run.sh：export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64; MVCAM_COMMON_RUNENV=/opt/MVS/lib;
#         CRYPTOGRAPHY_ALLOW_OPENSSL_102=1; python3 api.py

# systemd（开机自启，Restart=always；唯一单元）
systemctl restart detect-api.service
journalctl -u detect-api.service -n 50 --no-pager -l
tail -f /var/logs/server.log
```

> 日志由 `logger.py`（main/logger.py）写入 `/var/logs/server.log`（10MB 自动截半）。
> 监控页地址：`http://172.16.68.111:8080/`（前端 SPA + API + 视频流同端口）。

## 故障排查：铝片经过但网页无推理图

视频流（`/img_disc`）由 Producer 线程更新，与触发/推理链路（Consumer 线程）独立——
视频正常 ≠ 推理正常。按以下顺序定位断点：

1. 日志定位：`tail -n 200 /var/logs/server.log`，找
   `触发进入 TRACKING` / `触发回落` / `开始检测` / `推理服务耗费` /
   `consumer thread error` 各出现在哪一层：
   - 无 `触发进入 TRACKING` → **触发未发生**：空皮带暖机未完成（需 ≥30 帧空皮带）
     或前景信号不达标（调 `api.py` 的 `TRIGGER_K`/`OTSU_MIN_THRESHOLD`；铝片需
     **在皮带上移动**通过视野，静止铝片会卡在 TRACKING 不回落）。
   - 有 `触发进入 TRACKING` 但无 `触发回落` → **卡在 TRACKING**：退出条件（前景
     回落到基线）未满足，铝片未离开视野/皮带未动；超时保护
     `TRACKING_TIMEOUT_FRAMES` 会兜底强制推理。
   - 有 `开始检测` 但无图片 → 推理/存图失败：看 `consumer thread error` 与日志
     （FPGA 不可达会卡 30s 超时）。

> 机械臂排查提示：机械臂不动先确认 `/dev/XIPAN` 是否存在、`Arm_Lib` 是否安装；
> 抓取动作错误多数是**角度未标定**所致，与吸盘时序无关。

## 已知问题（当前代码状态）

1. **`/dev/BAOJING` 硬件类型未确认** —— 代码按串口打开（`serial.Serial` 9600 8N1）。
   若实际是 `/dev/input/event0` 输入事件设备，串口写入将无效果。**未修复**。
2. 历史/统计路由中若 `defect_list` 表未建（首次启动竞态）会查询报错——现已在
   `__main__` 中先 `create_database()` 再启动 cleanup/Consumer，**已修复**。
3. `_image_to_base64` 对文件已被 cleanup 删除的情况捕获异常返回空串（前端得到空
   img）——**已缓解**（不再 500，但仍有空图）。
4. GigE 设备名解析已按 `p != 0` 过滤 null 字节（camera.py `_log_device_info`）。
5. **机械臂角度是占位值**，必须现场标定（见上文"舵机角度预设"）。
6. 机械臂硬件（`/dev/XIPAN` + `Arm_Lib`）在服务启动时即初始化——机械臂未接/异常
   会导致整个服务起不来。
