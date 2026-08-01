# ArmControl/ — 机械臂控制服务（Flask :8899）

> 导航：上一级 [raspberryPi5/](../README.md) · 相关 [GrabImage/](../GrabImage/README.md)

5-DOF Yahboom 机械臂 + 气动吸盘的 OK/NG 分拣控制服务。接收 GrabImage 的
`POST /grab` 请求，按队列**串行**执行完整分拣动作序列，防止并发机械臂运动冲突；
也向前端提供单步手动控制与角度读取接口。

## 目录结构

| 文件 | 说明 |
|------|------|
| `api.py` | Flask 应用 + grab_task 动作序列 + GrabTaskConsumer 串行队列 |
| `api.service` | systemd 单元 |
| `run.sh` | 启动 api.py |

依赖：`Arm_Lib`（Yahboom 官方机械臂库，随机械臂提供，不在本仓库）、`pyserial`。

## 硬件接口

| 设备 | 接口 | 说明 |
|------|------|------|
| 机械臂 | `/dev/XIPAN`（ttyUSB0，9600 baud） | `Arm_Device`（Arm_Lib）写舵机角度 |
| 气泵继电器 | `/dev/XIPAN`（同一串口，9600） | Modbus-RTU 指令控制吸气/释放 |

气泵指令（api.py）：

```python
GRIP_ON  = bytes.fromhex('A0 01 01 A2')   # 继电器吸合 → 吸气
GRIP_OFF = bytes.fromhex('A0 01 00 A1')   # 继电器断开 → 释放
```

吸盘时序（秒）：`SUCTION_HOLD=0.6`（吸取保持）、`RELEASE_HOLD=0.4`（释放保持）、
`LIFT_PAUSE=0.3`、`POST_RELEASE=0.3`。

## 舵机角度预设（占位值 — 必须现场标定！）

`api.py` 顶部的角度常量是 **placeholder**，与实际物理机械臂的尺寸/安装位置无关，
**上线前必须逐点标定**：

| 常量 | 用途 | 格式 |
|------|------|------|
| `HOME` | 初始位（竖直收起，不遮挡相机、不干涉传送带） | `[基座, 肩, 肘, 腕, 末端]` 5 个角度（度） |
| `PICKUP` | 吸取位：正前方传送带面 200mm | `[90, 30, 62, 0, 90]` ← 需标定 |
| `PICKUP_LIFT` | 吸取后抬高，留旋转空间 | 需标定 |
| `OK_ABOVE` / `OK_PLACE` | OK 区上方 / OK 区放置位（基座右转） | 需标定 |
| `NG_ABOVE` / `NG_PLACE` | NG 区上方 / NG 区放置位（基座左转） | 需标定 |

`arm_move(angles, move_time)`：5 个舵机顺序写角度（ID5 末端舵机较慢，延迟与
1.2 倍时间补偿），随后 sleep `move_time/1000` 等待到位。

## grab_task 动作序列（单次分拣）

```
0. 等待传送延迟 time 秒（铝片从相机走到吸取点；time 即 GrabImage 的 GRAB_DELAY）
1. HOME → PICKUP（各 250ms）
2. 吸盘吸取（suction_on + 0.15s 稳定）
3. 抬起 PICKUP_LIFT（250ms）
4. 平移：OK → OK_ABOVE → OK_PLACE / NG → NG_ABOVE → NG_PLACE
5. 吸盘释放（suction_off + POST_RELEASE）
6. 抬起离开放置区（250ms + LIFT_PAUSE，避免复位碰撞工件）
7. 复位 HOME（500ms）
```

`_write_pump` 对串口写入异常自动重连（重新实例化 `serial.Serial`）。

## 任务队列：单槽缓存 + 串行消费者

- `POST /grab` 只把请求文本存入**单槽缓存** `_grab_pending`（新任务覆写旧缓存，
  最多缓存 1 个待执行任务），并 set `_grab_ready` Event —— 立即返回，不阻塞调用方。
- `GrabTaskConsumer`（daemon 线程）串行执行：等 Event → 取走缓存 → `clear()` →
  **竞态恢复**（clear 前若有新任务写入则重新 set）→ `grab_task(json)` → 循环。
- 效果：即使 GrabImage 同时来多个请求，机械臂也**一次只动一个**，动作不交叠。

## HTTP API

| 路由 | 说明 |
|------|------|
| `POST /grab` | 入队抓取任务，body `{"flags": "OK"/"NG", "time": <传送延迟秒>}`（GrabImage 调用） |
| `POST /use_arm` | 单步手动控制：`{"id": <1-5>, "angle": <度>}` 或 `{"switch": "true"/"false"}` 气泵 |
| `GET /get_arm?id=<1-5>` | 读取指定舵机当前角度 |

CORS：允许 `*`（`Content-Type,Authorization` 头、GET/POST）。

## 运行

```bash
# 手动
cd /opt/HaoYao/ArmControl && bash run.sh      # python3 api.py

# systemd（开机自启）
systemctl restart api.service
journalctl -u api.service -n 50 --no-pager -l
tail -f /var/logs/api_server.log
```

> 排查提示：机械臂不动先确认 `/dev/XIPAN` 是否存在、`Arm_Lib` 是否安装；
> 抓取动作错误多数是**角度未标定**所致，与吸盘时序无关。
