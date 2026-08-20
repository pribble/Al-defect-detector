"""
机械臂控制模块 — 主服务 (main/) 中负责机械臂分拣的部分.

  - grab_task(): OK/NG 分拣动作序列 (互斥锁串行执行, 防并发冲突)
  - HTTP API: /use_arm /get_arm (Blueprint 'arm', 由 api.py 挂载到同一 app)
  - enqueue_grab(): 本地触发接口, 供 api.py 的检测结果直接调用; 忙时丢弃不排队

依赖: Arm_Lib, pyserial
"""

import json
import threading
import time

from Arm_Lib import Arm_Device
from flask import Blueprint, request
import serial

from logger import setup_log

logger = setup_log('main', 'server.log')

# ============================================================
# 常量
# ============================================================

# --- 吸盘时序 (秒) ---
SUCTION_HOLD = 0.6  # 吸取后保持, 确保真空建立
RELEASE_HOLD = 0.4  # 释放后保持, 确保气压释放完毕
LIFT_PAUSE = 0.3  # 放置后抬起前短暂停顿, 避免带偏工件
POST_RELEASE = 0.3  # 释放后等待吸盘完全脱离

# 初始位: 竖直收起, 不遮挡相机视野, 不干涉传送带
HOME = [90, 50, 40, 0]

# 吸取位: 基座居中, 末端到达正前方 200mm 传送带面
PICKUP = [90, 30, 62, 0]

# 吸取后抬高: 从吸取点垂直抬起, 基座保持居中, 留出旋转空间
PICKUP_LIFT = [90, 50, 40, 0]

# OK 放置位上方: 基座右转, 末端在 OK 区正上方 (安全高度)
OK_ABOVE = [0, 50, 40, 0]

# OK 放置位: 基座右转, 末端到达右侧区域中心 (+60, 0) 传送带面
OK_PLACE = [0, 40, 55, 0]

# NG 放置位上方: 基座左转, 末端在 NG 区正上方 (安全高度)
NG_ABOVE = [180, 50, 40, 0]

# NG 放置位: 基座左转, 末端到达左侧区域中心 (-60, 0) 传送带面
NG_PLACE = [180, 40, 55, 0]

# --- 串口 ---
SERIAL_PORT = '/dev/XIPAN'

# 气泵控制指令 (Modbus-RTU)
GRIP_ON = bytes.fromhex('A0 01 01 A2')  # 继电器吸合 → 气泵吸气
GRIP_OFF = bytes.fromhex('A0 01 00 A1')  # 继电器断开 → 气泵释放

FAST_SERVO_DELAY = 0.01  # 普通舵机间延迟 (秒)

# ============================================================
# 初始化
# ============================================================

try:
    pump_serial = serial.Serial(SERIAL_PORT, 9600)
except Exception as e:
    pump_serial = None
    logger.error(f"吸盘串口初始化失败：{e}", exc_info=True)
arm_device = Arm_Device()
time.sleep(0.1)

# 机械臂动作互斥锁: 同一时刻只执行一次分拣; 忙时新信号直接丢弃 (不缓存)
_grab_lock = threading.Lock()

bp = Blueprint('arm', __name__)


# ============================================================
# 气泵控制
# ============================================================

def _pump(status: bool):
    """吸合/释放气泵继电器 (写串口)"""
    if pump_serial is None:
        return
    command = GRIP_ON if status else GRIP_OFF
    try:
        pump_serial.write(command)
    except Exception as e:
        logger.error("%s错误：%s", "吸取" if status else "释放", e, exc_info=True)


def suction_on():
    """吸盘吸取 — 打开真空气泵"""
    _pump(True)
    time.sleep(SUCTION_HOLD)


def suction_off():
    """吸盘释放 — 关闭真空气泵"""
    _pump(False)
    time.sleep(RELEASE_HOLD)


# ============================================================
# 机械臂运动
# ============================================================

def arm_move(angles: list, move_time: int = 500):
    """控制 4 个舵机同时运动到目标角度 (angles=[ID1..ID4] 度, move_time=ms)."""
    for i in range(4):
        servo_id = i + 1
        arm_device.Arm_serial_servo_write(servo_id, angles[i], move_time)
        time.sleep(FAST_SERVO_DELAY)
    time.sleep(move_time / 1000 + 0.05)


# ============================================================
# 核心抓取流程
# ============================================================

def grab_task(data: dict):
    """
    单次分拣动作序列:

      1. 等待传送延迟 (铝片从相机走到吸取点)
      2. HOME → 吸取点 (正前方 200mm)
      3. 吸盘吸取 (保持 SUCTION_HOLD 秒)
      4. 抬起至安全高度
      5. 平移至 OK/NG 区域上方
      6. 下降至放置位
      7. 吸盘释放 (保持 RELEASE_HOLD 秒)
      8. 抬起 → 回到 HOME
    """
    logger.info("分拣开始, flags=%s", data.get('flags'))

    delay = float(data.get('time', 0))  # 传送带延迟 (秒)
    flags = data['flags']  # "OK" 或 "NG"

    # --- 步骤0: 等待铝片从相机位置行进到吸取点 ---
    if delay > 0:
        logger.info("等待传送延迟: %d 秒", delay)
        time.sleep(delay)

    # --- 步骤1: HOME → 吸取点 ---
    logger.info("→ 移动到吸取点")
    arm_move(HOME, 250)
    arm_move(PICKUP, 250)

    # --- 步骤2: 吸取铝片 ---
    logger.info("→ 吸盘吸取")
    suction_on()
    time.sleep(0.15)  # 确认吸附稳定

    # --- 步骤3: 抬起至安全高度 ---
    logger.info("→ 抬起至运输高度")
    arm_move(PICKUP_LIFT, 250)

    # --- 步骤4: 平移 → 放置 ---
    if flags == "OK":
        logger.info("→ 移至 OK 区")
        arm_move(OK_ABOVE, 500)
        arm_move(OK_PLACE, 250)
    else:
        logger.info("→ 移至 NG 区")
        arm_move(NG_ABOVE, 500)
        arm_move(NG_PLACE, 250)

    # --- 步骤5: 释放吸盘 ---
    logger.info("→ 吸盘释放")
    suction_off()
    time.sleep(POST_RELEASE)

    # --- 步骤6: 抬起离开放置区, 避免复位时碰撞工件 ---
    logger.info("→ 抬起离开放置区")
    if flags == "OK":
        arm_move(OK_ABOVE, 250)
    else:
        arm_move(NG_ABOVE, 250)
    time.sleep(LIFT_PAUSE)

    # --- 步骤7: 复位 ---
    logger.info("→ 复位 HOME")
    arm_move(HOME, 500)
    time.sleep(0.3)
    logger.info("分拣完成")


# ============================================================
# 本地触发接口 (检测服务直接调用)
# ============================================================

def enqueue_grab(flags: str, delay: float = 0):
    """触发一次分拣 — 直接执行; 机械臂忙时丢弃本次信号 (不缓存/不排队).

    Args:
        flags: "OK" 或 "NG"
        delay: 传送带延迟秒数 (相机到吸取点), 即 shared.GRAB_DELAY
    """
    if not _grab_lock.acquire(blocking=False):
        logger.warning('机械臂忙, 丢弃本次分拣信号 (flags=%s)', flags)
        return
    try:
        grab_task({"flags": flags, "time": delay})
    except Exception as e:
        logger.error("抓取任务异常: %s", e)
    finally:
        _grab_lock.release()


# ============================================================
# HTTP 路由 (Blueprint 'arm')
# ============================================================

@bp.route('/use_arm', methods=['POST'])
def use_arm():
    """单步手动控制: 指定舵机 ID/角度, 或控制气泵开关"""
    data = json.loads(request.get_data(as_text=True))

    servo_id = data.get('id', '')
    servo_angle = data.get('angle', '')
    if servo_id and servo_angle:
        arm_device.Arm_serial_servo_write(int(servo_id), int(servo_angle), 1000)

    suction_switch = data.get('switch', '')
    if suction_switch:
        if suction_switch == "true":
            suction_on()
        else:
            suction_off()

    return data


@bp.route('/get_arm', methods=['GET'])
def get_arm():
    """读取指定舵机的当前角度"""
    servo_id = request.args.get("id")
    angle = arm_device.Arm_serial_servo_read(int(servo_id))
    return str(angle)
