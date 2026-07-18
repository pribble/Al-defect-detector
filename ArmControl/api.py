import time
import serial
import json
import os
import sys
from Arm_Lib import Arm_Device
from flask import Flask, request
from queue import Queue
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from logger import setup_log

logger = setup_log("arm", "api_server.log")

# ============================================================
# 物理布局 (俯视图, 机械臂基座中心为 XY 原点)
# ============================================================
#
#          NG区 (左侧)                    OK区 (右侧)
#     ┌──────────────────┐          ┌──────────────────┐
#     │   150 × 120 mm   │          │   150 × 120 mm   │
#     │ 长边 ⊥ 机械臂朝向 │          │ 长边 ⊥ 机械臂朝向 │
#     │ 中心 (-60, 0)    │          │ 中心 (+60, 0)    │
#     └────────┬─────────┘          └────────┬─────────┘
#              │                             │
#              └─────────────┬───────────────┘
#                            │
#                       ┌────┴────┐
#                       │ 机械臂   │
#                       │  基座    │
#                       └────┬────┘
#                            │ Y+
#                       ┌────┴────┐
#                       │ 吸取点   │
#                       │ (0, 200) │
#                       └─────────┘
#
# Z=0 参考面 = 传送带 / 工作台面
# 舵机 ID 对应 (以幻尔 5-DOF 总线舵机臂为例):
#   ID 1: 基座旋转 (yaw)
#   ID 2: 肩部俯仰 (pitch)
#   ID 3: 肘部俯仰 (pitch)
#   ID 4: 腕部俯仰 (pitch)
#   ID 5: 末端旋转 (控制吸盘朝向)
# ============================================================

# ---- 吸盘保持时间 (秒) ----
SUCTION_HOLD  = 0.6   # 吸取后保持, 确保真空建立
RELEASE_HOLD  = 0.4   # 释放后保持, 确保气压释放完毕
LIFT_PAUSE    = 0.3   # 放置后抬起前短暂停顿, 避免带偏工件
POST_RELEASE  = 0.3   # 释放后等待吸盘完全脱离

# ---- 舵机角度预设 (以下为占位值, 必须现场标定) ----
# 数据格式: [ID1_基座, ID2_肩, ID3_肘, ID4_腕, ID5_末端]

# 初始位: 竖直收起, 不遮挡相机视野, 不干涉传送带
HOME = [90, 90, 90, 0, 90]

# 吸取位: 基座居中, 末端到达正前方 200mm 传送带面
PICKUP = [90, 38, 26, 22, 90]       # ← 需标定

# 吸取后抬高: 从吸取点垂直抬起, 基座保持居中, 留出旋转空间
PICKUP_LIFT = [90, 90, 0, 90, 90]

# OK 放置位上方: 基座右转, 末端在 OK 区正上方 (安全高度)
OK_ABOVE = [-10, 90, 90, 0, 90]     # ← 需标定

# OK 放置位: 基座右转, 末端到达右侧区域中心(+60, 0)传送带面
OK_PLACE = [-10, 65, 0, 30, 90]     # ← 需标定

# NG 放置位上方: 基座左转, 末端在 NG 区正上方 (安全高度)
NG_ABOVE = [196, 90, 90, 0, 90]     # ← 需标定

# NG 放置位: 基座左转, 末端到达左侧区域中心(-60, 0)传送带面
NG_PLACE = [196, 55, 11, 24, 90]    # ← 需标定

# ============================================================

# 气泵继电器串口
pump_serial = serial.Serial('/dev/XIPAN', 9600)
# 气泵抓取 / 释放指令 (Modbus-RTU)
GRIP_ON  = bytes.fromhex('A0 01 01 A2')    # 继电器吸合 → 气泵吸气
GRIP_OFF = bytes.fromhex('A0 01 00 A1')    # 继电器断开 → 气泵释放

# 创建机械臂对象
arm_device = Arm_Device()
time.sleep(.1)

app = Flask(__name__)
task_queue = Queue(maxsize=0)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


# ---- 气泵控制 (带异常恢复) ----

def _write_pump(command, label):
    """向气泵继电器写入指令, 异常时自动重连"""
    global pump_serial
    try:
        pump_serial.write(command)
    except Exception as e:
        logger.error('%s 错误：%s', label, e, exc_info=True)
        pump_serial = serial.Serial('/dev/XIPAN', 9600)
        logger.info("重新实例化气泵串口")
        pump_serial.write(command)


def suction_on():
    """吸盘吸取 — 打开真空气泵"""
    _write_pump(GRIP_ON, "吸取")
    time.sleep(SUCTION_HOLD)


def suction_off():
    """吸盘释放 — 关闭真空气泵"""
    _write_pump(GRIP_OFF, "释放")
    time.sleep(RELEASE_HOLD)


# ---- 机械臂运动 ----

def arm_move(angles, move_time=500):
    """
    控制 5 个舵机同时运动到目标角度.
    angles: [id1, id2, id3, id4, id5] 单位: 度
    move_time: 运动时间, 单位 ms
    """
    for i in range(5):
        servo_id = i + 1
        if servo_id == 5:
            # ID=5 (末端旋转) 运动稍慢, 补偿 1.2x
            time.sleep(.1)
            arm_device.Arm_serial_servo_write(servo_id, angles[i], int(move_time * 1.2))
        else:
            arm_device.Arm_serial_servo_write(servo_id, angles[i], move_time)
            time.sleep(.01)
    time.sleep(move_time / 1000)


# ---- 单步手动控制接口 ----

@app.route('/use_arm', methods=['POST'])
def use_arm():
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


@app.route('/get_arm', methods=['GET'])
def get_arm():
    servo_id = request.args.get("id")
    angle = arm_device.Arm_serial_servo_read(int(servo_id))
    return str(angle)


# ---- 核心抓取流程 ----

def grab_task(data):
    """
    单次分拣动作序列:
      1. 等待传送延迟 (铝片从相机走到吸取点)
      2. HOME → 移至吸取点 (正前方 200mm)
      3. 吸盘吸取 (保持 SUCTION_HOLD 秒)
      4. 抬起至安全高度
      5. 平移至 OK/NG 区域上方
      6. 下降至放置位
      7. 吸盘释放 (保持 RELEASE_HOLD 秒)
      8. 抬起 → 回到 HOME
    """
    logger.info("分拣开始, flags=%s", data.get('flags'))

    speed = int(data['speed'])          # 舵机运动速度 (ms)
    delay = int(data.get('time', 0))    # 传送带延迟 (秒)
    flags = data['flags']               # "OK" 或 "NG"

    # --- 步骤0: 等待铝片从相机位置行进到吸取点 ---
    if delay > 0:
        logger.info("等待传送延迟: %d 秒", delay)
        time.sleep(delay)

    # --- 步骤1: 从 HOME 移动到吸取点 ---
    logger.info("→ 移动到吸取点 (正前方 200mm)")
    arm_move(HOME, speed)
    arm_move(PICKUP, speed)

    # --- 步骤2: 吸取铝片 ---
    logger.info("→ 吸盘吸取")
    suction_on()
    # 吸取后短暂确认吸附稳定
    time.sleep(0.15)

    # --- 步骤3: 抬起至安全高度 ---
    logger.info("→ 抬起至运输高度")
    arm_move(PICKUP_LIFT, speed)

    # --- 步骤4: 平移到目标区域上方, 然后下降到放置位 ---
    if flags == "OK":
        logger.info("→ 移至 OK 区 (右侧 +60mm)")
        arm_move(OK_ABOVE, speed)
        arm_move(OK_PLACE, speed)
    else:
        logger.info("→ 移至 NG 区 (左侧 -60mm)")
        arm_move(NG_ABOVE, speed)
        arm_move(NG_PLACE, speed)

    # --- 步骤5: 释放吸盘 ---
    logger.info("→ 吸盘释放")
    suction_off()
    time.sleep(POST_RELEASE)

    # --- 步骤6: 抬起, 避免复位时碰撞工件 ---
    logger.info("→ 抬起离开放置区")
    if flags == "OK":
        arm_move(OK_ABOVE, speed)
    else:
        arm_move(NG_ABOVE, speed)
    time.sleep(LIFT_PAUSE)

    # --- 步骤7: 复位 ---
    logger.info("→ 复位 HOME")
    arm_move(HOME, 1000)
    time.sleep(0.3)
    logger.info("分拣完成")


@app.route('/grab', methods=['POST'])
def grab():
    data = request.get_data(as_text=True)
    task_queue.put(data)
    return data


class GrabTaskConsumer(threading.Thread):
    """队列消费者: 串行执行抓取任务, 防止并发冲突"""
    def run(self):
        while True:
            try:
                raw_data = task_queue.get()
                logger.info("收到抓取请求: %s", raw_data)
                grab_task(json.loads(raw_data))
            except Exception as e:
                logger.error("抓取任务异常: %s", str(e))


_consumer = GrabTaskConsumer()
_consumer.setDaemon(True)
_consumer.start()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8899)
