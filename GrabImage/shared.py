"""
共享可变状态 — api.py 与 routes.py 之间通过此模块传递运行时状态.

避免因 __main__ vs import api 导致的两份独立模块副本问题.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import configparser

from database import FrameBuffer

# 配置 (由 api.py 在启动时初始化)
config = configparser.ConfigParser()
defect_name = {}
logger = None

# 图片路径 (由 api.py 在启动时设置)
original_image_path = None
detect_image_path = None

# 机械臂控制参数缓存 (路由 /change_conf 可更新)
GRAB_SPEED = None
GRAB_DELAY = None

# 标定模式状态 (路由 /calibration 控制, Consumer 读写)
calibration_active = False
calibration_samples = deque(maxlen=30)

# 全局运行时状态
capture_started = []
frame_queue = FrameBuffer()
stream_image_ref = [np.zeros((512, 512, 3), dtype=np.uint8)]
thread_pool = ThreadPoolExecutor()
