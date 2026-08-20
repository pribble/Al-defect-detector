"""
共享可变状态 — api.py 与 routes.py 之间通过此模块传递运行时状态.

避免因 __main__ vs import api 导致的两份独立模块副本问题.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# 缺陷中文映射 (原 config.ini [defect_name], 现改为 Python 直写)
defect_name = {
    'ca_shang': '划痕',
    'zhen_kong': '针孔',
    'zang_wu': '脏污',
    'zhe_zhou': '褶皱',
    'zheng_chang': '正常',
}

logger = None

# 图片路径 (由 api.py 在启动时设置)
original_image_path = None
detect_image_path = None

# 运行时可调配置 (原 config.ini [Configuration], 由路由 /change_conf 更新)
GRAB_DELAY = 2          # 传送带延迟 (秒) — 相机到抓取点的传输时间
CAMERA_DISTANCE = 200   # 相机到抓取点的距离 (mm), 标定模式计算建议 delay 用

# 标定模式状态 (路由 /calibration 控制, Consumer 读写)
calibration_active = False
calibration_samples = deque(maxlen=30)

# 全局运行时状态
stream_image_ref = [np.zeros((512, 512, 3), dtype=np.uint8)]
thread_pool = ThreadPoolExecutor()

# 圆识别 + 智能裁剪共享状态 (api.py 启动时同步, 供 /img_disc 视频流叠加)
last_disc = None            # (cx, cy, r) 最近一次检出的铝片圆
last_crop_box = None        # (x0, y0, side) 最近一次智能裁剪框 (全帧坐标)
disc_method = "mask"        # 定圆方法: mask | hough
disc_cfg = None             # [disc] 段参数字典 (api.py 启动时构建)
disc_margin_ratio = 0.10    # 裁剪黑边比例
debug_background = None     # Consumer 的 EMA 背景模型 (float32 低分辨率, 供背景差分定圆)
debug_mask = None           # 软处理前景掩码 (48×64, 供选帧打分)
