"""
缺陷检测服务 (Flask :7777)

通过 Hikvision 工业相机实时采集图像, 调用 FPGA 推理服务进行缺陷检测,
根据检测结果触发机械臂分拣 (OK/NG) 和蜂鸣器报警。

依赖: MVS SDK (MvImport), opencv-python, scikit-image (SSIM), pyserial
"""

import json
import os
import sys
import threading
import time
import uuid

import cv2
import numpy as np
import requests
import serial
from flask import Flask
from flask_cors import CORS
from PIL import Image, ImageFont, ImageDraw
from skimage.metrics import structural_similarity as compare_ssim

sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from logger import setup_log

from camera import frame_queue, Producer
import database
import shared

# ============================================================
# 路径常量
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, 'files')
ORIGINAL_DIR = os.path.join(BASE_DIR, 'original_files')
DETECT_DIR = os.path.join(BASE_DIR, 'detect_files')

for _dir in [FILES_DIR, DETECT_DIR, ORIGINAL_DIR]:
    os.makedirs(_dir, exist_ok=True)

original_image_path = os.path.join(ORIGINAL_DIR, 'original.jpg')
detect_image_path = os.path.join(DETECT_DIR, 'detect.jpg')
shared.original_image_path = original_image_path
shared.detect_image_path = detect_image_path

# ============================================================
# 日志与配置
# ============================================================

shared.logger = setup_log('detect', 'detect_server.log')
logger = shared.logger

shared.config.read('config.ini', encoding='utf-8')
shared.defect_name = dict(shared.config.items('defect_name'))

# ============================================================
# 常量
# ============================================================

FPGA_URL = 'http://172.16.68.110:8080/predict'
ARM_URL = 'http://172.16.68.111:8899/grab'
INFERENCE_TIMEOUT = 30

ALARM_PORT = "/dev/BAOJING"
ALARM_BAUD = 9600
ALARM_DATA_BITS = 8
ALARM_STOP_BITS = 1
ALARM_CMD = bytes.fromhex('7E FF 06 03 00 00 01 EF')

SAVE_IMAGE_NUM = 10
SSIM_WIDTH = 64
SSIM_HEIGHT = 48
REFERENCE_IMAGE = 'yuanshi.jpg'

SSIM_TRIGGER_THRESHOLD = 0.9
WHITE_RATIO_THRESHOLD = 0.1
STABILITY_STD_THRESHOLD = 0.01
STABILITY_MEAN_THRESHOLD = 0.8
SSIM_HISTORY_SIZE = 9
FRAME_SKIP_COUNT = 10
TRIGGER_COOLDOWN = 5  # 触发后冷却帧数, 防同一铝片重复触

LABEL_NORMAL = 'zheng_chang'
DEFECT_LABELS = ['ca_shang', 'zang_wu', 'zhe_zhou', 'zhen_kong']

GAUSSIAN_KERNEL = int(shared.config.get("Configuration", "gaussian_kernel", fallback="21"))
BINARY_THRESHOLD_1 = 100
DILATE_ITERATIONS = 4

# 机械臂控制参数缓存 (路由 /change_conf 同步更新)
GRAB_SPEED = shared.config.get("Configuration", "speed")
GRAB_DELAY = shared.config.get("Configuration", "time")
shared.GRAB_SPEED = GRAB_SPEED
shared.GRAB_DELAY = GRAB_DELAY

# ============================================================
# 全局状态
# ============================================================

alarm_serial = None

app = Flask(__name__)
CORS(app, supports_credentials=True)


# ============================================================
# 报警
# ============================================================

def trigger_alarm():
    """触发蜂鸣器报警 (异步调用, 异常时重连串口)"""
    global alarm_serial
    if alarm_serial is None:
        alarm_serial = serial.Serial(ALARM_PORT, ALARM_BAUD, ALARM_DATA_BITS, stopbits=ALARM_STOP_BITS)
    try:
        alarm_serial.write(ALARM_CMD)
    except Exception as e:
        logger.error('alarm error：%s', e)
        alarm_serial = None


# ============================================================
# 触发机械臂抓取
# ============================================================

def trigger_grab(flags: str):
    """发送 HTTP 请求至 ArmControl 服务, 触发机械臂分拣"""
    data = {"flags": flags, "speed": shared.GRAB_SPEED, "time": shared.GRAB_DELAY}
    requests.post(ARM_URL, json=data, timeout=INFERENCE_TIMEOUT)


# ============================================================
# SSIM 图像比较
# ============================================================

# 参考图内存缓存 (避免每帧读磁盘)
_cached_ref_gray = None


def compare_image(image_gray) -> float:
    """将当前帧与参考图 (yuanshi.jpg) 做 SSIM 比较, 返回相似度 (1=相同)."""
    global _cached_ref_gray
    if _cached_ref_gray is None:
        ref = cv2.imread(REFERENCE_IMAGE)
        ref = cv2.resize(ref, (SSIM_WIDTH, SSIM_HEIGHT), interpolation=cv2.INTER_AREA)
        _cached_ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    score = compare_ssim(_cached_ref_gray, image_gray)
    logger.debug("SSIM: {}".format(score))
    return score


# ============================================================
# FPS 标注
# ============================================================

def _draw_fps_label(image, fps: float):
    """在图片左上角绘制帧率标注"""
    label = "(Capture) {:.1f} FPS".format(fps)
    cv2.putText(image, label, (180, 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (38, 0, 255), 1)
    return image


# ============================================================
# 图片消费者 (检测 + 推理 + 分拣)
# ============================================================

class Consumer(threading.Thread):
    """
    从 frame_queue 消费帧, 执行 SSIM 触发检测、FPGA 推理、结果处理.
    每 FRAME_SKIP_COUNT 帧处理一次, 其余帧直接丢弃以平衡负载.
    """

    def __init__(self):
        super().__init__()
        self._diff_3ago = 0
        self._diff_2ago = 0
        self._diff_1ago = 0
        self._ssim_history = [0] * SSIM_HISTORY_SIZE
        self._recent_frames = [None, None]
        # 标定模式: 追踪 white_ratio 变化以计算传送带速度
        self._trigger_cooldown = 0
        self._cal_last_ratio = 0.0
        self._cal_entry_time = None
        self._cal_peak_ratio = 0.0

    def run(self):
        database.create_database()

        while True:
            time.sleep(0.01)
            try:
                frame_start_time = time.time()
                self._process_sampling_frame(frame_start_time)

            except Exception as e:
                logger.error('consumer thread error：{}'.format(str(e)))

    # ---- 单帧处理 ----

    def _process_sampling_frame(self, frame_start_time: float):
        global _cached_ref_gray
        image = frame_queue.get()
        if image is None:
            return
        self._recent_frames[0] = self._recent_frames[1]
        self._recent_frames[1] = image

        black_image = cv2.resize(image, (SSIM_WIDTH, SSIM_HEIGHT), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(black_image, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        _, img_binary = cv2.threshold(blurred, BINARY_THRESHOLD_1, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(img_binary, None, iterations=DILATE_ITERATIONS)

        white_ratio = np.count_nonzero(thresh) / thresh.size
        current_ssim = compare_image(black_image)

        # --- 标定模式: 追踪 white_ratio 变化以测量传送带速度 ---
        if shared.calibration_active:
            now = time.time()
            if white_ratio > 0.02:
                # 铝片开始进入: 记录进入时间和峰值
                if self._cal_last_ratio < 0.02:
                    self._cal_entry_time = now
                    self._cal_peak_ratio = white_ratio
                if white_ratio > self._cal_peak_ratio:
                    self._cal_peak_ratio = white_ratio
                # 铝片开始退出 (ratio 从峰值下降 > 10%)
                if (self._cal_entry_time is not None and
                        white_ratio < self._cal_peak_ratio * 0.85):
                    duration = now - self._cal_entry_time
                    if 0.1 < duration < 30:  # 合理性检查: 0.1s ~ 30s
                        speed = 100.0 / duration  # 100mm 直径 / 进入时间 = mm/s
                        shared.calibration_samples.append(speed)
                        logger.info('标定: 测得速度 {:.1f} mm/s (耗时 {:.2f}s)'.format(speed, duration))
                    self._cal_entry_time = None
            else:
                self._cal_entry_time = None
            self._cal_last_ratio = white_ratio

        self._ssim_history[:-1] = self._ssim_history[1:]
        self._ssim_history[-1] = current_ssim
        ssim_std = np.std(self._ssim_history)
        ssim_mean = np.mean(self._ssim_history)

        if ssim_std < STABILITY_STD_THRESHOLD and ssim_mean < STABILITY_MEAN_THRESHOLD and all(self._ssim_history):
            cv2.imwrite(REFERENCE_IMAGE, image)
            _cached_ref_gray = None  # 缓存失效, 下次重载

        diff_curr = current_ssim - self._diff_1ago
        diff_prev = self._diff_1ago - self._diff_2ago
        diff_prev2 = self._diff_2ago - self._diff_3ago

        if self._trigger_cooldown > 0:
            self._trigger_cooldown -= 1
        elif current_ssim < SSIM_TRIGGER_THRESHOLD:
            logger.info(
                'current_ssim:{},diff_1ago:{},diff_2ago:{},diff_3ago:{},white_ratio:{}'.format(
                    current_ssim, self._diff_1ago, self._diff_2ago, self._diff_3ago, white_ratio
                )
            )

            if diff_curr > 0 > diff_prev2 and diff_prev < 0 and white_ratio > WHITE_RATIO_THRESHOLD:
                self._trigger_cooldown = TRIGGER_COOLDOWN
                self._run_inference_pipeline(frame_start_time)

        self._diff_3ago = self._diff_2ago
        self._diff_2ago = self._diff_1ago
        self._diff_1ago = current_ssim

    # ---- 推理管线 ----

    def _run_inference_pipeline(self, frame_start_time: float):
        annotated_image = self._recent_frames[0]
        if annotated_image is None:
            annotated_image = self._recent_frames[1]
        original_image = annotated_image.copy()

        uid = str(uuid.uuid1())
        file_name = os.path.join(FILES_DIR, '{}.jpg'.format(uid))

        # Resize to 300×300 for FPGA inference (SSD MobileNet input size)
        inference_image = cv2.resize(annotated_image, (300, 300))
        # Send raw grayscale pixels — no JPEG encode/decode overhead
        image_bytes = inference_image.tobytes()

        inference_start_time = time.time()
        logger.info('开始检测')
        response = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=INFERENCE_TIMEOUT)
        logger.info('推理服务耗费：{}'.format(time.time() - inference_start_time))

        processing_rate = FRAME_SKIP_COUNT / (time.time() - frame_start_time)
        inference_result = json.loads(response.text)
        logger.info(inference_result)

        if inference_result['len'] > 0:
            self._handle_inference_result(
                inference_result, uid, file_name, annotated_image, original_image,
                processing_rate,
            )
        else:
            shared.thread_pool.submit(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image, processing_rate)

    def _handle_inference_result(
            self, inference_result, uid, file_name, annotated_image, original_image, processing_rate
    ):
        action = inference_result.get('action', 'OK')

        if action == "NG":
            self._handle_defect(inference_result, uid, file_name, annotated_image, original_image, processing_rate)
        else:
            shared.thread_pool.submit(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image, processing_rate)

    @staticmethod
    def _save_normal_result(uid, file_name, annotated_image, original_image, processing_rate):
        try:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass
        _draw_fps_label(annotated_image, processing_rate)
        cv2.imwrite(file_name, annotated_image)
        database.insert_data(uid, file_name, 'zheng_chang', None, None)
        cv2.imwrite(original_image_path, original_image)
        if os.path.exists(detect_image_path):
            os.remove(detect_image_path)

    def _handle_defect(self, inference_result, uid, file_name, annotated_image, original_image, processing_rate):
        logger.info("准备报警")
        shared.thread_pool.submit(trigger_alarm)
        logger.info("报警完成")
        time.sleep(0.1)
        shared.thread_pool.submit(trigger_grab, "NG")
        logger.info("检测到缺陷，触发报警和抓取动作")

        for detection in inference_result['result']:
            class_name = detection['class_name']
            if class_name == LABEL_NORMAL:
                continue

            annotated_image = self._draw_defect_box(
                annotated_image, detection['loc'], shared.defect_name[class_name],
                detection['score'], processing_rate
            )
            cv2.imwrite(file_name, annotated_image)
            database.insert_data(uid, file_name, class_name, detection['prediction_time'], detection['score'])
            database.insert_data(uid, 'detect.jpg', class_name, None, None)
            logger.info("数据库写入完成")

        cv2.imwrite(original_image_path, original_image)
        cv2.imwrite(detect_image_path, annotated_image)

    @staticmethod
    def _draw_defect_box(image, loc, class_name_cn: str, score: float, fps: float):
        # Scale coordinates from 300×300 inference space to actual image size
        h, w = image.shape[:2]
        scale_x = w / 300.0
        scale_y = h / 300.0
        x1, y1, x2, y2 = [int(v * (scale_x if i % 2 == 0 else scale_y)) for i, v in enumerate(loc)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        except Exception:
            pil_image = Image.fromarray(image)

        font = ImageFont.truetype(
            os.path.join(BASE_DIR, 'Font/platech.ttf'), 9, encoding="utf-8"
        )
        draw = ImageDraw.Draw(pil_image)
        draw.text(
            (x1 - 10, y1 - 10),
            '{} {:.2f}%'.format(class_name_cn, score * 100),
            font=font, fill="green",
        )
        image = np.array(pil_image)
        _draw_fps_label(image, fps)
        return image


# ============================================================
# 图片保留清理
# ============================================================

def cleanup():
    """保留最新 SAVE_IMAGE_NUM 张缺陷图片, 删除其余"""
    recent_rows = database.query(
        'distinct path', 'defect_list',
        "where path is not null and path != 'detect.jpg' order by id DESC limit {}".format(SAVE_IMAGE_NUM)
    )
    keep_files = [row[0] for row in recent_rows]
    for f in os.listdir(FILES_DIR):
        full_path = os.path.join(FILES_DIR, f)
        if full_path not in keep_files:
            os.remove(full_path)


# ============================================================
# 注册路由
# ============================================================

from routes import bp

app.register_blueprint(bp)

# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    database.create_database()  # 确保表已存在, 防 cleanup 竞态
    # 提前启动相机采集和检测线程 (避免 routes.py 因 import Consumer 形成循环)
    Producer(shared.stream_image_ref).start()
    Consumer().start()
    threading.Thread(target=cleanup).start()
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=7777)
