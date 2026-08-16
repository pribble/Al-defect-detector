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

SSIM_TRIGGER_THRESHOLD = 0.8
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


def _det_cfg(key, fallback):
    """读取 [detection] 段配置 (config.ini 缺段/缺键时回退默认值)."""
    return shared.config.get("detection", key, fallback=fallback)


# ---- 软处理参数 (config.ini [detection] 段, 现场可调; 改动后需重启服务) ----
USE_SOFT_PROCESSING = int(_det_cfg("use_soft_processing", "1"))
BINARIZE_MODE = _det_cfg("binarize_mode", "otsu")
OTSU_MIN_THRESHOLD = float(_det_cfg("otsu_min_threshold", "50"))
ADAPTIVE_BLOCK = int(_det_cfg("adaptive_block", "21"))
ADAPTIVE_C = int(_det_cfg("adaptive_c", "8"))
MORPH_OPEN = int(_det_cfg("morph_open", "1"))
MORPH_CLOSE = int(_det_cfg("morph_close", "2"))
MIN_COMPONENT_AREA = float(_det_cfg("min_component_area", "0.002"))
BACKGROUND_ALPHA = float(_det_cfg("background_alpha", "0.03"))
BASELINE_INIT_FRAMES = int(_det_cfg("baseline_init_frames", "30"))
BASELINE_WINDOW = int(_det_cfg("baseline_window", "60"))
TRIGGER_K = float(_det_cfg("trigger_k", "3.0"))
TRIGGER_CONFIRM = int(_det_cfg("trigger_confirm_frames", "3"))
CAL_RATIO_THRESHOLD = float(_det_cfg("cal_ratio_threshold", "0.01"))
USE_SSIM_GATE = int(_det_cfg("use_ssim_gate", "0"))

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
    data = {"flags": flags, "time": shared.GRAB_DELAY}
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
# 软处理: 光照鲁棒前景提取 (背景差分 → 自适应二值化 → 形态学 → 连通域)
# ============================================================

def _binarize(diff):
    """对背景差分图做二值化: Otsu(默认)/adaptive/fixed."""
    blurred = cv2.GaussianBlur(diff, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    if BINARIZE_MODE == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
    elif BINARIZE_MODE == "fixed":
        _, binary = cv2.threshold(blurred, BINARY_THRESHOLD_1, 255, cv2.THRESH_BINARY)
    else:  # otsu
        t, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if t < OTSU_MIN_THRESHOLD:  # 纯暗背景下限保护, 防把噪声当目标
            _, binary = cv2.threshold(blurred, OTSU_MIN_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def _morph(binary):
    """形态学: 开运算去噪 + 闭运算填洞连块."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if MORPH_OPEN:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN)
    if MORPH_CLOSE:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE)
    return binary


def _filter_components(binary):
    """连通域面积过滤: 只保留足够大的前景块 (铝片), 去掉散乱高光碎斑."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    mask = np.zeros_like(binary)
    min_area = MIN_COMPONENT_AREA * binary.size
    for i in range(1, num):  # 跳过 label=0 背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    return mask


def _extract_fg_ratio(gray, background):
    """软处理前景提取: 返回 (fg_ratio, 中间结果 dict 供调试)."""
    diff = cv2.absdiff(gray, background.astype(np.uint8))
    binary = _binarize(diff)
    opened = _morph(binary)
    mask = _filter_components(opened)
    fg_ratio = np.count_nonzero(mask) / mask.size
    return fg_ratio, {"diff": diff, "binary": binary, "mask": mask}


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
        self._ssim_history = [0] * SSIM_HISTORY_SIZE
        self._state = 0  # 0=IDLE, 1=TRACKING, 2=COOLDOWN
        self._cooldown = 0  # 冷却倒计帧数
        self._tracking_count = 0  # 跟踪周期帧总数
        self._tracking_frames = []  # 跟踪帧队列，头部始终是中间帧
        self._selected_frame = None  # 跟踪结束后选中的中间帧
        self._buf = 0  # 状态切换缓冲计数（入口/出口共用）
        # 标定模式: 追踪 white_ratio 变化以计算传送带速度
        self._cal_last_ratio = 0.0
        self._cal_entry_time = None
        self._cal_peak_ratio = 0.0
        # 软处理: EMA 背景模型 + fg_ratio 基线
        self._background = None   # float32 (48,64) EMA 背景
        self._baseline = []       # 空皮带 fg_ratio 滚动窗口
        self._baseline_mean = 0.0
        self._baseline_std = 0.0

    def run(self):
        database.create_database()

        while True:
            time.sleep(0.01)
            try:
                self._process_sampling_frame()

            except Exception as e:
                logger.error('consumer thread error：{}'.format(str(e)))

    # ---- 单帧处理 ----

    def _process_sampling_frame(self):
        """每帧入口: 按 USE_SOFT_PROCESSING 选软/硬检测, 再走统一状态机."""
        image = frame_queue.get()
        if image is None:
            return

        gray = cv2.resize(image, (SSIM_WIDTH, SSIM_HEIGHT), interpolation=cv2.INTER_AREA)

        if USE_SOFT_PROCESSING:
            enter, exit_cond, ratio = self._soft_detect(gray)
            self._calibration_update(ratio, CAL_RATIO_THRESHOLD)
        else:
            enter, exit_cond, ratio = self._hard_detect(gray, image)
            self._calibration_update(ratio, 0.02)

        self._run_state_machine(enter, exit_cond, ratio, image)

    # ---- 软处理: 光照鲁棒前景提取 (背景建模 + 自适应二值化) ----

    def _init_background(self, gray):
        """用参考图(存在时)或首帧初始化 EMA 背景."""
        if os.path.exists(REFERENCE_IMAGE):
            ref = cv2.imread(REFERENCE_IMAGE, cv2.IMREAD_GRAYSCALE)
            if ref is not None:
                ref = cv2.resize(ref, (SSIM_WIDTH, SSIM_HEIGHT), interpolation=cv2.INTER_AREA)
                self._background = ref.astype(np.float32)
                return
        self._background = gray.astype(np.float32)

    def _recalc_baseline(self):
        if len(self._baseline) >= BASELINE_INIT_FRAMES:
            arr = np.array(self._baseline)
            self._baseline_mean = float(arr.mean())
            self._baseline_std = float(arr.std())

    def _triggered(self, fg_ratio):
        """fg_ratio 是否显著高于空皮带基线 (z-score). 基线未建好时抑制触发."""
        if len(self._baseline) < BASELINE_INIT_FRAMES:
            return False
        if self._baseline_std < 1e-9:
            return fg_ratio > self._baseline_mean + TRIGGER_K * 0.01
        return fg_ratio > self._baseline_mean + TRIGGER_K * self._baseline_std

    def _update_background(self, gray, fg_ratio):
        """暖机期无条件更新; 稳定期只在无目标时更新, 避免吸收铝片/高光."""
        warmup = len(self._baseline) < BASELINE_INIT_FRAMES
        if not warmup and self._triggered(fg_ratio):
            return
        self._background = ((1 - BACKGROUND_ALPHA) * self._background
                            + BACKGROUND_ALPHA * gray.astype(np.float32))
        self._baseline.append(fg_ratio)
        if len(self._baseline) > BASELINE_WINDOW:
            self._baseline.pop(0)
        self._recalc_baseline()

    def _soft_detect(self, gray):
        """软处理: 背景差分→自适应二值化→连通域, 返回 (进入条件, 退出条件, fg_ratio)."""
        if self._background is None:
            self._init_background(gray)
        fg_ratio, stages = _extract_fg_ratio(gray, self._background)
        self._update_background(gray, fg_ratio)
        triggered = self._triggered(fg_ratio)

        # 归一化 SSIM (背景 vs 当前帧): 默认仅观测; USE_SSIM_GATE=1 时参与触发
        a = self._background.astype(np.float32)
        b = gray.astype(np.float32)
        sa, sb = float(a.std()), float(b.std())
        if sa < 1e-6 or sb < 1e-6:
            n_ssim = 0.0
        else:
            a = (a - a.mean()) / sa
            b = (b - b.mean()) / sb
            n_ssim = float(compare_ssim(a, b))

        shared.debug_mask = stages["mask"]
        shared.debug_intermediates = stages
        shared.last_ssim = n_ssim
        shared.last_white_ratio = fg_ratio

        if USE_SSIM_GATE:
            triggered = triggered and n_ssim < SSIM_TRIGGER_THRESHOLD

        return triggered, (not triggered), fg_ratio

    def _hard_detect(self, gray, image):
        """原硬处理链 (固定阈值+SSIM+单快照参考), 保留用于回滚."""
        global _cached_ref_gray
        blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        _, img_binary = cv2.threshold(blurred, BINARY_THRESHOLD_1, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(img_binary, None, iterations=DILATE_ITERATIONS)

        white_ratio = np.count_nonzero(thresh) / thresh.size
        current_ssim = compare_image(gray)

        shared.debug_mask = thresh
        shared.last_ssim = current_ssim
        shared.last_white_ratio = white_ratio

        self._ssim_history[:-1] = self._ssim_history[1:]
        self._ssim_history[-1] = current_ssim
        ssim_std = np.std(self._ssim_history)
        ssim_mean = np.mean(self._ssim_history)
        if ssim_std < STABILITY_STD_THRESHOLD and ssim_mean < STABILITY_MEAN_THRESHOLD and all(self._ssim_history):
            cv2.imwrite(REFERENCE_IMAGE, image)
            _cached_ref_gray = None  # 缓存失效, 下次重载

        enter = current_ssim < SSIM_TRIGGER_THRESHOLD and white_ratio > WHITE_RATIO_THRESHOLD
        exit_cond = current_ssim > SSIM_TRIGGER_THRESHOLD and white_ratio < WHITE_RATIO_THRESHOLD
        return enter, exit_cond, white_ratio

    def _calibration_update(self, ratio, threshold):
        """标定模式: 追踪 ratio 变化测量传送带速度 (语义与原 white_ratio 一致)."""
        if not shared.calibration_active:
            return
        now = time.time()
        if ratio > threshold:
            if self._cal_last_ratio < threshold:
                self._cal_entry_time = now
                self._cal_peak_ratio = ratio
            if ratio > self._cal_peak_ratio:
                self._cal_peak_ratio = ratio
            if (self._cal_entry_time is not None and
                    ratio < self._cal_peak_ratio * 0.85):
                duration = now - self._cal_entry_time
                if 0.1 < duration < 30:
                    speed = 100.0 / duration
                    shared.calibration_samples.append(speed)
                    logger.info('标定: 测得速度 {:.1f} mm/s (耗时 {:.2f}s)'.format(speed, duration))
                self._cal_entry_time = None
        else:
            self._cal_entry_time = None
        self._cal_last_ratio = ratio

    def _run_state_machine(self, enter, exit_cond, ratio, image):
        """统一触发状态机: 0=IDLE → 1=TRACKING → 2=COOLDOWN."""
        if self._state == 0:  # IDLE — 等待铝片进入
            if enter:
                self._buf += 1
                if self._buf >= TRIGGER_CONFIRM:
                    self._state = 1
                    self._tracking_count = 1
                    self._tracking_frames = [image.copy()]
                    self._buf = 0
            else:
                self._buf = 0

        elif self._state == 1:  # TRACKING — 收集跟踪帧，取中间帧
            if exit_cond:
                self._buf += 1
                if self._buf >= TRIGGER_CONFIRM:
                    self._selected_frame = self._tracking_frames[0]
                    self._buf = 0
                    logger.info(
                        '触发回落 ratio:{:.3f}, total_frames:{}'.format(
                            ratio, self._tracking_count,
                        )
                    )
                    self._run_inference_pipeline()
                    self._state = 2
                    self._cooldown = TRIGGER_COOLDOWN
            else:
                self._buf = 0
                self._tracking_count += 1
                # 单数帧 append，双数帧 pop 头部再 append
                if self._tracking_count % 2 == 0:
                    self._tracking_frames.pop(0)
                self._tracking_frames.append(image.copy())

        elif self._state == 2:  # COOLDOWN — 冷却
            self._cooldown -= 1
            if self._cooldown <= 0:
                self._state = 0

    # ---- 推理管线 ----

    def _run_inference_pipeline(self):
        annotated_image = self._selected_frame.copy()
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

        inference_result = json.loads(response.text)
        logger.info(inference_result)

        if inference_result['len'] > 0:
            self._handle_inference_result(inference_result, uid, file_name, annotated_image, original_image)
        else:
            shared.thread_pool.submit(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image)

    def _handle_inference_result(self, inference_result, uid, file_name, annotated_image, original_image):
        action = inference_result.get('action', 'OK')

        if action == "NG":
            self._handle_defect(inference_result, uid, file_name, annotated_image, original_image)
        else:
            shared.thread_pool.submit(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image)

    @staticmethod
    def _save_normal_result(uid, file_name, annotated_image, original_image):
        try:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass
        cv2.imwrite(file_name, annotated_image)
        database.insert_data(uid, file_name, 'zheng_chang', None, None)
        cv2.imwrite(original_image_path, original_image)
        if os.path.exists(detect_image_path):
            os.remove(detect_image_path)

    def _handle_defect(self, inference_result, uid, file_name, annotated_image, original_image):
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
                annotated_image, detection['loc'], shared.defect_name[class_name], detection['score']
            )
            cv2.imwrite(file_name, annotated_image)
            database.insert_data(uid, file_name, class_name, detection['prediction_time'], detection['score'])
            database.insert_data(uid, 'detect.jpg', class_name, None, None)
            logger.info("数据库写入完成")

        cv2.imwrite(original_image_path, original_image)
        cv2.imwrite(detect_image_path, annotated_image)

    @staticmethod
    def _draw_defect_box(image, loc, class_name_cn: str, score: float):
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
