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
from disc_detect import find_disc_robust, score_mask, in_frame_fraction
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
CLEANUP_INTERVAL = 60  # 周期清理间隔 (秒): 防止运行期间 files/ 被缺陷图占满磁盘
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
# TRACKING 超时帧数: 进入跟踪后超过该帧数仍不回落, 强制取中间帧推理 (防铝片停
# 留在视野内导致状态机永久卡死、后续铝片不再触发)
TRACKING_TIMEOUT_FRAMES = int(_det_cfg("tracking_timeout_frames", "90"))

# ---- 圆形铝片识别 + 智能裁剪参数 (config.ini [disc] 段, 改动后需重启服务) ----

def _disc_cfg(key, fallback):
    """读取 [disc] 段配置 (config.ini 缺段/缺键时回退默认值)."""
    return shared.config.get("disc", key, fallback=fallback)


DISC_ENABLED = int(_disc_cfg("enabled", "1"))
DISC_METHOD = _disc_cfg("method", "mask")
DISC_MARGIN_RATIO = float(_disc_cfg("margin_ratio", "0.10"))
DISC_DRAW_OVERLAY = int(_disc_cfg("draw_overlay", "1"))
# 裁剪最低完整度: 检出的圆在画面内占比低于此值时回退整帧 (低于该值说明铝片
# 未完整进入视野, 硬裁会切掉铝片)
DISC_MIN_COMPLETENESS = float(_disc_cfg("min_completeness", "0.7"))
# 横向居中优先级: 圆完整度 ≥ 此值的帧进入"完整帧"档, 档内选圆心最接近画面
# 水平中心的帧 (原长方形图中铝片横向最中间); 低于此值按完整度打分选帧
DISC_CENTER_COMPLETE = float(_disc_cfg("center_complete", "0.9"))
DISC_CFG = {
    "min_radius_ratio": float(_disc_cfg("min_radius_ratio", "0.15")),
    "max_radius_ratio": float(_disc_cfg("max_radius_ratio", "0.50")),
    "circularity": float(_disc_cfg("circularity", "0.60")),
    "mask_threshold": float(_disc_cfg("mask_threshold", "0")),
    "otsu_min_threshold": float(_disc_cfg("otsu_min_threshold", "50")),
    "hough_param1": float(_disc_cfg("hough_param1", "100")),
    "hough_param2": float(_disc_cfg("hough_param2", "30")),
    "hough_min_dist": float(_disc_cfg("hough_min_dist", "0")),
    "method_fallback": int(_disc_cfg("method_fallback", "1")),
}
shared.disc_method = DISC_METHOD
shared.disc_enabled = DISC_ENABLED
shared.disc_cfg = DISC_CFG
shared.disc_margin_ratio = DISC_MARGIN_RATIO

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


def _submit_pool(fn, *args):
    """有界提交到线程池: 任务积压超过阈值时丢弃本次任务并告警.

    机械臂/报警请求在目标服务无响应时会挂起 30s, 若铝片持续触发, 无界队列的
    任务只进不出会吃内存. 丢弃的是"补救性"动作 (抓取/报警), 不丢也不致命.
    """
    try:
        qsize = shared.thread_pool._work_queue.qsize()
    except Exception:
        qsize = 0
    if qsize > 200:
        logger.warning('线程池任务积压 %d, 丢弃 %s', qsize, getattr(fn, '__name__', str(fn)))
        return
    shared.thread_pool.submit(fn, *args)


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
        self._best_key = None  # 跟踪周期内"选帧排序键"最优值 (两阶段: 完整帧内比居中, 否则比完整度)
        self._best_frame = None  # 排序键最优帧, 推理时取它 (铝片最完整且横向居中的一帧)
        self._selected_frame = None  # 跟踪结束后选中的帧
        self._crop_box = None  # 本次推理的裁剪信息 (x0, y0, side); None=整帧缩放
        self._record_box = None  # 保存记录图时应用的裁剪框 (同 _crop_box); None=整帧
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
        frame_counter = 0

        while True:
            time.sleep(0.01)
            try:
                self._process_sampling_frame()
                frame_counter += 1
                # 周期性记录进程 RSS, 供排查内存缓慢增长 (Linux /proc)
                if frame_counter % 300 == 0:
                    self._log_rss()

            except Exception as e:
                # 记录最近一次异常到 shared, 供 /get_status 诊断; exc_info 输出完整堆栈
                shared.last_consumer_error = '{}'.format(e)
                logger.error('consumer thread error：{}'.format(str(e)), exc_info=True)

    @staticmethod
    def _log_rss():
        """读取 /proc/self/status 的 VmRSS 并记 INFO 日志 (非 Linux 环境静默跳过)."""
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS'):
                        logger.info('RSS: %s', line.strip())
                        break
        except Exception:
            pass

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

        # 供 /debug_disc 与智能裁剪使用: EMA 背景 (硬处理路径下为 None → 走原始阈值)
        shared.debug_background = getattr(self, '_background', None)
        # 供 /get_status 诊断: 空皮带基线统计 (暖机期 n < BASELINE_INIT_FRAMES)
        shared.baseline_stats = (self._baseline_mean, self._baseline_std, len(self._baseline))

        # 圆完整度打分 + 选帧排序键:
        #   完整帧 (in_frac ≥ DISC_CENTER_COMPLETE): 键 = (2, -|圆心x-画面宽/2|)
        #      —— 完整帧档内选"原长方形图中铝片横向最中间"的帧
        #   不完整帧: 键 = (1, r × 圆在画面内占比)  —— 按完整度兜底
        frame_score, in_frac, circ = score_mask(getattr(shared, 'debug_mask', None), DISC_CFG)
        frame_key = self._selection_key(frame_score, in_frac, circ, image)

        self._run_state_machine(enter, exit_cond, ratio, image, frame_key)

    def _selection_key(self, score, in_frac, circ, image):
        """两阶段选帧排序键: 完整帧优先, 完整帧内比横向居中; 否则按完整度."""
        if in_frac >= DISC_CENTER_COMPLETE and circ is not None:
            h, w = image.shape[:2]
            mask_w = getattr(shared, 'debug_mask', None)
            mask_w = mask_w.shape[1] if mask_w is not None else SSIM_WIDTH
            cx_full = circ[0] * (w / mask_w)  # 掩码坐标 → 全分辨率
            return (2, -abs(cx_full - w / 2.0))
        return (1, score)

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
            n_ssim = float(compare_ssim(a, b, data_range=1.0))

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

    def _run_state_machine(self, enter, exit_cond, ratio, image, frame_key):
        """统一触发状态机: 0=IDLE → 1=TRACKING → 2=COOLDOWN.

        frame_key: 当前帧的"选帧排序键" (见 _selection_key) — 完整帧内比横向居中,
        否则比完整度; 维护排序键最大的帧作为推理帧.
        """
        if self._state == 0:  # IDLE — 等待铝片进入
            if enter:
                self._buf += 1
                if self._buf >= TRIGGER_CONFIRM:
                    self._state = 1
                    self._tracking_count = 1
                    # 记录排序键最优帧 (完整 + 横向居中优先, 比 fg_ratio 抗反光)
                    self._best_key = frame_key
                    self._best_frame = image.copy()
                    self._buf = 0
                    shared.last_trigger_time = time.time()
                    logger.info('触发进入 TRACKING, ratio=%.3f, key=%s', ratio, frame_key)
            else:
                self._buf = 0

        elif self._state == 1:  # TRACKING — 收集帧, 维护"选帧排序键"最优帧
            if exit_cond:
                self._buf += 1
                if self._buf >= TRIGGER_CONFIRM:
                    self._selected_frame = self._best_frame
                    self._buf = 0
                    logger.info(
                        '触发回落 ratio:{:.3f}, total_frames:{}, 最佳key:{}'.format(
                            ratio, self._tracking_count, self._best_key,
                        )
                    )
                    self._run_inference_pipeline()
                    self._state = 2
                    self._cooldown = TRIGGER_COOLDOWN
            else:
                self._buf = 0
                self._tracking_count += 1
                # 记录排序键最优帧 (完整帧内横向最居中; 无完整帧时按完整度兜底)
                if frame_key > self._best_key:
                    self._best_key = frame_key
                    self._best_frame = image.copy()
                # 超时保护: 铝片停留/皮带停住时退出条件永不满足, 强制推理防卡死
                if self._tracking_count >= TRACKING_TIMEOUT_FRAMES:
                    self._selected_frame = self._best_frame
                    logger.warning('TRACKING 超时(%d帧), 取最佳帧推理', self._tracking_count)
                    self._run_inference_pipeline()
                    self._state = 2
                    self._cooldown = TRIGGER_COOLDOWN

        elif self._state == 2:  # COOLDOWN — 冷却
            self._cooldown -= 1
            if self._cooldown <= 0:
                self._state = 0

        # 实时反映状态机位置 (供 /get_status 诊断)
        shared.trigger_state = self._state

    # ---- 圆形铝片识别 + 智能裁剪 ----

    def _smart_crop(self, image):
        """
        定位铝片圆心, 以其为中心裁正方形(四周留黑边), 返回 (裁剪图, (x0, y0, side)).
        未启用 / 未检出圆 / 裁剪过小时返回 (None, None), 由调用方回退整帧缩放.

        定圆: find_disc_robust() 主方法(默认 mask) → 回退链(有背景时 hough;
        无背景时 mask 原始阈值 → hough), 优先使用 Consumer 的 EMA 背景做背景
        差分 (与触发链路同源, 光照鲁棒).

        理由: 整帧 768×512 (3:2) 被各向异性压缩成 300×300 会挤压缺陷形状;
        以铝片为中心裁方形再等比缩放, 内容不失真, 黑边也减少了背景干扰.

        注意: 整个方法被 try/except 包裹——即使定圆代码出意外, 也绝不中断
        推理/存图链路 (宁可靠后回退整帧, 不可让铝片经过时丢图).
        """
        shared.last_disc = None
        shared.last_crop_box = None
        if not DISC_ENABLED:
            return None, None
        try:
            h, w = image.shape[:2]
            circle, info = find_disc_robust(
                image, method=DISC_METHOD, cfg=DISC_CFG, background=self._background)
            if circle is None:
                logger.warning('智能裁剪: 未检出铝片圆 (%s), 回退整帧缩放',
                               (info or {}).get('reject') or '未知原因')
                return None, None
            cx, cy, r = circle
            # 圆完整度门槛: 圆在画面内占比 < min_completeness 时回退整帧 (铝片
            # 未完整进入视野, 硬裁会切掉铝片); 否则出界的圆也照裁, 黑边吸收小越界
            in_frac = in_frame_fraction((cx, cy, r), h, w)
            if in_frac < DISC_MIN_COMPLETENESS:
                logger.warning('智能裁剪: 圆完整度 %.2f < %.2f, 回退整帧缩放',
                               in_frac, DISC_MIN_COMPLETENESS)
                return None, None
            margin = int(r * DISC_MARGIN_RATIO)
            side = int(2 * (r + margin))
            side = min(side, w, h)  # 窗口不得超出画面
            if side < 32:
                logger.warning('智能裁剪: 裁剪边长过小(%d), 回退整帧缩放', side)
                return None, None
            x0 = min(max(int(cx) - side // 2, 0), w - side)
            y0 = min(max(int(cy) - side // 2, 0), h - side)
            crop = image[y0:y0 + side, x0:x0 + side]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                return None, None
            shared.last_disc = (cx, cy, r)
            shared.last_crop_box = (x0, y0, side)
            logger.info('智能裁剪: 圆(cx=%.1f, cy=%.1f, r=%.1f) [%s] → 裁剪(%d,%d,%d)',
                        cx, cy, r, (info or {}).get('used_method', '?'), x0, y0, side)
            return crop, (x0, y0, side)
        except Exception as e:
            logger.error('智能裁剪异常, 回退整帧缩放: %s', e, exc_info=True)
            return None, None

    def _draw_disc_overlay(self, image):
        """在标注图上叠加绿色圆与黄色裁剪框 (现场调参/验证用, draw_overlay 控制).

        在【全帧】上绘制 (坐标是全帧坐标); 若本次裁剪成功 (_record_box 已设),
        黄框就是记录图的边界, 不再重复画.
        """
        if not DISC_DRAW_OVERLAY:
            return image
        disc = shared.last_disc
        if disc is not None:
            cx, cy, r = (int(v) for v in disc)
            cv2.circle(image, (cx, cy), r, (0, 255, 0), 1)
            cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)
        if self._record_box is None:
            box = shared.last_crop_box
            if box is not None:
                x0, y0, side = box
                cv2.rectangle(image, (x0, y0), (x0 + side, y0 + side), (0, 255, 255), 1)
        return image

    def _make_record(self, image):
        """生成记录图: 裁剪成功时把(已标注/原)图裁成以铝片为中心的方形记录; 否则原样返回."""
        box = self._record_box
        if box is None:
            return image
        x0, y0, side = box
        h, w = image.shape[:2]
        return image[y0:min(y0 + side, h), x0:min(x0 + side, w)]

    # ---- 推理管线 ----

    def _run_inference_pipeline(self):
        shared.last_inference_time = time.time()  # 供 /get_status 诊断
        annotated_image = self._selected_frame.copy()
        original_image = annotated_image.copy()

        uid = str(uuid.uuid1())
        file_name = os.path.join(FILES_DIR, '{}.jpg'.format(uid))

        # 圆形铝片识别 + 智能裁剪: 以铝片中心裁正方形(四周留黑边), 避免整帧
        # 3:2 图被各向异性压缩成 300×300 导致缺陷变形 (未检出圆时回退整帧)
        inference_image, self._crop_box = self._smart_crop(annotated_image)
        if inference_image is None:
            inference_image = annotated_image
            self._crop_box = None
            self._record_box = None
        else:
            # 裁剪成功: 保存的记录图(original.jpg / files/ / detect.jpg)同样用
            # 以铝片为中心的方形图, 保证"提取的图片中铝片在中间"
            self._record_box = self._crop_box

        # Resize to 300×300 for FPGA inference (SSD MobileNet input size)
        # 裁剪图/整帧均为正方形缩放: 缩小用 INTER_AREA 防锯齿, 放大用 INTER_LINEAR
        interp = cv2.INTER_AREA if inference_image.shape[0] > 300 else cv2.INTER_LINEAR
        inference_image = cv2.resize(inference_image, (300, 300), interpolation=interp)
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
            _submit_pool(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image)

    def _handle_inference_result(self, inference_result, uid, file_name, annotated_image, original_image):
        action = inference_result.get('action', 'OK')

        if action == "NG":
            self._handle_defect(inference_result, uid, file_name, annotated_image, original_image)
        else:
            _submit_pool(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image)

    def _save_normal_result(self, uid, file_name, annotated_image, original_image):
        try:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass
        self._draw_disc_overlay(annotated_image)  # 全帧坐标叠加 (3 通道), 再裁记录图
        record = self._make_record(annotated_image)
        cv2.imwrite(file_name, record)
        database.insert_data(uid, file_name, 'zheng_chang', None, None)
        cv2.imwrite(original_image_path, self._make_record(original_image))
        if os.path.exists(detect_image_path):
            os.remove(detect_image_path)

    def _handle_defect(self, inference_result, uid, file_name, annotated_image, original_image):
        logger.info("准备报警")
        _submit_pool(trigger_alarm)
        logger.info("报警完成")
        time.sleep(0.1)
        _submit_pool(trigger_grab, "NG")
        logger.info("检测到缺陷，触发报警和抓取动作")

        for detection in inference_result['result']:
            class_name = detection['class_name']
            if class_name == LABEL_NORMAL:
                continue

            annotated_image = self._draw_defect_box(
                annotated_image, detection['loc'], shared.defect_name[class_name], detection['score']
            )
            database.insert_data(uid, file_name, class_name, detection['prediction_time'], detection['score'])
            database.insert_data(uid, 'detect.jpg', class_name, None, None)
            logger.info("数据库写入完成")

        # 记录图: 全帧叠加圆/框后, 裁成以铝片为中心的方形 (提取的图片铝片居中)
        try:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass
        self._draw_disc_overlay(annotated_image)
        record = self._make_record(annotated_image)
        cv2.imwrite(file_name, record)
        cv2.imwrite(original_image_path, self._make_record(original_image))
        cv2.imwrite(detect_image_path, record)

    def _draw_defect_box(self, image, loc, class_name_cn: str, score: float):
        # 推理坐标 (300×300 空间) → 全帧坐标
        h, w = image.shape[:2]
        box = self._crop_box
        if box is None:
            # 整帧缩放路径 (未启用裁剪 / 未检出圆)
            scale_x = w / 300.0
            scale_y = h / 300.0
            x1, y1, x2, y2 = [int(round(v * (scale_x if i % 2 == 0 else scale_y))) for i, v in enumerate(loc)]
        else:
            # 智能裁剪路径: 300×300 → 裁剪图 → 全帧
            x0, y0, side = box
            scale = side / 300.0
            x1, y1, x2, y2 = [int(round(v * scale)) + (x0 if i % 2 == 0 else y0) for i, v in enumerate(loc)]
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


def cleanup_loop():
    """周期清理 files/ 目录, 防止长时间运行磁盘被缺陷图占满 (原 cleanup 只在启动时跑一次)."""
    while True:
        try:
            cleanup()
        except Exception as e:
            logger.error('cleanup error: %s', e)
        time.sleep(CLEANUP_INTERVAL)


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
    threading.Thread(target=cleanup_loop).start()
    # threaded=True 必须开: MJPEG 流 (/img /debug_disc /debug_stages) 是长驻生成器,
    # 单线程下第一个流会占死唯一 HTTP 线程, 其余请求全部排队阻塞直至内存堆积
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=7777, threaded=True)
