"""
主服务 (Flask :8080) — 缺陷检测 + 机械臂分拣

通过 Hikvision 工业相机实时采集图像, 调用 FPGA 推理服务进行缺陷检测,
根据检测结果直接触发机械臂分拣任务 (本地调用) 和蜂鸣器报警。
机械臂控制逻辑见 arm_control.py (Blueprint 'arm' 挂载到同一 app)。
前端监控 SPA (frontend/) 由本服务静态托管, 与 API/视频流同端口同源。

依赖: MVS SDK (MvImport), opencv-python, numpy, pyserial, Arm_Lib
"""

import json
import os
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

from logger import setup_log

from camera import frame_queue, Producer
from disc_detect import find_disc_robust, score_mask, in_frame_fraction, crop_box
import arm_control
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

shared.logger = setup_log('main', 'server.log')
logger = shared.logger

# ============================================================
# 常量
# ============================================================

FPGA_URL = 'http://172.16.68.110:8080/predict'
INFERENCE_TIMEOUT = 30

ALARM_PORT = "/dev/BAOJING"
ALARM_CMD = bytes.fromhex('7E FF 06 03 00 00 01 EF')

SAVE_IMAGE_NUM = 100
CLEANUP_INTERVAL = 600  # 周期清理间隔 (秒): 防止运行期间 files/ 被缺陷图占满磁盘
SSIM_WIDTH = 64   # 处理用低分辨率 (宽)
SSIM_HEIGHT = 48  # 处理用低分辨率 (高)

TRIGGER_COOLDOWN_SECONDS = 0.5  # 触发后冷却时间 (秒), 防同一铝片重复触发

LABEL_NORMAL = 'zheng_chang'

GAUSSIAN_KERNEL = 21
BINARY_THRESHOLD_1 = 100


# ---- 软处理参数 (原 config.ini [detection], 现改为 Python 直写; 改动后需重启服务) ----
BINARIZE_MODE = "otsu"
OTSU_MIN_THRESHOLD = 50.0
ADAPTIVE_BLOCK = 21
ADAPTIVE_C = 8
MORPH_OPEN = 1
MORPH_CLOSE = 2
MIN_COMPONENT_AREA = 0.002
BACKGROUND_ALPHA = 0.03
BASELINE_INIT_FRAMES = 30
BASELINE_WINDOW = 60
TRIGGER_K = 3.0
TRIGGER_CONFIRM = 3
CAL_RATIO_THRESHOLD = 0.01
# TRACKING 超时帧数: 超过后强制取最佳帧推理, 防铝片停留导致状态机卡死
TRACKING_TIMEOUT_FRAMES = 180
# 灯光/环境突变保护: 帧均值偏离背景超过此值且 fg_ratio 超过上限门时重置背景
LIGHT_CHANGE_THRESHOLD = 40.0
LIGHT_CHANGE_FG_RATIO = 0.6  # fg_ratio 上限门: 区分铝片局部变亮与灯光全局变亮

# ---- 圆形铝片识别 + 智能裁剪参数 (原 config.ini [disc], 现改为 Python 直写; 改动后需重启服务) ----

DISC_ENABLED = 1
DISC_METHOD = "mask"
DISC_MARGIN_RATIO = 0.10
DISC_DRAW_OVERLAY = 1
DISC_MIN_COMPLETENESS = 0.7         # 圆在画面内占比低于此值回退整帧
DISC_CENTER_COMPLETE = 0.9          # ≥此值进入"完整帧"档, 档内比横向居中
DISC_MIN_RADIUS_RATIO_FRAME = 0.18  # 圆半径/帧短边低于此值的帧选帧软降级(档0)
DISC_CFG = {
    "min_radius_ratio": 0.15,
    "max_radius_ratio": 0.50,
    "circularity": 0.60,
    "mask_threshold": 0.0,
    "otsu_min_threshold": 20.0,
    "hough_param1": 100.0,
    "hough_param2": 30.0,
    "hough_min_dist": 0.0,
    "method_fallback": 1,
}
shared.disc_method = DISC_METHOD
shared.disc_cfg = DISC_CFG
shared.disc_margin_ratio = DISC_MARGIN_RATIO

# ============================================================
# 全局状态
# ============================================================

alarm_serial = None

FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app, supports_credentials=True)


def _submit_pool(fn, *args):
    """提交到线程池; 积压超阈值时丢弃本次任务 (抓取/报警属补救动作, 丢弃无碍)."""
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
        alarm_serial = serial.Serial(ALARM_PORT, 9600, 8, stopbits=1)
    try:
        alarm_serial.write(ALARM_CMD)
    except Exception as e:
        logger.error('alarm error：%s', e)
        alarm_serial = None


# ============================================================
# 触发机械臂抓取
# ============================================================

def trigger_grab(flags: str):
    """触发机械臂分拣 — 直接执行 (互斥锁串行, 忙时丢弃)"""
    arm_control.enqueue_grab(flags, float(shared.GRAB_DELAY))


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
    """软处理前景提取: 返回 (fg_ratio, 前景掩码 mask)."""
    diff = cv2.absdiff(gray, background.astype(np.uint8))
    binary = _binarize(diff)
    opened = _morph(binary)
    mask = _filter_components(opened)
    fg_ratio = np.count_nonzero(mask) / mask.size
    return fg_ratio, mask


# ============================================================
# 图片消费者 (检测 + 推理 + 分拣)
# ============================================================

class Consumer(threading.Thread):
    """
    从 frame_queue 消费帧, 执行前景触发检测、FPGA 推理、结果处理.
    """

    def __init__(self):
        super().__init__()
        self._state = 0  # 0=IDLE, 1=TRACKING, 2=COOLDOWN
        self._cooldown_until = 0.0  # 冷却截止时间戳 (time.time(), 秒)
        self._tracking_count = 0  # 跟踪周期帧总数
        self._best_key = None  # 跟踪周期内"选帧排序键"最优值 (两阶段: 完整帧内比居中, 否则比完整度)
        self._best_frame = None  # 排序键最优帧, 推理时取它 (铝片最完整且横向居中的一帧)
        self._selected_frame = None  # 跟踪结束后选中的帧
        self._crop_box = None  # 本次推理的裁剪信息 (x0, y0, side); None=整帧缩放
        self._record_box = None  # 保存记录图时应用的裁剪框 (同 _crop_box); None=整帧
        self._buf = 0  # 状态切换缓冲计数（入口/出口共用）
        # 标定模式: 追踪 fg_ratio 变化以计算传送带速度
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
                logger.error('consumer thread error：{}'.format(str(e)), exc_info=True)

    # ---- 单帧处理 ----

    def _process_sampling_frame(self):
        """每帧入口: 软处理检测 (背景差分→二值化→连通域), 再走统一状态机."""
        image = frame_queue.get()
        if image is None:
            return

        gray = cv2.resize(image, (SSIM_WIDTH, SSIM_HEIGHT), interpolation=cv2.INTER_AREA)

        enter, exit_cond, ratio = self._soft_detect(gray)
        self._calibration_update(ratio, CAL_RATIO_THRESHOLD)

        shared.debug_background = self._background  # 供 /img_disc 视频流定圆
        frame_score, in_frac, circ = score_mask(shared.debug_mask, DISC_CFG)
        frame_key = self._selection_key(frame_score, in_frac, circ, image)

        self._run_state_machine(enter, exit_cond, ratio, image, frame_key)

    def _selection_key(self, score, in_frac, circ, image):
        """两阶段选帧排序键: 完整帧内比横向居中(2), 否则按完整度(1); 过小圆降档(0)."""
        h, w = image.shape[:2]
        mask = shared.debug_mask
        mask_h = mask.shape[0] if mask is not None else SSIM_HEIGHT
        mask_w = mask.shape[1] if mask is not None else SSIM_WIDTH
        if circ is not None:
            r_full = circ[2] * (h / mask_h)  # 掩码坐标半径 → 全分辨率
            if r_full / min(h, w) < DISC_MIN_RADIUS_RATIO_FRAME:
                return (0, score)
        if in_frac >= DISC_CENTER_COMPLETE and circ is not None:
            cx_full = circ[0] * (w / mask_w)
            return (2, -abs(cx_full - w / 2.0))
        return (1, score)

    # ---- 软处理: 光照鲁棒前景提取 (背景建模 + 自适应二值化) ----

    def _init_background(self, gray):
        """用首帧初始化 EMA 背景."""
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
        """暖机期无条件更新; 稳定期仅在无目标时更新 (避免把铝片/高光吸收进背景)."""
        if self._background is not None and LIGHT_CHANGE_THRESHOLD > 0:
            delta = abs(float(gray.mean()) - float(self._background.mean()))
            # fg_ratio 上限门: 铝片经过只让局部变亮 (fg_ratio ≤ ~0.5), 关灯/大幅
            # 换灯是全画面变化 (fg_ratio ≥ ~0.7) —— 两者都满足才判环境突变
            if delta > LIGHT_CHANGE_THRESHOLD and fg_ratio > LIGHT_CHANGE_FG_RATIO:
                logger.warning('灯光/环境突变: 帧均值偏差 %.1f > %.0f, 重置背景并回 IDLE',
                               delta, LIGHT_CHANGE_THRESHOLD)
                self._background = gray.astype(np.float32)
                self._baseline = []
                self._baseline_mean = 0.0
                self._baseline_std = 0.0
                self._state = 0
                self._buf = 0
                self._tracking_count = 0
                self._best_key = None
                self._best_frame = None
                return
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
        fg_ratio, mask = _extract_fg_ratio(gray, self._background)
        self._update_background(gray, fg_ratio)
        triggered = self._triggered(fg_ratio)

        shared.debug_mask = mask
        return triggered, (not triggered), fg_ratio

    def _calibration_update(self, ratio, threshold):
        """标定模式: 追踪 ratio 变化测量传送带速度."""
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
        """触发状态机: IDLE → TRACKING → COOLDOWN, 维护排序键最优帧作为推理帧."""
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
                    self._cooldown_until = time.time() + TRIGGER_COOLDOWN_SECONDS
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
                    self._cooldown_until = time.time() + TRIGGER_COOLDOWN_SECONDS

        elif self._state == 2:  # COOLDOWN — 按时间冷却
            if time.time() >= self._cooldown_until:
                self._state = 0

    # ---- 圆形铝片识别 + 智能裁剪 ----

    def _smart_crop(self, image):
        """定位铝片圆心并居中裁正方形(四周留黑边), 避免整帧 3:2 图被压成 300×300 变形.

        未启用/未检出/裁剪过小时返回 (None, None), 调用方回退整帧缩放; 全程 try/except 保护.
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
            x0, y0, side = crop_box(circle, h, w, DISC_MARGIN_RATIO)
            if side < 32:
                logger.warning('智能裁剪: 裁剪边长过小(%d), 回退整帧缩放', side)
                return None, None
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

    def _draw_disc_overlay(self, image, force_crop_box=False):
        """在全帧标注图上叠加绿色圆与黄色裁剪框 (draw_overlay 控制)."""
        if not DISC_DRAW_OVERLAY:
            return image
        disc = shared.last_disc
        if disc is not None:
            cx, cy, r = (int(v) for v in disc)
            cv2.circle(image, (cx, cy), r, (0, 255, 0), 1)
            cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)
        if force_crop_box or self._record_box is None:
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
        annotated_image = self._selected_frame.copy()
        original_image = annotated_image.copy()

        uid = str(uuid.uuid1())
        file_name = os.path.join(FILES_DIR, '{}.jpg'.format(uid))

        # 圆门: 未检出铝片圆则跳过推理/抓取/存图 (防误触发; 宁漏检不误抓)
        inference_image, self._crop_box = self._smart_crop(annotated_image)
        if inference_image is None:
            logger.warning('圆门: 未检出铝片圆, 跳过推理/抓取/存图')
            return
        self._record_box = self._crop_box

        # 300×300 灰度 raw 送 FPGA (SSD MobileNet 输入)
        interp = cv2.INTER_AREA if inference_image.shape[0] > 300 else cv2.INTER_LINEAR
        inference_image = cv2.resize(inference_image, (300, 300), interpolation=interp)
        image_bytes = inference_image.tobytes()

        inference_start_time = time.time()
        logger.info('开始检测')
        response = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=INFERENCE_TIMEOUT)
        logger.info('推理服务耗费：{}'.format(time.time() - inference_start_time))

        inference_result = json.loads(response.text)
        logger.info(inference_result)

        # action (FPGA 判定): NONE=无任何检测框; NG=检出缺陷类; OK=检出 zheng_chang
        action = inference_result.get('action')

        if action == 'NG':
            self._handle_defect(inference_result, uid, file_name, annotated_image, original_image)
        else:
            if action == 'NONE':
                logger.info('推理无检测框 (len=0), 按正常(OK)处理')
            _submit_pool(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image)

    def _save_normal_result(self, uid, file_name, annotated_image, original_image):
        try:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass
        self._draw_disc_overlay(annotated_image, force_crop_box=True)  # 全帧叠加绿圆+黄框
        cv2.imwrite(file_name, annotated_image)  # 历史图: 未裁剪全帧标注图
        database.insert_data(uid, file_name, 'zheng_chang', None, None)
        cv2.imwrite(original_image_path, original_image)  # 原始图片: 全帧原图(无标注)
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
                annotated_image, detection['loc'], shared.defect_name.get(class_name, class_name), detection['score']
            )
            database.insert_data(uid, file_name, class_name, detection['prediction_time'], detection['score'])
            database.insert_data(uid, 'detect.jpg', class_name, None, None)
            logger.info("数据库写入完成")

        # 记录图: 全帧叠加圆/框 (files=全帧标注图, original=全帧+绿圆黄框, detect=裁剪标注图)
        try:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass
        self._draw_disc_overlay(annotated_image, force_crop_box=True)
        cv2.imwrite(file_name, annotated_image)  # 历史图: 未裁剪全帧标注图
        cv2.imwrite(original_image_path, original_image)  # 原始图片: 全帧原图(无标注)
        record = self._make_record(annotated_image)  # detect.jpg 保持裁剪标注图
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
    keep_files = {row[0] for row in recent_rows}
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
app.register_blueprint(arm_control.bp)

# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    database.create_database()  # 确保表已存在, 防 cleanup 竞态
    # 提前启动相机采集和检测线程 (避免 routes.py 因 import Consumer 形成循环)
    Producer(shared.stream_image_ref).start()
    Consumer().start()
    threading.Thread(target=cleanup_loop).start()
    # threaded=True 必须开: MJPEG 流 (/img_disc) 是长驻生成器,
    # 单线程下会占死唯一 HTTP 线程, 其余请求全部排队阻塞直至内存堆积
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=8080, threaded=True)
