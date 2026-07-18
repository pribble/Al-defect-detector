"""
缺陷检测服务 (Flask :7777)

通过 Hikvision 工业相机实时采集图像, 调用 FPGA 推理服务进行缺陷检测,
根据检测结果触发机械臂分拣 (OK/NG) 和蜂鸣器报警。

依赖: MVS SDK (MvImport), opencv-python, scikit-image (SSIM), pyserial
"""

import base64
import configparser
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import cv2
import datetime
import numpy as np
import requests
import serial
from flask import Flask, render_template, Response, request
from flask_cors import CORS
from PIL import Image, ImageFont, ImageDraw
from skimage.metrics import structural_similarity as compare_ssim

sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from logger import setup_log

import database
from camera import init_camera, Producer, CAMERA_NEED_RESTART

# ============================================================
# 路径常量
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, 'files')                     # 缺陷结果图片
ORIGINAL_DIR = os.path.join(BASE_DIR, 'original_files')         # 原始捕获图片
DETECT_DIR = os.path.join(BASE_DIR, 'detect_files')             # 缺陷标注图片

for _dir in [FILES_DIR, DETECT_DIR, ORIGINAL_DIR]:
    os.makedirs(_dir, exist_ok=True)

# 当前图片文件路径 (会被后续检测更新覆盖)
original_image_path = os.path.join(ORIGINAL_DIR, 'original.jpg')
detect_image_path = os.path.join(DETECT_DIR, 'detect.jpg')

# ============================================================
# 日志与配置
# ============================================================

logger = setup_log('detect', 'detect_server.log')

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
defect_name = dict(config.items('defect_name'))     # 拼音 → 中文标签映射

# ============================================================
# 常量
# ============================================================

# --- 网络 ---
FPGA_URL = 'http://172.16.68.110:8080/predict'
ARM_URL = 'http://172.16.68.111:8899/grab'
INFERENCE_TIMEOUT = 30

# --- 报警器串口 ---
ALARM_PORT = "/dev/BAOJING"
ALARM_BAUD = 9600
ALARM_DATA_BITS = 8
ALARM_STOP_BITS = 1
ALARM_CMD = bytes.fromhex('7E FF 06 03 00 00 01 EF')

# --- 图片保留 ---
SAVE_IMAGE_NUM = 10

# --- SSIM / 图像比较 ---
SSIM_WIDTH = 64
SSIM_HEIGHT = 48
REFERENCE_IMAGE = 'yuanshi.jpg'

# --- 推理触发 ---
SSIM_TRIGGER_THRESHOLD = 0.9          # 低于此值表示视野中有物体
WHITE_RATIO_THRESHOLD = 0.1           # 白色像素比例推断阈值
STABILITY_STD_THRESHOLD = 0.01        # SSIM 历史标准差阈值 (判断画面是否稳定)
STABILITY_MEAN_THRESHOLD = 0.8        # SSIM 历史均值阈值
SSIM_HISTORY_SIZE = 9                 # SSIM 滑动窗口大小

# --- 帧处理 ---
FRAME_SKIP_COUNT = 10                 # 每 N 帧处理一次

# --- 缺陷标签 ---
LABEL_NORMAL = 'zheng_chang'          # FPGA 返回的 "正常" 标签
DEFECT_LABELS = ['ca_shang', 'zang_wu', 'zhe_zhou', 'zhen_kong']

# --- 相机错误码 (MVS SDK) ---
# 图像预处理参数
GAUSSIAN_KERNEL = 21                  # 高斯滤波核
BINARY_THRESHOLD_1 = 100              # 初始二值化阈值
BINARY_THRESHOLD_2 = 0.3              # 第二次二值化阈值
DILATE_ITERATIONS = 4                 # 膨胀迭代次数

# --- 机械臂控制参数 (来自配置文件, 运行时可能通过 /change_conf 更新) ---
GRAB_SPEED = config.get("Configuration", "speed")
GRAB_DELAY = config.get("Configuration", "time")

# ============================================================
# 全局状态
# ============================================================

alarm_serial = None
frame_queue: Queue = Queue(maxsize=0)
stream_image_ref = [np.zeros((512, 512, 3), dtype=np.uint8)]

capture_started = []                  # 哨兵, 标记 /img 是否已启动采集
_thread_pool = ThreadPoolExecutor()

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
        logger.error('alarm error：{}'.format(str(e)))
        alarm_serial = None


# ============================================================
# 触发机械臂抓取
# ============================================================

def trigger_grab(flags: str):
    """发送 HTTP 请求至 ArmControl 服务, 触发机械臂分拣"""
    data = {"flags": flags, "speed": GRAB_SPEED, "time": GRAB_DELAY}
    requests.post(ARM_URL, json=data, timeout=INFERENCE_TIMEOUT)


# ============================================================
# SSIM 图像比较
# ============================================================

def compare_image(image_gray) -> float:
    """
    将当前帧灰度图与参考图 (yuanshi.jpg) 做 SSIM 比较.

    Returns: SSIM 值 (1=完全相同, 越小差异越大)
    """
    image_a = cv2.imread(REFERENCE_IMAGE)
    image_a = cv2.resize(image_a, (SSIM_WIDTH, SSIM_HEIGHT), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    (score, _diff) = compare_ssim(gray_a, image_gray, full=True)
    logger.info("SSIM: {}".format(score))
    return score


# ============================================================
# 数据查询辅助
# ============================================================

def _day_get(d):
    """生成最近 7 天的日期字符串 (MM-DD), 供统计使用"""
    for i in range(0, 7):
        oneday = datetime.timedelta(days=i)
        day = d - oneday
        date_to = datetime.datetime(day.year, day.month, day.day)
        yield str(date_to)[5:10]


def _get_week_labels() -> list:
    """返回最近 7 天的标签列表 (今日 → 6 天前)"""
    d = datetime.datetime.now()
    days_list = [obj for obj in _day_get(d)]
    return days_list[::-1]


# ============================================================
# 图片 / 视频流辅助函数
# ============================================================

def _read_image_base64(path: str) -> str:
    """读取图片文件并编码为 data:image/jpg;base64, ..."""
    with open(path, 'rb') as f:
        return "data:image/jpg;base64," + str(base64.b64encode(f.read()), encoding='utf-8')


def _draw_fps_label(image, fps: float):
    """在图片左上角绘制帧率标注"""
    label = "(Capture) {:.1f} FPS".format(fps)
    cv2.putText(image, label, (180, 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (38, 0, 255), 1)
    return image


def _encode_frame():
    """将当前 stream_image 编码为 JPEG 字节"""
    frame = None
    try:
        _ret, jpeg = cv2.imencode('.jpg', stream_image_ref[0])
        frame = jpeg.tobytes()
    except Exception as e:
        logger.error('get frame error：%s', e, exc_info=True)
    return frame


def _generate_frames():
    """视频流生成器: MJPEG multipart 响应"""
    while True:
        time.sleep(0.1)
        frame = _encode_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# ============================================================
# Flask 路由 — 配置
# ============================================================

@app.route('/get_conf', methods=['GET'])
def get_conf():
    """读取当前配置文件 (Configuration + defect_name)"""
    config_item = dict(config.items('Configuration'))
    config_item['defect_name'] = defect_name
    return json.dumps([config_item])


@app.route('/change_conf', methods=['POST'])
def change_conf():
    """修改配置文件并写入磁盘, 同时更新内存缓存"""
    global GRAB_SPEED, GRAB_DELAY
    data = json.loads(request.get_data(as_text=True))

    for key in ['time', 'speed', 'grab_position', 'release_position']:
        value = data.get(key, '')
        if len(value) > 0:
            config.set("Configuration", key, value)

    with open("config.ini", 'w', encoding='utf-8') as f:
        config.write(f)

    # 同步更新内存缓存
    GRAB_SPEED = config.get("Configuration", "speed")
    GRAB_DELAY = config.get("Configuration", "time")
    return data


# ============================================================
# Flask 路由 — 历史 / 图片
# ============================================================

@app.route('/get_history', methods=['GET'])
def get_history():
    """获取最新 4 张历史检测图片 (base64)"""
    recent_paths = database.query(
        'distinct path', 'defect_list',
        "where path is not null and path != 'detect.jpg' order by id DESC limit 4"
    )

    image_files = []
    for row in recent_paths:
        image_file = row[0]
        name_rows = database.query(
            'name', 'defect_list', "where path='{}'".format(image_file)
        )
        image_files.append({"name": name_rows, "img": _read_image_base64(image_file)})
    return json.dumps(image_files)


@app.route('/get_original_pic', methods=['GET'])
def get_original_pic():
    """获取最近一次检测的原始图片 (base64)"""
    if os.path.exists(original_image_path):
        return json.dumps([_read_image_base64(original_image_path)])
    return json.dumps([])


@app.route('/get_detect_pic', methods=['GET'])
def get_detect_pic():
    """获取最新一次检测的缺陷标注图片 (base64)"""
    if not os.path.exists(detect_image_path):
        return json.dumps([])

    uuid_rows = database.query(
        'distinct uuid', 'defect_list',
        "where path='detect.jpg' order by id DESC limit 1"
    )
    latest_uuid = uuid_rows[0][0]
    name_rows = database.query(
        'name', 'defect_list',
        "where uuid='{}' and path='detect.jpg'".format(latest_uuid)
    )
    return json.dumps([{"name": name_rows, "img": _read_image_base64(detect_image_path)}])


# ============================================================
# Flask 路由 — 统计
# ============================================================

@app.route('/get_num', methods=['GET'])
def get_num():
    """按缺陷类型统计总数"""
    defect_counts = database.query(
        'name, count(1) AS counts', 'defect_list',
        "where path is not null and path != 'detect.jpg' group by name"
    )
    count_items = [{row[0]: row[1]} for row in defect_counts]
    return json.dumps([count_items])


@app.route('/get_this_month_num', methods=['GET'])
def get_this_month_num():
    """按缺陷类型统计当月总数"""
    defect_counts = database.query(
        'name, count(1) AS counts', 'defect_list',
        "where path is not null and path != 'detect.jpg'"
        " and DATE(CreatedTime) >= DATE('now', 'start of month', '+1 seconds')"
        " group by name"
    )
    count_items = [{row[0]: row[1]} for row in defect_counts]
    return json.dumps([count_items])


@app.route('/get_seven_days_num', methods=['GET'])
def get_seven_days_num():
    """返回最近 7 天每日检测量"""
    today_num = database.select_day_data("+0", "+1")
    yesterday_num = database.select_day_data("-1", "+0")
    two_days_ago_num = database.select_day_data("-2", "-1")
    three_days_ago_num = database.select_day_data("-3", "-2")
    four_days_ago_num = database.select_day_data("-4", "-3")
    five_days_ago_num = database.select_day_data("-5", "-4")
    six_days_ago_num = database.select_day_data("-6", "-5")
    week_labels = _get_week_labels()
    result = {
        week_labels[6]: today_num,
        week_labels[5]: yesterday_num,
        week_labels[4]: two_days_ago_num,
        week_labels[3]: three_days_ago_num,
        week_labels[2]: four_days_ago_num,
        week_labels[1]: five_days_ago_num,
        week_labels[0]: six_days_ago_num,
    }
    return json.dumps(result)


@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    """返回总体统计: 平均推理时间 / 平均得分 / 总数 / 缺陷数"""
    sum_time_rows = database.query(
        'sum(prediction_time)', 'defect_list', 'where prediction_time is not null'
    )
    total_prediction_time = sum_time_rows[0][0]

    sum_score_rows = database.query(
        'sum(score)', 'defect_list', 'where score is not null'
    )
    total_score = sum_score_rows[0][0]

    count_rows = database.query('count()', 'defect_list', 'where prediction_time is not null')
    num = count_rows[0][0]

    average_prediction_time = 0
    average_score = 0
    if num:
        average_prediction_time = total_prediction_time / num
        average_score = (total_score / num) * 100

    total_rows = database.query('count(distinct uuid)', 'defect_list', '')
    total_num = total_rows[0][0]

    defect_rows = database.query(
        'count(distinct uuid)', 'defect_list',
        "where name='ca_shang' or name='zang_wu' or name='zhe_zhou' or name='zhen_kong'"
    )
    defect_num = defect_rows[0][0]

    result = {
        "average_score": average_score,
        "average_prediction_time": average_prediction_time,
        "total_num": total_num,
        "defect_num": defect_num,
    }
    return json.dumps(result)


# ============================================================
# Flask 路由 — 视频流 & 主页
# ============================================================

@app.route('/img')
def video_feed():
    """实时视频流 (MJPEG)"""
    if not capture_started:
        p = Producer(frame_queue, stream_image_ref)
        capture_started.append(p)
        p.start()
        c = Consumer()
        c.start()
    return Response(_generate_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/')
def index():
    """主页面 (Bootstrap 模板)"""
    return render_template('index.html')


# ============================================================
# 图片消费者 (检测 + 推理 + 分拣)
# ============================================================

class Consumer(threading.Thread):
    """
    从 frame_queue 消费帧, 执行 SSIM 触发检测、FPGA 推理、结果处理.

    每 FRAME_SKIP_COUNT 帧处理一次, 其余帧直接丢弃以平衡负载.
    """

    def run(self):
        self._init_tracking()
        database.create_database()

        while True:
            time.sleep(0.01)
            if frame_queue.empty():
                continue

            try:
                frame_start_time = time.time()

                if self._frame_counter == 0:
                    self._process_sampling_frame(frame_start_time)
                else:
                    # 跳过的帧直接出队丢弃
                    frame_queue.get()

                self._frame_counter = (self._frame_counter + 1) % FRAME_SKIP_COUNT

            except Exception as e:
                logger.error('consumer thread error：{}'.format(str(e)))

    # ---- 内部状态 ----

    def _init_tracking(self):
        """初始化帧间追踪变量"""
        self._frame_counter = 0
        self._diff_3ago = 0
        self._diff_2ago = 0
        self._diff_1ago = 0
        self._ssim_history = [0] * SSIM_HISTORY_SIZE
        self._recent_frames = [0, 0]

    # ---- 单帧处理 ----

    def _process_sampling_frame(self, frame_start_time: float):
        """
        处理一个采样帧 (每 FRAME_SKIP_COUNT 帧触发一次).

        流程: 读取 → 预处理 → SSIM → 触发判定 → 推理 → 结果处理
        """
        image = frame_queue.get()
        self._recent_frames[0] = self._recent_frames[1]
        self._recent_frames[1] = image

        # 预处理: 高斯模糊 + 二值化 + 膨胀, 用于计算白色像素占比
        black_image = cv2.resize(image, (SSIM_WIDTH, SSIM_HEIGHT), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(black_image, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        _, img_binary_1 = cv2.threshold(blurred, BINARY_THRESHOLD_1, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(img_binary_1, None, iterations=DILATE_ITERATIONS)
        _, img_binary_2 = cv2.threshold(thresh, BINARY_THRESHOLD_2, 255, cv2.THRESH_BINARY)

        white_ratio = np.count_nonzero(img_binary_2) / img_binary_2.size
        current_ssim = compare_image(black_image)

        # 更新 SSIM 滑动窗口 (左移)
        self._ssim_history[:-1] = self._ssim_history[1:]
        self._ssim_history[-1] = current_ssim
        ssim_std = np.std(self._ssim_history)
        ssim_mean = np.mean(self._ssim_history)

        # 画面长期稳定 → 更新背景参考图
        if ssim_std < STABILITY_STD_THRESHOLD and ssim_mean < STABILITY_MEAN_THRESHOLD and all(self._ssim_history):
            cv2.imwrite(REFERENCE_IMAGE, image)

        # 连续帧差分, 用于定位铝片进入视野后的最小 SSIM 峰值
        diff_curr = current_ssim - self._diff_1ago
        diff_prev = self._diff_1ago - self._diff_2ago
        diff_prev2 = self._diff_2ago - self._diff_3ago

        if current_ssim < SSIM_TRIGGER_THRESHOLD:
            logger.info(
                'current_ssim:{},diff_1ago:{},diff_2ago:{},diff_3ago:{},white_ratio:{}'.format(
                    current_ssim, self._diff_1ago, self._diff_2ago, self._diff_3ago, white_ratio
                )
            )

            # 触发条件: 当前帧差异上升 + 前两帧差异下降 + 白色像素足够
            if diff_curr > 0 > diff_prev2 and diff_prev < 0 and white_ratio > WHITE_RATIO_THRESHOLD:
                self._run_inference_pipeline(image, frame_start_time)

        self._diff_3ago = self._diff_2ago
        self._diff_2ago = self._diff_1ago
        self._diff_1ago = current_ssim

    # ---- 推理管线 ----

    def _run_inference_pipeline(self, image, frame_start_time: float):
        """
        执行 FPGA 推理并处理结果.

        使用前一帧 (recent_frames[0]) 进行标注, 确保看到的是完整的铝片图像.
        """
        annotated_image = self._recent_frames[0]
        original_image = annotated_image.copy()

        uid = str(uuid.uuid1())
        file_name = os.path.join(FILES_DIR, '{}.jpg'.format(uid))
        image_bytes = cv2.imencode(".jpg", annotated_image)[1].tobytes()

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
            # 推理结果为空 → 视为正常
            _thread_pool.submit(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image, processing_rate)

    def _handle_inference_result(
            self, inference_result, uid, file_name, annotated_image, original_image, processing_rate
    ):
        """处理 FPGA 返回的推理结果"""
        all_normal = all(
            detection['class_name'] == LABEL_NORMAL
            for detection in inference_result['result']
        )

        if all_normal:
            _thread_pool.submit(trigger_grab, "OK")
            self._save_normal_result(uid, file_name, annotated_image, original_image, processing_rate)
        else:
            self._handle_defect(inference_result, uid, file_name, annotated_image, original_image, processing_rate)

    def _save_normal_result(self, uid, file_name, annotated_image, original_image, processing_rate):
        """保存正常检测结果: 写入文件 + 数据库"""
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
        """处理缺陷检测结果: 报警 + 抓取 + 标注 + 入库"""
        logger.info("准备报警")
        _thread_pool.submit(trigger_alarm)
        logger.info("报警完成")
        time.sleep(0.1)
        _thread_pool.submit(trigger_grab, "NG")
        logger.info("检测到缺陷，触发报警和抓取动作")

        for detection in inference_result['result']:
            class_name = detection['class_name']
            if class_name == LABEL_NORMAL:
                continue

            class_name_cn = defect_name[class_name]
            score = detection['score']
            loc = detection['loc']
            inference_time = detection['prediction_time']

            annotated_image = self._draw_defect_box(
                annotated_image, loc, class_name_cn, score, processing_rate
            )
            cv2.imwrite(file_name, annotated_image)
            database.insert_data(uid, file_name, class_name, inference_time, score)
            database.insert_data(uid, 'detect.jpg', class_name, None, None)
            logger.info("数据库写入完成")

        cv2.imwrite(original_image_path, original_image)
        cv2.imwrite(detect_image_path, annotated_image)

    def _draw_defect_box(self, image, loc, class_name_cn: str, score: float, fps: float):
        """在图片上绘制缺陷边框和标签"""
        x1, y1, x2, y2 = [int(v) for v in loc]
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

# 保留最新 SAVE_IMAGE_NUM 张缺陷图片, 删除其余
def cleanup():
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
# 入口
# ============================================================

if __name__ == "__main__":
    threading.Thread(target=cleanup).start()
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=7777)
