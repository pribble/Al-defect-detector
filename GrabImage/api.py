from concurrent.futures import ThreadPoolExecutor
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import sys
import time
import uuid
import requests
import numpy as np
import threading
import os
import cv2
from ctypes import *
from flask import Flask, render_template, Response, request
from PIL import Image, ImageFont, ImageDraw
import json
import base64
import configparser
from queue import Queue
import datetime
import serial
import database
from flask_cors import CORS
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from logger import setup_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, 'files')
ORIGINAL_DIR = os.path.join(BASE_DIR, 'original_files')
DETECT_DIR = os.path.join(BASE_DIR, 'detect_files')

logger = setup_log('detect', 'detect_server.log')
# 缺陷对应关系，将拼音转化为中文
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
defect_name = dict(config.items('defect_name'))

# 报警器串口指令
alarm_cmd_hex = '7E FF 06 03 00 00 01 EF'
alarm_cmd = bytes.fromhex(alarm_cmd_hex)
alarm_serial = None

sys.path.append("../MvImport")
from MvCameraControl_class import *

app = Flask(__name__)
CORS(app, supports_credentials=True)
# 保留缺陷图片数量
save_image_num = 10
# 缺陷图片路径
defect_image_file = FILES_DIR + '/'
# 当前原始图片路径
original_image_path = os.path.join(ORIGINAL_DIR, 'original.jpg')
# 当前缺陷图片存放路径
detect_image_path = os.path.join(DETECT_DIR, 'detect.jpg')

# 创建存放缺陷检测图片目录
os.makedirs(FILES_DIR, exist_ok=True)
# 创建存放当前缺陷检测图片存放目录
os.makedirs(DETECT_DIR, exist_ok=True)
# 创建存放当前原始图片存放目录
os.makedirs(ORIGINAL_DIR, exist_ok=True)

_thread_pool = ThreadPoolExecutor()

capture_started = []

# 图片处理队列，先进先出
frame_queue = Queue(maxsize=0)
# 初始化图片
stream_image = np.zeros((512, 512, 3), dtype=np.uint8)

# 报警 (异步)
def trigger_alarm():
    global alarm_serial
    if alarm_serial is None:
        alarm_serial = serial.Serial("/dev/BAOJING", 9600, 8, stopbits=1)
    try:
        alarm_serial.write(alarm_cmd)
    except Exception as e:
        logger.error('bao jing error：{}'.format(str(e)))
        alarm_serial = serial.Serial("/dev/BAOJING", 9600, 8, stopbits=1)
        time.sleep(0.1)
        alarm_serial.write(alarm_cmd)
        logger.info("reboot init alarm serial")

def initMvCamera():
    while True:
        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        logger.info("SDKVersion[0x%x]" % SDKVersion)

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            logger.error("enum devices fail! ret[0x%x], 10秒后重试..." % ret)
            time.sleep(10)
            continue

        if deviceList.nDeviceNum == 0:
            logger.error("find no device! 10秒后重试...")
            time.sleep(10)
            continue

        logger.info("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                logger.info("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                logger.info("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                logger.info("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                logger.info("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                logger.info("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                logger.info("user serial number: %s" % strSerialNumber)

        nConnectionNum = 0

        if int(nConnectionNum) >= deviceList.nDeviceNum:
            logger.error("intput error! 10秒后重试...")
            time.sleep(10)
            continue

        # ch:创建相机实例 | en:Creat Camera Object
        cam = MvCamera()
        # ch:选择设备并创建句柄| en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            logger.error("create handle fail! ret[0x%x], 10秒后重试..." % ret)
            time.sleep(10)
            continue

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            logger.error("open device fail! ret[0x%x], 10秒后重试..." % ret)
            time.sleep(10)
            continue

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            logger.error("set trigger mode fail! ret[0x%x], 10秒后重试..." % ret)
            time.sleep(10)
            continue

        # ch:获取数据包大小 | en:Get payload size
        stParam =  MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            logger.error("get payload size fail! ret[0x%x], 10秒后重试..." % ret)
            time.sleep(10)
            continue
        nPayloadSize = stParam.nCurValue

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            logger.error("start grabbing fail! ret[0x%x], 10秒后重试..." % ret)
            time.sleep(10)
            continue

        data_buf = (c_ubyte * nPayloadSize)()

        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        return cam, data_buf, nPayloadSize, stFrameInfo

# 获取7天数据
def day_get(d):
    for i in range(0, 7):
        oneday = datetime.timedelta(days=i)
        day = d - oneday
        date_to = datetime.datetime(day.year, day.month, day.day)
        yield str(date_to)[5:10]

def get_day():
    d = datetime.datetime.now()
    days_generator = day_get(d)
    days_list = []
    for obj in days_generator:
        days_list.append(obj)
    week_labels = days_list[::-1]
    return week_labels


# 抓取缺陷铝片
def trigger_grab(flags):
    speed = config.get("Configuration", "speed")
    delay = config.get("Configuration", "time")
    url = 'http://172.16.68.111:8899/grab'
    data = {"flags": flags, "speed": speed, "time": delay}
    requests.post(url, json=data, timeout=30)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


# 获取抓取配置文件信息
@app.route('/get_conf', methods=['GET'])
def get_conf():
    config_list = []
    config_item = dict(config.items('Configuration'))
    config_item['defect_name'] = defect_name
    config_list.append(config_item)
    return json.dumps(config_list)


# 修改抓取配置文件信息
@app.route('/change_conf', methods=['POST'])
def change_conf():
    data = json.loads(request.get_data(as_text=True))
    delay = data['time']
    if len(delay) > 0:
        config.set("Configuration", "time", delay)
    speed = data['speed']
    if len(speed) > 0:
        config.set("Configuration", "speed", speed)
    grab_position = data['grab_position']
    if len(grab_position) > 0:
        config.set("Configuration", "grab_position", grab_position)
    release_position = data['release_position']
    if len(release_position) > 0:
        config.set("Configuration", "release_position", release_position)
    # 写入配置文件
    with open("config.ini", 'w', encoding='utf-8') as f:
        config.write(f)
    return data


# 获取最新4张历史检测图片
@app.route('/get_history', methods=['GET'])
def get_history():
    sql_non_null = database.select_instructions('*', 'defect_list', 'where path is not null')
    sql_no_detect = database.select_instructions('*', '(' + sql_non_null + ')',
        "where path is not 'detect.jpg' order by id DESC")
    sql_recent = database.select_instructions('distinct path', '(' + sql_no_detect + ')', 'limit 4')
    recent_paths = database.select_data(sql_recent)
    image_files = []
    for i in range(0, len(recent_paths)):
        image_file = recent_paths[i][0]
        with open(image_file, 'rb') as f:
            image = f.read()
            image_base64 = "data:image/jpg;base64," + str(base64.b64encode(image), encoding='utf-8')
            sql_name = database.select_instructions('name', 'defect_list',
                "where path='{}'".format(image_file))
            name_rows = database.select_data(sql_name)
            name = name_rows
            result_item = {"name": name, "img": image_base64}
            image_files.append(result_item)
    return json.dumps(image_files)


# 获取检测原始图片
@app.route('/get_original_pic', methods=['GET'])
def get_original_pic():
    image_files = []
    if os.path.exists(original_image_path):
        with open(original_image_path, 'rb') as f:
            image = f.read()
            image_base64 = "data:image/jpg;base64," + str(base64.b64encode(image), encoding='utf-8')
            image_files.append(image_base64)
    return json.dumps(image_files)


# 获取检测后带有缺陷的图片
@app.route('/get_detect_pic', methods=['GET'])
def get_detect_pic():
    image_files = []
    if os.path.exists(detect_image_path):
        sql_by_path = database.select_instructions('*', 'defect_list',
            "where path='detect.jpg' order by id DESC")
        sql_latest_uuid = database.select_instructions('distinct uuid', '(' + sql_by_path + ')', 'limit 1')
        uuid_rows = database.select_data(sql_latest_uuid)
        latest_uuid = uuid_rows[0][0]
        sql_names = database.select_instructions('name', 'defect_list',
            "where uuid='{}' and path='detect.jpg'".format(latest_uuid))
        name_rows = database.select_data(sql_names)
        defect_names = name_rows
        with open(detect_image_path, 'rb') as f:
            image = f.read()
            image_base64 = "data:image/jpg;base64," + str(base64.b64encode(image), encoding='utf-8')
            result_item = {"name": defect_names, "img": image_base64}
            image_files.append(result_item)
    return json.dumps(image_files)


# 获取检测数
@app.route('/get_num', methods=['GET'])
def get_num():
    sql_non_null = database.select_instructions('*', 'defect_list', 'where path is not null')
    sql_no_detect = database.select_instructions('*', '(' + sql_non_null + ')',
        "where path is not 'detect.jpg'")
    sql_counts = database.select_instructions('name, count(1) AS counts', '(' + sql_no_detect + ')',
        'group by name')
    defect_counts = database.select_data(sql_counts)
    wrapper = []
    count_items = []
    for i in range(0, len(defect_counts)):
        count_item = {defect_counts[i][0]: defect_counts[i][1]}
        count_items.append(count_item)
    wrapper.append(count_items)
    return json.dumps(wrapper)


# 获取30天内检测信息
@app.route('/get_this_month_num', methods=['GET'])
def get_this_month_num():
    sql_non_null = database.select_instructions('*', 'defect_list', 'where path is not null')
    sql_no_detect = database.select_instructions('*', '(' + sql_non_null + ')',
        "where path is not 'detect.jpg'")
    sql_this_month = database.select_instructions('*', '(' + sql_no_detect + ')',
        "WHERE DATE(CreatedTime) >= DATE('now', 'start of month', '+1 seconds')")
    sql_counts = database.select_instructions('name, count(1) AS counts', '(' + sql_this_month + ')',
        'group by name')
    defect_counts = database.select_data(sql_counts)
    wrapper = []
    count_items = []
    for i in range(0, len(defect_counts)):
        count_item = {defect_counts[i][0]: defect_counts[i][1]}
        count_items.append(count_item)
    wrapper.append(count_items)
    return json.dumps(wrapper)


# 获取一周内的检测信息
@app.route('/get_seven_days_num', methods=['GET'])
def get_seven_days_num():
    today_num = database.select_day_data("+0", "+1")
    yesterday_num = database.select_day_data("-1", "+0")
    two_days_ago_num = database.select_day_data("-2", "-1")
    three_days_ago_num = database.select_day_data("-3", "-2")
    four_days_ago_num = database.select_day_data("-4", "-3")
    five_days_ago_num = database.select_day_data("-5", "-4")
    six_days_ago_num = database.select_day_data("-6", "-5")
    week_labels = get_day()
    result = {
        week_labels[6]: today_num, week_labels[5]: yesterday_num,
        week_labels[4]: two_days_ago_num, week_labels[3]: three_days_ago_num,
        week_labels[2]: four_days_ago_num, week_labels[1]: five_days_ago_num,
        week_labels[0]: six_days_ago_num,
    }
    return json.dumps(result)


# 获取统计信息
@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    sql_with_time = database.select_instructions('*', 'defect_list', 'where prediction_time is not null')
    sql_sum_time = database.select_instructions('sum(prediction_time)', '(' + sql_with_time + ')', '')
    sum_time_rows = database.select_data(sql_sum_time)
    total_prediction_time = sum_time_rows[0][0]
    sql_with_score = database.select_instructions('*', 'defect_list', 'where score is not null')
    sql_sum_score = database.select_instructions('sum(score)', '(' + sql_with_score + ')', '')
    sum_score_rows = database.select_data(sql_sum_score)
    total_score = sum_score_rows[0][0]
    sql_count = database.select_instructions('count()', '(' + sql_with_time + ')', '')
    count_rows = database.select_data(sql_count)
    num = count_rows[0][0]
    average_prediction_time = 0
    average_score = 0
    if num:
        average_prediction_time = total_prediction_time / num
        average_score = (total_score / num) * 100
    sql_total = database.select_instructions('count(distinct uuid)', 'defect_list', '')
    total_rows = database.select_data(sql_total)
    total_num = total_rows[0][0]
    sql_defects = database.select_instructions('count(distinct uuid)', 'defect_list',
        "where name='ca_shang' or name='zang_wu' or name='zhe_zhou' or name='zhen_kong'")
    defect_rows = database.select_data(sql_defects)
    defect_num = defect_rows[0][0]
    result = {
        "average_score": average_score,
        "average_prediction_time": average_prediction_time,
        "total_num": total_num,
        "defect_num": defect_num,
    }
    return json.dumps(result)


# 将视频流图片和原始图片进行比较，获取对应的差异值。值为1表示两张图片一样，值越小，差异越大。
def compare_image(image):
    # 传入图片路径，读取图片
    image_a = cv2.imread('yuanshi.jpg')
    image_a = cv2.resize(image_a, (64, 48), interpolation=cv2.INTER_AREA)
    #image_b = cv2.imread(opt.image)
    # 使用色彩空间转化函数 cv2.cvtColor( )进行色彩空间的转换
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    #gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    # 计算图像相似度并圈出不同处
    t0 = time.time()
    (score, diff) = compare_ssim(gray_a, image, full=True)
    t1 = time.time()
    #logger.info("Compare_time is {}ms ".format((t1-t0)*1000))
    logger.info("SSIM: {}".format(score))
    return score

class Producer(threading.Thread):
    def run(self):
        cam, data_buf, nPayloadSize, stFrameInfo = initMvCamera()
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 10000)
            if ret != 0:
                logger.info('海康相机状态：{}'.format(ret))
            if ret == 2147483655:
                logger.info("相机重启")
                ret = cam.MV_CC_StopGrabbing()
                #ch:关闭设备 | Close device
                ret = cam.MV_CC_CloseDevice()
                # ch:销毁句柄 | Destroy handle
                ret = cam.MV_CC_DestroyHandle()
                cam, data_buf, nPayloadSize, stFrameInfo = initMvCamera()
                continue
            time.sleep(0.01)
            if ret == 0:
                image = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                #3072*2048 768 * 512
                image = cv2.resize(image, (int(stFrameInfo.nWidth/4), int(stFrameInfo.nHeight/4)), interpolation=cv2.INTER_AREA)
                #图片处理队列
                frame_queue.put(image)
                #视频流队列
                global stream_image
                stream_image = image


class Consumer(threading.Thread):
    def run(self):
        frame_counter = 0
        frame_skip_count = 10
        diff_3ago, diff_2ago, diff_1ago = 0, 0, 0
        ssim_history = [0] * 9
        recent_frames = [0, 0]
        database.create_database()
        while True:
            time.sleep(0.01)
            if not frame_queue.empty():
                try:
                    frame_start_time = time.time()
                    # 每10帧处理一次
                    if frame_counter == 0:
                        image = frame_queue.get()
                        # 保存最近的两张图片,当触发检测之后,获取前一张图片
                        recent_frames[0] = recent_frames[1]
                        recent_frames[1] = image
                        # 图片处理，高斯滤波、膨胀、二值化，用于得到只有黑白像素的图片，用于判断是否有铝片进入视野
                        black_image = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA)
                        blurred = cv2.GaussianBlur(black_image, (21, 21), 0)
                        _, img_binary_1 = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
                        thresh = cv2.dilate(img_binary_1, None, iterations=4)
                        _, img_binary_2 = cv2.threshold(thresh, 0.3, 255, cv2.THRESH_BINARY)
                        # 获取白色像素的占比
                        white_pixel_count = np.count_nonzero(img_binary_2)
                        x, y = img_binary_2.shape
                        white_ratio = white_pixel_count / (x * y)
                        current_ssim = compare_image(black_image)
                        ssim_history[:-1] = ssim_history[1:]
                        ssim_history[-1] = current_ssim
                        # 计算标准差
                        ssim_std = np.std(ssim_history)
                        # 计算均值
                        ssim_mean = np.mean(ssim_history)
                        # 判断视野是否长期处于稳定状态
                        if ssim_std < 0.01 and ssim_mean < 0.8 and all(ssim_history):
                            #yuanshi_img = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
                            cv2.imwrite("yuanshi.jpg", image)
                        # 获取当前帧和前一帧的差异值大小
                        diff_curr = current_ssim - diff_1ago
                        # 获取前一帧和前两帧的差异值大小
                        diff_prev = diff_1ago - diff_2ago
                        # 获取前两帧和前三帧的差异值大小
                        diff_prev2 = diff_2ago - diff_3ago
                        if current_ssim < 0.9:
                            # 判断铝片进入视野后的差异值大小，其中铝片全部进入视野差异值最小，判断最小峰值处用于推理
                            logger.info('current_ssim:{},diff_1ago:{},diff_2ago:{},diff_3ago:{},white_ratio:{}'.format(
                                str(current_ssim), str(diff_1ago), str(diff_2ago), str(diff_3ago), str(white_ratio)))
                            if diff_curr > 0 and diff_prev < 0 and diff_prev2 < 0 and white_ratio > 0.1:
                                uid = str(uuid.uuid1())
                                file_name = os.path.join(FILES_DIR, '{}.jpg'.format(uid))
                                annotated_image = recent_frames[0]
                                original_image = annotated_image.copy()
                                image_bytes = cv2.imencode(".jpg", annotated_image)[1].tobytes()
                                url = 'http://172.16.68.110:8080/predict'
                                files = {'image_file': image_bytes}
                                inference_start_time = time.time()
                                logger.info('开始检测')
                                response = requests.post(url, files=files, timeout=30)
                                logger.info('推理服务耗费：{}'.format(time.time() - inference_start_time))
                                processing_rate = 10 / (time.time() - frame_start_time)
                                inference_result = json.loads(response.text)
                                logger.info(inference_result)
                                if inference_result['len'] > 0:
                                    all_normal = all(
                                        detection['class_name'] == 'zheng_chang'
                                        for detection in inference_result['result'])
                                    if all_normal:
                                        _thread_pool.submit(trigger_grab, "OK")
                                        try:
                                            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
                                        except:
                                            annotated_image = annotated_image
                                        processing_rate_label = "(Capture) {:.1f} FPS".format(processing_rate)
                                        cv2.putText(annotated_image, processing_rate_label, (180, 30),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.3, (38, 0, 255), 1)
                                        cv2.imwrite(file_name, annotated_image)
                                        database.insert_data(uid, file_name, 'zheng_chang', None, None)
                                        cv2.imwrite(original_image_path, original_image)
                                        if os.path.exists(detect_image_path):
                                            os.remove(detect_image_path)
                                    else:
                                        logger.info("准备报警")
                                        _thread_pool.submit(trigger_alarm)
                                        logger.info("报警完成")
                                        time.sleep(0.1)
                                        _thread_pool.submit(trigger_grab, "NG")
                                        logger.info("检测到缺陷，触发报警和抓取动作")
                                        for detection in inference_result['result']:
                                            class_name = detection['class_name']
                                            if class_name != 'zheng_chang':
                                                class_name_cn = defect_name[class_name]
                                                score = detection['score']
                                                loc = detection['loc']
                                                inference_time = detection['prediction_time']
                                                processing_rate_label = "(Capture) {:.1f} FPS".format(processing_rate)
                                                x1 = int(loc[0])
                                                y1 = int(loc[1])
                                                x2 = int(loc[2])
                                                y2 = int(loc[3])
                                                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                                                try:
                                                    pil_image = Image.fromarray(
                                                        cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB))
                                                except:
                                                    pil_image = Image.fromarray(annotated_image)
                                                font = ImageFont.truetype(
                                                    os.path.join(BASE_DIR, 'Font/platech.ttf'), 9, encoding="utf-8")
                                                draw = ImageDraw.Draw(pil_image)
                                                draw.text((x1 - 10, y1 - 10),
                                                    '{} {:.2f}%'.format(class_name_cn, score * 100),
                                                    font=font, fill="green")
                                                annotated_image = np.array(pil_image)
                                                cv2.putText(annotated_image, processing_rate_label, (180, 30),
                                                    cv2.FONT_HERSHEY_COMPLEX, 0.3, (38, 0, 255), 1)
                                                cv2.imwrite(file_name, annotated_image)
                                                # 缺陷信息插入数据库
                                                database.insert_data(uid, file_name, class_name, inference_time, score)
                                                database.insert_data(uid, 'detect.jpg', class_name, None, None)
                                                logger.info("数据库写入完成")
                                                cv2.imwrite(original_image_path, original_image)
                                                cv2.imwrite(detect_image_path, annotated_image)
                                else:
                                    _thread_pool.submit(trigger_grab, "OK")
                                    try:
                                        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
                                    except:
                                        annotated_image = annotated_image
                                    processing_rate_label = "(Capture) {:.1f} FPS".format(processing_rate)
                                    cv2.putText(annotated_image, processing_rate_label, (180, 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (38, 0, 255), 1)
                                    cv2.imwrite(file_name, annotated_image)
                                    database.insert_data(uid, file_name, 'zheng_chang', None, None)
                                    cv2.imwrite(original_image_path, original_image)
                                    if os.path.exists(detect_image_path):
                                        os.remove(detect_image_path)

                        diff_3ago = diff_2ago
                        diff_2ago = diff_1ago
                        diff_1ago = current_ssim
                    else:
                        image = frame_queue.get()

                    frame_counter = frame_counter + 1
                    frame_counter = frame_counter % frame_skip_count
                except Exception as e:
                    logger.error('consumer thread error：{}'.format(str(e)))


def get_frame():
    frame = None
    try:
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', stream_image)
        frame = jpeg.tobytes()
    except Exception as e:
        logger.error('get frame error：%s', e, exc_info=True)
    return frame


def gen():
    while True:
        time.sleep(0.1)
        frame = get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/img')  # 这个地址返回视频流响应
def video_feed():
    if not capture_started:
        p = Producer()
        capture_started.append(p)
        p.start()
        c = Consumer()
        c.start()
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


class ImageRetentionCleanup:
    """清理旧图片: 保留最新10张缺陷图片, 删除其余"""
    def __init__(self):
        self.__delete_thread = threading.Thread(target=self._delete)
        self.__delete_thread.start()

    def _delete(self):
        sql_recent = database.select_instructions('distinct path', 'defect_list',
            "where path is not null and path != 'detect.jpg' order by id DESC limit 10")
        recent_rows = database.select_data(sql_recent)
        keep_files = []
        for i in range(0, len(recent_rows)):
            image_file = recent_rows[i][0]
            keep_files.append(image_file)
        file_list = os.listdir(defect_image_file)
        for f in file_list:
            if (defect_image_file + f) not in keep_files:
                os.remove(defect_image_file + f)

if __name__ == "__main__":
    ImageRetentionCleanup()
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=7777)
