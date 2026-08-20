"""
Flask 路由模块 — 主服务 (main/) 的检测类 HTTP API.

所有路由通过 Blueprint 注册, 由 api.py 挂载到 Flask app (与机械臂 Blueprint 'arm' 同 app).
共享运行时状态通过 shared.py 访问.
"""

import base64
import datetime
import json
import os
import time

import cv2
import numpy as np
from flask import Blueprint, Response, render_template, request

import database
import shared
from disc_detect import find_disc_robust, crop_box

bp = Blueprint('main', __name__)


# ============================================================
# 路由 — 配置
# ============================================================

@bp.route('/get_conf', methods=['GET'])
def get_conf():
    """读取当前配置文件 (Configuration + defect_name)"""
    config_item = dict(shared.config.items('Configuration'))
    config_item['defect_name'] = shared.defect_name
    return json.dumps([config_item])


@bp.route('/change_conf', methods=['POST'])
def change_conf():
    """修改配置文件并写入磁盘, 同时更新内存缓存"""
    data = json.loads(request.get_data(as_text=True))

    for key in ['time', 'speed', 'camera_distance']:
        value = data.get(key, '')
        if len(value) > 0:
            shared.config.set("Configuration", key, value)

    with open("config.ini", 'w', encoding='utf-8') as f:
        shared.config.write(f)

    # 同步内存缓存 (routes 和 api 共享同一份 shared 模块)
    shared.GRAB_SPEED = shared.config.get("Configuration", "speed")
    shared.GRAB_DELAY = shared.config.get("Configuration", "time")
    return data


# ============================================================
# 路由 — 历史 / 图片
# ============================================================

@bp.route('/get_history', methods=['GET'])
def get_history():
    """分页获取历史检测图片 (base64). 参数: page(1起)/page_size(默认5, 上限100)."""
    try:
        page = int(request.args.get('page', 1))
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int(request.args.get('page_size', 5))
    except (TypeError, ValueError):
        page_size = 5
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)

    total = database.query_value(
        'count(distinct path)', 'defect_list',
        "where path is not null and path != 'detect.jpg'"
    ) or 0
    total_pages = (total + page_size - 1) // page_size if total else 0

    recent_paths = database.query(
        'path', 'defect_list',
        "where path is not null and path != 'detect.jpg'"
        " group by path order by max(id) DESC limit ? offset ?",
        params=(page_size, (page - 1) * page_size)
    )
    image_files = []
    for row in recent_paths:
        image_file = row[0]
        detail_rows = database.query(
            'name, CreatedTime', 'defect_list',
            "where path=? order by id", params=(image_file,)
        )
        image_files.append({
            "name": [[r[0]] for r in detail_rows],
            "img": _image_to_base64(image_file),
            "time": detail_rows[0][1] if detail_rows else ''
        })
    return json.dumps({
        "items": image_files, "total": total,
        "page": page, "page_size": page_size, "total_pages": total_pages,
    }, ensure_ascii=False)


@bp.route('/get_original_pic', methods=['GET'])
def get_original_pic():
    """获取最近一次检测的原始图片 (base64)"""
    if os.path.exists(shared.original_image_path):
        return json.dumps([_image_to_base64(shared.original_image_path)])
    return json.dumps([])


@bp.route('/get_detect_pic', methods=['GET'])
def get_detect_pic():
    """获取最新一次检测的缺陷标注图片 (base64)"""
    if not os.path.exists(shared.detect_image_path):
        return json.dumps([])

    uuid_rows = database.query(
        'distinct uuid', 'defect_list',
        "where path=? order by id DESC limit 1",
        params=('detect.jpg',)
    )
    latest_uuid = uuid_rows[0][0]
    name_rows = database.query(
        'name', 'defect_list',
        "where uuid=? and path=?",
        params=(latest_uuid, 'detect.jpg')
    )
    return json.dumps([{"name": name_rows, "img": _image_to_base64(shared.detect_image_path)}])


# ============================================================
# 路由 — 统计
# ============================================================

@bp.route('/get_num', methods=['GET'])
def get_num():
    """按缺陷类型统计总数"""
    defect_counts = database.query(
        'name, count(1) AS counts', 'defect_list',
        "where path is not null and path != 'detect.jpg' group by name"
    )
    return json.dumps([[{row[0]: row[1]} for row in defect_counts]])


@bp.route('/get_this_month_num', methods=['GET'])
def get_this_month_num():
    """按缺陷类型统计当月总数"""
    defect_counts = database.query(
        'name, count(1) AS counts', 'defect_list',
        "where path is not null and path != 'detect.jpg'"
        " and DATE(CreatedTime) >= DATE('now', 'start of month', '+1 seconds')"
        " group by name"
    )
    return json.dumps([[{row[0]: row[1]} for row in defect_counts]])


@bp.route('/get_today_num', methods=['GET'])
def get_today_num():
    """按缺陷类型统计当日总数"""
    defect_counts = database.query(
        'name, count(1) AS counts', 'defect_list',
        "where path is not null and path != 'detect.jpg'"
        " and CreatedTime >= datetime('now', 'start of day')"
        " group by name"
    )
    return json.dumps([[{row[0]: row[1]} for row in defect_counts]])


@bp.route('/get_num_by_range', methods=['GET'])
def get_num_by_range():
    """按时间范围统计各缺陷类型数量: range=day|week|month|quarter (默认 month).

    day=当日; week=近7天; month=当月; quarter=近3个月 (近似一个季度).
    """
    rng = request.args.get('range', 'month')
    if rng == 'day':
        cond = " and CreatedTime >= datetime('now', 'start of day')"
    elif rng == 'week':
        cond = " and CreatedTime >= datetime('now', '-7 days')"
    elif rng == 'quarter':
        cond = " and CreatedTime >= datetime('now', '-3 months')"
    else:  # month
        cond = " and CreatedTime >= datetime('now', 'start of month', '+1 seconds')"
    defect_counts = database.query(
        'name, count(1) AS counts', 'defect_list',
        "where path is not null and path != 'detect.jpg'" + cond + " group by name"
    )
    return json.dumps([[{row[0]: row[1]} for row in defect_counts]])


def _get_week_labels() -> list:
    """返回最近 7 天的标签列表 (今日 → 6 天前)"""
    d = datetime.datetime.now()
    return [(d - datetime.timedelta(days=i)).strftime('%m-%d') for i in range(6, -1, -1)]


@bp.route('/get_seven_days_num', methods=['GET'])
def get_seven_days_num():
    """返回最近 7 天每日检测量"""
    offsets = ["+0", "-1", "-2", "-3", "-4", "-5", "-6"]
    counts = [database.select_day_data(o, str(int(o) + 1)) for o in offsets]
    week_labels = _get_week_labels()
    result = {week_labels[i]: counts[6 - i] for i in range(7)}
    return json.dumps(result)


@bp.route('/get_statistics', methods=['GET'])
def get_statistics():
    """返回总体统计: 平均推理时间 / 平均得分 / 总数 / 缺陷数"""
    sum_time = database.query_value('sum(prediction_time)', 'defect_list', 'where prediction_time is not null') or 0
    sum_score = database.query_value('sum(score)', 'defect_list', 'where score is not null') or 0
    num = database.query_value('count()', 'defect_list', 'where prediction_time is not null') or 0

    avg_time = 0
    avg_score = 0
    if num:
        avg_time = sum_time / num
        avg_score = (sum_score / num) * 100

    total_num = database.query_value('count(distinct uuid)', 'defect_list', '') or 0
    defect_num = database.query_value(
        'count(distinct uuid)', 'defect_list',
        "where name='ca_shang' or name='zang_wu' or name='zhe_zhou' or name='zhen_kong'"
    ) or 0

    return json.dumps({
        "average_score": avg_score,
        "average_prediction_time": avg_time,
        "total_num": total_num,
        "defect_num": defect_num,
    })


@bp.route('/get_seven_days_by_type', methods=['GET'])
def get_seven_days_by_type():
    """返回最近 7 天每种类型的每日数量, 用于多线趋势图"""
    defect_types = ['ca_shang', 'zhen_kong', 'zang_wu', 'zhe_zhou', 'zheng_chang']
    offsets = ["+0", "-1", "-2", "-3", "-4", "-5", "-6"]
    week_labels = _get_week_labels()

    result = {}
    for i, offset in enumerate(offsets):
        label = week_labels[6 - i]
        day_data = {}
        for dtype in defect_types:
            count = database.query_value(
                'count(1)', 'defect_list',
                "where path is not null and path != 'detect.jpg'"
                " and name=?"
                " and CreatedTime >= datetime('now', 'start of day', ? || ' day')"
                " and CreatedTime <  datetime('now', 'start of day', ? || ' day')",
                params=(dtype, offset, str(int(offset) + 1))
            ) or 0
            day_data[dtype] = count
        result[label] = day_data

    return json.dumps(result)


# ============================================================
# 路由 — 标定模式
# ============================================================

@bp.route('/calibration', methods=['POST'])
def calibration():
    """启动/停止传送带速度标定"""
    data = json.loads(request.get_data(as_text=True))
    mode = data.get('mode', '')
    if mode == 'start':
        shared.calibration_active = True
        shared.calibration_samples.clear()
    elif mode == 'stop':
        shared.calibration_active = False
    return json.dumps({'mode': mode, 'active': shared.calibration_active})


@bp.route('/calibration_status', methods=['GET'])
def calibration_status():
    """获取标定状态与结果"""
    samples = list(shared.calibration_samples)
    if len(samples) >= 3:
        median_speed = sorted(samples)[len(samples) // 2]
        camera_dist = float(shared.config.get("Configuration", "camera_distance"))
        suggested_delay = round(camera_dist / median_speed, 1) if median_speed > 0 else 0
    else:
        median_speed = 0
        suggested_delay = 0
    return json.dumps({
        'active': shared.calibration_active,
        'sample_count': len(samples),
        'speed': round(median_speed, 1),
        'suggested_delay': suggested_delay,
    })


# ============================================================
# 路由 — 视频流 & 主页
# ============================================================

def _image_to_base64(path: str) -> str:
    """读取图片文件并编码为 data:image/jpg;base64, ... (文件已删时返回空串)"""
    try:
        with open(path, 'rb') as f:
            return "data:image/jpg;base64," + str(base64.b64encode(f.read()), encoding='utf-8')
    except (FileNotFoundError, IOError):
        return ""


def _encode_frame() -> bytes:
    """将当前视频帧编码为 JPEG 字节"""
    try:
        _ret, jpeg = cv2.imencode('.jpg', shared.stream_image_ref[0])
        return jpeg.tobytes()
    except Exception as e:
        shared.logger.error('get frame error：%s', e, exc_info=True)
        return None


def _generate_frames():
    """视频流生成器: MJPEG multipart 响应"""
    while True:
        time.sleep(0.1)
        frame = _encode_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def _generate_debug_frames():
    """SSIM 调试帧生成器: 二值掩码 + SSIM/ratio 叠加"""
    while True:
        time.sleep(0.1)
        mask = getattr(shared, 'debug_mask', None)
        if mask is None:
            continue
        display = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
        ssim = getattr(shared, 'last_ssim', -1)
        wr = getattr(shared, 'last_white_ratio', -1)
        cv2.putText(display, 'SSIM={:.3f}  ratio={:.2f}'.format(ssim, wr),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 1)
        _ret, jpeg = cv2.imencode('.jpg', display)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@bp.route('/debug_mask')
def debug_mask():
    """SSIM 调试流: 实时二值掩码 + SSIM/ratio"""
    return Response(_generate_debug_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


def _generate_stage_frames():
    """软处理中间结果调试流: diff / binary / mask 三列并排"""
    while True:
        time.sleep(0.1)
        stages = getattr(shared, 'debug_intermediates', None)
        if not stages:
            continue
        diff = stages.get('diff')
        if diff is None:
            continue
        panels = []
        for key in ('diff', 'binary', 'mask'):
            img = stages.get(key)
            if img is None:
                img = np.zeros_like(diff)
            up = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
            panels.append(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR))
        canvas = np.hstack(panels)
        _ret, jpeg = cv2.imencode('.jpg', canvas)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@bp.route('/debug_stages')
def debug_stages():
    """软处理中间结果调试流: diff/binary/mask 逐段可视化"""
    return Response(_generate_stage_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


def _disc_frame_overlay(display, frame):
    """对实时帧定圆, 在 display (640×480) 上叠加绿圆与黄裁剪框.

    Returns: (found, detail). detail: 检出时为圆信息/方法, 否则为 reject 原因.
    """
    method = getattr(shared, 'disc_method', 'mask')
    cfg = getattr(shared, 'disc_cfg', None)
    circle, info = find_disc_robust(
        frame, method=method, cfg=cfg,
        background=getattr(shared, 'debug_background', None),
    )
    h, w = frame.shape[:2]
    sx, sy = 640.0 / w, 480.0 / h
    if circle is not None:
        cx, cy, r = (int(v) for v in circle)
        cv2.circle(display, (int(cx * sx), int(cy * sy)), max(int(r * sx), 1), (0, 255, 0), 2)
        cv2.circle(display, (int(cx * sx), int(cy * sy)), 2, (0, 255, 0), -1)
        # 实时预览裁剪框 (与 api.Consumer._smart_crop 相同逻辑, 共用 crop_box)
        margin_ratio = getattr(shared, 'disc_margin_ratio', 0.10)
        x0, y0, side = crop_box((cx, cy, r), h, w, margin_ratio)
        cv2.rectangle(
            display,
            (int(x0 * sx), int(y0 * sy)),
            (int((x0 + side) * sx), int((y0 + side) * sy)),
            (0, 255, 255), 2,
        )
        detail = 'circle=({:.0f},{:.0f}) r={:.0f} [{}]'.format(
            cx, cy, r, (info or {}).get('used_method', '?'))
        return True, detail
    detail = 'reject: {}'.format((info or {}).get('reject') or '未知')
    return False, detail


def _generate_disc_debug_frames():
    """圆识别调试流: 对实时帧直接跑 find_disc_robust, 叠加检出圆(绿)与裁剪框(黄).

    注意: 这里是实时检测, 与生产推理管线(SSIM 触发后才定圆)解耦——调试时把
    铝片/任意物体放进画面即可立刻看到检出结果与失败原因.
    """
    while True:
        time.sleep(0.1)
        frame = shared.stream_image_ref[0]
        if frame is None:
            continue
        if frame.ndim == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()
        display = cv2.resize(display, (640, 480), interpolation=cv2.INTER_AREA)

        found, detail = _disc_frame_overlay(display, frame)
        enabled = getattr(shared, 'disc_enabled', 1)
        method = getattr(shared, 'disc_method', 'mask')
        cv2.putText(display, 'disc: enabled={} method={} found={}'.format(enabled, method, found),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(display, detail, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        _ret, jpeg = cv2.imencode('.jpg', display)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@bp.route('/debug_disc')
def debug_disc():
    """圆识别 + 智能裁剪调试流: 实时帧叠加检出圆(绿)与裁剪框(黄)"""
    return Response(_generate_disc_debug_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


def _generate_disc_stream_frames():
    """主监控页视频流: 实时帧 + 圆检测叠加 (绿圆 + 黄裁剪框, 无文字)"""
    while True:
        time.sleep(0.1)
        frame = shared.stream_image_ref[0]
        if frame is None:
            continue
        if frame.ndim == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()
        display = cv2.resize(display, (640, 480), interpolation=cv2.INTER_AREA)

        _found, _detail = _disc_frame_overlay(display, frame)
        _ret, jpeg = cv2.imencode('.jpg', display)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@bp.route('/img_disc')
def img_disc():
    """主监控页视频流: 实时帧叠加检出圆(绿)与裁剪框(黄), 不带调试文字"""
    return Response(_generate_disc_stream_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


def _fmt_ts(ts):
    """时间戳 → 本地时间字符串 (None/0 返回 None)"""
    if not ts:
        return None
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))


@bp.route('/get_status')
def get_status():
    """触发/推理链路健康状态 (JSON): 现场诊断"铝片经过但无推理图"断点位置.

    判读:
      - trigger_state 恒为 0 且 last_trigger_time=None → 触发未发生 (基线/前景信号问题)
      - last_trigger_time 有更新但 last_inference_time=None → 卡在 TRACKING (铝片未离开视野)
      - last_inference_time 有更新但无图片 → 推理/存图失败 (看 last_consumer_error)
      - last_consumer_error 非 None → Consumer 线程异常详情
    """
    st = getattr(shared, 'trigger_state', 0)
    baseline = getattr(shared, 'baseline_stats', None)
    return json.dumps({
        'trigger_state': st,
        'state_name': {0: 'IDLE', 1: 'TRACKING', 2: 'COOLDOWN'}.get(st, '?'),
        'last_trigger_time': _fmt_ts(getattr(shared, 'last_trigger_time', None)),
        'last_inference_time': _fmt_ts(getattr(shared, 'last_inference_time', None)),
        'last_consumer_error': getattr(shared, 'last_consumer_error', None),
        'last_ssim': getattr(shared, 'last_ssim', None),
        'last_white_ratio': getattr(shared, 'last_white_ratio', None),
        'baseline': None if baseline is None else {
            'mean': round(float(baseline[0]), 5),
            'std': round(float(baseline[1]), 5),
            'n': int(baseline[2]),
        },
        'disc_enabled': getattr(shared, 'disc_enabled', 1),
        'disc_method': getattr(shared, 'disc_method', 'mask'),
        'last_disc': getattr(shared, 'last_disc', None),
        'last_crop_box': getattr(shared, 'last_crop_box', None),
    }, ensure_ascii=False)


@bp.route('/img')
def video_feed():
    """实时视频流 (MJPEG)"""
    return Response(_generate_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')


@bp.route('/')
def index():
    """主页面 (Bootstrap 模板)"""
    return render_template('index.html')
