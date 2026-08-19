"""
Hikvision 工业相机模块

封装 MVS SDK 的相机枚举、初始化和帧读取。
"""

import sys
import threading
import time
from ctypes import *

import cv2
import numpy as np

sys.path.append("../MvImport")
from MvCameraControl_class import *

from logger import setup_log

logger = setup_log('main', 'server.log')

# 相机错误码 (MVS SDK)
CAMERA_NEED_RESTART = 2147483655

# 宽高比上限 4:3 (640:480)
_MAX_ASPECT_RATIO = 4.0 / 3.0


def _crop_to_max_aspect(image: np.ndarray) -> np.ndarray:
    """宽高比超过 4:3 时, 裁长边使其压缩到 4:3; 未超过则不变."""
    h, w = image.shape[:2]
    ratio = w / h
    if ratio > _MAX_ASPECT_RATIO:
        # 太宽 → 裁左右
        new_w = int(h * _MAX_ASPECT_RATIO)
        offset = (w - new_w) // 2
        return image[:, offset:offset + new_w]
    elif ratio < 1.0 / _MAX_ASPECT_RATIO:
        # 太高 → 裁上下
        new_h = int(w * _MAX_ASPECT_RATIO)
        offset = (h - new_h) // 2
        return image[offset:offset + new_h, :]
    return image


class FrameBuffer:
    """单槽帧缓冲, 线程安全, 丢弃旧帧不阻塞。

    Producer (相机线程) 写, Consumer (检测线程) 读。
    """

    __slots__ = ('_frame', '_lock')

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            self._frame = item

    def get(self):
        with self._lock:
            return self._frame


# 模块级单例: Producer 写入, Consumer 读取
frame_queue = FrameBuffer()


# ============================================================
# 初始化辅助
# ============================================================

def _mv_ok(step_name: str, ret: int) -> bool:
    """检查 MVS SDK 返回值, 失败时 log + sleep 10 s, 返回 False 表示继续."""
    if ret == 0:
        return True
    logger.error("%s fail! ret[0x%x], 10秒后重试...", step_name, ret)
    time.sleep(10)
    return False


def _log_device_info(deviceList):
    """打印所有枚举到的相机设备信息"""
    for i in range(0, deviceList.nDeviceNum):
        dev = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if dev.nTLayerType == MV_GIGE_DEVICE:
            model = "".join(chr(p) for p in dev.SpecialInfo.stGigEInfo.chModelName if p != 0)
            ip = ".".join(str((dev.SpecialInfo.stGigEInfo.nCurrentIp >> shift) & 0xff)
                          for shift in [24, 16, 8, 0])
            logger.info("gige device [%d]: %s @ %s", i, model, ip)
        elif dev.nTLayerType == MV_USB_DEVICE:
            model = "".join(chr(p) for p in dev.SpecialInfo.stUsb3VInfo.chModelName if p != 0)
            serial = "".join(chr(p) for p in dev.SpecialInfo.stUsb3VInfo.chSerialNumber if p != 0)
            logger.info("u3v device [%d]: %s (sn: %s)", i, model, serial)


# ============================================================
# 初始化相机
# ============================================================

def init_camera():
    """
    枚举并初始化 Hikvision 工业相机.

    Returns: (cam, data_buf, nPayloadSize, stFrameInfo)
    失败时循环重试 (从枚举开始重新执行).
    """
    while True:
        logger.info("SDKVersion[0x%x]", MvCamera.MV_CC_GetSDKVersion())

        # --- 枚举设备 ---
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            logger.error("find no device! sleep 10 s...")
            time.sleep(10)
            continue

        logger.info("Find %d devices!", deviceList.nDeviceNum)
        _log_device_info(deviceList)

        # --- 选第一个设备, 创建句柄 + 打开 ---
        cam = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        if not _mv_ok("create handle", cam.MV_CC_CreateHandle(stDeviceList)):
            continue
        if not _mv_ok("open device", cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)):
            continue
        if not _mv_ok("set trigger mode", cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)):
            continue

        # --- 手动固定曝光 (避免自动曝光导致 SSIM 误触) ---
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 0=Off
        cam.MV_CC_SetEnumValue("GainAuto", 0)  # 0=Off
        cam.MV_CC_SetFloatValue("ExposureTime", 6000.0)
        cam.MV_CC_SetFloatValue("Gain", 0.0)  # 0 dB
        logger.info("set manual exposure: ExposureTime=5000us Gain=0dB")

        # --- 获取 payload, 准备取流 ---
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        if not _mv_ok("get payload size", cam.MV_CC_GetIntValue("PayloadSize", stParam)):
            continue
        nPayloadSize = stParam.nCurValue

        if not _mv_ok("start grabbing", cam.MV_CC_StartGrabbing()):
            continue

        data_buf = (c_ubyte * nPayloadSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        return cam, data_buf, nPayloadSize, stFrameInfo


# ============================================================
# 相机读取线程
# ============================================================

class Producer(threading.Thread):
    """从 Hikvision 相机持续读取帧, 放入 frame_queue"""

    def __init__(self, stream_image_container):
        """
        Args:
            stream_image_container: list of [np.ndarray], 用于更新视频流 (传引用绕开 global)
        """
        super().__init__()
        self._stream_image = stream_image_container

    def run(self):
        cam, data_buf, nPayloadSize, stFrameInfo = init_camera()
        while True:
            try:
                ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 10000)
                if ret != 0:
                    logger.info('海康相机状态：{}'.format(ret))

                if ret == CAMERA_NEED_RESTART:
                    logger.info("相机重启")
                    cam.MV_CC_StopGrabbing()
                    cam.MV_CC_CloseDevice()
                    cam.MV_CC_DestroyHandle()
                    cam, data_buf, nPayloadSize, stFrameInfo = init_camera()
                    continue

                time.sleep(0.01)

                if ret == 0:
                    if stFrameInfo.nWidth == 0 or stFrameInfo.nHeight == 0:
                        logger.error("frame with zero dimension! w=%d h=%d", stFrameInfo.nWidth, stFrameInfo.nHeight)
                        continue
                    image = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    # 先裁宽高比到 4:3 以内, 再下采样 (减少 resize 处理的像素)
                    image = _crop_to_max_aspect(image)
                    h, w = image.shape[:2]
                    image = cv2.resize(
                        image,
                        (int(w / 4), int(h / 4)),
                        interpolation=cv2.INTER_AREA,
                    )
                    frame_queue.put(image)
                    self._stream_image[0] = image
            except Exception as e:
                # 单帧异常不杀死采集线程, 记日志继续 (防视频流静默中断)
                logger.error('producer frame error: %s', e, exc_info=True)
                time.sleep(0.1)
