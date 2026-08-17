"""
圆形铝片识别 — 推理前智能裁剪的定位模块

功能:
  在选中帧上定位铝片圆心 (cx, cy) 与半径 r, 供 api.py 做:
    1. 以圆心为中心的方形智能裁剪 (四周留黑边), 避免整帧 3:2 图被各向异性
       压缩成 300×300 导致缺陷变形;
    2. 圆心信息本身可用于确认铝片位于裁剪图中央.

定圆方法 (config.ini [disc] method):
  mask (默认): 直接对灰度帧做 Otsu 二值化(带下限保护) → 形态学开/闭 → 最大
               外轮廓 → minEnclosingCircle. 反射高光位于圆内部, 不影响外
               边界; 圆形度过滤可剔除机械臂等非圆亮斑.
  hough: Canny + HoughCircles 圆弧投票, 边缘残缺/局部反光也能定圆, 但参数
               需要现场调整 (hough_param1/param2).

调试: probe_disc() 返回 (circle, info), info["reject"] 给出未检出/被过滤的
      原因(中文), 供 /debug_disc 调试流实时展示; find_disc() 是它的薄封装.

依赖: opencv-python, numpy. 不依赖 api.py, 可独立测试.
"""

import cv2
import numpy as np

# 默认参数 (与 config.ini [disc] 段对应, 便于独立使用)
DEFAULTS = {
    "min_radius_ratio": 0.15,     # 半径下限 = 该值 × min(宽,高)
    "max_radius_ratio": 0.50,     # 半径上限 = 该值 × min(宽,高)
    "circularity": 0.60,          # 圆形度下限 (凸包面积/(πr²)), 过滤非圆亮斑
    "mask_threshold": 0,          # 0=Otsu 自适应; >0=固定阈值
    "otsu_min_threshold": 50,     # Otsu 下限保护, 防把噪声当目标
    "hough_param1": 100,          # HoughCircles 高阈值 (Canny)
    "hough_param2": 30,           # HoughCircles 累加器阈值
    "hough_min_dist": 0,          # 圆心最小间距, 0=自动 max(w,h)/4
}


def _mask_circle(gray, cfg):
    """mask 方法: 二值化 → 形态学 → 最大外轮廓 → 外接圆.

    Returns: (circle, diag). circle=(cx,cy,r) 或 None; diag 为诊断 dict
    (threshold/area/circularity/reject), 供调试流展示失败原因.
    """
    diag = {"threshold": None, "area": None, "circularity": None, "reject": None}
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    if cfg["mask_threshold"] > 0:
        _, binary = cv2.threshold(blurred, cfg["mask_threshold"], 255, cv2.THRESH_BINARY)
        diag["threshold"] = cfg["mask_threshold"]
    else:
        t, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if t < cfg["otsu_min_threshold"]:  # 纯暗背景下限保护
            _, binary = cv2.threshold(blurred, cfg["otsu_min_threshold"], 255, cv2.THRESH_BINARY)
            t = cfg["otsu_min_threshold"]
        diag["threshold"] = float(t)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # 去散斑
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)  # 填反射暗洞

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        diag["reject"] = "二值图无前景轮廓 (阈值 t={})".format(diag["threshold"])
        return None, diag
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    diag["area"] = float(area)
    if area < 8:
        diag["reject"] = "最大轮廓过小 (面积 {:.0f})".format(area)
        return None, diag
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    diag["radius"] = float(r)
    circ = area / (np.pi * r * r)
    diag["circularity"] = float(circ)
    if circ < cfg["circularity"]:
        diag["reject"] = "圆形度 {:.2f} < {:.2f} (可能是非圆亮斑)".format(circ, cfg["circularity"])
        return None, diag
    return (float(cx), float(cy), float(r)), diag


def _hough_circle(gray, cfg):
    """hough 方法: Canny + HoughCircles, 取半径最大(画面主导)的候选圆.

    Returns: (circle, diag).
    """
    diag = {"radius": None, "reject": None}
    blurred = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    min_dist = int(cfg["hough_min_dist"]) if cfg["hough_min_dist"] else max(h, w) // 4
    min_r = int(cfg["min_radius_ratio"] * min(h, w))
    max_r = int(cfg["max_radius_ratio"] * min(h, w))
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
        param1=cfg["hough_param1"], param2=cfg["hough_param2"],
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        diag["reject"] = "HoughCircles 未检出候选圆"
        return None, diag
    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    diag["radius"] = float(best[2])
    return (float(best[0]), float(best[1]), float(best[2])), diag


def _sanity_check(cx, cy, r, h, w, cfg):
    """通用合理性过滤: 半径范围 + 圆心大致落在帧内."""
    if r <= 0 or not np.isfinite(r):
        return False
    min_r = cfg["min_radius_ratio"] * min(h, w)
    max_r = cfg["max_radius_ratio"] * min(h, w)
    if not (min_r <= r <= max_r):
        return False
    if not (-r <= cx <= w + r and -r <= cy <= h + r):
        return False
    return True


def probe_disc(gray, method="mask", cfg=None):
    """
    定位铝片圆心与半径, 并附带诊断信息 (供 /debug_disc 调试流展示).

    Args:
        gray: 灰度帧 (BGR 会自动转灰度).
        method: "mask" | "hough".
        cfg: dict, 缺省用 DEFAULTS (与 config.ini [disc] 段一致).

    Returns:
        (circle, info): circle=(cx, cy, r) 或 None;
        info 含 threshold/area/circularity/radius 与 reject(失败原因, 中文).
    """
    info = {"threshold": None, "area": None, "circularity": None, "radius": None, "reject": None}
    if gray is None:
        info["reject"] = "无图像帧"
        return None, info
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    merged = dict(DEFAULTS)
    if cfg:
        merged.update(cfg)
    h, w = gray.shape[:2]

    if method == "mask":
        circle, diag = _mask_circle(gray, merged)
    else:
        circle, diag = _hough_circle(gray, merged)
    info.update(diag or {})

    if circle is None:
        info["reject"] = info.get("reject") or "未检出圆"
        return None, info

    cx, cy, r = circle
    if not _sanity_check(cx, cy, r, h, w, merged):
        min_r = merged["min_radius_ratio"] * min(h, w)
        max_r = merged["max_radius_ratio"] * min(h, w)
        info["reject"] = "半径 {:.0f} 超出合理范围 [{:.0f}, {:.0f}]".format(r, min_r, max_r)
        return None, info
    return (cx, cy, r), info


def find_disc(gray, method="mask", cfg=None):
    """
    定位铝片圆心与半径.

    Args:
        gray: 灰度帧 (BGR 会自动转灰度).
        method: "mask" | "hough".
        cfg: dict, 缺省用 DEFAULTS (与 config.ini [disc] 段一致).

    Returns:
        (cx, cy, r) 或 None (未检出 / 合理性过滤不通过).
        需要失败原因时用 probe_disc().
    """
    circle, _ = probe_disc(gray, method=method, cfg=cfg)
    return circle
