"""
圆形铝片识别 — 推理前智能裁剪的定位模块

功能:
  在选中帧上定位铝片圆心 (cx, cy) 与半径 r, 供 api.py 做:
    1. 以圆心为中心的方形智能裁剪 (四周留黑边), 避免整帧 3:2 图被各向异性
       压缩成 300×300 导致缺陷变形;
    2. 圆心信息本身可用于确认铝片位于裁剪图中央.

定圆方法 (config.ini [disc] method):
  mask (默认): 二值化 → 形态学开/闭 → 最大外轮廓 → minEnclosingCircle.
       - 传入 background(EMA 背景) 时走【背景差分】路径: 在 48×64 低分辨率下
         absdiff → 高斯模糊 → Otsu(带下限) → 放大掩码 → 形态学。与触发链路
         同源且低分辨率下静态高对比物体(反光斑)被背景抵消, 光照变化/皮带反光
         下最稳, 反射高光在圆内部不影响外边界;
       - 无 background 时走【原始帧阈值】路径 (亮铝片/暗背景场景), 兼容调试。
  hough: Canny + HoughCircles 圆弧投票, 边缘残缺/局部反光也能定圆, 但参数
               需要现场调整 (hough_param1/param2).

回退链: find_disc_robust() 按 主方法 → 另一方法 依次尝试, method_fallback=1 时
       全部失败才返回 None. 有 EMA 背景时 mask 只走背景差分(不回退 raw, 防止
       锁定静止反光斑导致裁剪错位); 圆形度过滤剔除机械臂等非圆亮斑, 半径范围
       过滤剔除超大/超小误检.

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
    "method_fallback": 1,         # 主方法未检出时是否尝试另一方法
}


def _threshold_binary(blurred, cfg, diag):
    """Otsu(带下限保护) 或固定阈值, 返回二值图; 同时记录所用阈值到 diag."""
    if cfg["mask_threshold"] > 0:
        _, binary = cv2.threshold(blurred, cfg["mask_threshold"], 255, cv2.THRESH_BINARY)
        diag["threshold"] = cfg["mask_threshold"]
    else:
        t, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if t < cfg["otsu_min_threshold"]:  # 纯暗背景下限保护
            _, binary = cv2.threshold(blurred, cfg["otsu_min_threshold"], 255, cv2.THRESH_BINARY)
            t = cfg["otsu_min_threshold"]
        diag["threshold"] = float(t)
    return binary


def _circle_from_binary(binary, cfg, diag):
    """形态学 → 最大外轮廓 → 外接圆 + 圆形度过滤. 返回 (cx,cy,r) 或 None."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # 去散斑
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)  # 填反射暗洞

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        diag["reject"] = "二值图无前景轮廓 (阈值 t={})".format(diag["threshold"])
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    diag["area"] = float(area)
    if area < 8:
        diag["reject"] = "最大轮廓过小 (面积 {:.0f})".format(area)
        return None
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    diag["radius"] = float(r)
    circ = area / (np.pi * r * r)
    diag["circularity"] = float(circ)
    if circ < cfg["circularity"]:
        diag["reject"] = "圆形度 {:.2f} < {:.2f} (可能是非圆亮斑)".format(circ, cfg["circularity"])
        return None
    return (float(cx), float(cy), float(r))


def _mask_circle(gray, cfg, background=None):
    """mask 方法: 背景差分(有 background) 或 原始帧阈值(无 background).

    Returns: (circle, diag). circle=(cx,cy,r) 或 None; diag 为诊断 dict
    (threshold/area/circularity/reject), 供调试流展示失败原因.
    """
    diag = {"threshold": None, "area": None, "circularity": None, "reject": None}
    if background is not None:
        # 背景差分路径: 与触发链路同源, 光照鲁棒.
        # 关键: 差分在【低分辨率】(48×64, 与 EMA 背景同尺寸)上做——全分辨率帧与
        # 上采样背景之间的分辨率不匹配会在静态高对比物体(反光斑/皮带边缘)上产生
        # 虚假 diff; 低分辨率下两者对齐, 静态边缘被背景抵消, 只有真正的新前景
        # (铝片)留下. 掩码再放大回全分辨率拟合圆.
        bh, bw = background.shape[:2]
        small = cv2.resize(gray, (bw, bh), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(small, np.clip(background, 0, 255).astype(np.uint8))
        blurred = cv2.GaussianBlur(diff, (7, 7), 0)
        binary = _threshold_binary(blurred, cfg, diag)
        binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        # 原始帧阈值路径: 亮铝片/暗背景场景
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        binary = _threshold_binary(blurred, cfg, diag)
    circle = _circle_from_binary(binary, cfg, diag)
    return circle, diag


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


def probe_disc(gray, method="mask", cfg=None, background=None):
    """
    定位铝片圆心与半径, 并附带诊断信息 (供 /debug_disc 调试流展示).

    Args:
        gray: 灰度帧 (BGR 会自动转灰度).
        method: "mask" | "hough".
        cfg: dict, 缺省用 DEFAULTS (与 config.ini [disc] 段一致).
        background: 可选 EMA 背景模型 (float32 低分辨率); mask 方法给定时走
                    背景差分路径, 光照鲁棒性更强.

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
        circle, diag = _mask_circle(gray, merged, background=background)
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


def find_disc_robust(gray, method="mask", cfg=None, background=None):
    """
    主方法 + 回退链定圆 (生产与 /debug_disc 共用的入口).

    尝试顺序 (method_fallback=1 时):
      method=mask + 有背景: mask(背景差分) → hough
           —— 有背景时【不回退 raw 原始阈值】: 静止高亮物体(反光斑)会被 raw
              当成圆, 裁剪错位比不裁剪更糟; 背景差分与触发链路同源, 能触发
              就一定能差分出铝片.
      method=mask + 无背景: mask(原始阈值) → hough   (硬处理路径/暖机期)
      method=hough:        hough → mask(背景差分或原始阈值)
    全部失败返回 (None, 最后一次的 info).

    Returns:
        (circle, info): circle=(cx,cy,r) 或 None;
        info["used_method"] 记录最终生效的方法 (如 "mask+bg").
    """
    merged = dict(DEFAULTS)
    if cfg:
        merged.update(cfg)
    fallback = int(merged.get("method_fallback", 1))

    if method == "mask":
        if background is not None:
            order = [("mask", background)]
            if fallback:
                order.append(("hough", None))
        else:
            order = [("mask", None)]
            if fallback:
                order.append(("hough", None))
    else:
        order = [("hough", None)]
        if fallback:
            order.append(("mask", background))

    last_info = None
    for m, bg in order:
        circle, info = probe_disc(gray, method=m, cfg=merged, background=bg)
        info["used_method"] = m + ("+bg" if bg is not None else "")
        if circle is not None:
            return circle, info
        last_info = info
    return None, last_info


def find_disc(gray, method="mask", cfg=None, background=None):
    """
    定位铝片圆心与半径 (单方法, 不回退).

    Args:
        gray: 灰度帧 (BGR 会自动转灰度).
        method: "mask" | "hough".
        cfg: dict, 缺省用 DEFAULTS.
        background: 可选 EMA 背景 (mask 方法背景差分路径用).

    Returns:
        (cx, cy, r) 或 None (未检出 / 合理性过滤不通过).
        需要回退链或失败原因时用 find_disc_robust() / probe_disc().
    """
    circle, _ = probe_disc(gray, method=method, cfg=cfg, background=background)
    return circle
