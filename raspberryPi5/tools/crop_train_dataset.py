#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集预处理: 将长方形 VOC 标注图裁剪为"铝片居中正方形 + 黑边", 与线上智能裁剪一致.

用途:
  旧训练集图片是长方形(整帧), 线上推理输入是"以铝片为中心的方形裁剪+黑边"。
  训练/推理输入分布不一致会引入误检, 本脚本把旧数据集切成同样的形态, 并同步
  变换 VOC XML 里的缺陷框坐标, 供重新训练。输出到新目录, 绝不动原文件。

裁剪规则 (与 GrabImage/Consumer._smart_crop 一致):
  1. 定圆心: find_disc_robust (mask 背景差分→原始阈值→hough; 训练图无背景走
     原始阈值, hough 可从"被画面切掉的圆"的可见弧恢复完整圆心)。
  2. 检出圆 → 正方形边长 side = 2×(r + r×margin_ratio), 圆心居中, 越界钳制。
  3. 未检出圆 → 回退"图片中心取最大正方形" (本数据集铝片几乎占满整帧,
     中心正方形≈正确裁剪; --no-fallback-center 可改为跳过)。
  4. 缺陷框: 与裁剪区相交保留并裁剪越界部分, 完全在外丢弃, 过小(<--min-box)丢弃。
  5. 输出保持裁剪后的原始分辨率 (不缩放); 源图本身是灰度 (VOC depth=1),
     输出灰度图 (与 FPGA 推理输入一致)。

用法:
  python crop_train_dataset.py --images-dir D:\\HAOYAO\\images1 \
      --xmls-dir D:\\HAOYAO\\annotations1 \
      --out-images D:\\HAOYAO\\images_square --out-xmls D:\\HAOYAO\\annotations_square
  加 --dry-run 先预览不写文件
"""

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../GrabImage'))
from disc_detect import find_disc_robust

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
DEFAULT_MARGIN = 0.10   # 与 config.ini [disc] margin_ratio 一致


# ============================================================
# VOC XML 读写
# ============================================================

def load_voc(xml_path):
    """解析 VOC XML, 返回 (tree, root, [(obj_el, name, xmin, ymin, xmax, ymax), ...])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall('object'):
        name = obj.findtext('name', '')
        bb = obj.find('bndbox')
        if bb is None:
            continue
        try:
            xmin = int(round(float(bb.findtext('xmin', '0'))))
            ymin = int(round(float(bb.findtext('ymin', '0'))))
            xmax = int(round(float(bb.findtext('xmax', '0'))))
            ymax = int(round(float(bb.findtext('ymax', '0'))))
        except (TypeError, ValueError):
            continue
        objs.append((obj, name, xmin, ymin, xmax, ymax))
    return tree, root, objs


def transform_voc(root, objs, x0, y0, side, out_w, out_h, out_name, out_path, min_box):
    """按裁剪框变换/过滤缺陷框, 更新 <size>/<filename>/<path>. 返回 (kept, dropped)."""
    kept = 0
    dropped = 0
    kept_objs = []
    for obj, name, xmin, ymin, xmax, ymax in objs:
        # 与裁剪区相交
        nxmin = max(xmin, x0)
        nymin = max(ymin, y0)
        nxmax = min(xmax, x0 + side)
        nymax = min(ymax, y0 + side)
        if nxmax <= nxmin or nymax <= nymin:
            dropped += 1
            continue
        nxmin -= x0
        nymin -= y0
        nxmax -= x0
        nymax -= y0
        # 过小框丢弃 (裁剪后可能只剩一条缝)
        if (nxmax - nxmin) < min_box or (nymax - nymin) < min_box:
            dropped += 1
            continue
        kept_objs.append((obj, nxmin, nymin, nxmax, nymax))
        kept += 1

    # 重写 <size> / <filename> / <path>
    size = root.find('size')
    if size is None:
        size = ET.SubElement(root, 'size')
    for tag, val in (('width', out_w), ('height', out_h), ('depth', 1)):
        el = size.find(tag)
        if el is None:
            el = ET.SubElement(size, tag)
        el.text = str(val)
    fn = root.find('filename')
    if fn is None:
        fn = ET.SubElement(root, 'filename')
    fn.text = out_name
    p = root.find('path')
    if p is None:
        p = ET.SubElement(root, 'path')
    p.text = out_path

    # 删除所有旧 object, 重写保留的
    for obj, *_ in objs:
        root.remove(obj)
    for obj, nxmin, nymin, nxmax, nymax in kept_objs:
        bb = obj.find('bndbox')
        if bb is None:
            bb = ET.SubElement(obj, 'bndbox')
        for tag, val in (('xmin', nxmin), ('ymin', nymin), ('xmax', nxmax), ('ymax', nymax)):
            el = bb.find(tag)
            if el is None:
                el = ET.SubElement(bb, tag)
            el.text = str(val)
        root.append(obj)
    return kept, dropped


# ============================================================
# 单图处理
# ============================================================

def crop_box_for(gray, margin_ratio, fallback_center, logger):
    """返回 (x0, y0, side, source) — source: 'disc' | 'center-fallback' | None(跳过)."""
    h, w = gray.shape[:2]
    circle, info = find_disc_robust(gray, method='mask', cfg=None, background=None)
    if circle is not None:
        cx, cy, r = circle
        side = int(2 * (r + r * margin_ratio))
        side = min(side, w, h)
        if side >= 32:
            x0 = min(max(int(cx) - side // 2, 0), w - side)
            y0 = min(max(int(cy) - side // 2, 0), h - side)
            return x0, y0, side, 'disc(r={:.0f}) [{}]'.format(r, (info or {}).get('used_method', '?'))
        logger.append('  检出圆但边长过小({}), 回退中心'.format(side))
    else:
        logger.append('  未检出圆 ({}), 回退中心'.format((info or {}).get('reject') or '未知'))
    if not fallback_center:
        return None
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return x0, y0, side, 'center-fallback'


def process_one(img_path, xml_path, out_img_dir, out_xml_dir, margin_ratio, min_box,
                fallback_center, dry_run, logger):
    """处理一对 图+xml. 返回 status: 'ok' | 'skip' | 'dry' | 'err'."""
    base, ext = os.path.splitext(os.path.basename(img_path))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 'err', '图片读取失败'

    box = crop_box_for(img, margin_ratio, fallback_center, logger)
    if box is None:
        return 'skip', '未检出圆且未启用中心回退'
    x0, y0, side, source = box
    h, w = img.shape[:2]
    x1, y1 = min(x0 + side, w), min(y0 + side, h)
    crop = img[y0:y1, x0:x1]

    out_name = base + ext
    if xml_path and os.path.exists(xml_path):
        tree, root, objs = load_voc(xml_path)
        out_xml_path = os.path.join(out_xml_dir, base + '.xml')
        kept, dropped = transform_voc(
            root, objs, x0, y0, side, crop.shape[1], crop.shape[0],
            out_name, out_xml_path, min_box)
    else:
        kept, dropped = 0, 0
        tree = root = None

    if dry_run:
        logger.append('  [dry-run] {} | 裁剪({},{},{}) source={} 框:保留{} 丢弃{}'.format(
            base, x0, y0, side, source, kept, dropped))
        return 'dry', base

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_xml_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_img_dir, out_name), crop)
    if root is not None:
        tree.write(os.path.join(out_xml_dir, base + '.xml'), encoding='utf-8', xml_declaration=True)
    logger.append('  [OK] {} | 裁剪({},{},{}) source={} 框:保留{} 丢弃{}'.format(
        base, x0, y0, side, source, kept, dropped))
    return 'ok', base


# ============================================================
# 主入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='VOC 训练集裁剪为铝片居中正方形 (与线上智能裁剪一致)')
    ap.add_argument('--images-dir', required=True, help='原图片目录 (如 D:\\HAOYAO\\images1)')
    ap.add_argument('--xmls-dir', required=True, help='原标注目录 (如 D:\\HAOYAO\\annotations1)')
    ap.add_argument('--out-images', required=True, help='输出图片目录 (新建, 如 D:\\HAOYAO\\images_square)')
    ap.add_argument('--out-xmls', required=True, help='输出标注目录 (新建, 如 D:\\HAOYAO\\annotations_square)')
    ap.add_argument('--margin-ratio', type=float, default=DEFAULT_MARGIN,
                    help='黑边 = 半径 × 该值 (默认 {})'.format(DEFAULT_MARGIN))
    ap.add_argument('--min-box', type=int, default=2, help='裁剪后过小的框丢弃 (像素)')
    ap.add_argument('--no-fallback-center', action='store_true',
                    help='找不到圆时跳过该图 (默认用图片中心取最大正方形)')
    ap.add_argument('--dry-run', action='store_true', help='只预览不写文件')
    args = ap.parse_args()

    img_files = []
    for ext in IMAGE_EXTS:
        img_files.extend(glob.glob(os.path.join(args.images_dir, '*' + ext)))
    img_files = sorted(img_files)
    if not img_files:
        print('未在 {} 找到图片'.format(args.images_dir))
        return 1

    stats = {'ok': 0, 'skip': 0, 'err': 0, 'no_xml': 0}
    src_count = {'disc': 0, 'center-fallback': 0}
    for img_path in img_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(args.xmls_dir, base + '.xml')
        logger = []
        if not os.path.exists(xml_path):
            stats['no_xml'] += 1
            logger.append('  [WARN] {} 无对应 .xml'.format(base))
        status, note = process_one(
            img_path, xml_path if os.path.exists(xml_path) else None,
            args.out_images, args.out_xmls, args.margin_ratio, args.min_box,
            not args.no_fallback_center, args.dry_run, logger)
        if status not in stats:
            stats[status] = 0
        stats[status] += 1
        print('\n'.join(logger))
        if 'source=center-fallback' in note:
            src_count['center-fallback'] += 1
            print('  ^ 该图走中心回退')
        elif status == 'ok':
            src_count['disc'] += 1

    mode = 'DRY-RUN (未写文件)' if args.dry_run else '完成'
    print('\n[{}] 共 {} 张: 成功 {} / 跳过 {} / 出错 {} / 缺XML {}'.format(
        mode, len(img_files), stats.get('ok', 0), stats.get('skip', 0),
        stats.get('err', 0), stats.get('no_xml', 0)))
    if not args.dry_run:
        print('定圆来源: 检出圆 {} 张 / 中心回退 {} 张'.format(src_count['disc'], src_count['center-fallback']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
