#!/usr/bin/env python3
"""ref_cnn_top.py — cnn_top 黑盒行为模型（numpy 版）

对照硬件语义（实测确认，见 cnn_rtl/model_profile.md）：
  - 每层由 struct parameter 驱动（维度 / kernel / stride / pad / act / type / 偏移）
  - 输出分块：output_row_tile = min(750//out_w, out_h)；
    input_row_tile = (tile-1)*stride + dil*(k-1) + 1
  - 双层循环：row_block（行块）× chn_block（输出通道块，8 一组）
  - 输入/输出/权重均为 NHWC8 [C/8,H,W,8]，param 偏移为 64-bit 字（字节=偏移*8）

验证目标：
  1. 「分块循环执行 == 整图直算」bit-exact（RTL 对拍基准）——随机小层 + 真实 47 层
  2. 47 层形状链 / 偏移链 / 行块公式（model_profile.md 实证复算）

依赖：numpy（python3 -m pip install --user --break-system-packages numpy）
"""
import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ref_int8 import quantize_params

# ---------------------------------------------------------------------------
# 47 层真实参数（model_profile.md 第 2 节）
#   (type, in_c, in_h, in_w, out_c, out_h, out_w, k, s, p, act)
#   type: 1=conv, 4=dw_conv；act: 0=none, 1=relu, 2=relu6
# ---------------------------------------------------------------------------
MODEL_LAYERS = [
    (1, 3, 300, 300, 32, 150, 150, 3, 2, 1, 1),
    (4, 32, 150, 150, 32, 150, 150, 3, 1, 1, 1),
    (1, 32, 150, 150, 64, 150, 150, 1, 1, 0, 1),
    (4, 64, 150, 150, 64, 75, 75, 3, 2, 1, 1),
    (1, 64, 75, 75, 128, 75, 75, 1, 1, 0, 1),
    (4, 128, 75, 75, 128, 75, 75, 3, 1, 1, 1),
    (1, 128, 75, 75, 128, 75, 75, 1, 1, 0, 1),
    (4, 128, 75, 75, 128, 38, 38, 3, 2, 1, 1),
    (1, 128, 38, 38, 256, 38, 38, 1, 1, 0, 1),
    (4, 256, 38, 38, 256, 38, 38, 3, 1, 1, 1),
    (1, 256, 38, 38, 256, 38, 38, 1, 1, 0, 1),
    (4, 256, 38, 38, 256, 19, 19, 3, 2, 1, 1),
    (1, 256, 19, 19, 512, 19, 19, 1, 1, 0, 1),
    (4, 512, 19, 19, 512, 19, 19, 3, 1, 1, 1),
    (1, 512, 19, 19, 512, 19, 19, 1, 1, 0, 1),
    (4, 512, 19, 19, 512, 19, 19, 3, 1, 1, 1),
    (1, 512, 19, 19, 512, 19, 19, 1, 1, 0, 1),
    (4, 512, 19, 19, 512, 19, 19, 3, 1, 1, 1),
    (1, 512, 19, 19, 512, 19, 19, 1, 1, 0, 1),
    (4, 512, 19, 19, 512, 19, 19, 3, 1, 1, 1),
    (1, 512, 19, 19, 512, 19, 19, 1, 1, 0, 1),
    (4, 512, 19, 19, 512, 19, 19, 3, 1, 1, 1),
    (1, 512, 19, 19, 512, 19, 19, 1, 1, 0, 1),
    (1, 512, 19, 19, 16, 19, 19, 1, 1, 0, 0),
    (1, 512, 19, 19, 20, 19, 19, 1, 1, 0, 0),
    (4, 512, 19, 19, 512, 10, 10, 3, 2, 1, 1),
    (1, 512, 10, 10, 1024, 10, 10, 1, 1, 0, 1),
    (4, 1024, 10, 10, 1024, 10, 10, 3, 1, 1, 1),
    (1, 1024, 10, 10, 1024, 10, 10, 1, 1, 0, 1),
    (1, 1024, 10, 10, 24, 10, 10, 1, 1, 0, 0),
    (1, 1024, 10, 10, 30, 10, 10, 1, 1, 0, 0),
    (1, 1024, 10, 10, 256, 10, 10, 1, 1, 0, 2),
    (1, 256, 10, 10, 512, 5, 5, 3, 2, 1, 2),
    (1, 512, 5, 5, 24, 5, 5, 1, 1, 0, 0),
    (1, 512, 5, 5, 30, 5, 5, 1, 1, 0, 0),
    (1, 512, 5, 5, 128, 5, 5, 1, 1, 0, 2),
    (1, 128, 5, 5, 256, 3, 3, 3, 2, 1, 2),
    (1, 256, 3, 3, 24, 3, 3, 1, 1, 0, 0),
    (1, 256, 3, 3, 30, 3, 3, 1, 1, 0, 0),
    (1, 256, 3, 3, 128, 3, 3, 1, 1, 0, 2),
    (1, 128, 3, 3, 256, 2, 2, 3, 2, 1, 2),
    (1, 256, 2, 2, 24, 2, 2, 1, 1, 0, 0),
    (1, 256, 2, 2, 30, 2, 2, 1, 1, 0, 0),
    (1, 256, 2, 2, 64, 2, 2, 1, 1, 0, 2),
    (1, 64, 2, 2, 128, 1, 1, 3, 2, 1, 2),
    (1, 128, 1, 1, 16, 1, 1, 1, 1, 0, 0),
    (1, 128, 1, 1, 20, 1, 1, 1, 1, 0, 0),
]

OUTPUT_BUFF_SIZE = 150 * 5  # MAX_OUTPUT_W * OUTPUT_ROW_TILE
SHIFT = 30


def output_row_tile(out_h, out_w):
    return min(OUTPUT_BUFF_SIZE // out_w, out_h)


def input_row_tile(tile, stride, dilation, kernel):
    return (tile - 1) * stride + dilation * (kernel - 1) + 1


def chn_block_num(out_c):
    return (out_c + 7) // 8


def row_block_num(out_h, tile):
    return (out_h + tile - 1) // tile


def nhwc8_bytes(c, h, w):
    return ((c + 7) // 8) * h * w * 8


# ---------------------------------------------------------------------------
# 层参数（含量化参数：每输出通道 mult/bias_mul/shift + rcl6_mul，乘后域）
# ---------------------------------------------------------------------------
class LayerParam:
    def __init__(self, typ, in_c, in_h, in_w, out_c, out_h, out_w,
                 k, s, p, act, in_off=0, w_off=0, out_off=0, rng=None):
        self.type = typ
        self.in_c, self.in_h, self.in_w = in_c, in_h, in_w
        self.out_c, self.out_h, self.out_w = out_c, out_h, out_w
        self.k, self.s, self.p, self.act = k, s, p, act
        self.in_off, self.w_off, self.out_off = in_off, w_off, out_off
        rng = rng or random.Random()
        ws = [10 ** rng.uniform(-2.5, -0.5) for _ in range(out_c)]
        is_ = 10 ** rng.uniform(-2.5, -1.5)
        os_ = 10 ** rng.uniform(-1.5, 0.0)
        b = [rng.uniform(-5.0, 5.0) for _ in range(out_c)]
        self.mult, self.bias_mul, self.rcl6, self.shift = quantize_params(
            ws, is_, os_, b, SHIFT)
        self.out_row_tile = output_row_tile(out_h, out_w)
        self.in_row_tile = input_row_tile(self.out_row_tile, s, 1, k)
        self.chn_block = chn_block_num(out_c)
        self.row_block = row_block_num(out_h, self.out_row_tile)

    @property
    def in_cb(self):
        return (self.in_c + 7) // 8

    @property
    def w_bytes(self):
        return self.chn_block * self.in_cb * 8 * self.k * self.k * 8


# ---------------------------------------------------------------------------
# CMA 内存：bytearray + 64-bit 字寻址（硬件语义）
# ---------------------------------------------------------------------------
class CmaMem:
    def __init__(self, nbytes):
        self.buf = bytearray(nbytes)

    @staticmethod
    def _s8(v):
        return v - 256 if v >= 128 else v

    def wr(self, byte_idx, data8):  # data8: int / numpy int（写入低字节）
        self.buf[byte_idx] = int(data8) & 0xFF

    def rd(self, byte_idx):
        return CmaMem._s8(self.buf[byte_idx])


def in_addr(p, cb, h, w, m):
    return (p.in_off * 8 + cb * p.in_h * p.in_w * 8 + h * p.in_w * 8
            + w * 8 + m)


def out_addr(p, cb, h, w, m):
    return (p.out_off * 8 + cb * p.out_h * p.out_w * 8 + h * p.out_w * 8
            + w * 8 + m)


def w_addr(p, cb_out, cb_in, m_out, k, m_in):
    return (p.w_off * 8
            + cb_out * p.in_cb * 8 * p.k * p.k * 8
            + cb_in * 8 * p.k * p.k * 8
            + m_out * p.k * p.k * 8
            + k * 8 + m_in)


# ---------------------------------------------------------------------------
# numpy 计算核心：NHWC8 分块窗口 → einsum → requant/act → NHWC8
# 返回 [Ho, Wo, Co] int8（Co = 本块输出通道数）
# ---------------------------------------------------------------------------
def conv_tile_np(p, in_hw, w_np):
    """对 [Hi', Wi'] 输入区域（含 pad 补零）与权重做一次卷积。
    in_hw: [C, Hi', Wi'] int32（已 pad）；w_np: [Co, C, K, K] int32"""
    C, Hi, Wi = in_hw.shape
    K = p.k
    Ho = (Hi - K) // p.s + 1
    Wo = (Wi - K) // p.s + 1
    win = np.lib.stride_tricks.sliding_window_view(
        in_hw, (C, K, K), axis=(0, 1, 2))
    # win: [1, Hi-K+1, Wi-K+1, C, K, K]（axis0 为 C 窗口，恒为 1）
    win = win[:, ::p.s, ::p.s]                   # [1, Ho, Wo, C, K, K]
    win = win.reshape(Ho, Wo, C, K * K)          # 合并 kh,kw
    acc = np.einsum('hwck,ock->hwo', win,
                    w_np.reshape(w_np.shape[0], C, K * K),
                    optimize=True)               # [Ho,Wo,Co]
    return acc


def post_np(p, acc, co0):
    """raw+bias → act → requant → 饱和 int8。acc: [Ho,Wo,Co] int32"""
    out_c = min(8, p.out_c - co0)
    bias = np.array(p.bias_mul[co0:co0 + out_c], dtype=np.int64)
    mult = np.array(p.mult[co0:co0 + out_c], dtype=np.int64)
    acc = acc[:, :, :out_c]  # 块末不足 8 通道：截取有效通道
    v = acc.astype(np.int64) * mult[None, None, :] + (bias[None, None, :] << 8)  # 乘后域 q30 对齐（bias_mul 为 q22）
    if p.act == 1:
        v = np.maximum(v, 0)
    elif p.act == 2:
        rcl6 = np.array(p.rcl6[co0:co0 + out_c], dtype=np.int64)
        v = np.clip(v, 0, rcl6[None, None, :] << 8)
    v = (v + (1 << (p.shift - 1))) >> p.shift
    return np.clip(v, -128, 127).astype(np.int8)


def load_input_block(p, mem, cb_in, h0, h1):
    """读输入块 cb_in 的 [h0,h1) 行 → [C_eff, h1-h0, W] int32
    C_eff = min(8, in_c - cb_in*8)（块末不足 8 通道只取实际通道）"""
    c_eff = min(8, p.in_c - cb_in * 8)
    rows = h1 - h0
    arr = np.zeros((c_eff, rows, p.in_w), dtype=np.int32)
    for m in range(c_eff):
        for h in range(rows):
            for w in range(p.in_w):
                arr[m, h, w] = mem.rd(in_addr(p, cb_in, h0 + h, w, m))
    return arr


def load_weight_np(p, wmem, cb_out):
    """读输出块 cb_out 的权重（独立 cb_weight 区）→ [Co, C, K, K] int32"""
    co0 = cb_out * 8
    out_c = min(8, p.out_c - co0)
    if p.type == 1:
        w = np.zeros((8, p.in_c, p.k, p.k), dtype=np.int32)
        for mo in range(out_c):
            for ci in range(p.in_c):
                for kh in range(p.k):
                    for kw in range(p.k):
                        w[mo, ci, kh, kw] = wmem.rd(
                            w_addr(p, cb_out, ci // 8, mo,
                                   kh * p.k + kw, ci % 8))
    else:  # dw 对角：输出通道 == 输入通道
        w = np.zeros((8, 8, p.k, p.k), dtype=np.int32)
        for mo in range(out_c):
            for kh in range(p.k):
                for kw in range(p.k):
                    w[mo, mo, kh, kw] = wmem.rd(
                        w_addr(p, cb_out, cb_out, mo, kh * p.k + kw, mo))
    return w


def store_output(p, mem, cb_out, r0, out_np):
    """out_np: [Ho, Wo, Co] int8 → 写 NHWC8 输出（从全图行 r0 起）"""
    Ho, Wo, Co = out_np.shape
    for h in range(Ho):
        for w in range(Wo):
            for m in range(Co):
                mem.wr(out_addr(p, cb_out, r0 + h, w, m), out_np[h, w, m])


# ---------------------------------------------------------------------------
# 分块执行（硬件语义：row_block × chn_block；data 区与 weight 区独立）
# ---------------------------------------------------------------------------
def _layer_cb_list(p, cb):
    """conv 遍历全部输入块；dw 只取对角块 cb"""
    return range(p.in_cb) if p.type == 1 else [cb]


def _w_slice(p, w_np, cb, cbi):
    c_eff = min(8, p.in_c - cbi * 8)
    if p.type == 1:
        return w_np[:, cbi * 8:cbi * 8 + c_eff, :, :]
    return w_np[:, :c_eff, :, :]  # dw：对角（mo==mi 非零）


def run_layer_tiled(p, dm, wm):
    for cb in range(p.chn_block):
        w_np = load_weight_np(p, wm, cb_out=cb)
        for rb in range(p.row_block):
            r0 = rb * p.out_row_tile
            rows = min(p.out_row_tile, p.out_h - r0)
            # 本输出行块需要的输入行范围（含 pad）：
            # 窗口行数 = (rows-1)*s + k（最后行块 rows < tile 时更小）
            i0 = r0 * p.s - p.p
            i1 = i0 + (rows - 1) * p.s + p.k
            acc = None
            co0 = cb * 8
            for cbi in _layer_cb_list(p, cb):
                in_blk = load_input_block(p, dm, cbi, max(0, i0),
                                          min(p.in_h, i1))
                # pad 顶部/底部
                tp = max(0, -i0)
                bt = max(0, i1 - p.in_h)
                in_pad = np.pad(in_blk, ((0, 0), (tp, bt), (0, 0)))
                # 左右 pad
                in_pad = np.pad(in_pad, ((0, 0), (0, 0), (p.p, p.p)))
                part = conv_tile_np(p, in_pad, _w_slice(p, w_np, cb, cbi))
                acc = part if acc is None else acc + part
            out_np = post_np(p, acc, co0)
            store_output(p, dm, cb, r0, out_np)


# ---------------------------------------------------------------------------
# 整图直算（真值；分块=1 的极限）
# ---------------------------------------------------------------------------
def run_layer_whole(p, dm, wm):
    for cb in range(p.chn_block):
        w_np = load_weight_np(p, wm, cb_out=cb)
        acc = None
        for cbi in _layer_cb_list(p, cb):
            in_blk = load_input_block(p, dm, cbi, 0, p.in_h)
            in_pad = np.pad(in_blk, ((0, 0), (p.p, p.p), (p.p, p.p)))
            part = conv_tile_np(p, in_pad, _w_slice(p, w_np, cb, cbi))
            acc = part if acc is None else acc + part
        out_np = post_np(p, acc, cb * 8)
        store_output(p, dm, cb, 0, out_np)


# ---------------------------------------------------------------------------
# 验证 1：随机小层 tiled vs whole bit-exact
# ---------------------------------------------------------------------------
def verify_random_layers(n=80, seed=1):
    rng = random.Random(seed)
    errs = 0
    for i in range(n):
        typ = 4 if rng.random() < 0.4 else 1
        if typ == 4:
            in_c = out_c = rng.choice([8, 16, 32, 64])
        else:
            in_c = rng.choice([3, 8, 16, 32])
            out_c = rng.choice([8, 16, 20, 24, 32])
        in_h = in_w = rng.choice([16, 24, 32])
        k = rng.choice([1, 3])
        s = rng.choice([1, 2])
        p = 1 if k == 3 else 0
        out_h = (in_h + 2 * p - k) // s + 1
        out_w = (in_w + 2 * p - k) // s + 1
        act = rng.choice([0, 1, 2])
        lp = LayerParam(typ, in_c, in_h, in_w, out_c, out_h, out_w,
                        k, s, p, act, rng=rng)
        # 输入/输出偏移必须不重叠（硬件层间线性布局，避免行块间覆盖输入）
        lp.in_off = 0
        lp.out_off = nhwc8_bytes(in_c, in_h, in_w) // 8
        dm = CmaMem(nhwc8_bytes(in_c, in_h, in_w) +
                    nhwc8_bytes(out_c, out_h, out_w) + 64)
        wm = CmaMem(lp.w_bytes + 64)
        # 权重（独立 weight 区，硬件布局）
        for cb_out in range(lp.chn_block):
            for cb_in in range(lp.in_cb):
                for mo in range(8):
                    for kk in range(k * k):
                        for mi in range(8):
                            if typ == 4 and (cb_out != cb_in or mo != mi):
                                val = 0
                            else:
                                val = rng.randint(-127, 127)
                            wm.buf[w_addr(lp, cb_out, cb_in, mo, kk,
                                          mi)] = val & 0xFF
        # 输入（data 区）
        for cb in range(lp.in_cb):
            for h in range(in_h):
                for w in range(in_w):
                    for m in range(8):
                        dm.buf[in_addr(lp, cb, h, w, m)] = rng.randint(0, 255)
        ref = bytearray(dm.buf)
        run_layer_whole(lp, dm, wm)
        whole_out = bytes(dm.buf)
        dm.buf = bytearray(ref)
        run_layer_tiled(lp, dm, wm)
        if bytes(dm.buf) != whole_out:
            errs += 1
            print(f"  random layer {i} MISMATCH: typ={typ} in={in_c}x{in_h} "
                  f"out={out_c}x{out_h} k={k} s={s} act={act}")
            # 首个差异坐标（诊断行块边界问题）
            base = lp.out_off * 8
            wb = bytes(whole_out)
            tb = bytes(dm.buf)
            for j in range(base, min(len(wb), len(tb))):
                if wb[j] != tb[j]:
                    off = j - base
                    cb = off // (lp.out_h * lp.out_w * 8)
                    rem = off % (lp.out_h * lp.out_w * 8)
                    h = rem // (lp.out_w * 8)
                    w = (rem % (lp.out_w * 8)) // 8
                    m = rem % 8
                    rb = h // lp.out_row_tile
                    r0 = rb * lp.out_row_tile
                    rows = min(lp.out_row_tile, lp.out_h - r0)
                    print(f"    first diff out({cb},{h},{w},{m}) "
                          f"whole={wb[j]} tiled={tb[j]} "
                          f"row_block={rb} r0={r0} rows={rows} "
                          f"tile={lp.out_row_tile} "
                          f"i0={r0 * lp.s - lp.p} "
                          f"i1={r0 * lp.s - lp.p + (rows - 1) * lp.s + lp.k}")
                    break
    print(f"verify_random_layers: {n - errs}/{n} pass")
    return errs == 0


# ---------------------------------------------------------------------------
# 验证 2：真实 47 层——形状链 + 偏移链 + 行块公式 + tiled==whole 数值
# ---------------------------------------------------------------------------
def verify_full_model():
    rng = random.Random(7)
    w_off = 0
    in_off = 0
    errs = 0
    mem_words = 0
    for i, (typ, in_c, in_h, in_w, out_c, out_h, out_w, k, s, p, act) in \
            enumerate(MODEL_LAYERS):
        lp = LayerParam(typ, in_c, in_h, in_w, out_c, out_h, out_w,
                        k, s, p, act, rng=rng)
        lp.in_off, lp.w_off = in_off, w_off
        lp.out_off = in_off + nhwc8_bytes(in_c, in_h, in_w) // 8
        exp_h = (in_h + 2 * p - k) // s + 1
        if (out_h, out_w) != (exp_h, exp_h):
            print(f"  layer {i}: shape mismatch {out_h}x{out_w} != {exp_h}")
            errs += 1
        if lp.out_row_tile != output_row_tile(out_h, out_w) or \
                lp.in_row_tile != input_row_tile(lp.out_row_tile, s, 1, k):
            print(f"  layer {i}: tile mismatch")
            errs += 1
        if lp.w_bytes // 8 != (lp.chn_block * lp.in_cb * 8 * k * k):
            print(f"  layer {i}: w size mismatch")
            errs += 1
        mem_words = max(mem_words, lp.out_off +
                        nhwc8_bytes(out_c, out_h, out_w) // 8)
        w_off += lp.w_bytes // 8
        in_off = lp.out_off
    # 分配全模型内存（data 区），逐层跑（链式：上层输出作下层输入）
    dm = CmaMem(mem_words * 8 + 1024)
    wm = CmaMem(w_off * 8 + 1024)
    # 构造每层参数（偏移链：out_off[i] = in_off[i] + 输入 NHWC8 字）
    params = []
    in_off = w_off = 0
    for (typ, in_c, in_h, in_w, out_c, out_h, out_w, k, s, p, act) in \
            MODEL_LAYERS:
        lp = LayerParam(typ, in_c, in_h, in_w, out_c, out_h, out_w,
                        k, s, p, act, rng=rng)
        lp.in_off, lp.w_off, lp.out_off = in_off, w_off, \
            in_off + nhwc8_bytes(in_c, in_h, in_w) // 8
        params.append(lp)
        w_off += lp.w_bytes // 8
        in_off = lp.out_off
    # 第一层输入
    p0 = params[0]
    for cb in range(p0.in_cb):
        for h in range(p0.in_h):
            for w in range(p0.in_w):
                for m in range(8):
                    dm.buf[in_addr(p0, cb, h, w, m)] = rng.randint(0, 255)
    for i, lp in enumerate(params):
        # 权重（独立 weight 区）
        for cb_out in range(lp.chn_block):
            for cb_in in range(lp.in_cb):
                for mo in range(8):
                    for kk in range(lp.k * lp.k):
                        for mi in range(8):
                            if lp.type == 4 and (cb_out != cb_in or
                                                 mo != mi):
                                val = 0
                            else:
                                val = rng.randint(-127, 127)
                            wm.buf[w_addr(lp, cb_out, cb_in, mo, kk,
                                          mi)] = val & 0xFF
        ref = bytearray(dm.buf)
        run_layer_whole(lp, dm, wm)
        whole_out = bytes(dm.buf)
        dm.buf = bytearray(ref)
        run_layer_tiled(lp, dm, wm)
        if bytes(dm.buf) != whole_out:
            errs += 1
            print(f"  layer {i} ({'DW' if lp.type == 4 else 'CONV'}) "
                  f"{lp.in_c}x{lp.in_h}x{lp.in_w}->{lp.out_c}x{lp.out_h}"
                  f"x{lp.out_w}: tiled != whole")
    print(f"verify_full_model: layers={len(params)} "
          f"({'OK' if errs == 0 else str(errs) + ' errors'})")
    return errs == 0


if __name__ == "__main__":
    ok1 = verify_random_layers()
    ok2 = verify_full_model()
    print("ALL PASS" if (ok1 and ok2) else "FAILED")
