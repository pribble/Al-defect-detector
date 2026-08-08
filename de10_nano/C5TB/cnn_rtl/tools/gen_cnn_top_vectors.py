#!/usr/bin/env python3
"""cnn_top 端到端向量生成器（阶段 5）。

输出 vec_cnn_top/layer_NN/：
  param.hex    27 个 32-bit struct parameter（每行 16 hex，低 32 位有效）
  scale.hex    4×out_c 个 32-bit 定点 requant（mult/bias_mul/shift/rcl6）
  in.hex       完整 NHWC8 全图输入（64-bit/行）
  w.hex        权重 slice 序（64-bit/行）
  expect.hex   完整 NHWC8 全图输出期望（64-bit/行）

DDR 布局（tb 摆放基址）：
  PARAM → param.hex、SCALE → scale.hex、DDRIN → in.hex、DDRW → w.hex、
  DDROUT → expect 比对区。
"""
import os, sys, random, struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_cnn_top import (LayerParam, CmaMem, nhwc8_bytes, in_addr, out_addr,
                         w_addr, run_layer_tiled)


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]


def gen_layer(idx, rng, out_dir):
    typ = rng.choice([1, 4])
    in_c = rng.choice([3, 8, 16])
    out_c = rng.choice([8, 16, 20, 24])
    if typ == 4:
        out_c = in_c
    k, s, p = 3, rng.choice([1, 2]), 1
    act = rng.choice([0, 1, 2])
    while True:
        in_h = rng.choice([8, 10, 12, 16, 38, 45, 55, 75])   # 含非 tile 整数倍（覆盖 S_WR_TILE 最后行块裁剪）
        in_w = in_h
        out_h = (in_h + 2 * p - k) // s + 1
        out_w = out_h
        if out_h < 1:
            in_h += 2
            continue
        tile = min(750 // out_w, out_h)
        in_tile = (tile - 1) * s + k
        if tile <= 20 and in_tile <= 41:
            break

    in_bytes = nhwc8_bytes(in_c, in_h, in_w)
    w_bytes_ = nhwc8_bytes(out_c, 8, 9) * ((in_c + 7) // 8)
    lp = LayerParam(typ, in_c, in_h, in_w, out_c, out_h, out_w, k, s, p, act,
                    in_off=0, out_off=(in_bytes + w_bytes_) // 8, rng=rng)

    dm = CmaMem(nhwc8_bytes(in_c, in_h, in_w) +
                nhwc8_bytes(out_c, out_h, out_w) + 64 +
                ((in_bytes + w_bytes_) // 8) * 8)
    wm = CmaMem(lp.w_bytes + 64)

    # 权重（conv 全随机；dw 对角，slice 在 (cb_out, cb_out)）
    for cb_out in range(lp.chn_block):
        for cb_in in ([cb_out] if typ == 4 else range(lp.in_cb)):
            for mo in range(8):
                for kk in range(k * k):
                    for mi in range(8):
                        if typ == 4:
                            v = 0 if mo != mi else rng.randint(-127, 127)
                        else:
                            v = rng.randint(-127, 127)
                        wm.buf[w_addr(lp, cb_out, cb_in, mo, kk, mi)] = v & 0xFF

    # 输入（全图 NHWC8，pad 通道补 0）
    for cb in range(lp.in_cb):
        for h in range(in_h):
            for w in range(in_w):
                for m in range(8):
                    if cb * 8 + m < in_c:
                        dm.buf[in_addr(lp, cb, h, w, m)] = rng.randint(0, 255)
                    else:
                        dm.buf[in_addr(lp, cb, h, w, m)] = 0

    run_layer_tiled(lp, dm, wm)

    # ---- param 块（27 个 32-bit，struct parameter 字段序）----
    w_off_words = 0                       # 权重偏移（word=8B）
    in_off_words = 0
    out_off_words = in_bytes // 8         # 输出区（tb 内存布局）
    param = [
        in_off_words, w_off_words, 0, out_off_words,
        in_c, in_h, in_w,
        out_c, out_h, out_w,
        k, p, 0, s, act, typ,
        lp.chn_block,
        lp.out_row_tile, lp.in_row_tile, lp.row_block,
        f32_bits(0.0078), f32_bits(0.5),     # input_scale / output_scale
        0, 1,                                # lr / dilation
        lp.w_bytes, in_bytes, nhwc8_bytes(out_c, out_h, out_w),
    ]
    assert len(param) == 27

    # ---- scale 区：4×out_c 定点（mult/bias_mul/shift/rcl6）----
    scale = []
    for ch in range(out_c):
        scale.append(lp.mult[ch] & 0xFFFFFFFF)
    for ch in range(out_c):
        scale.append(lp.bias_mul[ch] & 0xFFFFFFFF)
    for ch in range(out_c):
        scale.append(lp.shift)
    for ch in range(out_c):
        scale.append(lp.rcl6[ch] & 0xFFFFFFFF)

    # ---- 输入全图（64-bit NHWC8）----
    in_words = []
    for cb in range(lp.in_cb):
        for h in range(in_h):
            for w in range(in_w):
                by = bytes(dm.buf[in_addr(lp, cb, h, w, m)] for m in range(8))
                in_words.append(struct.unpack('<Q', by)[0])

    # ---- 权重 slice 序（64-bit）----
    # 2026-08-10：同 gen_cnn_core_v2_vectors.py——改软件布局字节序
    # [mi][k][mo]（mi 主序，每 8 字节 = mo），与 conv_op.cc / RTL 装载一致
    w_words = []
    for cb_out in range(lp.chn_block):
        for cb_in in ([cb_out] if typ == 4 else range(lp.in_cb)):
            for mi in range(8):
                for kk in range(k * k):
                    by = bytes(wm.buf[w_addr(lp, cb_out, cb_in, mo, kk, mi)]
                               for mo in range(8))
                    w_words.append(struct.unpack('<Q', by)[0])

    # ---- 期望输出全图（64-bit NHWC8）----
    exp_words = []
    for cb_out in range(lp.chn_block):
        c_eff = min(8, out_c - cb_out * 8)
        for h in range(out_h):
            for w in range(out_w):
                by = bytes(dm.buf[out_addr(lp, cb_out, h, w, m)]
                           if m < c_eff else 0 for m in range(8))
                exp_words.append(struct.unpack('<Q', by)[0])

    d = os.path.join(out_dir, f"layer_{idx:02d}")
    os.makedirs(d, exist_ok=True)

    def emit(name, words, width=16):
        with open(os.path.join(d, name), 'w') as f:
            for w in words:
                if isinstance(w, str):
                    f.write(w + "\n")
                else:
                    f.write(f"{w:0{width}x}\n")

    emit("param.hex", param, 8)
    emit("scale.hex", scale, 8)
    emit("in.hex", in_words + ["f" * 16] * 3)
    emit("w.hex", w_words + ["f" * 16] * 3)
    emit("expect.hex", exp_words + ["f" * 16] * 3)

    print(f"layer {idx:02d}: {'DW' if typ == 4 else 'CONV'} "
          f"{in_c}x{in_h}->{out_c}x{out_h} k={k} s={s} act={act} "
          f"rb={lp.row_block} tile={lp.out_row_tile} in={len(in_words)}w "
          f"w={len(w_words)}w exp={len(exp_words)}w")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    out_dir = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "verification", "vct")
    rng = random.Random(seed)
    for i in range(n):
        gen_layer(i, rng, out_dir)
    print(f"OK: {n} layers -> {out_dir}")


if __name__ == "__main__":
    main()
