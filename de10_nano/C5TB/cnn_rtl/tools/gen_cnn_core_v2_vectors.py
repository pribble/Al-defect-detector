#!/usr/bin/env python3
"""cnn_core_v2 向量生成器：行块驻留（NHWC8 块序输入、slice 权重、行块拼接）。

每个 layer_NN 目录输出：
  cfg.hex     54-bit cfg 序列（标量 + requant 数组；base_row 由 tb 每行块写）
  in.hex      64-bit NHWC8 输入，行块序列拼接（每行块 = 有效行 × in_cb × W）
  w.hex       64-bit 权重 slice 序列（单行块，tb 每行块循环重放）
  expect.hex  64-bit 输出（块序），行块序列拼接

与 ref_cnn_top.py 的内存布局/数值完全一致（ref 为裁判）。
"""
import os, sys, random, struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_cnn_top import (LayerParam, CmaMem, nhwc8_bytes, in_addr, out_addr,
                         w_addr, run_layer_tiled)


def pack_cfg(sel, addr, wdata):
    """64-bit 打包：wdata[31:0] | addr[19:0]<<32 | sel[1:0]<<52，16 hex。
    注意 reg 位宽必须是整 hex（64bit），否则 readmemh 截断会丢 sel 位。"""
    v = (wdata & 0xFFFFFFFF) | ((addr & 0x3FFFF) << 32) | ((sel & 7) << 52)
    return f"{v:016x}"


def gen_layer(idx, rng, out_dir):
    typ = rng.choice([1, 4])
    in_c = rng.choice([3, 8, 16])
    out_c = rng.choice([8, 16, 20, 24])
    if typ == 4:
        out_c = in_c
    k, s, p = 3, rng.choice([1, 2]), 1
    act = rng.choice([0, 1, 2])
    while True:
        in_h = rng.choice([8, 10, 12, 16, 38])
        in_w = in_h
        out_h = (in_h + 2 * p - k) // s + 1
        out_w = out_h
        if out_h < 1:
            in_h = in_h + 2
            continue
        # 行块上限约束（对齐 G_MAX_OROWS=20 / G_MAX_IN_ROWS=41）
        tile = min(750 // out_w, out_h)
        in_tile = (tile - 1) * s + k
        if tile <= 20 and in_tile <= 41:
            break

    in_bytes = nhwc8_bytes(in_c, in_h, in_w)
    w_bytes_ = nhwc8_bytes(out_c, 8, 9) * ((in_c + 7) // 8)  # 近似权重字节（仅偏移规划）
    # 输入/输出区分离（out_off 默认 0 会与输入重叠，run_layer_tiled 写输出破坏输入）
    lp = LayerParam(typ, in_c, in_h, in_w, out_c, out_h, out_w, k, s, p, act,
                    in_off=0,
                    out_off=(in_bytes + w_bytes_) // 8,
                    rng=rng)
    tag = f"{'DW' if typ == 4 else 'CONV'} {in_c}x{in_h}->{out_c}x{out_h} " \
          f"k={k} s={s} act={act}"

    dm = CmaMem(nhwc8_bytes(in_c, in_h, in_w) +
                nhwc8_bytes(out_c, out_h, out_w) + 64 +
                ((in_bytes + w_bytes_) // 8) * 8)
    wm = CmaMem(lp.w_bytes + 64)

    # 权重（布局 [Co/8][Ci/8][8][9][8]；dw 对角，slice 位于 (cb_out, cb_out)——与 ref 一致）
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

    # 输入（全图，NHWC8；块末超出 in_c 的通道补 0——黑盒布局语义）
    for cb in range(lp.in_cb):
        for h in range(in_h):
            for w in range(in_w):
                for m in range(8):
                    if cb * 8 + m < in_c:
                        dm.buf[in_addr(lp, cb, h, w, m)] = rng.randint(0, 255)
                    else:
                        dm.buf[in_addr(lp, cb, h, w, m)] = 0

    # 期望（行块 tiled 执行）
    run_layer_tiled(lp, dm, wm)

    # ---- cfg 序列 ----
    scalars = [
        (0, typ), (1, act), (2, in_c), (3, in_h), (4, in_w),
        (5, out_c), (6, out_h), (7, out_w), (8, k), (9, p), (10, s),
        (11, lp.out_row_tile), (12, lp.in_row_tile),
        (13, lp.in_cb), (14, lp.chn_block), (16, lp.row_block),
    ]
    cfg = [pack_cfg(0, a, v) for a, v in scalars]
    # requant 参数直接用 ref（LayerParam.quantize_params）生成的，保证与期望一致
    for ch in range(out_c):
        cfg.append(pack_cfg(1, ch, lp.bias_int[ch] & 0xFFFFFFFF))
        cfg.append(pack_cfg(2, ch, lp.mult[ch] & 0xFFFFFFFF))
        cfg.append(pack_cfg(3, ch, lp.shift))
        cfg.append(pack_cfg(4, ch, lp.rcl6[ch] & 0xFFFFFFFF))
    cfg.append("f" * 16 * 3)  # 哨兵（tb 扫描用）

    # ---- 输入行块拼接 + 期望行块拼接 ----
    in_words, exp_words = [], []
    for rb in range(lp.row_block):
        base = rb * lp.out_row_tile * s - p
        r0, r1 = max(0, base), min(in_h, base + lp.in_row_tile)
        for cb in range(lp.in_cb):
            for r in range(r0, r1):
                for w in range(in_w):
                    by = bytes(dm.buf[in_addr(lp, cb, r, w, m)]
                               for m in range(8))
                    in_words.append(struct.unpack('<Q', by)[0])
        r_out0 = rb * lp.out_row_tile
        r_out1 = min(out_h, r_out0 + lp.out_row_tile)
        for cb_out in range(lp.chn_block):
            c_eff = min(8, out_c - cb_out * 8)
            for r in range(r_out0, r_out1):
                for w in range(out_w):
                    by = bytes(dm.buf[out_addr(lp, cb_out, r, w, m)]
                               if m < c_eff else 0
                               for m in range(8))
                    exp_words.append(struct.unpack('<Q', by)[0])

    # ---- 权重 slice 序列（单行块）----
    w_words = []
    for cb_out in range(lp.chn_block):
        for cb_in in ([cb_out] if typ == 4 else range(lp.in_cb)):
            for mo in range(8):
                for kk in range(k * k):
                    by = bytes(wm.buf[w_addr(lp, cb_out, cb_in, mo, kk, mi)]
                               for mi in range(8))
                    w_words.append(struct.unpack('<Q', by)[0])

    d = os.path.join(out_dir, f"layer_{idx:02d}")
    os.makedirs(d, exist_ok=True)
    def emit(name, lines):
        with open(os.path.join(d, name), 'w') as f:
            for ln in lines:
                f.write(ln + "\n")
    emit("cfg.hex", cfg)
    emit("in.hex", [f"{w:016x}" for w in in_words] + ["f" * 16] * 3)
    emit("w.hex", [f"{w:016x}" for w in w_words] + ["f" * 16] * 3)
    emit("expect.hex", [f"{w:016x}" for w in exp_words])
    print(f"layer {idx:02d}: {tag} in_tile={lp.in_row_tile} "
          f"out_tile={lp.out_row_tile} row_block={lp.row_block} "
          f"in={len(in_words)}w exp={len(exp_words)}w w={len(w_words)}w")
    return d


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    out_dir = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "verification", "v2")
    rng = random.Random(seed)
    for i in range(n):
        gen_layer(i, rng, out_dir)
    print(f"OK: {n} layers -> {out_dir}")


if __name__ == "__main__":
    main()
