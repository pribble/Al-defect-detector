#!/usr/bin/env python3
"""gen_cnn_core_vectors.py — cnn_core.v 对拍向量生成（阶段 3）

为随机层（conv/dw）生成 RTL 流式输入：
  vec_core/layer_XX/cfg.hex    cfg 序列：每行 48bit {sel[1:0], addr[19:0], wdata[31:0]}
  vec_core/layer_XX/in.hex     输入流：行→列→通道（NHWC8 线性序）
  vec_core/layer_XX/w.hex      权重流：按 RTL 执行轨迹的 slice 序列（8×k² 字节/slice）
  vec_core/layer_XX/expect.hex 期望输出：按 RTL 输出事件顺序 (h, g, w, m有效)

数值基准 = ref_cnn_top.run_layer_tiled（cnn_top 行为模型）。
权重 slice 消费轨迹（RTL 确定性 FSM）：
  conv:  for row: for g(输出块): for w: for ci: slice(g, ci)
  dw:    for row: for g: for w: slice_dw(g)      （每窗口 1 slice，块=输出块）
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ref_cnn_top as R

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "verification", "vec_core")


def u32(v):
    return v & 0xFFFFFFFF


def s32(v):
    return u32(v)


def s8(v):
    return v & 0xFF


def gen_layer(dirpath, rng, typ, in_c, in_h, out_c, k, s, p, act):
    """生成一层向量（RTL 执行 + ref_cnn_top 期望）。"""
    os.makedirs(dirpath, exist_ok=True)
    in_w = in_h
    out_h = (in_h + 2 * p - k) // s + 1
    out_w = (in_w + 2 * p - k) // s + 1

    lp = R.LayerParam(typ, in_c, in_h, in_w, out_c, out_h, out_w,
                      k, s, p, act, rng=rng)
    lp.in_off = 0
    lp.out_off = R.nhwc8_bytes(in_c, in_h, in_w) // 8

    dm = R.CmaMem(R.nhwc8_bytes(in_c, in_h, in_w) +
                  R.nhwc8_bytes(out_c, out_h, out_w) + 64)
    wm = R.CmaMem(lp.w_bytes + 64)
    for cb_out in range(lp.chn_block):
        for cb_in in range(lp.in_cb):
            for mo in range(8):
                for kk in range(k * k):
                    for mi in range(8):
                        wm.buf[R.w_addr(lp, cb_out, cb_in, mo, kk, mi)] = (
                            0 if (typ == 4 and (cb_out != cb_in or mo != mi))
                            else rng.randint(-127, 127)) & 0xFF
    for cb in range(lp.in_cb):
        for h in range(in_h):
            for w in range(in_w):
                for m in range(8):
                    dm.buf[R.in_addr(lp, cb, h, w, m)] = rng.randint(0, 255)

    R.run_layer_tiled(lp, dm, wm)  # 期望（含行块循环，小尺寸 row_block 多=1 或少量）

    # ---- cfg 序列 ----
    cfg = []
    def push(sel, addr, data):
        cfg.append((sel & 0x3) | ((addr & 0xFFFFF) << 2) |
                   ((data & 0xFFFFFFFF) << 22))
    push(0, 0, (typ & 0xF) | ((act & 0x3) << 4))   # type/act
    push(0, 1, in_c); push(0, 2, in_h); push(0, 3, in_w)
    push(0, 4, out_c); push(0, 5, out_h); push(0, 6, out_w)
    push(0, 7, k); push(0, 8, p); push(0, 9, s)
    push(0, 10, lp.out_row_tile); push(0, 11, lp.in_row_tile)
    push(0, 12, 1)  # row_block（阶段 3 固定 1）
    for c in range(out_c):
        push(1, c, s32(lp.bias_int[c]))
        push(2, c, u32(lp.mult[c]))
        push(3, c, lp.shift)  # shift 为全层标量
        push(0, 16 + c, s32(lp.rcl6[c]))  # rcl6 经 sel=00 addr≥16
    # ---- 输入流：行→列→通道（in 区线性字节）----
    in_seq = []
    for h in range(in_h):
        for w in range(in_w):
            for c in range(in_c):
                in_seq.append(dm.buf[R.in_addr(lp, c // 8, h, w, c % 8)])
    in_n = len(in_seq)
    with open(f"{dirpath}/in.hex", "w") as f:
        for b in in_seq:
            f.write(f"{b:02x}\n")

    # ---- 权重流（RTL 执行轨迹）----
    w_seq = []
    def slice_bytes(g, ci=None):
        out = bytearray()
        for lane in range(8):
            for kk in range(k * k):
                if typ == 1:
                    out.append(wm.buf[R.w_addr(lp, g, ci // 8, lane, kk,
                                               ci % 8)])
                else:
                    out.append(wm.buf[R.w_addr(lp, g, g, lane, kk, lane)])
        return out
    for row in range(out_h):
        for g in range(lp.chn_block):
            for w in range(out_w):
                if typ == 1:
                    for ci in range(in_c):
                        w_seq += slice_bytes(g, ci)
                else:
                    w_seq += slice_bytes(g)
    w_n_len = len(w_seq)
    with open(f"{dirpath}/w.hex", "w") as f:
        for b in w_seq:
            f.write(f"{b:02x}\n")

    # ---- 期望输出：按 RTL 事件顺序 (h, g, w, m有效) ----
    exp = []
    out_base = lp.out_off * 8
    for h in range(out_h):
        for g in range(lp.chn_block):
            co = min(8, out_c - g * 8)
            for w in range(out_w):
                for m in range(co):
                    exp.append(dm.buf[out_base + g * out_h * out_w * 8 +
                                      h * out_w * 8 + w * 8 + m])
    with open(f"{dirpath}/expect.hex", "w") as f:
        for b in exp:
            f.write(f"{b:02x}\n")

    # cfg.hex（含长度哨兵：addr=0xFFFFF → in_n, 0xFFFFE → w_n_len）
    with open(f"{dirpath}/cfg.hex", "w") as f:
        for v in cfg:
            f.write(f"{v:014x}\n")
        f.write(f"{0x3 | (0xFFFFF << 2) | (in_n << 22):014x}\n")
        f.write(f"{0x3 | (0xFFFFE << 2) | (w_n_len << 22):014x}\n")

    return dict(typ=typ, in_c=in_c, in_h=in_h, out_c=out_c, out_h=out_h,
                k=k, s=s, p=p, act=act, wlen=len(w_seq), explen=len(exp))


def main():
    rng = random.Random(11)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    os.makedirs(OUT_ROOT, exist_ok=True)
    for i in range(n):
        typ = 4 if rng.random() < 0.4 else 1
        if typ == 4:
            in_c = out_c = rng.choice([8, 16, 24, 32])
        else:
            in_c = rng.choice([3, 8, 16])
            out_c = rng.choice([8, 16, 20, 24])
        in_h = rng.choice([6, 8, 12, 16])
        k = 3  # 阶段 3 仅支持 k=3（k=1 行缓冲适配属阶段 3b 已知缺口）
        s = rng.choice([1, 2])
        p = 1
        act = rng.choice([0, 1, 2])
        info = gen_layer(f"{OUT_ROOT}/layer_{i:02d}", rng, typ, in_c, in_h,
                         out_c, k, s, p, act)
        print(f"layer {i:02d}: {'DW' if typ == 4 else 'CONV'} "
              f"{in_c}x{in_h}->{out_c}x{info['out_h']} k={k} s={s} act={act} "
              f"w={info['wlen']}B exp={info['explen']}B")


if __name__ == "__main__":
    main()
