#!/usr/bin/env python3
"""cpu_ref_check.py — 用 cpu_ref float 公式重算参考输出，与 RTL [LOUTD] 对比。

与 rtl_ref_check.py 的区别：本脚本用 **float** 公式（cpu_ref cvt_kernel 语义）
    y = round_half_away(acc·ws·is/os + bias/os)
    act: relu = max(y,0)；relu6 = min(max(y,0), 6/os)
    out = clamp(y, -127, 127)
float 参数从 rtl_ref.log 的 [RSCALE] 定点数反推：
    ws·is/os = mult / 2^shift
    bias/os  = bias_mul / 2^22
    6/os     = rcl6 / 2^22

用途：
  - rtl_ref_check.py（定点参考）OK 但本脚本 mismatch → 差异来自 q30 定点近似
    （mult/bias_mul 量化误差），而非 RTL 逻辑
  - 两层都 OK → RTL 与 cpu_ref float 一致

用法：
  python3 cpu_ref_check.py rtl_ref.log rtl_layers.log
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtl_ref_check import parse_rtl_ref, parse_rtl_out
from ref_cnn_top import (LayerParam, CmaMem, nhwc8_bytes, w_addr,
                         run_layer_tiled, post_np_float)


def check_layer_float(name, L, rtl_out):
    typ = L['type']
    in_c, in_h, in_w = L['in_c'], L['in_h'], L['in_w']
    out_c, out_h, out_w = L['out_c'], L['out_h'], L['out_w']
    k, s, p, act = L['k'], L['s'], L['p'], L['act']
    out_bytes = nhwc8_bytes(out_c, out_h, out_w)
    in_bytes = nhwc8_bytes(in_c, in_h, in_w)

    lp = LayerParam(typ, in_c, in_h, in_w, out_c, out_h, out_w, k, s, p, act,
                    in_off=0, w_off=0, out_off=0,
                    rng=np.random.RandomState(0))
    w_dump_bytes = lp.chn_block * (1 if typ == 4 else lp.in_cb) * 8 * \
        k * k * 8
    out_off = (in_bytes + w_dump_bytes) // 8
    lp.out_off = out_off

    sc = L['scale']
    oc = out_c
    if len(sc) != 4 * oc:
        return f"scale len={len(sc)} != 4*oc={4*oc}"
    shifts = [int(sc[2 * oc + i]) for i in range(oc)]
    if len(set(shifts)) != 1:
        return f"per-channel shift 不一致: {sorted(set(shifts))[:4]}"
    shift = shifts[0]
    scale_f = [float(sc[i]) / (1 << shift) for i in range(oc)]
    bias_f = [float(sc[oc + i]) / (1 << 22) for i in range(oc)]
    relu6_f = [float(sc[3 * oc + i]) / (1 << 22) for i in range(oc)]

    dm = CmaMem(in_bytes + out_bytes + 64 + out_off * 8)
    if len(L['input']) < in_bytes:
        return f"input dump {len(L['input'])} < {in_bytes}"
    dm.buf[:in_bytes] = L['input'][:in_bytes]

    wm = CmaMem(lp.w_bytes + 64)
    wb = L['weight']
    if len(wb) < w_dump_bytes:
        return f"weight dump {len(wb)} < {w_dump_bytes}"
    idx = 0
    for cb_out in range(lp.chn_block):
        for cb_in in ([cb_out] if typ == 4 else range(lp.in_cb)):
            for mi in range(8):
                for kk in range(k * k):
                    for mo in range(8):
                        wm.buf[w_addr(lp, cb_out, cb_in, mo, kk, mi)] = wb[idx]
                        idx += 1

    def post(p, acc, co0):
        return post_np_float(p, acc, co0, scale_f, bias_f, relu6_f)

    run_layer_tiled(lp, dm, wm, post)
    exp = bytes(dm.buf[out_off * 8: out_off * 8 + out_bytes])
    got = rtl_out.get(name)
    if got is None:
        return "RTL 输出缺失"
    if len(got) != out_bytes:
        return f"RTL 输出 {len(got)} != 期望 {out_bytes}"
    if got == exp:
        return None
    ndiff = 0
    first = -1
    maxd = 0
    hist = {}
    for i in range(out_bytes):
        if got[i] != exp[i]:
            if first < 0:
                first = i
            ndiff += 1
            d = int(got[i]) - int(exp[i])
            maxd = max(maxd, abs(d))
            hist[d] = hist.get(d, 0) + 1
    return (f"mismatch first={first} ndiff={ndiff}/{out_bytes} "
            f"got=0x{got[first]:02x} exp=0x{exp[first]:02x} "
            f"max|d|={maxd} hist={dict(sorted(hist.items())[:6])}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ref = parse_rtl_ref(sys.argv[1])
    rtl_out = parse_rtl_out(sys.argv[2])
    print(f"ref layers={len(ref)}, rtl layers={len(rtl_out)}")
    nmis = 0
    for name, L in ref.items():
        err = check_layer_float(name, L, rtl_out)
        if err is None:
            print(f"  {name:<20} OK")
        else:
            print(f"  {name:<20} {err}")
            nmis += 1
            if nmis >= 8:
                print("  ... 后续层不再计算")
                break
    print(f"mismatch layers: {nmis}")


if __name__ == '__main__':
    main()
