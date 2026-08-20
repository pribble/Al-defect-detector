#!/usr/bin/env python3
"""rtl_ref_check.py — 用 RTL 自 dump 的输入/权重/定点 scale 重算参考输出，
与 rtl_layers.log 的 [LOUTD] 逐层对比，定位 MAC II=1 首个出错层。

不依赖旧黑盒 log（新模型权重变化后仍可用）。

输入文件（设备上 DUMP_LAYER_OUT=1 生成）：
  rtl_ref.log     [RINFO]/[RSCALE]/[RIN]/[RWD]（intelfpga.cc dbg_dump_layer_ref）
  rtl_layers.log  [LOUTD]（RTL 层输出）

用法：
  python3 rtl_ref_check.py rtl_ref.log rtl_layers.log
"""
import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_cnn_top import (LayerParam, CmaMem, nhwc8_bytes, w_addr,
                         run_layer_tiled)


def parse_hex_block(lines, i):
    """解析 [HEADER] 之后的 hex 块：'hexoff: aa bb ...' 行，直到空行/下一段。"""
    data = bytearray()
    while i < len(lines):
        line = lines[i].strip()
        if not line or ':' not in line:
            break
        hexpart = line.split(':', 1)[1].strip()
        for tok in hexpart.split():
            data.append(int(tok, 16))
        i += 1
    return bytes(data), i


def parse_rtl_ref(path):
    layers = {}
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        m = re.match(r'\[RINFO\] (\S+) type=(\d+) in=(\d+),(\d+),(\d+) '
                     r'out=(\d+),(\d+),(\d+) k=(\d+) s=(\d+) p=(\d+) act=(\d+)'
                     r' in_off=(\d+) w_off=(\d+) out_off=(\d+)', l)
        if m:
            name = m.group(1)
            layers[name] = {
                'type': int(m.group(2)),
                'in_c': int(m.group(3)), 'in_h': int(m.group(4)),
                'in_w': int(m.group(5)),
                'out_c': int(m.group(6)), 'out_h': int(m.group(7)),
                'out_w': int(m.group(8)),
                'k': int(m.group(9)), 's': int(m.group(10)),
                'p': int(m.group(11)), 'act': int(m.group(12)),
                'in_off': int(m.group(13)), 'w_off': int(m.group(14)),
                'out_off': int(m.group(15)),
            }
            i += 1
            continue
        m = re.match(r'\[RSCALE\] (\S+) n=(\d+)', l)
        if m:
            name = m.group(1)
            data, i = parse_hex_block(lines, i + 1)
            layers.setdefault(name, {})['scale'] = np.frombuffer(
                data, dtype=np.int32).copy()
            continue
        m = re.match(r'\[RIN\] (\S+) off=(\d+) n=(\d+)', l)
        if m:
            name = m.group(1)
            data, i = parse_hex_block(lines, i + 1)
            layers.setdefault(name, {})['input'] = data
            continue
        m = re.match(r'\[RWD\] (\S+) off=(\d+) n=(\d+)', l)
        if m:
            name = m.group(1)
            data, i = parse_hex_block(lines, i + 1)
            layers.setdefault(name, {})['weight'] = data
            continue
        i += 1
    return layers


def parse_rtl_out(path):
    """解析 rtl_layers.log 的 [LOUTD]，返回 {name: bytes}。"""
    from compare_rtl_vs_blackbox import parse_rtl
    return parse_rtl(path)


def check_layer(name, L, rtl_out):
    typ = L['type']
    in_c, in_h, in_w = L['in_c'], L['in_h'], L['in_w']
    out_c, out_h, out_w = L['out_c'], L['out_h'], L['out_w']
    k, s, p, act = L['k'], L['s'], L['p'], L['act']
    out_bytes = nhwc8_bytes(out_c, out_h, out_w)
    in_bytes = nhwc8_bytes(in_c, in_h, in_w)

    lp = LayerParam(typ, in_c, in_h, in_w, out_c, out_h, out_w, k, s, p, act,
                    in_off=0, w_off=0, out_off=0,
                    rng=np.random.RandomState(0))
    wm_bytes = lp.w_bytes  # 参考模型权重区（conv 全量；DW 只用对角）
    # DDR 软件权重区实际大小：DW 只分配对角块（out_cb×1×8×k²×8）
    w_dump_bytes = lp.chn_block * (1 if typ == 4 else lp.in_cb) * 8 * \
        k * k * 8
    out_off = (in_bytes + w_dump_bytes) // 8
    lp.out_off = out_off

    sc = L['scale']
    oc = out_c
    if len(sc) != 4 * oc:
        return f"scale len={len(sc)} != 4*oc={4*oc}"
    lp.mult = [int(sc[i]) for i in range(oc)]
    lp.bias_mul = [int(sc[oc + i]) for i in range(oc)]
    shifts = [int(sc[2 * oc + i]) for i in range(oc)]
    if len(set(shifts)) != 1:
        return f"per-channel shift 不一致: {sorted(set(shifts))[:4]}"
    lp.shift = shifts[0]
    lp.rcl6 = [int(sc[3 * oc + i]) for i in range(oc)]

    dm = CmaMem(in_bytes + out_bytes + 64 + out_off * 8)
    if len(L['input']) < in_bytes:
        return f"input dump {len(L['input'])} < {in_bytes}"
    dm.buf[:in_bytes] = L['input'][:in_bytes]

    wm = CmaMem(wm_bytes + 64)
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

    run_layer_tiled(lp, dm, wm)
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
    for i in range(out_bytes):
        if got[i] != exp[i]:
            if first < 0:
                first = i
            ndiff += 1
    return (f"mismatch first={first} ndiff={ndiff}/{out_bytes} "
            f"got[{first}]=0x{got[first]:02x} exp=0x{exp[first]:02x}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ref = parse_rtl_ref(sys.argv[1])
    rtl_out = parse_rtl_out(sys.argv[2])
    print(f"ref layers={len(ref)}, rtl layers={len(rtl_out)}")
    nmis = 0
    for name, L in ref.items():
        err = check_layer(name, L, rtl_out)
        if err is None:
            print(f"  {name:<20} OK")
        else:
            print(f"  {name:<20} {err}")
            nmis += 1
            if nmis >= 5:
                print("  ... 后续层不再计算")
                break
    print(f"mismatch layers: {nmis}")


if __name__ == '__main__':
    main()
