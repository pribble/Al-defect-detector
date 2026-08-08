#!/usr/bin/env python3
"""verify_rtl_vs_blackbox.py — 用黑盒 log 的输入/权重/scale，在相同采样点上
分别用 float 公式（黑盒语义）与 RTL 定点公式（mult/bias_mul q30/q22）重算，
对比：
  - float 公式 vs blackbox_layers2.log [LOUT]  → 验证解析正确（应 ~100%）
  - RTL 定点公式 vs rtl_layers.log（DUMP_LAYER_OUT 的 RTL 输出）→ 定位 RTL 错误层

用法：python3 verify_rtl_vs_blackbox.py <blackbox.log> <rtl_layers.log>
"""
import re
import sys
import math

sys.path.insert(0, '/opt/HAOYAO/_cnn_rtl_debug/tools')
sys.path.insert(0, '/opt/HAOYAO/_cnn_rtl_debug/bigchan_test')
from ref_cnn_top import LayerParam, CmaMem, in_addr, w_addr


def parse_blackbox_all(path):
    lines = open(path, encoding='utf-8', errors='replace').readlines()
    lout_starts = [i for i, l in enumerate(lines) if l.startswith('[LOUT] relu_0')]
    seg = lines[lout_starts[1]:]
    layers = []
    cur = None
    for l in seg:
        l = l.strip()
        if l.startswith('[LOUT]'):
            m = re.match(
                r'\[LOUT\] (\S+) type=(\d+) in=(\d+),(\d+),(\d+) out=(\d+),(\d+),(\d+) '
                r'k=(\d+) s=(\d+) p=(\d+) act=(\d+) in_off=(\d+) w_off=(\d+) out_off=(\d+) '
                r'in_scale=([0-9a-fx.p+-]+) out_scale=([0-9a-fx.p+-]+)', l)
            cur = {'name': m.group(1), 'type': int(m.group(2)),
                   'in_c': int(m.group(3)), 'in_h': int(m.group(4)), 'in_w': int(m.group(5)),
                   'out_c': int(m.group(6)), 'out_h': int(m.group(7)), 'out_w': int(m.group(8)),
                   'k': int(m.group(9)), 's': int(m.group(10)), 'p': int(m.group(11)),
                   'act': int(m.group(12)), 'in_off': int(m.group(13)), 'w_off': int(m.group(14)),
                   'out_off': int(m.group(15)),
                   'in_scale': float.fromhex(m.group(16)), 'out_scale': float.fromhex(m.group(17)),
                   'output_data': bytearray(), 'weight_data': bytearray(),
                   'scale_ws': [], 'scale_b': []}
            layers.append(cur)
        elif l.startswith('[LOUTD]') and cur is not None:
            cur['output_data'] += bytes.fromhex(l.replace('[LOUTD]', '').replace(' ', ''))
        elif l.startswith('[LSCALE]') and cur is not None:
            parts = l.split()[2:]
            n = len(parts) // 2
            cur['scale_ws'] = [float.fromhex(p) for p in parts[:n]]
            cur['scale_b'] = [float.fromhex(p) for p in parts[n:]]
        elif l.startswith('[LWD]') and cur is not None:
            cur['weight_data'] += bytes.fromhex(l.replace('[LWD]', '').replace(' ', ''))
    lin_starts = [i for i, l in enumerate(lines) if l.startswith('[LIN]')]
    lin1 = bytearray()
    for l in lines[lin_starts[1]:]:
        l = l.strip()
        if l.startswith('[LIND]'):
            lin1 += bytes.fromhex(l.replace('[LIND]', '').replace(' ', ''))
        elif l.startswith('[LOUT]'):
            break
    return layers, bytes(lin1)


def rnd(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def main():
    bb_path, rtl_path = sys.argv[1], sys.argv[2]
    layers, lin1 = parse_blackbox_all(bb_path)
    # rtl_layers.log 解析
    from compare_rtl_vs_blackbox import parse_rtl
    rtl = parse_rtl(rtl_path)

    bufs = {0: lin1}
    print(f"{'层':>3} {'名':>20} {'typ':>3} {'k':>2}  float-vs-黑盒  rtl定点-vs-RTL")
    for li, L in enumerate(layers):
        name = L['name']
        is_, os_ = L['in_scale'], L['out_scale']
        lp = LayerParam(L['type'], L['in_c'], L['in_h'], L['in_w'], L['out_c'],
                        L['out_h'], L['out_w'], L['k'], L['s'], L['p'], L['act'],
                        in_off=0, out_off=0)
        dm = CmaMem(6_000_000)
        in_b = L['in_off'] * 8
        inb = bufs.get(in_b)
        if inb is None:
            print(f"{li:3d} {name:>20} 无输入区 in_off={in_b}")
            bufs[L['out_off'] * 8] = bytes(L['output_data'])
            continue
        for cb in range(lp.in_cb):
            for h in range(L['in_h']):
                for w in range(L['in_w']):
                    for m in range(8):
                        src = cb * L['in_h'] * L['in_w'] * 8 + h * L['in_w'] * 8 + w * 8 + m
                        dm.buf[in_addr(lp, cb, h, w, m)] = inb[src] if src < len(inb) else 0
        wm = CmaMem(lp.w_bytes + 64)
        wb = L['weight_data']
        kk_max = L['k'] * L['k']
        n_blk = lp.chn_block * (1 if L['type'] == 4 else lp.in_cb)
        need = n_blk * 8 * kk_max * 8
        idx = 0
        for cb_out in range(lp.chn_block):
            for cb_in in (range(1) if L['type'] == 4 else range(lp.in_cb)):
                for mi in range(8):
                    for kk in range(kk_max):
                        for mo in range(8):
                            wm.buf[w_addr(lp, cb_out, (cb_out if L['type'] == 4 else cb_in),
                                          mo, kk, mi)] = wb[idx] if idx < len(wb) else 0
                            idx += 1
        bb = bytes(L['output_data'])
        rtl_b = rtl.get(name)
        OH, OW = L['out_h'], L['out_w']
        ok_bb = tot = ok_rtl = 0
        nch = 0
        mults = [rnd(L['scale_ws'][ch] * is_ / os_ * (1 << 30)) for ch in range(L['out_c'])]
        # 黑盒 [LSCALE] 第二半 = bias/os（黑盒预转值，不是 bias）→ bias_mul = scale_b × 2^22
        bmuls = [rnd(L['scale_b'][ch] * (1 << 22)) for ch in range(L['out_c'])]
        for ch in range(L['out_c']):
            A = L['scale_ws'][ch] * is_ / os_
            if abs(A) < 1e-12:
                continue
            cb = ch // 8
            mo = ch % 8
            import numpy as np
            rng = np.random.RandomState(ch + li * 100)
            for r, c in rng.randint(2, max(OH - 3, 3), (12, 2)):
                r, c = int(r), int(c)
                a = 0
                for kh in range(L['k']):
                    for kw in range(L['k']):
                        ir = r * L['s'] + kh - L['p']
                        ic = c * L['s'] + kw - L['p']
                        for cbi in (range(1) if L['type'] == 4 else range(lp.in_cb)):
                            for mi in range(8):
                                x = 0
                                if 0 <= ir < L['in_h'] and 0 <= ic < L['in_w']:
                                    v = dm.buf[in_addr(lp, (cb if L['type'] == 4 else cbi), ir, ic, mi)]
                                    x = v - 256 if v >= 128 else v
                                w = wm.buf[w_addr(lp, cb, (cb if L['type'] == 4 else cbi),
                                                  mo, kh * L['k'] + kw, mi)]
                                if w >= 128:
                                    w -= 256
                                a += x * w
                bidx = (cb * OH * OW + r * OW + c) * 8 + mo
                if bidx >= len(bb):
                    continue
                y = bb[bidx]
                pv = round(a * A + L['scale_b'][ch])
                if pv < 0:
                    pv -= 1
                if L['act'] in (1, 2):
                    pv = max(0, pv)
                pred_bb = int(pv) & 0xFF
                ok_bb += (pred_bb == y)
                # RTL 定点：round-half-up（加半右移）+ 黑盒实测负值 -1
                v = a * mults[ch] + (bmuls[ch] << 8)
                if L['act'] in (1, 2):
                    v = max(0, v)
                y_rtl = (v + (1 << 29)) >> 30
                if y_rtl < 0:
                    y_rtl -= 1          # 黑盒实测：round 后负值 -1（2026-08-10 已入 RTL）
                y_rtl = y_rtl & 0xFF
                if rtl_b is not None:
                    y_rtl_act = rtl_b[(cb * OH * OW + r * OW + c) * 8 + mo]
                    ok_rtl += (y_rtl == y_rtl_act)
                tot += 1
            nch += 1
        m_bb = 100.0 * ok_bb / max(tot, 1)
        m_rtl = 100.0 * ok_rtl / max(tot, 1) if rtl_b is not None else -1.0
        print(f"{li:3d} {name:>20} {L['type']:3d} {L['k']:2d}  {m_bb:6.1f}%        {m_rtl:6.1f}%")
        bufs[L['out_off'] * 8] = bytes(L['output_data'])


if __name__ == '__main__':
    main()
