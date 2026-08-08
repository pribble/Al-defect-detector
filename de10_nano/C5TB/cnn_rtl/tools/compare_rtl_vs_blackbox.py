#!/usr/bin/env python3
"""compare_rtl_vs_blackbox.py — 解析 RTL 逐层输出 dump（rtl_layers.log），
与黑盒 dump（blackbox_layers2.log）逐层对比，定位第一个不匹配层。

RTL dump 格式（intelfpga.cc dbg_dump_layer_output，DUMP_LAYER_OUT=1）：
    [LOUTD] <layer_name> off=<byte_offset> n=<nbytes>
    <hex_offset>: <32 个字节 hex>

黑盒格式未知时，先跑 `--rtl-only` 打印每层首 16 字节与哈希；
拿到 blackbox 格式后，把 parse_blackbox() 按实际格式实现即可逐层对比。

用法：
    python3 compare_rtl_vs_blackbox.py --rtl-only rtl_layers.log
    python3 compare_rtl_vs_blackbox.py rtl_layers.log blackbox_layers2.log
"""
import hashlib
import re
import sys

RTL_RE = re.compile(r"^\[LOUTD\] (\S+) off=(\d+) n=(\d+)$")


def parse_rtl(path):
    """返回 {layer_name: bytes}（4.jpg 第二次推理的层输出）"""
    layers = {}
    cur = None
    cur_bytes = bytearray()
    cur_n = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("latin1").rstrip("\n")
            m = RTL_RE.match(line)
            if m:
                if cur is not None:
                    layers[cur] = bytes(cur_bytes)
                cur = m.group(1)
                cur_n = int(m.group(3))
                cur_bytes = bytearray()
                continue
            if cur is not None and ":" in line:
                hexpart = line.split(":", 1)[1].strip()
                for tok in hexpart.split():
                    if len(cur_bytes) < cur_n:
                        cur_bytes.append(int(tok, 16))
    if cur is not None:
        layers[cur] = bytes(cur_bytes)
    return layers


def parse_blackbox(path):
    """黑盒 log 解析（格式见 _cnn_rtl_debug/tools/all_layers.py）：
    [LOUT] <name> type=.. in=.. out=.. k=.. s=.. p=.. act=.. in_off=.. w_off=..
           out_off=.. in_scale=.. out_scale=..
    [LOUTD] <hex 字节>
    取第 2 次推理段（lout_starts[1] = 4.jpg，跳过 warm up），与 RTL 的
    g_pass==2 dump 对齐。返回 {layer_name: bytes}。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\n") for l in f]
    starts = [i for i, l in enumerate(lines) if l.startswith("[LOUT] relu_0")]
    if len(starts) < 2:
        print("blackbox: only %d [LOUT] relu_0 segments found" % len(starts))
        return {}
    seg = lines[starts[1]:]
    layers = {}
    cur = None
    for l in seg:
        l = l.strip()
        if l.startswith("[LOUT]"):
            m = re.match(r"\[LOUT\] (\S+)\s+type=", l)
            if not m:
                print("blackbox: unparsed [LOUT] line:", l[:80])
                break
            cur = m.group(1)
            layers[cur] = bytearray()
        elif l.startswith("[LOUTD]") and cur is not None:
            layers[cur] += bytes.fromhex(l.replace("[LOUTD]", "").replace(" ", ""))
        elif l.startswith("[LOUT]") is False and cur is not None and \
                (l.startswith("[LSCALE]") or l.startswith("[LW") or l.startswith("[LIN")):
            pass  # 其他段不解析
    return {k: bytes(v) for k, v in layers.items()}


def layer_digest(b):
    if not b:
        return "EMPTY"
    h = hashlib.md5(b).hexdigest()
    first = " ".join("%02x" % x for x in b[:16])
    return "%s n=%d first=[%s]" % (h[:16], len(b), first)


def main():
    args = sys.argv[1:]
    rtl_only = "--rtl-only" in args
    args = [a for a in args if a != "--rtl-only"]
    if len(args) < 1:
        print(__doc__)
        sys.exit(1)
    rtl = parse_rtl(args[0])
    if rtl_only or len(args) == 1:
        print("RTL layers (first=%d):" % len(rtl))
        for name, b in rtl.items():
            print("  %-16s %s" % (name, layer_digest(b)))
        return
    bb = parse_blackbox(args[1])
    print("compare: rtl=%d layers, blackbox=%d layers" % (len(rtl), len(bb)))
    mism = 0
    for name, b in rtl.items():
        if name not in bb:
            print("  %-16s : MISSING in blackbox" % name)
            mism += 1
            continue
        bb_b = bb[name]
        if b == bb_b:
            print("  %-16s : MATCH (%d bytes)" % (name, len(b)))
        else:
            mism += 1
            n = min(len(b), len(bb_b))
            diff_at = next((i for i in range(n) if b[i] != bb_b[i]), min(len(b), len(bb_b)))
            print("  %-16s : MISMATCH rtl=%d bb=%d first_diff@%d" %
                  (name, len(b), len(bb_b), diff_at))
            print("           rtl[%d:]=%s" % (diff_at, " ".join("%02x" % x for x in b[diff_at:diff_at + 16])))
            print("           bb [%d:]=%s" % (diff_at, " ".join("%02x" % x for x in bb_b[diff_at:diff_at + 16])))
    print("mismatch layers: %d" % mism)


if __name__ == "__main__":
    main()
