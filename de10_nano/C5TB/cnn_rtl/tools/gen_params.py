#!/usr/bin/env python3
"""gen_params.py — float 模型参数 → conv_layer_s8 定点参数（per-channel）

独立 CLI（供阶段 2 集成 intelfpga 工具链 / 人工调试）：

    python3 gen_params.py --ws 0.02,0.015 --is 0.0078 --os 0.5 --b 1.0,-2.5
    python3 gen_params.py --json params.json

输出 JSON：{shift, mult:[...], bias_int:[...], raw_clamp6:[...]}
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ref_int8 import quantize_params


def parse_floats(s):
    return [float(t) for t in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", help="per-channel weight_scale, 逗号分隔")
    ap.add_argument("--is", type=float, help="input_scale")
    ap.add_argument("--os", type=float, help="output_scale")
    ap.add_argument("--b", help="per-channel bias, 逗号分隔")
    ap.add_argument("--shift", type=int, default=30)
    ap.add_argument("--json", help="从 JSON 文件读取 {ws,is,os,b[,shift]}")
    args = ap.parse_args()

    if args.json:
        d = json.load(open(args.json))
        ws, is_, os_, b = d["ws"], d["is"], d["os"], d["b"]
        shift = d.get("shift", 30)
    else:
        args_is = getattr(args, "is")
        assert args.ws and args_is is not None and args.os is not None and args.b
        ws, is_, os_, b = parse_floats(args.ws), args_is, args.os, parse_floats(args.b)
        shift = args.shift

    mult, bias_int, raw_clamp6, sh = quantize_params(ws, is_, os_, b, shift)
    out = {
        "shift": sh,
        "mult": mult,
        "bias_int": bias_int,
        "raw_clamp6": raw_clamp6,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
