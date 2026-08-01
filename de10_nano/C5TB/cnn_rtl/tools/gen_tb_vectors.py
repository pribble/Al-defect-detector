#!/usr/bin/env python3
"""gen_tb_vectors.py — 生成 tb_conv_layer_s8.v 用的测试向量

期望值用 ref_int8.conv_s8 的**定点同构**计算生成（与 RTL 严格一致），
不是 float 参考。输出文件供 $readmemh 读入：

  in.hex        输入流：行→列→通道（C 最内层）
  w.hex         权重流：slice = [lane(输出通道)][kernel 9]，按输入通道顺序
  bias.hex      bias_int（per-channel int32，signed hex）
  mult.hex      requant mult（per-channel uint32）
  shift.hex     requant shift（per-channel uint8）
  expect.hex    期望输出流：(ho, wo, lane) 行优先，每窗口 G_C_PAR 字节
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ref_int8 import quantize_params, conv_s8

OUT = os.path.join(os.path.dirname(__file__), "..", "verification", "vec")


def u32(v):
    return v & 0xFFFFFFFF


def s32(v):
    return u32(v)


def s8(v):
    return v & 0xFF


def main():
    rng = random.Random(7)
    Ci, Hi, Wi = 3, 6, 6
    Co, kh = 4, 3
    pad, stride, act = 1, 2, 1
    shift = 30

    x = [[[rng.randint(-128, 127) for _ in range(Wi)] for _ in range(Hi)] for _ in range(Ci)]
    w = [[[[rng.randint(-127, 127) for _ in range(kh)] for _ in range(kh)]
          for _ in range(Ci)] for _ in range(Co)]
    ws = [10 ** rng.uniform(-2.5, -0.5) for _ in range(Co)]
    is_ = 10 ** rng.uniform(-2.5, -1.5)
    os_ = 10 ** rng.uniform(-1.5, 0.0)
    b = [rng.uniform(-5.0, 5.0) for _ in range(Co)]

    mult, bias_int, rcl6, sh = quantize_params(ws, is_, os_, b, shift)
    out, raw = conv_s8(x, w, bias_int, mult, sh, act=act, raw_clamp6=rcl6,
                       pad=pad, stride=stride)

    os.makedirs(OUT, exist_ok=True)

    # 输入流：行→列→通道
    with open(f"{OUT}/in.hex", "w") as f:
        for h in range(Hi):
            for ww in range(Wi):
                for c in range(Ci):
                    f.write(f"{s8(x[c][h][ww]):02x}\n")

    # 权重流：slice = for lane: for k(row*3+col)；按输入通道
    with open(f"{OUT}/w.hex", "w") as f:
        for ci in range(Ci):
            for lane in range(Co):
                for k in range(9):
                    f.write(f"{s8(w[lane][ci][k // 3][k % 3]):02x}\n")

    with open(f"{OUT}/bias.hex", "w") as f:
        for c in range(Co):
            f.write(f"{s32(bias_int[c]):08x}\n")
    with open(f"{OUT}/mult.hex", "w") as f:
        for c in range(Co):
            f.write(f"{u32(mult[c]):08x}\n")
    with open(f"{OUT}/shift.hex", "w") as f:
        for c in range(Co):
            f.write(f"{sh & 0xFF:02x}\n")

    # 期望输出：(ho, wo) 行优先，每窗口 G_C_PAR 字节（lane = 输出通道）
    with open(f"{OUT}/expect.hex", "w") as f:
        for ho in range(len(out[0])):
            for wo in range(len(out[0][0])):
                for co in range(Co):
                    f.write(f"{s8(out[co][ho][wo]):02x}\n")

    print(f"vectors -> {OUT}")
    print(f"  in={Ci*Hi*Wi} w={Ci*Co*9} out={Co}*{len(out[0])}x{len(out[0][0])}")
    print(f"  act={act} shift={sh}")
    print(f"  ws={[round(v,5) for v in ws]} is={is_:.6f} os={os_:.6f} b={[round(v,3) for v in b]}")
    print(f"  mult={mult} bias_int={bias_int} clamp6={rcl6}")


if __name__ == "__main__":
    main()
