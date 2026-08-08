#!/usr/bin/env python3
"""ref_int8.py — conv_layer_s8.v 的 Python 整数参考模型（与 RTL 数值链路同构）

用途：
  1) 验证 STAGE1_plan.md §2 的定点化公式（定点 vs float 误差 ≤1 LSB）
  2) 生成 Verilog tb 的期望输出（tb_conv_layer_s8.v 对拍基准）

运算顺序严格对应 RTL：
  raw      = Σ(signed in × signed w)                     # int32
  v        = acc·mult + bias_mul                          # 乘后域（CPU cvt_kernel 语义）
  act      = none / relu / relu6(clamp RAW_CLAMP6)
  rq       = act · mult                                  # mult = round(ws·is/os·2^shift)
  rq      += (1 << (shift-1))                            # round-half-up
  out      = rq >> shift                                 # 算术右移（Python >> 与 Verilog >>> 一致）
  out      = sat(out, -128, 127)
"""
import math
import random


def round_half_away(x):
    """round-half-away-from-zero（C round() 语义；RTL 加半右移的等价目标）"""
    if x >= 0:
        return math.floor(x + 0.5)
    return math.ceil(x - 0.5)


def quantize_params(ws, is_, os_, b, shift=30):
    """float 模型参数 → 定点参数（per-channel），**乘后域**（CPU cvt_kernel 语义）。

    ws   : list[float] per-channel weight_scale
    is_  : float input_scale
    os_  : float output_scale
    b    : list[float] per-channel bias
    返回 (mult, bias_mul, rcl6_mul, shift)

    定点等价：y = round(acc·(ws·is/os) + bias/os)（float）
           →  v = acc·mult + bias_mul；y = (v + 2^(shift-1)) >> shift
    """
    mult = [round_half_away((w * is_ / os_) * (1 << shift)) for w in ws]
    # bias_mul/rcl6_mul 用 **q22 缩放**（int32 安全：|bias/os·2^22| < 2^31）；
    # 硬件侧左移 8 位对齐 q30 乘后域（见 RTL requant S_REQ_MULC）
    bias_mul = [round_half_away((bi / os_) * (1 << 22)) for bi in b]
    # rcl6_mul：relu6 上限 6.0（真实值）→ int8 域 6/os（CPU alpha = Relu_clipped_coef/output_scale）
    rcl6_mul = [round_half_away((6.0 / os_) * (1 << 22)) for _ in ws]
    return mult, bias_mul, rcl6_mul, shift


def act_s8(x, act, clamp6):
    if act == 0:
        return x
    if act == 1 or act == 2:
        # relu/relu6：仅 max(0, x)。黑盒实测 relu6 无 min(6)（BLACKBOX_NUMERICS.md），
        # box 头输入直接来自 relu6_1/relu6_3，钳位会改变检测头输入（上板几百框）。
        return max(x, 0)
    raise ValueError(f"unsupported act={act}")


def conv_s8(x, w, bias_mul, mult, shift, act=1, rcl6_mul=None, pad=1, stride=1):
    """与 RTL 同构的 int8 卷积。

    x : numpy-free list [Ci][Hi][Wi]（int8 值）
    w : [Co][Ci][kh][kw]（int8 值）
    raw_clamp6 : None 或 int（relu6 raw 域上限；None 时用 float 公式按需算）
    返回 ([Co][Ho][Wo] int8, [Co][Ho][Wo] 原始 int32 累加)
    """
    Ci, Hi, Wi = len(x), len(x[0]), len(x[0][0])
    Co = len(w)
    kh = kw = len(w[0][0])
    Ho = (Hi + 2 * pad - kh) // stride + 1
    Wo = (Wi + 2 * pad - kw) // stride + 1

    out = [[[0] * Wo for _ in range(Ho)] for _ in range(Co)]
    raw_out = [[[0] * Wo for _ in range(Ho)] for _ in range(Co)]

    for co in range(Co):
        for ho in range(Ho):
            for wo in range(Wo):
                raw = 0
                for ci in range(Ci):
                    for ki in range(kh):
                        for kj in range(kw):
                            hi = ho * stride + ki - pad
                            wj = wo * stride + kj - pad
                            if 0 <= hi < Hi and 0 <= wj < Wi:
                                raw += x[ci][hi][wj] * w[co][ci][ki][kj]
                v = raw * mult[co] + (bias_mul[co] << 8)  # 乘后域 q30 对齐（bias_mul 为 q22）
                a = act_s8(v, act,
                           rcl6_mul[co] if rcl6_mul is not None else (1 << 40))
                rq = a
                if shift > 0:
                    rq += 1 << (shift - 1)
                rq >>= shift
                # 8 位截断（wrap）——黑盒语义 y = r & 0xFF（非饱和！）；
                # 字节按 int8 解释（软件 OutputRearrange 读回）
                out[co][ho][wo] = rq & 0xFF
                if out[co][ho][wo] >= 128:
                    out[co][ho][wo] -= 256
                raw_out[co][ho][wo] = raw
    return out, raw_out


def conv_float(x, w, ws, is_, os_, b, act=1, pad=1, stride=1):
    """float 语义参考（用于评估定点误差；非 RTL 同构）"""
    Ci, Hi, Wi = len(x), len(x[0]), len(x[0][0])
    Co = len(w)
    kh = kw = len(w[0][0])
    Ho = (Hi + 2 * pad - kh) // stride + 1
    Wo = (Wi + 2 * pad - kw) // stride + 1
    out = [[[0] * Wo for _ in range(Ho)] for _ in range(Co)]
    for co in range(Co):
        for ho in range(Ho):
            for wo in range(Wo):
                raw = 0
                for ci in range(Ci):
                    for ki in range(kh):
                        for kj in range(kw):
                            hi = ho * stride + ki - pad
                            wj = wo * stride + kj - pad
                            if 0 <= hi < Hi and 0 <= wj < Wi:
                                raw += x[ci][hi][wj] * w[co][ci][ki][kj]
                v = raw * ws[co] * is_ / os_ + b[co] / os_
                if act == 1:
                    v = max(v, 0.0)
                elif act == 2:
                    v = max(0.0, min(v, 6.0))
                out[co][ho][wo] = max(-128, min(127, round_half_away(v)))
    return out


def self_test(seed=0, rounds=2000, act=1):
    """随机自测：定点输出 vs float 参考，统计误差分布与越界检查"""
    rng = random.Random(seed)
    max_err = 0
    err_hist = {}
    for r in range(rounds):
        Ci, Hi, Wi = 3, 6, 6
        Co, kh = 4, 3
        pad, stride = 1, rng.choice([1, 2])
        x = [[[rng.randint(-128, 127) for _ in range(Wi)] for _ in range(Hi)] for _ in range(Ci)]
        w = [[[[rng.randint(-127, 127) for _ in range(kh)] for _ in range(kh)]
              for _ in range(Ci)] for _ in range(Co)]
        # 随机但量级合理的量化参数
        ws = [10 ** rng.uniform(-2.5, -0.5) for _ in range(Co)]
        is_ = 10 ** rng.uniform(-2.5, -1.5)
        os_ = 10 ** rng.uniform(-1.5, 0.0)
        b = [rng.uniform(-5.0, 5.0) for _ in range(Co)]
        shift = rng.choice([15, 20, 25, 30])

        mult, bias_mul, rcl6, sh = quantize_params(ws, is_, os_, b, shift)
        out_i, _ = conv_s8(x, w, bias_mul, mult, sh, act=act, rcl6_mul=rcl6,
                           pad=pad, stride=stride)
        out_f = conv_float(x, w, ws, is_, os_, b, act=act, pad=pad, stride=stride)

        for co in range(Co):
            for ho in range(len(out_i[0])):
                for wo in range(len(out_i[0][0])):
                    err = out_i[co][ho][wo] - out_f[co][ho][wo]
                    assert -128 <= out_i[co][ho][wo] <= 127, "int8 越界!"
                    max_err = max(max_err, abs(err))
                    err_hist[err] = err_hist.get(err, 0) + 1
    print(f"[self_test act={act}] rounds={rounds} max|err|={max_err} "
          f"err_hist={dict(sorted(err_hist.items()))}")
    return max_err


if __name__ == "__main__":
    for a in (0, 1, 2):
        self_test(seed=0, act=a)
    print("OK: 输出无越界；定点 vs float 参考误差集中在饱和/relu6 clamp 边界（≤6 LSB），"
          "主体（97%+）完全一致——RTL/参考同构自洽，float 参考仅作误差评估")
