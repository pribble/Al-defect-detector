# CPU 同源算法对照（Paddle-Lite ARM int8 conv）

来源：`/opt/_HAOYAO_bak/nna/Paddle-Lite`（备份仓库，未改动）
复制到：`cpu_ref/paddle_lite_arm/`（保留原始相对路径）

## 一、FPGA 算子 → CPU 同源实现映射

FPGA 子图只含 conv2d / depthwise_conv2d（`intel_fpga/bridges/paddle_use_bridges.h`），
CPU 侧按形状选择 kernel（`kernels/arm/conv_compute.cc:116-137`）：

| FPGA 层类型（实测） | CPU 同源 kernel | 关键文件 |
|---|---|---|
| 3×3 s=2（首层 conv，k=3 s=2 p=1） | `DirectConv` → `conv_3x3s2_direct_int8` | `backends/arm/math/conv3x3s2_direct_int8.cc` |
| 1×1 s=1（主干 k=1 conv） | `GemmLikeConv` → `conv1x1s1_gemm_int8`（flag_1x1gemm_） | `backends/arm/math/conv_impl.cc:631`、`gemm_prepacked_int8.*` |
| 3×3 s=1（relu6_1/6_3 等 k=3 s=1） | `WinogradConv`/`GemmLikeConv`（im2col+gemm） | `conv3x3_winograd_int8.cc`、`conv_impl.cc` |
| DW 3×3 s=1（depthwise） | `DepthwiseConv` → `conv3x3s1_depthwise_int8` | `backends/arm/math/conv3x3s1_depthwise_int8.cc` |

## 二、算法核心（所有 int8 conv 共用同一 requant 出口）

### 1. scale/bias 准备（`kernels/arm/conv_gemmlike.cc:52-92`）

```cpp
w_scale_ = param.weight_scale;              // 模型 per-channel 权重 scale（ws）
for (auto& ws : w_scale_) ws *= input_scale;            // ws·is
// int8→int8 时再除 output_scale：
ws = ws * input_scale / output_scale;                   // ★ A = ws·is/os
// bias 同样换算（kFloat 版）：
ptr[i] = ptr_in[i] / param.output_scale;                // ★ B = bias/os
// relu6 上限换算：
alpha = Relu_clipped_coef / param.output_scale;         // ★ rcl6 = 6/os（CPU 存在！）
```

### 2. 输出转换（`backends/arm/math/conv_block_utils.h:3723-3749`，`cvt_kernel<int8_t>`）

```cpp
// act==1（relu）:
tmp = saturate_cast<int8_t>(round(LITEMAX(din*scale + bias, 0)));
// act==0（无）:
tmp = saturate_cast<int8_t>(round(din*scale + bias));
// act==2（relu6）:
tmp = saturate_cast<int8_t>(round(LITEMIN(LITEMAX(din*scale+bias, 0), alpha)));
// act==3（leakyrelu）:
tmp = saturate_cast<int8_t>(round(result > 0 ? result : alpha*result));
return tmp < -127 ? -127 : tmp;
```

- `din` = int32 累加结果（int8×int8 全精度累加）
- `round()` = C++ round（half-away-from-zero）
- `saturate_cast<int8_t>(float) = (int8_t)float`（`saturate.h:57`；ARM 上 float→int8 强转
  由 `vcvt` 实现，溢出行为平台相关，通常饱和到 [-128,127]）
- `tmp < -127 ? -127 : tmp`：下限钳 -127（上限不钳——依赖 saturate_cast 饱和）

### 3. 累加（`conv_block_utils.h` 的 MAC 内联 / NEON）

int8 输入 × int8 权重 → int32 累加（每通道 scale/bias 在输出阶段施加，累加阶段无缩放）。

## 三、CPU 同源算法 vs 黑盒实测 vs 复现核现状

| 环节 | CPU（cvt_kernel，清晰可验证） | 黑盒实测（上板推导） | 复现核现状（cnn-rtl） |
|---|---|---|---|
| 累加 | int32 精确 | int32 精确（一致） | int32 精确（一致） |
| requant | `round(acc·ws·is/os + bias/os)`（float） | 同左（100% 匹配） | 定点预转 `((acc+bias_int)·mult+2^29)>>30` |
| bias | 输出域 float 直接加 | 同左 | 乘前域 bias_int（双重舍入） |
| round | C++ round（half-away） | round 后负值 -1 | round-half-away（乘前域） |
| relu | `max(0, x)` | 同左 | 同左 |
| relu6 | `min(max(0,x), 6/os)`（CPU 有上限） | **无 min6**（实测输出到 127） | rcl6 钳制（有上限——与黑盒不符，与 CPU 相符） |
| 溢出 | saturate_cast（饱和） | **wrap 截断** | 饱和 [-128,127] |
| 输出钳制 | `tmp < -127 ? -127` | 无（wrap） | 饱和 |

## 四、CPU 公式 vs 黑盒实测匹配率（tools/all_layers_cpu.py 实测）

用 CPU `cvt_kernel` 语义（C++ round half-away + saturate_cast + act）在黑盒 47 层数据上验证：

| 层类型 | CPU 公式匹配率 | 说明 |
|---|---|---|
| 全部 relu 激活的 conv/DW（主干，层 0-28） | **100.0%** | CPU 与黑盒完全一致（relu 后非负小值，饱和/wrap 无区别） |
| relu6 层（relu6_0/1/2/3/4） | 34-92% | 差异 = CPU 有 min(6)，黑盒无 |
| box 头（conv2d_69-76，act=0） | 30-56% | 差异 = CPU 饱和 [-128,127]，黑盒 wrap 截断 |

**结论**：CPU 同源算法与黑盒在主干逻辑上 100% 一致（float requant + round + relu），
仅"溢出处理"（wrap）与"relu6 上限"两处不同——复现核以 CPU 公式为基准，再按目标
（黑盒 bit-exact 或正确语义）决定这两处。

## 五、RTL 复现推荐逻辑（优先 CPU 同源，清晰可验证）

1. **累加**：int8×int8 → int32 精确累加（现有复现核已一致）。
2. **requant**（核心修改）：采用 CPU 语义的定点等价：
   - 软件预转：`mult = round(ws·is/os × 2^30)`、`bias_mul = round(bias/os × 2^30)`、
     `rcl6_mul = round(6/os × 2^30)`（若实现 relu6 上限）
   - RTL：`y = ((acc × mult) + bias_mul + 2^29) >> 30`——**bias 在乘后域**（与 CPU float 公式等价）
   - 或更稳妥：RTL 保留软件预转，但**预转公式改为输出域 bias**（bias_mul 而非 bias_int），
     并验证与 CPU `cvt_kernel` 逐位一致（用 CPU 公式做仿真对拍基准）。
3. **round**：round-half-up（`+2^29 >> 30`）对正数 = C++ round；负数差异需按 CPU `cvt_kernel`
   对齐（C++ round half-away 对负数是远离零——`(v + 2^29) >> 30` 算术右移是向下——两者对
   负数不同，仿真对拍确定）。
4. **relu/relu6**：relu = `max(0, y)`；relu6 上限（rcl6）**以 CPU 语义为准**（黑盒无 min6 是
   黑盒硬件缺陷/特性，CPU 语义更正确、清晰）；如目标是 bit-exact 复现黑盒则去掉上限——
   **两者二选一需用户确认**。
5. **溢出**：饱和 [-128,127]（CPU 语义）还是 wrap（黑盒语义）——同上，默认 CPU 饱和，
   若要黑盒 bit-exact 改 wrap。

> 推荐：复现核以 **CPU 同源算法为基准**（清晰、可验证、正确），黑盒实测仅作参考；
> 若最终要"逐位等于黑盒"，再按实测差异（wrap、无 min6、负值-1）单独加开关。

## 五、文件清单（cpu_ref/paddle_lite_arm/）

```
kernels/arm/conv_compute.cc|.h        kernel 选择（int8 分支）
kernels/arm/conv_gemmlike.cc|.h       1×1/通用 conv（scale 准备 + run）
kernels/arm/conv_direct.cc|.h         3×3 s2 直接卷积
kernels/arm/conv_winograd.cc          3×3 s1 winograd
kernels/arm/conv_depthwise.cc         DW kernel 选择
backends/arm/math/conv_block_utils.h  ★ cvt_kernel（requant 出口）+ 输出写回
backends/arm/math/conv_impl.cc|.h     conv1x1s1_gemm_int8、im2col、gemm 封装
backends/arm/math/conv3x3s2_direct_int8.cc   3×3 s2 int8
backends/arm/math/conv3x3s1_direct_int8.cc   3×3 s1 int8
backends/arm/math/conv3x3_winograd_int8.cc   winograd int8
backends/arm/math/conv3x3s1_depthwise_int8.cc  DW s1 int8
backends/arm/math/conv3x3s2_depthwise_int8.cc  DW s2 int8
backends/arm/math/conv_depthwise.h
backends/arm/math/gemm_prepacked_int8.cc|.h  int8 gemm（1×1 用）
backends/arm/math/packed_sgemm.h / sgemm.h
backends/arm/math/saturate.h          ★ saturate_cast
backends/arm/math/quantize.h
```
