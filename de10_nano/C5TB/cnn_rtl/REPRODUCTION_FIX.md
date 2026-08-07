# cnn-rtl 复现核修复方案（完整分析文档）

日期：2026-08-08
范围：main 分支（黑盒 qxp）vs cnn-rtl 实验分支（RTL 复现）的数值语义对比与修复方案
全部工作在 `_cnn_rtl_debug/` 完成，未改动原项目。

---

## 1. 背景与目标

- main 分支：`cnn_top.qxp`（无源码黑盒 IP，上板可检测出目标，score 0.926）。
- cnn-rtl 分支：纯 RTL 复现（`cnn_top_core.v`/`cnn_core_v2.v` + 软件定点预转），
  47 层仿真与 ref 模型 bit-exact（自洽），但**上板从未成功**。
- 目标：找出复现核与黑盒的数值差异，按"CPU 同源算法（清晰可验证）"修复 RTL。

## 2. 黑盒实测方法

1. main 分支副本（`ssd_main_copy/`）加 dump 代码（`DUMP_LAYER_OUT=1` 环境变量开启）：
   - `[LIN]/[LIND]`：首层输入区（NHWC8，前两次推理 warm up + 4.jpg）
   - `[LOUT]/[LOUTD]`：每层输出区（NHWC8 补齐 8 通道）
   - `[LSCALE]`：scale 数组（float[2×out_c] = [ws, bias/os]）
   - `[LW]/[LWD]`：重排后权重
   - `[LIA]/[LIAD]`：量化后原始 NCHW 输入
2. 设备上板跑 `DUMP_LAYER_OUT=1 ./ssd_detection config.txt data`，
   得到 `blackbox_layers2.log`（111MB，47 层 × 2 次推理）。
3. 本地精确重建每层 `acc = Σ(int8输入 × int8权重)`（int32），
   逐层逐通道验证候选公式（每通道 12 采样点）。

**关键教训**（分析过程中的坑）：
- warm up 输入是未初始化 Mat（全 0）→ 输入区全 `0xbd`（-67），不是真实图片；
  必须 dump 4.jpg（第 2 次推理）的输入。
- 权重布局是 `[cb_out][cb_in][mi][k][mo]`（**mi 主序**、k 次、mo 字节），
  解析顺序错会导致 acc 全错（曾误按 k 主序，仅 ch17 巧合匹配）。
- SSD 检测头（conv2d_69-76）是多分支并行，输入按 `in_off` 查（不是前层输出）。
- 对比必须按**位模式**（`pred & 0xFF == y`），不能按 int8 数值对比（wrap 的
  字节 = int8 负值原样）。

## 3. 黑盒 IP 数值语义（47/47 层实测确认）

```
acc  = Σ(input_int8 × weight_int8)                    // int32 精确累加
fv   = acc × (ws·is/os) + (bias/os)                   // float 乘加
r    = round_half_even(fv)                            // 实测匹配的舍入
if fv < 0: r -= 1                                     // 负值修正（floor 除法特性）
if act ∈ {relu, relu6}: r = max(0, r)                 // relu；黑盒 relu6 无 min(6)
y    = r & 0xFF                                       // 8 位截断（wrap）
```

- 输出布局 `[cb][h][w][8]` 全图序；scale 区 = float[2×out_c]。
- 黑盒公式 47 层全部 ≥95.8%（主干与最后一层 100%）。

## 4. CPU 同源算法（Paddle-Lite ARM int8 conv）

来源：`/opt/_HAOYAO_bak/nna/Paddle-Lite/lite/...`（备份仓库），
已复制到 `cpu_ref/paddle_lite_arm/`。

FPGA 算子 → CPU 同源 kernel：
| FPGA 层 | CPU 同源 |
|---|---|
| conv 3×3 s=2 | `conv_3x3s2_direct_int8`（conv3x3s2_direct_int8.cc） |
| conv 1×1 | `conv1x1s1_gemm_int8`（conv_impl.cc:631） |
| conv 3×3 s=1 | winograd / im2col+gemm |
| DW 3×3 s=1 | `conv3x3s1_depthwise_int8` |

requant 出口（全共用，`conv_block_utils.h:3723` `cvt_kernel<int8_t>`）：
```cpp
// act==1（relu）:
tmp = saturate_cast<int8_t>(round(LITEMAX(din*scale + bias, 0)));
// act==0:
tmp = saturate_cast<int8_t>(round(din*scale + bias));
// act==2（relu6）:
tmp = saturate_cast<int8_t>(round(LITEMIN(LITEMAX(din*scale+bias, 0), alpha)));
return tmp < -127 ? -127 : tmp;
```
scale/bias 准备（conv_gemmlike.cc）：
```cpp
ws = ws * input_scale / output_scale;    // A = ws·is/os
bias: /output_scale                      // B = bias/os
alpha = Relu_clipped_coef / output_scale // relu6 上限（int8 域 = 6/os）
```

## 5. 匹配率验证结果

| 公式 | 主干 relu 层 | box 头 | relu6 层 |
|---|---|---|---|
| 黑盒实测公式（round half-even + 负值-1 + wrap + 无 min6） | 100% | 95.8-100% | 100% |
| CPU 公式（round half-away + 饱和 + min6）**数值对比** | 100% | 30-56% | 34-92% |
| CPU 公式**位模式对比**（pred & 0xFF == y） | 100% | 30-56%（采样差异分布全为差 1） | 34-92% |

**关键理解**：
- 主干 relu 层输出值域 0-91（无溢出），饱和/wrap 位模式相同 → CPU 与黑盒 100% 一致；
- box 头 wrap 区（>127）字节 = int8 负值原样，CPU 按 int8 读回一致；
  box 头真实差异仅 ±1 LSB（负值舍入），对 softmax/NMS 鲁棒；
- relu6 层差异（CPU min6 vs 黑盒无）影响中间特征图，被后续层压缩，不致命。

## 6. 为什么 CPU 和黑盒逻辑不同仍能出正确结果

1. 主干 100% 位级一致（值域不触发饱和/wrap 差异，relu 截断负值舍入差异）；
2. box 头 wrap 位模式 = int8 负值原样，差异仅 ±1 LSB + 极少数 |fv|>128（实测 0.00%）；
3. relu6 无 min6 只影响中间特征，被卷积压缩；
4. 检测后处理（decode + softmax + NMS）对 ±1 LSB 级误差鲁棒（只取 top-k 框）。

## 7. 复现核失败根因（确认）

复现核/ref 模型用**定点预转（bias 乘前域）**：
```
mult     = round_half_away(ws·is/os × 2^30)
bias_int = round_half_away(b/(ws·is))       ← bias 先除再乘（双重舍入）
out      = saturate(((acc + bias_int) × mult + 2^29) >> 30)
```
与 float 公式（CPU/黑盒）`round(acc·ws·is/os + bias/os)` 在**主干就有系统性偏差**
（首层仅 28% 位匹配，差 1-11，不是 ±1）。主干特征图从第 1 层就错 → 检测必然失败。
**这不是饱和/wrap/relu6 等边缘差异造成的，而是 requant 公式本身错误。**

## 8. 修复方案（以 CPU 同源算法为基准）

### 8.1 软件预转（conv_op.cc，scale 区 4×out_c int32 不变）

**位宽约束**（实测黑盒模型参数）：mult（q30）最大 5.07e8 < 2^31 ✓；
但 bias/os 最大 ±250 → q30 的 bias_mul 达 ±2.68e11 **超 int32**。
故 bias/rcl6 用 **q22 缩放**（最大 1.05e9/5.9e8 < 2^31 ✓），硬件左移 8 位对齐 q30 域：
精度 2^-22 ≈ 2.4e-7（输出域），远小于 float 精度，不影响舍入结果（实测主干仍 100%）。

```
scale[0..out_c)          = mult      = round_half_away(ws·is/os × 2^30)
scale[out_c..2·out_c)    = bias_mul  = round_half_away(bias/os × 2^22)   ← q22（原为 bias_int 乘前域）
scale[2·out_c..3·out_c)  = shift     = 30
scale[3·out_c..4·out_c)  = rcl6_mul  = round_half_away(6/os × 2^22)      ← q22（relu6 上限，int8 域）
```

### 8.2 RTL requant（cnn_core_v2.v 已改，cnn_top_core.v 无 requant 逻辑不动）

流水重排（乘法提前、act 移到乘后，新增 S_REQ_ACT 拍）：

```
S_REQ_MULB : v_act_l = acc                    （原：acc+bias_int，bias 已移除）
S_REQ_MUL2 : DSP 部分积（v_act_l × mult，16-bit 分块不变）
S_REQ_MUL3 : 两组 64-bit 中间和
S_REQ_MUL4 : v_rq64_l = v_sum_lo + v_sum_hi
S_REQ_MULC : v_rq64_l += bias_mul << 8         （乘后 bias，q22 对齐 q30，64-bit 加法）
S_REQ_MULC2: relu/rcl6 64-bit 比较（符号位 + v_rq64_l > rcl6_mul<<8）
S_REQ_ACT  : mux → v_rq64_l（relu: max(0)；relu6: min(max(0), rcl6<<8)）   ← 新增拍
S_REQ_OUT  : v_rnd_delta = 1 << (shift-1)
S_REQ_ROUND2: v_round_l = v_rq64_l + v_rnd_delta
S_REQ_OUT2 : v_shifted = v_round_l >>> shift
S_REQ_OUT3 : 饱和 [-128,127] → o_data
```

与原流水的差异：
1. bias 从乘前（acc+bias_int，双重舍入）移到乘后（bias_mul 直接加，float 公式对齐）；
2. relu/rcl6 从乘前（raw 域）移到乘后（q30 域）——CPU cvt_kernel 顺序
   round(max(fv,0)) / round(min(max(fv,0), alpha)) 一致；
3. 新增 S_REQ_ACT 状态（64-bit mux 单独一拍，避免组合链过深）。

### 8.3 修改点清单

| 文件 | 修改 |
|---|---|
| `cnn_rtl/ip/cnn_core_v2.v`（src 仿真版同步） | requant 流水：bias 乘后域、relu/rcl6 乘后、新增 S_REQ_ACT |
| `cnn_rtl/ip/cnn_top_core.v` | 无 requant 逻辑，不改 |
| `ssd_detection/.../bridges/conv_op.cc` | scale 预转：bias_int → bias_mul(q22)、rcl6(q22) |
| `tools/ref_int8.py` + `ref_cnn_top.py` | 参考模型同步（乘后域） |
| 验证 tb | 对拍基准改为 CPU 公式（float 模拟） |

### 8.4 验证计划

1. ✅ 软件层：ref 模型改乘后域 q22 → 黑盒 47 层：主干 relu/relu6 全部 100%，
   box 头 30-56%（负值 half-up vs 黑盒 round-1 的 ±1 LSB，检测鲁棒）；
2. ✅ 仿真：cnn_core_v2（乘后域）vs ref（乘后域 q22）16 层随机回归 **pass=16 fail=0**；
   待办：cnn_top 顶层 tb 回归、真实 47 层仿真；
3. ⏳ 上板：cnn-rtl 分支 + 新软件（conv_op.cc 已改）→ 编译 + 检测出结果（下一步）；
4. （可选）对比黑盒输出：主干 100% 位一致、box 头 ±1 差异不影响检测。

## 9. 数据与工具清单

- `ssd_main_copy/`：main 分支副本（dump 修改 + 构建产物 + 上板 log）
- `ssd_main_copy/blackbox_layers2.log`：黑盒 47 层完整 dump（111MB）
- `cpu_ref/paddle_lite_arm/`：CPU 同源源码（24 文件）
- `cpu_ref/CPU_ALGO_REFERENCE.md`：CPU 算法对照
- `tools/all_layers.py`：黑盒公式 47 层验证（100%）
- `tools/all_layers_cpu.py`：CPU 公式 47 层验证
- `tools/analyze_v2.py`、`chan_map.py` 等：分析工具
- `BLACKBOX_NUMERICS.md`：黑盒数值语义速查
