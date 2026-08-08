# cnn_top 黑盒规格书（复现参考）

> 导航：上一级 [README.md](README.md) · 相关 [ssd_detection/](../ssd_detection/README.md)、[intelfpga.cc](../ssd_detection/intelfpga.cc)、[intelfpga.h](../ssd_detection/include/intelfpga.h)

本文档从**黑盒外部可观测行为**出发，描述 `cnn_top` 加速器的接口、寄存器协议、
指令格式、数据布局与数值语义。它是纯 RTL 复现（`cnn_rtl/`，已替代黑盒
`ip/cnn_top.qxp`）的**历史规格依据**——复现已完成（仿真 16/16+6/6 对拍全过），
RTL 上板正确性 2026-08 调试中，RTL 实现见
`cnn_rtl/README.md` 与 `ip/` 下三个 `.v`。所有信息均来自仓库内可读文件
（接口/参数、软件协议、QSys 连接），**不包含任何内部微架构/时序的猜测性承诺**，
未决点见 §11。

## 1. 背景与现状

- `cnn_top` 是 QSys 自定义 IP（Component Editor 18.1，v2.0，`ip/cnn_top_hw.tcl`）。
  原综合网表为 QXP 归档（Netlist Only，`ip/cnn_top.qxp`，8.3MB，无 RTL 源码）；
  **2026-08 已由开源复现 RTL（`ip/cnn_top.v`/`cnn_top_core.v`/`cnn_core_v2.v`）替代，
  qxp 已删除**（注意：`soc_system/synthesis/submodules/cnn_top.qxp` 为 7/31 QSys
  生成残留，.gitignore 忽略；QSys 未重新 Generate 时综合会引用它而非 `ip/` 下的
  自研 RTL——需重新 Generate HDL 或手动同步 5 个 .v 到 submodules/）。
- 系统级连接完整可读：`soc_system.qsys` 中 `cnn_top_0` 的接口连接与基地址；
  `soc_system.xml`（QSys 生成日志）保留全部 IP 参数值。
- 软件侧协议完整可读：`ssd_detection/intelfpga.cc`（寄存器读写、DMA、重排）、
  `ssd_detection/include/intelfpga.h`（指令结构）、
  `ssd_detection/paddle_lite/lite/kernels/intel_fpga/bridges/conv_op.cc`（参数/scale 填充）。
- 现状：复现 RTL 已上板联调（`test.sh` 全链路可跑通）；检测正确性于 2026-08 流式化 + param 偏移 + M10K 容量系列修复后**待最终验证**（调试历史见 `cnn_rtl/README.md`；时序收敛见其中"2026-08-05 时序收敛"节）。

## 2. 顶层接口

来源：`ip/cnn_top_hw.tcl`（接口定义）、`db/C5TB_top.soc_system.cnn_top.qxp.txt`（边界端口）。

### 2.1 时钟/复位

| 端口 | 方向 | 说明 |
|---|---|---|
| `sysclk` | in | 工作时钟，接 QSys `clk_cnn`（PLL `fpga_clk_cnn`） |
| `rst_n` | in | 低有效异步复位，接 `clk_cnn.clk_reset` |

### 2.2 `hps2cnn_avs`（Avalon 从接口，寄存器）

- 属性：`addressUnits=WORDS`，地址宽度 **8-bit**（256 个 32-bit 寄存器），
  `readLatency=1`、`readWaitTime=1`、带 `waitrequest`。
- 连接：`mm_bridge_lw_axi.m0 → cnn_top_0.hps2cnn_avs`，基地址 0x0000
  （HPS 侧经 lw H2F 桥，软件映射基址 `FPGAREG_CNN_BASE_ADDR`，DE10-Nano 上为 0xFF200000）。

| 信号 | 方向 | 宽度 |
|---|---|---|
| `as_address` | in | 8（字地址） |
| `as_write` / `as_read` | in | 1 |
| `as_writedata` | in | 32 |
| `as_readdata` | out | 32 |
| `as_data_waitquest` | out | 1 |

### 2.3 四个 Avalon 主接口（自带头 DMA，直读 DDR）

全部连到 `mm_bridge_sdram0.s0`（→ `hps_0.f2h_sdram0_data` → DDR），
`burstcount` 5-bit **恒 1**（单拍读写，无 burst 传输），`readWaitTime=1`。
**load master 由输入/权重读分时复用**（`lr_read`/`wr_read` 不同时拉高，
`load_avm_readdata` 同时回灌两路；黑盒无独立权重 master）。

| 接口 | 数据宽度 | 方向 | 用途 |
|---|---|---|---|
| `param_read_avalon` | 32-bit | 读 | 读取指令块（`struct parameter`） |
| `load_read_avalon` | 64-bit | 读 | 读取输入特征图（每周期 8 个 int8） |
| `output_read_avalon` | 64-bit | **写** | 写回输出特征图 |
| `scale_avm_avalon` | 32-bit | 读 | 读取 scale/bias 数组 |

> 无 reorg 主接口：输入/输出重排由 ARM 软件完成（`REOGANIZE_TYPE=REOGANIZE_ARM`），
> FPGA 仅执行卷积本体。`hw.tcl` 中残留的 `REORG_M_AXI_*` 参数无对应接口，属历史遗留。

## 3. IP 参数表（复现必须保持一致）

来源：`ip/cnn_top_hw.tcl`；DE10-Nano 实际取值见 `soc_system.xml` 中 cnn_top 参数化。

| 参数 | 值 | 语义 |
|---|---|---|
| `INPUT_CHANNEL_TILE` | 8 | 输入通道分块（= MAC 阵列输入侧并行度） |
| `OUTPUT_CHANNEL_TILE` | 8 | 输出通道分块（= MAC 阵列输出侧并行度） |
| `INPUT_ROW_TILE` | 11 | 输入行块高度（stride=2 时对应 5 输出行） |
| `OUTPUT_ROW_TILE` | 5 | 输出行块高度 |
| `IMAGE_MAX_W` | 302 | 输入最大宽度（300 + 2 pad） |
| `OUTPUT_MAX_W` | 150 | 输出最大宽度（300/2） |
| `INPUT_WIDTH` / `WEIGHT_WIDTH` | 8 | int8 量化 |
| `OUTPUT_WIDTH` | 32 | 累加器位宽 |
| `KERNAL_SIZE` | 9 | 3×3 kernel（硬件按 9 权重排列） |
| `KERNEL_SIZE_WIDTH` | 4 | kernel 尺寸位宽 |
| `INPUT_PAD_WIDTH` / `OUTPUT_PAD_WIDTH` | 2 | pad 位宽（支持 0~3） |
| `CNN_TYPE_WIDTH` | 3 | op type 位宽（conv/dw_conv） |
| `CNN_STRIDE_WIDTH` | 2 | stride 位宽（支持 1~3） |
| `CNN_RELU_WIDTH` | 3 | activation 类型位宽 |
| `IN_C_WIDTH`/`IN_H_WIDTH`/`IN_W_WIDTH` | 11/9/9 | 输入维度位宽 |
| `OUTPUT_C_WIDTH`/`OUTPUT_H_WIDTH`/`OUTPUT_W_WIDTH` | 11/9/9 | 输出维度位宽 |
| `OUTPUT_ADDR_WIDTH` | 10 | 输出缓冲地址位宽 |
| `INPUT_RAM_DATA_WIDTH` | 64 | 输入片上 RAM 宽度（8×int8） |
| `CFG_PARAM_WIDTH` / `CFG_M_AXI_*` | 32 | 指令/寄存器数据宽度 |
| `LOAD/OUTPUT/SCALE/REORG_M_AXI_*` | 32/64 | 主接口地址/数据宽度 |
| `SCALE_6` | 1086324736 | 34-bit；位模式 = float **6.0**，relu6 量化上限常量（黑盒内部常量；复现核由软件预转 `rcl6` 取代，见 §6.5） |
| `REORGANIZE_BUFF_DEEP`/`REORG_RAM_*` | — | reorg 残留参数，当前未使用 |

## 4. 寄存器协议

来源：`ssd_detection/include/common.h`（宏定义）、`ssd_detection/intelfpga.cc`
（`fpga_init`/`start_fpga`）。

**DE10-Nano 实际生效分支为 `ARCH_ABI_ARM32`**（偏移按 `(reg<<4)` 展开）：

| 寄存器 | 字地址 | 字节偏移 | 功能 |
|---|---|---|---|
| `FPGAREG_CNN_START` | 0x00 | 0x00 | 写 bit0=1 启动；**硬件完成后自清 0** |
| `FPGAREG_CNN_DDRIN` | 0x10 | 0x40 | 输入数据 CMA 物理基址 |
| `FPGAREG_CNN_DDRW` | 0x1C | 0x70 | 权重 CMA 物理基址 |
| `FPGAREG_CNN_DDROUT` | 0x28 | 0xA0 | 输出数据 CMA 物理基址（与 DDRIN 同块） |
| `FPGAREG_CNN_PARAM` | 0x34 | 0xD0 | 指令块 CMA 物理基址 |
| `FPGAREG_CNN_SCALE` | 0x40 | 0x100 | scale/bias 数组 CMA 物理基址 |

> ARM64 分支使用 `(reg<<2)` 偏移（字节地址 = 字地址×4），寄存器布局不同，
> 复现版仅需保证当前 ARM32 平台正确。

**启动时序**（`start_fpga`，`intelfpga.cc:212`）：

```
HPS: 写 param 块 / scale 数组 → 写 START.bit0=1
HW:  读取并执行一层 → 完成后把 START.bit0 清 0
HPS: 轮询 START.bit0（5s 超时判失败）
```

## 5. 指令块格式（`struct parameter`）

来源：`ssd_detection/include/intelfpga.h:98`（字段与顺序即硬件解析布局）；
一次执行对应一个该结构体（约 108 字节），HPS 整体 `memcpy` 到 PARAM 地址，
硬件经 `param_read_avalon` 读取。

| 字段 | 类型 | 语义 |
|---|---|---|
| `input_offset` | int | 输入数据偏移（word，1 word = 8 字节 = 1 通道 tile） |
| `weight_offset` | int | 权重偏移（word） |
| `scale_offset` | int | 恒为 0（scale 从 SCALE 基址起） |
| `output_offset` | int | 输出偏移（word） |
| `in_c`/`in_h`/`in_w` | int | 输入维度 |
| `output_c`/`output_h`/`output_w` | int | 输出维度 |
| `kernel` | int | 核边长（3） |
| `in_pad`/`out_pad` | int | padding（当前 in_pad=1，out_pad 未用） |
| `stride` | int | 步长（1/2） |
| `relu` | int | 0 none / 1 relu / 2 relu6 / 3 leakyrelu |
| `type` | int | 1 = conv2d，4 = depthwise_conv2d |
| `output_channel_block_num` | int | ceil(output_c / 8) |
| `output_row_tile` | int | 每次处理输出行数（见 §8） |
| `input_row_tile` | int | 对应输入行数（见 §8） |
| `output_row_block_num` | int | ceil(output_h / output_row_tile)，硬件内部循环 |
| `input_scale`/`output_scale` | float | 量化 scale（软件透传模型值） |
| `lr` | float | LeakyRelu alpha |
| `dilation` | int | 膨胀系数（SSD 只用 1） |
| `weight_size`/`input_size`/`output_size` | int | 各 buffer 字节数 |

## 6. 数据布局

### 6.1 输入（软件写，硬件读）

NCHW int8 → **NHWC8**：`[ceil(C/8), H, W, 8]`，通道补 0。
无 pad（pad 由硬件按 `in_pad` 片上处理）。
ARM 端实现：`InputRearrange`（`intelfpga.cc:337`，`tran_8` NEON 转置）。
字节地址 = `DDRIN + input_offset*8`。

### 6.2 权重 conv2d（软件写，硬件读）

原始 `[Co, Ci, 3, 3]` → **`[ceil(Co/8), ceil(Ci/8), 8, 9, 8]`**，
即按输出通道块 × 输入通道块组织，每块内是 8 输入通道 × 9 权重 × 8 输出通道的
"外积 tile"（`conv2d_weight_reorganize`，`intelfpga.cc:525`）。
字节地址 = `DDRW + weight_offset*8`。

### 6.3 权重 depthwise（软件写，硬件读）

原始 `[Co, 1, 3, 3]` → **`[ceil(Co/8), 8, 9, 8]`** 稀疏对角化：
仅当 输入通道下标 == 输出通道下标（`ti==m`）时填入权重，其余 0
（`dw_conv2d_weight_reorganize`，`intelfpga.cc:568`）。
**该布局直接表明 MAC 阵列 = 8 in-ch × 8 out-ch（64 MAC/周期）。**

### 6.4 输出（硬件写，软件读）

硬件写 **NHWC8**：`[ceil(Co/8), H, W, 8]`；
字节地址 = `DDROUT + output_offset*8`。
软件 `OutputRearrange`（`intelfpga.cc:366`）用 `tran_8` 转回 NCHW 后再拷贝到
Paddle 张量（`global_mem_cfg` 拷贝逻辑，`intelfpga.cc:640/667`）。

### 6.5 scale/bias（软件写，硬件读）

**2026 定点化后**：每输出通道 4 个 int32，共 `4*output_c`，从 SCALE 基址连续存放
（`conv_op.cc:120-147` 生成，`intelfpga.cc:638` 整块 memcpy）：

```
scale[0..out_c)          = mult      = round_half_away((ws·is/os)·2^30)  // 乘数（q30）
scale[out_c..2·out_c)    = bias_mul  = round_half_away(b/os·2^22)        // 乘后域 bias（q22，int32 安全）
scale[2·out_c..3·out_c)  = shift     = 30                                // 右移位数
scale[3·out_c..4·out_c)  = rcl6      = round_half_away(6/os·2^22)        // relu6 上限（q22，int8 域）
```

公式与舍入（away-from-zero）对齐 `cnn_rtl/tools/ref_int8.py` 的 `quantize_params`
（随机回归 bit-exact 验证一致）。RTL 侧 `S_RD_SCALE` 按此布局边读边写 core requant 数组，
顺序即 mult/bias_mul/shift/rcl6。**乘后域语义**：硬件把 bias_mul/rcl6 左移 8 位对齐
q30 乘后域（`v = acc·mult + bias_mul<<8`；relu6 钳 `v ≤ rcl6<<8`），
与 float 公式 `round(acc·ws·is/os + bias/os)` 一致——黑盒实测 47 层主干位匹配 100%。

> 黑盒时代为每通道 2 个 float（`scale[i]=ws`、`scale[out_c+i]=bias/os`），已被定点化取代，
> 仅作历史参考。

## 7. 量化/数值语义

**2026 定点化后**（软件预转，RTL 无浮点），单层计算（int8 输入×int8 权重 → int8 输出）：

```
acc_int32 = Σ (input_int8 × weight_int8)            // 64 MAC/周期，kernel=3×3
                                                    // 2026-08-10 修复：乘法器 b 索引交换
                                                    // （w_q[mac_m_i][mac_lane_i]）——原
                                                    // w_q[mac_lane_i][mac_m_i] 使输入/输出
                                                    // 通道索引交叉（acc[lane]=Σ_m 输入[m]·W[输入 lane][输出 m]），
                                                    // 上板实测与黑盒逐层不符；修复后 = 正确卷积
                                                    // acc[o]=Σ_i 输入[i]·W[输入 i][输出 o]（黑盒 100% 匹配）
v_rq64    = acc_int32 × mult + (bias_mul << 8)      // 乘后域：mult = round((ws·is/os)·2^30)，
                                                    // bias_mul = round(b/os·2^22)（q22 左移 8 对齐）
act（可选，乘后施加）：
  relu        v = max(v, 0)
  relu6       v = max(v, 0)                          // 黑盒实测无 min(6)（BLACKBOX_NUMERICS.md）；
                                                      // 2026-08-10 移除 rcl6 钳位（原 min(v, rcl6<<8)）——
                                                      // box 头输入直接来自 relu6_1/relu6_3，钳位改变
                                                      // 检测头输入 → 上板几百框误检
out_int8  = ((v + 2^(shift-1)) >> shift) & 0xFF     // shift=30，round-half-up + 8 位截断（wrap，黑盒语义）
                                                     // 2026-08-10 追加：移位后负值 -1（黑盒实测"floor 除法特性"），
                                                     // box 头负 logits 对齐黑盒（conv2d_69-76 黑盒 log 验证 95.8-100%）
```

- `ws`/`b` 为模型 per-channel weight_scale/bias，`is`/`os` 为 input/output scale；
  mult/bias_mul/shift/rcl6 由 `conv_op.cc` 预转（§6.5），公式与舍入
  （away-from-zero）对齐 `cnn_rtl/tools/ref_int8.py`（随机回归 bit-exact 一致）。
- RTL 实现（2026-08 乘后域重排后为 10 拍单操作流水，每拍仅加法/移位/比较之一，
  150 MHz 收敛）：`S_REQ_MUL`（参数打拍）→ `S_REQ_MULB`（acc 打拍）→
  `S_REQ_MUL2`（4×16×16 DSP 部分积）→ `S_REQ_MUL3`（两组中间和）→
  `S_REQ_MUL4`（中间和相加）→ `S_REQ_MULC`（+bias_mul<<8，64-bit 加法）→
  `S_REQ_MULC2`（保留空拍；relu6 rcl6 比较 2026-08-10 已随钳位移除）→ `S_REQ_ACT`（relu mux）→
  `S_REQ_OUT`（round 桶形移位）→ `S_REQ_ROUND2`（round 加法）→
  `S_REQ_OUT2`（算术右移）→ `S_REQ_OUT3`（8 位截断 wrap `y = r & 0xFF` + 输出）。
  数值语义与 float 公式（CPU cvt_kernel 同源）完全一致（tb 按事件对拍）。
  **输出为 8 位截断（wrap）而非饱和**：黑盒实测（BLACKBOX_NUMERICS.md）为
  `y = r & 0xFF`；饱和会把 box 头（act=0）超界 logits 全部钳成 ±127，抹平
  softmax 区分度 → 上板实测几百个高 score 误检框（2026-08-09 修复，`cnn_core_v2.v`
  S_REQ_OUT3 饱和改截断）。relu 层输出值域 [0,127] 不受影响。

## 8. 执行时序与分块

来源：`intelfpga_subgraph`（`intelfpga.cc:601`）+ `conv_op.cc`。

软件为每个 conv 节点构建 `DeviceGraphNode`，运行时**逐节点**（拓扑序，`node->next_`）：

```
1. 若节点是图输入：InputRearrange 写输入到 DDRIN+input_offset*8
2. 计算分块参数：
   output_row_tile      = min(OUTPUT_BUFF_SIZE / output_w, output_h)   // OUTPUT_BUFF_SIZE=150*5
   input_row_tile       = (output_row_tile-1)*stride + dilation*(kernel-1) + 1   // stride=2 → 11
   output_channel_block_num = ceil(output_c / 8)
   output_row_block_num     = ceil(output_h / output_row_tile)
3. memcpy param 块 → PARAM；memcpy scale（`4*output_c` int32 定点参数）→ SCALE
4. 若上一节点有未拷贝输出：先 memcpy 回 Paddle 张量
5. 写 START → 轮询完成
6. 若节点是图输出：OutputRearrange 读回并写 Paddle 张量
```

- 一层一次 START；硬件按 `output_row_block_num`、`output_channel_block_num`
  自行循环行块/通道块。
- 输出缓冲容量 `OUTPUT_BUFF_SIZE = MAX_OUTPUT_W * OUTPUT_ROW_TILE = 150*5`，
  行块大小受此约束（`intelfpga.h:78`）。

### 8.1 分块公式实测验证（2026-08 设备实测 `[FPGA-DUMP] run` 行）

47 层全部输出尺寸的分块参数与公式**逐层复算一致**：

| 输出尺寸 | row_tile | in_row_tile（k=3,s=1 / k=3,s=2 / k=1,s=1） | row_block |
|---|---|---|---|
| 150×150 | 5 | 7 / 11 / 5 | 30 |
| 75×75 | 10 | 12 / 21 / 10 | 8 |
| 38×38 | 19 | 21 / 39 / 19 | 2 |
| 19×19 | 19（受 output_h 限） | 21 / 39 / 19 | 1 |
| 10×10 | 10 | 12 / 21 / 10 | 1 |
| 5×5 | 5 | — / 11 / 5 | 1 |
| 3×3 | 3 | — / 7 / 3 | 1 |
| 2×2 | 2 | — / 5 / 2 | 1 |
| 1×1 | 1 | — / 3 / 1 | 1 |

`chn_block = ceil(out_c/8)` 实测：32→4、64→8、128→16、256→32、512→64、1024→128、16→2、20→3、24→3、30→4。
完整 47 层参数见 `cnn_rtl/model_profile.md`。

## 9. 微架构线索与性能预期

以下为从参数/布局反推的**结构线索**（非承诺，复现可据此设计）：

| 线索 | 依据 |
|---|---|
| 64 MAC/周期（8 in-ch × 8 out-ch） | dw 权重 8×8 对角化布局 |
| 输入每周期 8 像素（64-bit） | `INPUT_RAM_DATA_WIDTH=64` |
| 行块流水：5 输出行 × ≤150 宽 | `OUTPUT_ROW_TILE=5`、`OUTPUT_MAX_W=150`、输出缓冲 150×5 |
| 输入行缓冲 11×302 | `INPUT_ROW_TILE=11`、`IMAGE_MAX_W=302` |
| 独立时钟域 `clk_cnn` | QSys `clk_cnn.clk` 驱动 cnn_top + mm_bridge_sdram0 |

复现版只需功能等价、吞吐量级相近；逐周期时序不要求一致（软件为同步轮询模型，
对时序无依赖）。

## 10. 复现与验收方法

1. **接口冻结**：新 IP 保持 `cnn_top` 名称、§2/§3 的接口与参数不变，
   `soc_system.qsys`、cmadrv、intelfpga.cc 全部零改动。
2. **RTL 模块划分建议**：
   - Avalon 从接口 + 寄存器堆（START/DDRIN/DDRW/DDROUT/PARAM/SCALE）
   - param 解析器（param_read_avalon 突发读指令块）
   - 行块 DMA/流水控制（load 读、output 写、行块/通道块循环）
   - MAC 阵列（8×8×3×3）与累加
   - 量化器（scale/bias、relu/relu6/leakyrelu、舍入饱和）
3. **验证**：tb 位精确对拍（`cnn_rtl/verification/`，小通道参数覆盖）＋上板 `test.sh`
   全链路运行、检测结果正确性比对（黑盒 qxp 已删除，不再有黑盒对拍手段）。
4. **性能对标**：`intelfpga_subgraph` 打印 `fpga_time`，与旧 IP 对比量级。

## 11. 未决问题与风险

| # | 问题 | 影响 | 确认手段 |
|---|---|---|---|
| 1 | ~~int32 累加的舍入/饱和方式、relu6 对 `SCALE_6` 的具体用法~~ | **已解决**：软件定点预转（§6.5），舍入 away-from-zero 与 `ref_int8.quantize_params` 500 轮随机验证一致；RTL 无浮点 | `ref_int8.py` + 上板对拍 |
| 2 | 内部流水级数/频率未知 | 仅性能差异，不影响功能 | 复现后调 `clk_cnn` 与阵列规模 |
| 3 | ~~`param` 结构体 offset 字段（input/weight/output）硬件是否读取~~ | **已解决**：2026-08 起 FPGA 解析 `param[0/1/3]` 并加入 DDR 基址（`897a40f`，此前忽略导致第 1 层起读写错位）；尾部 `weight_size` 等仍不读，无影响 | `conv_op.cc` + 上板对拍 |
| 4 | dilation>1、stride=3 等边界参数是否被硬件支持 | SSD 模型只用 dilation=1/stride∈{1,2}，首版可裁剪 | 对拍时覆盖全部 47 层参数组合 |
| 5 | ~~`output_row_tile` 由软件按缓冲容量计算，硬件假定与参数一致~~ | **已解决**：§8.1 对 47 层实测逐层复算一致 | 设备 `[FPGA-DUMP]` 实测 |

## 附：证据来源索引

| 信息 | 位置 |
|---|---|
| IP 接口与参数 | `C5TB/ip/cnn_top_hw.tcl` |
| QXP 边界端口（黑盒事实） | `C5TB/db/C5TB_top.soc_system.cnn_top.qxp.txt` |
| QSys 连接与基地址 | `C5TB/soc_system.qsys`（cnn_top_0 各 connection） |
| IP 参数实际取值 | `C5TB/soc_system/soc_system.xml` |
| 寄存器协议 | `ssd_detection/include/common.h:59-87` |
| 启动/轮询/DMA/重排 | `ssd_detection/intelfpga.cc`（`fpga_init`/`start_fpga`/`InputRearrange`/`OutputRearrange`/`intelfpga_subgraph`） |
| 指令结构体 | `ssd_detection/include/intelfpga.h:98-130` |
| 权重重排布局 | `ssd_detection/intelfpga.cc:525-599` |
| 参数/scale 填充 | `ssd_detection/paddle_lite/lite/kernels/intel_fpga/bridges/conv_op.cc:96-175` |
| **47 层实测参数/行块/偏移画像** | `C5TB/cnn_rtl/model_profile.md`（设备 `[FPGA-DUMP]` 输出） |
