# cnn_top 规格书（接口 / 协议 / 数值语义）

> 导航：上一级 [README.md](README.md) · 相关 [ssd_detection/](../ssd_detection/README.md)、[intelfpga.cc](../ssd_detection/intelfpga.cc)、[intelfpga.h](../ssd_detection/include/intelfpga.h)

本文档描述 `cnn_top` 加速器的**现行行为契约**：顶层接口（§2）、IP 参数（§3）、
寄存器协议（§4）、指令格式（§5）、数据布局（§6）、量化/数值语义（§7）、
执行时序与分块（§8）。RTL 实现（`cnn_rtl/src/` 下五个 `.v`，见 `cnn_rtl/README.md`）与
软件栈（`intelfpga.cc` / `conv_op.cc`）均以本文档为准；47 层实测参数见附 A，
数值语义实测方法与 CPU 同源算法见附 B。

> 历史沿革：本文档原为黑盒 qxp（2026-08 已删除）的复现规格依据；
> 复现已完成并上板验证，"黑盒/复现参考"表述已随之移除。

## 1. 背景与现状

- `cnn_top` 是 QSys 自定义 IP（Component Editor 18.1，v2.0，`ip/cnn_top_hw.tcl`）。
  原黑盒网表 `ip/cnn_top.qxp`（8.3MB，无 RTL 源码）已于 **2026-08 由开源复现 RTL
  （`cnn_rtl/src/` 下 `cnn_top.v`/`cnn_top_core.v`/`cnn_core_v2.v`/`mac8x8_dsp.v`/`mac8x8_lut.v`
  五个 `.v`，hw.tcl fileset 唯一引用）替代并删除**。
- 软件侧协议：`ssd_detection/intelfpga.cc`（寄存器读写、DMA、重排）、
  `ssd_detection/include/intelfpga.h`（指令结构）、
  `ssd_detection/paddle_lite/lite/kernels/intel_fpga/conv_op.cc`（参数/scale 填充）。
- 现状：RTL 已上板全链路跑通，检测正确性已确认（2026-08-10 修复后 47 层黑盒 log
  逐层对比：主干 100%、box 头 97-100%）；**性能较黑盒慢**（`fpga_time` 约 2670 ms vs
  黑盒 475 ms，瓶颈与优化方向见 `cnn_rtl/README.md` 遗留节）。
- 编译注意：QSys Generate 会把 `cnn_rtl/src/` 的 HDL 复制到
  `soc_system/synthesis/submodules/` 后才参与编译——更新 `cnn_rtl/src/` 下 `.v` 后必须重新
  Generate QSys（或手动覆盖 submodules/ 同名文件），否则综合仍引用旧副本。

## 2. 顶层接口

来源：`ip/cnn_top_hw.tcl`（接口定义）。原 `db/C5TB_top.soc_system.cnn_top.qxp.txt`
（黑盒边界端口）已随黑盒删除——`db/` 为 gitignore 构建产物，仓库中不存在。

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
> FPGA 仅执行卷积本体。黑盒时代的 `REORG_M_AXI_*` 参数已随 RTL 化删除
> （当前 `hw.tcl` 无 reorg 相关参数）。

## 3. IP 参数表

来源：`ip/cnn_top_hw.tcl`（当前 RTL 版 IP 的全部参数）。

| 参数 | 值 | 语义 |
|---|---|---|
| `CFG_M_AXI_ADDR_WIDTH` / `CFG_M_AXI_DATA_WIDTH` | 32 / 32 | param 读接口地址 / 数据宽度 |
| `LOAD_M_AXI_ADDR_WIDTH` / `LOAD_M_AXI_DATA_WIDTH` | 32 / 64 | 输入读接口地址宽度 / 数据宽度（8×int8） |
| `SCALE_M_AXI_ADDR_WIDTH` / `SCALE_M_AXI_DATA_WIDTH` | 32 / 32 | scale 读接口地址 / 数据宽度 |
| `OUTPUT_M_AXI_ADDR_WIDTH` / `OUTPUT_M_AXI_DATA_WIDTH` | 32 / 64 | 输出写接口地址宽度 / 数据宽度 |

> **黑盒时代参数（已随 RTL 化删除，仅历史参考）**：`INPUT_CHANNEL_TILE=8`、
> `OUTPUT_CHANNEL_TILE=8`、`INPUT_ROW_TILE=11`、`OUTPUT_ROW_TILE=5`、
> `IMAGE_MAX_W=302`、`OUTPUT_MAX_W=150`、`INPUT_WIDTH/WEIGHT_WIDTH=8`、
> `OUTPUT_WIDTH=32`、`KERNAL_SIZE=9`、`KERNEL_SIZE_WIDTH=4`、
> `INPUT_PAD_WIDTH/OUTPUT_PAD_WIDTH=2`、`CNN_TYPE_WIDTH=3`、`CNN_STRIDE_WIDTH=2`、
> `CNN_RELU_WIDTH=3`、`IN_C_WIDTH=11`/`IN_H_WIDTH=9`/`IN_W_WIDTH=9`、
> `OUTPUT_C_WIDTH=11`/`OUTPUT_H_WIDTH=9`/`OUTPUT_W_WIDTH=9`、`OUTPUT_ADDR_WIDTH=10`、
> `INPUT_RAM_DATA_WIDTH=64`、`CFG_PARAM_WIDTH=32`、`SCALE_6=1086324736`
> （34-bit，位模式 = float 6.0，relu6 上限常量）、`REORGANIZE_BUFF_DEEP`/`REORG_RAM_*`
> （reorg 残留）。RTL 化后这些尺寸/常量参数全部删除——各层尺寸由运行时 param 块（§5）
> 驱动，relu6 上限由软件预转 `rcl6` 取代（§6.5）。原 `soc_system.xml`（QSys Generate
> 产物，gitignore，仓库不含）记录黑盒时代 DE10-Nano 实例参数取值。

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

**启动时序**（`start_fpga`，`intelfpga.cc:223`）：

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
ARM 端实现：`InputRearrange`（`intelfpga.cc:328`，`tran_8` NEON 转置）。
字节地址 = `DDRIN + input_offset*8`。

### 6.2 权重 conv2d（软件写，硬件读）

原始 `[Co, Ci, 3, 3]` → **`[ceil(Co/8), ceil(Ci/8), 8, 9, 8]`**，
即按输出通道块 × 输入通道块组织，每块内是 8 输入通道 × 9 权重 × 8 输出通道的
"外积 tile"（`conv2d_weight_reorganize`，`intelfpga.cc:467`）。
字节地址 = `DDRW + weight_offset*8`。

### 6.3 权重 depthwise（软件写，硬件读）

原始 `[Co, 1, 3, 3]` → **`[ceil(Co/8), 8, 9, 8]`** 稀疏对角化：
仅当 输入通道下标 == 输出通道下标（`ti==m`）时填入权重，其余 0
（`dw_conv2d_weight_reorganize`，`intelfpga.cc:510`）。
**该布局直接表明 MAC 阵列 = 8 in-ch × 8 out-ch（64 MAC/周期）。**

### 6.4 输出（硬件写，软件读）

硬件写 **NHWC8**：`[ceil(Co/8), H, W, 8]`；
字节地址 = `DDROUT + output_offset*8`。
软件 `OutputRearrange`（`intelfpga.cc:339`）用 `tran_8` 转回 NCHW 后再拷贝到
Paddle 张量（`global_mem_cfg` 拷贝逻辑，`intelfpga.cc:656`）。

### 6.5 scale/bias（软件写，硬件读）

**2026 定点化后**：每输出通道 4 个 int32，共 `4*output_c`，从 SCALE 基址连续存放
（`conv_op.cc:122-162` 生成，`intelfpga.cc:583` 整块 memcpy）：

```
scale[0..out_c)          = mult      = round_half_away((ws·is/os)·2^30)  // 乘数（q30）
scale[out_c..2·out_c)    = bias_mul  = round_half_away(b/os·2^22)        // 乘后域 bias（q22，int32 安全）
scale[2·out_c..3·out_c)  = shift     = 30                                // 右移位数
scale[3·out_c..4·out_c)  = rcl6      = round_half_away(6/os·2^22)        // relu6 上限（q22，int8 域）
```

公式与舍入（away-from-zero）对齐 `cnn_rtl/tools/ref_int8.py` 的 `quantize_params`
（随机回归 bit-exact 验证一致）。RTL 侧 `S_RD_SCALE` 按此布局边读边写 core requant 数组，
顺序即 mult/bias_mul/shift/rcl6。**乘后域语义**：硬件把 bias_mul 左移 8 位对齐
q30 乘后域（`v = acc·mult + bias_mul<<8`），与 float 公式
`round(acc·ws·is/os + bias/os)` 一致——黑盒实测 47 层主干位匹配 100%。
**rcl6 仍按布局写入 core，但 2026-08-10 起已不使用**（relu6 钳位随
`S_REQ_MULC2` 状态一并移除，见 §7；`conv_op.cc` 保留写入以维持布局不变）。

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
  relu6       v = max(v, 0)                          // 黑盒实测无 min(6)（附 B.1）；
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
- RTL 实现（2026-08 乘后域重排后为 12 拍单操作流水，每拍仅加法/移位/比较之一；
  当前 PLL `clk_cnn` = 100 MHz，见 `cnn_rtl/README.md`"最终时钟状态"）：
  `S_REQ_ADDR`（采样 acc）→ `S_REQ_MUL`（参数打拍）→ `S_REQ_MULB`（acc 打拍）→
  `S_REQ_MUL2`（4×16×16 DSP 部分积）→ `S_REQ_MUL3`（两组中间和）→
  `S_REQ_MUL4`（中间和相加）→ `S_REQ_MULC`（+bias_mul<<8，64-bit 加法）→
  `S_REQ_ACT`（relu mux；原 `S_REQ_MULC2` 已随 2026-08-10 rcl6 钳位移除）→
  `S_REQ_OUT`（round 桶形移位）→ `S_REQ_ROUND2`（round 加法）→
  `S_REQ_OUT2`（算术右移）→ `S_REQ_OUT3`（8 位截断 wrap `y = r & 0xFF` + 输出）。
  数值语义与 float 公式（CPU cvt_kernel 同源）完全一致（tb 按事件对拍）。
  **输出为 8 位截断（wrap）而非饱和**：黑盒实测（附 B.1）为
  `y = r & 0xFF`；饱和会把 box 头（act=0）超界 logits 全部钳成 ±127，抹平
  softmax 区分度 → 上板实测几百个高 score 误检框（2026-08-09 修复，`cnn_core_v2.v`
  S_REQ_OUT3 饱和改截断）。relu 层输出值域 [0,127] 不受影响。

## 8. 执行时序与分块

来源：`intelfpga_subgraph`（`intelfpga.cc:543`）+ `conv_op.cc`。

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
完整 47 层参数见附 A。

## 9. 微架构事实与性能预期

复现核已实现的结构（与黑盒参数反推一致）：

- **64 MAC/周期**（8 in-ch × 8 out-ch；`mac8x8_dsp.v` lane 0-3 + `mac8x8_lut.v` lane 4-7）
- 输入每周期 8 像素（64-bit）；行块流水 5 输出行 × ≤150 宽（黑盒 `OUTPUT_ROW_TILE=5`、
  `IMAGE_MAX_W=302` 等尺寸参数已 RTL 化，见 §3 历史注记）
- 独立时钟域 `clk_cnn`，当前 **100 MHz**（`cnn_rtl/README.md`"最终时钟状态"）

性能现状：`fpga_time` 约 **2670 ms/帧**（黑盒 475 ms）；优化方向见
`cnn_rtl/README.md` 遗留节（装载/MAC 未重叠、pr/sr 单笔在途、requant 12 拍流水）。
软件为同步轮询模型，对逐周期时序无依赖。

## 10. 验证与回归方法

1. **tb 位精确对拍**：`cnn_rtl/verification/`（`run_cnn_core_v2.sh` 24 层、
   `run_cnn_top.sh` 6 层；对拍基准 `tools/ref_cnn_top.py`：80 随机层 + 真实 47 层 ALL PASS）。
2. **上板验证**：`test.sh` 全链路运行 + 检测结果比对；数值语义用 `DUMP_LAYER_OUT=1`
   dump 逐层对比（方法见附 B.1）。
3. **性能对标**：`intelfpga_subgraph` 打印 `fpga_time`，与黑盒 475 ms 基线对比。

## 11. 遗留风险

| # | 风险 | 状态 |
|---|---|---|
| 1 | dilation>1、stride=3 等边界参数未验证（SSD 模型只用 dilation=1/stride∈{1,2}） | RTL 未覆盖，新增层型需先验证 |
| 2 | 性能待优化：`fpga_time` 约 2670 ms（黑盒 475 ms） | 见 `cnn_rtl/README.md` 遗留节 |
| 3 | tb 覆盖仅小通道参数（in_c/out_c ≤ 32），大通道（最大 1024）靠上板验证 | 见 `cnn_rtl/README.md` 验证覆盖边界 |

> 原"未决问题"表中的 int32 舍入方式、`param` offset 读取、`output_row_tile` 一致性等
> 已随复现完成解决（历史见 git），不再列出。

## 附 A：47 层实测参数画像（设备 `[FPGA-DUMP]` 实测）

> 来源：FPGA 板（172.16.68.110）`bash test.sh` 的 `[FPGA-DUMP]` 输出（paddle_lite + libvnna
> 带日志版本）。本附录是 47 层实测参数的**最终规格依据**——算子集合、维度分布、行块切分、
> 数据布局全部由实测确认（原 `cnn_rtl/model_profile.md`，2026-08 合并至此）。

### A.1 全图结构（360 ops）

```
[0-2]   feed ×3（image / im_shape / scale_factor）
[3]     calib fp32→int8（image → image/precision_trans0）
[4]     subgraph —— 47 个 conv/dw_conv 全部进 FPGA（下方 A.2）
[5-40]  12 组 calib+transpose2+reshape2（SSD 12 个输出头：conv2d_69~80）
[41-58] 6 组 calib+prior_box+reshape2（6 个 feature map 的 prior box）
[59-352] SSD 后处理：slice / elementwise_sub|mul|add|div / scale / exp / stack（box decode + variance）
[353-354] concat ×2
[355]   softmax
[356]   transpose2
[357]   multiclass_nms3 → save_infer_model/scale_{0,1}.tmp_1
[358-359] fetch ×2
```

### A.2 进 FPGA 的 47 层（全部 bridge → FPGA，实测参数）

| # | type | in(C×H) | out(C×H) | k | s | pad | act | 说明 |
|---|------|---------|----------|---|---|-----|-----|------|
| 0  | CONV | 3×300 | 32×150 | 3 | 2 | 1 | relu | 首层 |
| 1  | DW   | 32×150 | 32×150 | 3 | 1 | 1 | relu | |
| 2  | CONV | 32×150 | 64×150 | 1 | 1 | 0 | relu | 1×1 |
| 3  | DW   | 64×150 | 64×75  | 3 | 2 | 1 | relu | |
| 4  | CONV | 64×75  | 128×75 | 1 | 1 | 0 | relu | |
| 5  | DW   | 128×75 | 128×75 | 3 | 1 | 1 | relu | |
| 6  | CONV | 128×75 | 128×75 | 1 | 1 | 0 | relu | |
| 7  | DW   | 128×75 | 128×38 | 3 | 2 | 1 | relu | |
| 8  | CONV | 128×38 | 256×38 | 1 | 1 | 0 | relu | |
| 9  | DW   | 256×38 | 256×38 | 3 | 1 | 1 | relu | |
| 10 | CONV | 256×38 | 256×38 | 1 | 1 | 0 | relu | |
| 11 | DW   | 256×38 | 256×19 | 3 | 2 | 1 | relu | |
| 12 | CONV | 256×19 | 512×19 | 1 | 1 | 0 | relu | |
| 13 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 14 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | |
| 15 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 16 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | |
| 17 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 18 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | |
| 19 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 20 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | |
| 21 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 22 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | 主干结束 |
| 23 | CONV | 512×19 | 16×19  | 1 | 1 | 0 | none | conv2d_69 分类头 |
| 24 | CONV | 512×19 | 20×19  | 1 | 1 | 0 | none | conv2d_70 位置头 |
| 25 | DW   | 512×19 | 512×10 | 3 | 2 | 1 | relu | |
| 26 | CONV | 512×10 | 1024×10| 1 | 1 | 0 | relu | |
| 27 | DW   | 1024×10| 1024×10| 3 | 1 | 1 | relu | |
| 28 | CONV | 1024×10| 1024×10| 1 | 1 | 0 | relu | |
| 29 | CONV | 1024×10| 24×10  | 1 | 1 | 0 | none | conv2d_71 |
| 30 | CONV | 1024×10| 30×10  | 1 | 1 | 0 | none | conv2d_72 |
| 31 | CONV | 1024×10| 256×10 | 1 | 1 | 0 | relu6 | |
| 32 | CONV | 256×10 | 512×5  | 3 | 2 | 1 | relu6 | |
| 33 | CONV | 512×5  | 24×5   | 1 | 1 | 0 | none | conv2d_73 |
| 34 | CONV | 512×5  | 30×5   | 1 | 1 | 0 | none | conv2d_74 |
| 35 | CONV | 512×5  | 128×5  | 1 | 1 | 0 | relu6 | |
| 36 | CONV | 128×5  | 256×3  | 3 | 2 | 1 | relu6 | |
| 37 | CONV | 256×3  | 24×3   | 1 | 1 | 0 | none | conv2d_75 |
| 38 | CONV | 256×3  | 30×3   | 1 | 1 | 0 | none | conv2d_76 |
| 39 | CONV | 256×3  | 128×3  | 1 | 1 | 0 | relu6 | |
| 40 | CONV | 128×3  | 256×2  | 3 | 2 | 1 | relu6 | |
| 41 | CONV | 256×2  | 24×2   | 1 | 1 | 0 | none | conv2d_77 |
| 42 | CONV | 256×2  | 30×2   | 1 | 1 | 0 | none | conv2d_78 |
| 43 | CONV | 256×2  | 64×2   | 1 | 1 | 0 | relu6 | |
| 44 | CONV | 64×2   | 128×1  | 3 | 2 | 1 | relu6 | |
| 45 | CONV | 128×1  | 16×1   | 1 | 1 | 0 | none | conv2d_79 |
| 46 | CONV | 128×1  | 20×1   | 1 | 1 | 0 | none | conv2d_80 |

**实测结论**：
- 算子集合：仅 `CONV(1)` + `DW_CONV(4)`；**无 leaky、无 dilation≠1、无 k≠1/3、无 pool**
- act 实际值：0（none，12 个分类/位置头）、1（relu，主干 29 层）、2（relu6，SSD head 6 层）——leaky(3) 从未使用
- 非 8 倍数输出通道：16/20/24/30 → `chn_block = up_round(out_c,8)` 实测为 2/3/3/4

### A.3 行块切分（intelfpga_subgraph 运行时计算，实测验证）

```
output_row_tile = min(OUTPUT_BUFF_SIZE / output_w, output_h)   # OUTPUT_BUFF_SIZE = 150*5 = 750
input_row_tile  = (output_row_tile-1)*stride + dilation*(kernel-1) + 1
output_channel_block_num = up_round(output_c, 8)
output_row_block_num     = up_round(output_h, output_row_tile)
```

> 各输出尺寸的 row_tile / in_row_tile / row_block 实测值见 §8.1 表。

### A.4 数据布局（param 偏移为 64-bit 字，字节 = word×8，全部实测验证）

- 输入/输出：NHWC8 `[C/8, H, W, 8]`，字节数 = `ceil(C/8)*H*W*8`
  - 例：层 0 输入 3×300×300 = 720000 B；输出 32×150×150 = 720000 B（out_off 90000 word = 720000 B ✓）
  - 层间输出/输入缓冲**线性复用**（层 i out_off = 层 i-1 out_off + 层 i-1 输出字节/8）
- 权重：`[Co/8, Ci/8, 8, K², 8]`（K=1 或 3），字节数 = `ceil(Co/8)*ceil(Ci/8)*8*K²*8`
  - 例：层 0（3→32,k3）= 1*4*8*9*8 = 2304 B ✓；层 2（32→64,k1）= 8*4*8*1*8 = 2048 B ✓
  - 总权重 ≈ 5.9 MB（word 偏移 0 ~ 735104）
- scale 块：每输出通道 2 个 float（`ws` + `bias/os`），共 `2*out_c` word

### A.5 性能基线（黑盒实测，一帧 300×300）

```
input_organize_time: 64.3 ms（ARM 重排 NCHW→NHWC8）
fpga_time:          475.1 ms（47 层，平均 ~10.1 ms/层）
output_organize_time: 23.8 ms
```

### A.6 对硬件实现的约束

1. **硬件必须支持 k=1 与 k=3**（kernel 参数运行时给定，1×1 占 29/47 层）
2. **行块循环是硬件行为**：row_block（行方向）+ chn_block（通道方向 8 一组）双层循环，由参数驱动
3. act 需支持 0/1/2（relu6 clamp 边界即 `6·os/(ws·is)` 的 raw 域量化值）
4. depthwise = 对角权重（`ti==m` 才非零），复用同一数据通路
5. 无跨层融合，逐层启动；输入输出/权重偏移全由 param 提供

## 附 B：数值语义实测方法与 CPU 同源算法

### B.1 黑盒实测方法（2026-08-08，基于 blackbox_layers2.log）

- main 分支副本加 dump 代码（`DUMP_LAYER_OUT=1` 环境变量开启）：`[LIN]/[LIND]`（每层输入区）、
  `[LOUT]/[LOUTD]`（每层输出区）、`[LSCALE]`（scale 数组 float[2×out_c]）、`[LW]/[LWD]`（重排后权重）。
- 设备上板跑 `DUMP_LAYER_OUT=1 ./ssd_detection config.txt data` 得到 `blackbox_layers2.log`
  （111MB，47 层 × 2 次推理，第 2 段 = 4.jpg）。
- 本地用 Python 精确重建每层 `acc = Σ(int8输入 × int8权重)`（int32），逐层逐通道验证候选公式
  （每通道 12 采样点）。

实测要点（易错）：
- warm up 输入是未初始化 Mat（全 0）→ 输入区全 `0xbd`（-67）；必须 dump 4.jpg（第 2 次推理）的输入
- 权重布局是 `[cb_out][cb_in][mi][k][mo]`（**mi 主序**、k 次、mo 字节）
- 对比必须按**位模式**（`pred & 0xFF == y`），不能按 int8 数值对比
- SSD 检测头（conv2d_69-76）是多分支并行，输入按 `in_off` 查（不是前层输出）

> 数值语义结论见 §7（wrap 截断、负值 -1、无 min6 均为本实测确定）；修复过程见 `cnn_rtl/README.md`。

### B.2 CPU 同源算法（Paddle-Lite ARM int8 conv，源码在 `cnn_rtl/cpu_ref/paddle_lite_arm/`）

FPGA 算子 → CPU 同源 kernel（`kernels/arm/conv_compute.cc:116-137` 选择）：

| FPGA 层类型（实测） | CPU 同源 kernel | 关键文件 |
|---|---|---|
| 3×3 s=2（首层 conv） | `DirectConv` → `conv_3x3s2_direct_int8` | `backends/arm/math/conv3x3s2_direct_int8.cc` |
| 1×1 s=1（主干 k=1 conv） | `GemmLikeConv` → `conv1x1s1_gemm_int8` | `backends/arm/math/conv_impl.cc:631`、`gemm_prepacked_int8.*` |
| 3×3 s=1 | `WinogradConv`/`GemmLikeConv`（im2col+gemm） | `conv3x3_winograd_int8.cc`、`conv_impl.cc` |
| DW 3×3 s=1（depthwise） | `DepthwiseConv` → `conv3x3s1_depthwise_int8` | `backends/arm/math/conv3x3s1_depthwise_int8.cc` |

requant 出口（全共用，`backends/arm/math/conv_block_utils.h:3723` `cvt_kernel<int8_t>`）：

```cpp
// act==1（relu）:   tmp = saturate_cast<int8_t>(round(LITEMAX(din*scale + bias, 0)));
// act==0（无）:     tmp = saturate_cast<int8_t>(round(din*scale + bias));
// act==2（relu6）:  tmp = saturate_cast<int8_t>(round(LITEMIN(LITEMAX(din*scale+bias, 0), alpha)));
// 统一收尾:         return tmp < -127 ? -127 : tmp;
```

scale/bias 准备（`kernels/arm/conv_gemmlike.cc`）：`ws = ws*input_scale/output_scale`、
`bias: /output_scale`、`alpha = Relu_clipped_coef/output_scale`——即 §6.5 定点预转的 float 原型。

> 注意：CPU 语义为饱和 + relu6 上限（min6）；黑盒实测为 wrap + 无 min6（§7）。复现核最终
> 按**黑盒实测语义**实现（与黑盒逐位一致优先）；CPU 公式仅作清晰可验证的参照。本模型主干
> relu 层（值域 0-91 不触发溢出）CPU 与黑盒 100% 位一致，差异仅 box 头 wrap 区与 relu6 层。

文件清单（`cnn_rtl/cpu_ref/paddle_lite_arm/`）：

```
kernels/arm/conv_compute.cc|.h        kernel 选择（int8 分支）
kernels/arm/conv_gemmlike.cc|.h       1×1/通用 conv（scale 准备 + run）
kernels/arm/conv_direct.cc|.h         3×3 s2 直接卷积
kernels/arm/conv_winograd.cc          3×3 s1 winograd
kernels/arm/conv_depthwise.cc         DW kernel 选择
backends/arm/math/conv_block_utils.h  ★ cvt_kernel（requant 出口）+ 输出写回
backends/arm/math/conv_impl.cc|.h     conv1x1s1_gemm_int8、im2col、gemm 封装
backends/arm/math/conv3x3s2_direct_int8.cc / conv3x3s1_direct_int8.cc / conv3x3_winograd_int8.cc
backends/arm/math/conv3x3s1_depthwise_int8.cc / conv3x3s2_depthwise_int8.cc / conv_depthwise.h
backends/arm/math/gemm_prepacked_int8.cc|.h / packed_sgemm.h / sgemm.h
backends/arm/math/saturate.h          ★ saturate_cast
backends/arm/math/quantize.h
```

## 附 C：证据来源索引

| 信息 | 位置 |
|---|---|
| IP 接口与参数 | `C5TB/ip/cnn_top_hw.tcl` |
| QXP 边界端口（黑盒事实，已失效） | ~~`C5TB/db/C5TB_top.soc_system.cnn_top.qxp.txt`~~（黑盒已删除；`db/` 为 gitignore 构建产物，仓库不存在） |
| QSys 连接与基地址 | `C5TB/soc_system.qsys`（cnn_top_0 各 connection） |
| IP 参数实际取值 | `C5TB/soc_system/soc_system.xml`（QSys Generate 产物，gitignore，仓库不含；当前参数见 §3） |
| 寄存器协议 | `ssd_detection/include/common.h:60-87` |
| 启动/轮询/DMA/重排 | `ssd_detection/intelfpga.cc`（`fpga_init`/`start_fpga`/`InputRearrange`/`OutputRearrange`/`intelfpga_subgraph`） |
| 指令结构体 | `ssd_detection/include/intelfpga.h:98-130` |
| 权重重排布局 | `ssd_detection/intelfpga.cc:467-542` |
| 参数/scale 填充 | `ssd_detection/paddle_lite/lite/kernels/intel_fpga/conv_op.cc:122-162` |
| **47 层实测参数/行块/偏移画像** | 附 A（设备 `[FPGA-DUMP]` 输出） |
| **黑盒数值语义实测方法** | 附 B.1（`blackbox_layers2.log` 等调试数据在 `_cnn_rtl_debug/`，不在仓库） |
| **CPU 同源算法（参照实现）** | 附 B.2 + `cnn_rtl/cpu_ref/paddle_lite_arm/` |
