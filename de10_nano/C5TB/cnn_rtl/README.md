# cnn_rtl/ — cnn_top 开源复现工作区（基于 DE1-SoC 卷积数据通路）

> 导航：上一级 [README.md](../README.md) · 规格依据 [cnn_top_spec.md](../cnn_top_spec.md)

本目录是 **`cnn_top` 加速器的开源 RTL 复现工作区**。目标：用纯 RTL（替代
`ip/cnn_top.qxp` 黑盒）实现功能等价的 int8 卷积加速器，接口/协议/布局完全对齐
[cnn_top_spec.md](../cnn_top_spec.md)，替换后 QSys 系统与软件栈零改动。

**策略**：不从头写。以开源项目 **DE1-SoC FPGA-Accelerated Video Pipeline** 的
卷积数据通路（`conv_layer.vhd`，Cyclone V SoC 原生、int-only 量化链完整、GHDL 验证
框架齐全）为底子，按规格书改造接口、布局与 depthwise 支持。

## 来源与许可证

- 上游：<https://github.com/AlbianSalihu/de1-soc-fpga-accelerated-video-pipeline>
  （MIT License，Copyright (c) 2026 Albian Salihu，许可证原文见
  `upstream/LICENSE`，**必须保留**）
- `upstream/` 下为**原样引入**的上游文件（不改动，便于对照升级）；改造后的源码
  放在本目录顶层（`src/` 等，规划见下）。
- 上游仅引入 `conv_layer.vhd`（卷积层数据通路）、`docs/`（架构文档）、
  `verification/`（GHDL 验证框架）。`ml/`（AlexNet64Gray 训练/量化流程）不引入——
  与我们的 MobileNet SSD 无关，仅借其"Python 整数模型对拍"方法论。

## 目录结构

```
cnn_rtl/
├── README.md                    # 本文件：工作区说明 + 执行进度 + 调试历史
├── src/                         # 仿真版 RTL（verilator 回归用）
├── tools/                       # numpy 行为模型（ref_cnn_top/ref_int8）+ 向量生成器
├── verification/                # tb + 回归脚本（run_cnn_core_v2.sh）
├── cpu_ref/                     # CPU 同源算子参考（Paddle-Lite ARM int8 conv，说明见 cnn_top_spec.md 附 B.2）
```

## 设计来源（历史）

本分支数值/架构底子源自 Intel intelfpga 开源卷积数据通路
（`conv_layer.vhd`，MIT 协议）：单层流式卷积（行缓冲 3+1 行、int32 累加、
定点乘移位 requant、valid/ready 流控）。2026-08 已完全重写为自研 Verilog
（`src/cnn_core.v`，仿真与综合共用同一份，NHWC8 块序流 + 乘后域定点；
2026-08 由 `cnn_core_v2.v` 重命名，lane requant 参数存储拆分为 `src/requant_store.v` ×8 实例），
上游 VHDL 源码及其 GHDL 验证框架已从分支移除（历史见 git）。

## 差距矩阵（上游 → cnn_top 规格）

| 维度 | 上游 conv_layer.vhd（历史参考） | cnn_top（[spec](../cnn_top_spec.md)） | 改造动作 |
|---|---|---|---|
| 数值符号 | uint8 激活 / int8 权重 / uint8 输出 | **int8 输入/输出** | 数据通路改 signed |
| 量化 | 定点乘移位（uint32 乘 + shift） | per-channel float scale（`scale[2*out_c]`，含 bias/output_scale） | 软件把 float scale 预转定点乘移位（RTL 改动最小），或 RTL 加 float 乘 |
| 维度 | generic 综合期固定 | **运行期每层不同**（param 块） | generic → 运行时寄存器/计数器比较（关键结构改造） |
| 控制 | 流式单层自动重启 | 指令式：DDR 读 `struct parameter` + START 寄存器轮询 | 新增 param 解析器 + 指令状态机 |
| 接口 | 无总线 | Avalon 从（6 寄存器）+ 4 个 Avalon 主（param/load/output/scale） | 新增 Avalon 封装 + DMA |
| 输入布局 | 行→列→通道流 | **NHWC8**（DDR，`[C/8,H,W,8]`） | DMA 侧按 tile 地址读取，行缓冲逻辑复用 |
| 权重 | 流式 slice（G_C_PAR×9） | DDR tile 化（conv: `[Co/8,Ci/8,8,9,8]`；dw: 8×8 对角） | DMA 读 + 权重缓冲布局 |
| depthwise | 无 | 支持（type=4） | 权重对角化 + MAC 复用 |
| 分块 | 整帧 | 行块调度（`output_row_tile=5`/`input_row_tile=11`，输出缓冲 150×5） | 行块循环状态机 |
| relu | relu | relu / **relu6** / leakyrelu | 量化器扩展（relu6 上限 6.0 由 `SCALE_6` 提供） |

## 实际执行进度（Verilog 化，verilator 对拍）

> 早期 VHDL 规划（差距矩阵 + 分阶段路线图）已随 2026-08 完全 Verilog 化而删除；
> 历史见 git。各阶段以 `ref_cnn_top.py` 位精确对拍为闸门。

| 阶段 | 交付 | 验证 |
|---|---|---|
| 1 ✅ | `src/conv_layer_s8.v`（定点卷积数据通路，Verilog，阶段 1 中间产物）——**2026-08 清理删除**（历史见 git） | `verification/sim/run_conv_s8.sh`（iverilog/verilator）全绿 |
| 2 ✅ | `tools/ref_cnn_top.py`（numpy 行为模型，裁判） | 80 随机层 + 真实 47 层 ALL PASS |
| 3 ✅ | `src/cnn_core.v`（参数驱动单层执行器，k=3 子集）——**2026-08 清理删除**（历史见 git） | `verification/sim/run_cnn_core.sh` 24 层 bit-exact |
| 4 ✅ | `src/cnn_core.v`（**NHWC8 块序流**：slice 权重、行块 acc 缓冲、单行块执行供 DMA 调度；2026-08 由 `cnn_core_v2.v` 重命名，requant 参数存储拆为 `requant_store.v` ×8） | `verification/sim/run_cnn_core_v2.sh` 24 层 bit-exact（**均为 k=3**；k=1 层 2026-08 权重布局修复后未重新对拍） |
| 5 ✅ | `cnn_top.v` Avalon 顶层（从接口 + 6 寄存器 + param/scale 解析 + DMA + 行块调度）；requant 定点参数**软件侧预转**（`conv_op.cc` scale 区每通道 4 字：mult/bias_mul(q22)/shift/rcl6(q22)，乘后域，RTL 无浮点） | 端到端 tb：**6/6 层 bit-exact**（单块/多块/多组、38 高长层） |
| 6 ✅ | depthwise（type=4）：权重对角化 + MAC 复用 | v2 回归覆盖 dw 层 |
| 7 ✅ | 打包 QSys IP 上板替换（黑盒 `cnn_top.qxp` 已删除） | 上板 `test.sh` 全链路运行；**检测正确性验证中**（见下节调试历史） |

> **验证覆盖边界（重要）**：tb 各层向量均为**小通道参数**（`in_c`/`out_c` ≤ 32，见
> `verification/v2/layer_*/cfg.hex`），**大通道（in_c/out_c 最大 1024）未经仿真覆盖**，
> 其正确性依赖上板验证与下述 2026-08 容量修复。
>
> **2026-08 清理**：阶段 1/3 的中间产物（`conv_layer_s8.v`/`cnn_core.v` 及其 tb、
> run 脚本、向量生成器）已被 v2/top 取代，随 `STAGE1_plan.md` 一并删除，历史见 git；
> `tools/ref_int8.py` 保留（`conv_op.cc` 定点公式的对齐依据）。

**阶段 5 交付（Quartus 接入）**：`cnn_top.v`（QSys 适配层，端口与 `ip/cnn_top_hw.tcl` 一致）+ `cnn_top_core.v`（RTL 本体，tb 直接对拍）+ `cnn_core.v`（2026-08 由 `cnn_core_v2.v` 重命名，requant 参数存储拆为 `requant_store.v`）+ `mac8x8_dsp.v`/`mac8x8_lut.v`（8×8 乘法器，lane 0-3 DSP / 4-7 LUT）；`ip/cnn_top_hw.tcl` 已把 `cnn_top.qxp` 黑盒引用改为这 6 个 `.v`（fileset 指向 `../cnn_rtl/src/`，RTL 仅此一份）。接入步骤（Windows Quartus）：
1. RTL 源随工程提交（`cnn_rtl/src/`）；Platform Designer 打开 `soc_system.qsys` → Generate HDL（顶层会例化 `cnn_top` = 新 wrapper）
   **⚠ 编译实际引用 `soc_system/synthesis/submodules/` 下的副本**（`soc_system.qip:7110-7114` 指向 submodules 而非 `cnn_rtl/src/`）——改了 `cnn_rtl/src/` 的 `.v` 后**必须**重新 Generate QSys（或手动覆盖 submodules/ 同名文件），否则编译用的还是旧副本（现象：Flow Summary 数字与旧版逐位相同）
2. 全量编译（Analysis & Synthesis → Fitter → Timing；黑盒换 RTL 不能增量）
3. 关注 `clk_cnn`（50MHz）时序；board 验证：寄存器读写 → 单层对拍（软件侧跑一层，比对 DDR 输出）
- 适配要点：burstcount 固定 1、byteenable 全 1；**读完成以 `readdatavalid` 为准**（2026-08 起，见调试历史；早期实现曾错误地以 `read && !waitrequest` 判完成）；lr/wr 共用 load master（黑盒无独立权重 master），多笔在途独立计数

**阶段 5 关键设计**：
- requant 定点参数**软件侧预转**（量化优先、RTL 无浮点）：`conv_op.cc` 把 float scale（ws/bias/os）转成每通道 4 个 int32（mult/bias_mul/shift/rcl6）写入 scale 区，**乘后域**（`v = acc·mult + bias_mul<<8`，与 float 公式 `round(acc·ws·is/os + bias/os)` 对齐；bias/rcl6 用 q22 缩放避免 int32 溢出），公式与舍入（away-from-zero）对齐 `ref_int8.quantize_params`（随机回归 bit-exact 一致）；`intelfpga.cc` memcpy 4×out_c 字
- **requant 输出语义 = cpu_ref cvt_kernel（2026-08 起，弃用黑盒 wrap）**：`round-half-away`（v≥0 `(v+half)>>s`；v<0 `(v+half-1)>>s`，对应 `fcvtas`/`round()`）→ 饱和 `[-127,127]`（`saturate_cast<int8_t>` + `-128→-127`）；act：relu = `max(v,0)`、relu6 = `min(max(v,0), rcl6<<8)`（上限 6/os，q22→q30）。旧 wrap（`& 0xFF`）与 relu6 无上限已删除——越界 logits 翻转符号、relu6 特征超 6 会让 box 头 score 偏差（`cpu_ref/` 为权威参考）
- `cnn_top.v` 端到端：寄存器（START=0x00/DDRIN=0x10/DDRW=0x1C/DDROUT=0x28/PARAM=0x34/SCALE=0x40）→ param 块 20 字（0..19）边读边解析 → scale 读取 → cfg 配 cnn_core_v2 → 行块循环（base_row 重定位 + DMA 跟随 core 流握手）
- **DMA 关键修复**（对拍暴露）：地址预取（避免组合读重复）、`lr_read/wr_read/ow_write` 组合直连 core 握手（避免 1 拍残留导致段边界偏移）、cfg 最后一项写入时序（cfg_we 提前拉低丢 rcl6 末通道）
- **阶段 5 收尾（缺口已关闭）**：长层 1 LSB 差异根因 = **tb DDR 基址重叠**（DDRIN_BASE=0x2000 需 11552B 到 0x4D20，与 DDRW_BASE=0x4000 冲突；rb1 输入段拍 340 起读到 w.hex 数据）——tb 布局问题，非 RTL bug。基址拉开（DDRW=0x10000/DDROUT=0x20000）后 6/6 层 bit-exact；core 24 层回归仍全绿。真实软件内存由 CmaMem 连续分配不重叠，无此问题

**阶段 4 关键设计**（对齐黑盒，见 [cnn_top_spec.md §6/§8](../cnn_top_spec.md)）：
- 输入流 64-bit NHWC8 块序 `[C/8][H][W][8]`，DDR 顺序 burst 读；pad 行由模块补 0（不拉 `i_ready`，DMA 自动跳过，地址=已发字节数）
- 输入行缓冲：2026-08 前按通道块组织 `[in_cb][in_row_tile][W][8]`（曾假设模型 in_c≤32、4 cb 驻留 ≈ 309 块 M10K）——**该假设与实测不符**（模型 in_c 最大 1024），已改为**单 cb 流式驻留**（`G_MAX_IN_CB=1`，79 块 M10K），每 o_group 轮由顶层 DMA 重读全部输入通道块（见下节调试历史）；计算序 = 输出组 × 输入组 × 窗口（对齐 ref）
- **可综合性（同步读改造，2025；单维展平，2026）**：
  - 2025 同步读改造：原 `lb`/`acc` 为组合（异步）读，Quartus 无法推断 M10K，约 4Mbit 存储被展开成寄存器+读端口 mux，`quartus_map` 内存爆到 66GB 仍 OOM。改造后 `lb` 为 64-bit 字数组（8 lane/字）、`acc` 为 256-bit 字数组（8 lane×int32/字），读改为同步采样（`lb_q`/`acc_q` 晚一拍），`S_MAC` 拆 `S_MAC_RD`/`S_MAC_ACC` 两拍、`S_REQUANT` 拆 `S_REQ_ADDR`/`S_REQ_OUT` 两拍。tap 率/事件率减半。
  - 2026 单维展平（解决 40GB/1h A&S）：同步读改造后仍多维 unpacked 数组（`lb[cb][row][col]` 三维 + `acc` 256-bit 超宽），Quartus 对多维数组的 RAM 推断仍不可靠，A&S 实测 40GB+ / 1h。**改为单维数组**：`lb[0:CB*ROWS*W-1]`、`acc[0:OROWS*OW-1]`，索引用常量乘法拼接（`lb_waddr`/`lb_raddr`/`acc_waddr_*`/`acc_raddr_*` wire，综合折叠为移位/加法），保留 `(* ramstyle="M10K" *)`。功能/时序不变（读写仍同一拍采样），`run_cnn_core_v2.sh` 24/24 层回归通过。
  - 若换回多维写法，A&S 内存会复现爆炸——**不要改回多维 unpacked 数组**。
- 权重 slice 布局 `[Co/8][Ci/8][8][9][8]` 在 DDR 连续 → DMA 顺序读，slice 序号 = 权重流计数/72
- 单行块执行（`start→o_done`），行块循环/地址重定位留待 cnn_top 层
- **已修 RTL bug 记录**（对拍暴露）：cfg 打包 54→64-bit（readmemh 半 hex 截断丢 sel 位）、`cfg_sel` 3 位（sel=4 截断）、输入 pad 通道补 0、输出/输入区内存重叠（`out_off` 分离）、signed 输入语义回归（`$signed(lb_m)`）、窗口行索引多 `+pad`、DW slice 位置 `(cb_out,cb_out)`、输出末事件 `o_valid` 时序

## 上板调试历史（2026-08，复现核替代黑盒的关键修复）

黑盒 qxp 时代上板正常（`debug_test.log`，fpga_time 475ms/帧）。复现核上板后依次暴露以下问题，按修复顺序记录：

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `wait ip fail`（START 写 1 后 5s 未自清） | 读握手用 `read && !waitrequest` 判完成，而 `mm_bridge_sdram0` 是**流水读桥**（`MAX_PENDING_RESPONSES=4`、`PIPELINE_RESPONSE=1`）：waitrequest 拉低仅表示命令被接受，数据由 `readdatavalid` 延迟返回——原实现采到垃圾数据 → param/scale 解析错 → 状态机死锁 | `c7099ac` 改 `readdatavalid` 完成判定；`78cbe65` 补单笔在途（命令接受即拉低 read，防流水桥重复发命令） |
| 2 | （隔离验证） | 用"收到 START 立即自清、不做任何计算"的验证版（`305ae17`，临时提交）上板：通过 → 从接口/QSys/时钟/FPGA 配置全部正常，问题锁定在正式核内部 | 验证后回退正式版（`121b4da`） |
| 3 | 编译报 `Quartus 10028` 多驱动 | `lr_pending`/`wr_pending` 被两个 always 赋值（自动机 + 主状态机 S_START 清零） | `401eec6` 行块重置并入自动机（`core_start` 拍触发） |
| 4 | 编译报 `10161 not declared` | requant M10K 读数据 wire 声明在 generate 实例内，作用域只在实例内，主逻辑引用不到 | `fb90f2f` 提升到模块顶层 wire 数组（`rq_bias_q[0:7]` 等） |
| 5 | 有输出但检测不出目标 | **容量假设与实测模型不符**（47 层实测画像，见 [cnn_top_spec.md 附 A](../cnn_top_spec.md)：in_c/out_c 最大 1024）：① `G_MAX_C=128` 的 requant 数组从 conv8（out_c=256）起丢弃参数 → 输出全 0；② `req_buf[0:511]` 中转装不下 4×1024 字 | `d64c2cb`：`G_MAX_C` 128→1024，requant 数组改 8 lane 独立 M10K（8 lane 并行读 8 个通道地址，单端口 RAM 每拍 1 地址 → 每 lane 一份，约 104 块 M10K）；`S_RD_SCALE` 改边读边写 core（删 `req_buf`，4096 字寄存器堆在 5CSEBA6U23 上不可行） |
| 6 | （架构）lb 全通道驻留不可行 | `G_MAX_IN_CB=4`（in_c≤32）下 lb 4cb 驻留；模型 in_c≤1024 → 需 128cb ≈ 10160 块 M10K > 器件 553 块。层 4（in_c=64）起地址越界错乱，比 G_MAX_C 更早触发 | `368568f` **输入流式化**（对齐黑盒流式部分和架构）：`G_MAX_IN_CB` 4→1，`i_group+1 → S_LOAD` 重装下一 cb（部分和留 acc）、`o_group+1 → S_LOAD` 重装新组 cb；顶层 lr/wr 单笔→**多笔在途**（≤4，桥 `MAX_PENDING_RESPONSES=4`），每 o_group 轮末重置地址（CONV 回行块首、DW +1 cb） |
| 7 | （编译警告暴露）**k=1 权重错位**：core `S_WEIGHT` 固定消费 72 拍/slice，而软件 k=1 slice 仅 64B=8 拍（`conv2d_weight_reorganize` block_size=8×k×8）→ 层 2 起权重流跨 slice 错位；`w_rb_beats_r` 固定 ×72 多算 9 倍；`dma_wbeat` 16-bit 在 `w_rb_beats>65535` 层（层 12 起）回绕 | `2eb5d8d`：`S_WEIGHT` 按 k 分支（k=1：8 拍、lane 主序跳 72B 到 t=0 组）；`w_rb_beats_r` 按 `p_k==1 ? 8 : 72`；`dma_wbeat` 加宽 20-bit；`mac_t` 末 tap 按 k（k=1 单 tap，避免读 wbuf 未初始化区 + MAC 快 9 倍）；tb 24 层均为 k=3（k=1 未覆盖） |
| 8 | Flow Summary 异常：memory bits 仅 1.57Mbit（= lb+acc 恰好）、registers 119,480 超器件 55,856 LE、Logic utilization N/A（Fitter 无法完成）——即上次 A&S 后 ALM 超报错 | **requant 数组 M10K 推断失败**：写地址 `cfg_addr` 为 20-bit reg、读地址 `o_group*8+rqg` 为 32-bit 表达式，均超 RAM 深度位宽（1024 深 → 10-bit），Quartus 拒绝推断 → 数组展开成 LE/寄存器 | `c566453`：requant 写地址截断 `cfg_addr[9:0]`、读地址改 10-bit（`rq_raddr_base = {o_group[6:0],3'd0}` + 常数 `rqg`），值域 ≤1023 无损；修复后 memory bits 应 ≈2.7Mbit、寄存器 <25K |
| 9 | （软硬件接口审查暴露）**param offset 被忽略**：`conv_op.cc` 每层累加分配 input/weight/output 偏移（字偏移×8 为字节），FPGA 却只用 `fpga_init` 固定的 `reg_ddrin/ddrw/ddrout` 基址——层 0 偏移恰好全 0 掩盖问题，**第 1 层起输入/权重/输出全部错位**（检测不出目标的根因之一） | `897a40f`：S_CFG 解析 `param_buf[0/1/3]`，`S_START` 与 `lr_round_reset`(conv 分支) 地址 = 基址 + `(offset<<3)` + 行块偏移；`scale_offset` 不需用（软件每层 memcpy 到同一 `cb_scale`） |
| 10 | Flow Summary 依旧异常（memory bits 1,568,896、registers 119,711 逐位不变）——即使 `ip/` 与 `submodules/` 文件已是最新 | **写端口 case 多分支写多数组**：`case(cfg_sel)` 里 4 个 requant 数组共享一个写 always，Quartus 视为多写端口 RAM → M10K（单写端口）推断失败 → 32 个数组全展开（map.rpt 实测仅有 lb/acc 两个 altsyncram；`bias_store[1023][0]` 等逐 bit 展开） | `c5d0d6a`：写 always 拆成每数组独立（`if (cfg_we && cfg_sel==N && ...)`，去掉 case）；verilator 回归 16/16 层通过 |

**遗留**：上板检测正确性已确认（2026-08-10 修复后 47 层黑盒 log 逐层对比：主干 100%、box 头 97-100%，见上方已知问题表）；性能（装载/MAC 未重叠）待优化；tb 覆盖仅小通道参数（见上节验证覆盖边界）。

## 流水线化改造（分步进行，每步 verilator 对拍兜底）

| 步骤 | 内容 | 验证 |
|---|---|---|
| 1 ✅ | pr/sr 参数读单笔在途 → **多笔在途（≤4）流水读**：`cnn_top_core` 命令计数/返回计数分离，read 连续拉高直至发完或在途满；`tb_cnn_top` pr/sr 读模型改 4 深队列 | `run_cnn_top.sh` 6/6 PASS、`run_cnn_core_v2.sh` 16/16 PASS |
| 2 ✅ | `cnn_core` **S_ACC_CLR 与 S_LOAD 并行**：首 cb 装载期间同拍清 acc（lb/acc 独立 RAM，写口互斥），清完直接进 S_WEIGHT，未清完保留 S_ACC_CLR 兜底续清 | `run_cnn_core_v2.sh` 16/16 PASS、`run_cnn_top.sh` 6/6 PASS |
| 3 ✅ | `cnn_core` **lb 双缓冲 + 输入预取**：MAC 期间预取下一 cb 到 ~mac_buf_sel（新增 `i_pf_ready`、预取控制器、`S_PF_WAIT`/`S_LOAD_FLUSH`）；`cnn_top_core` `lr_read` 支持预取就绪与中间段解锁；**M10K 超量修复：`G_MAX_W` 512→304**（软件实际 ≤302）；**时序修复：`S_MAC_ACC` 预计算 `mac_r`/`mac_c_cl`**，拆 stride mux→lb_raddr 长路径 | `run_cnn_core_v2.sh` 16/16 PASS、`run_cnn_top.sh` 6/6 PASS（Quartus 待复验） |
| 4 ✅ | `cnn_top_core` **load master 改 burst 读**：lr/wr 各加 32×64b FWFT FIFO 解耦返回与消费；cmd FIFO 扩展为 `{type,beats_left}`；load master 发 ≤4 拍突发（wr 优先、busy 锁存保 waitrequest 稳定）；删除 `lr_round_end_q`/`wr_round_end_q` 旧轮停等状态机；容量预算按 `fifo_cnt + inflight` 防 FIFO 溢出；**时序修复：段尾判断改预计算阈值比较 + 满突发恒 4/尾拍恒 1，拆 `dma_ibeat→lr_address` 长组合链** | `run_cnn_top.sh` 6/6 PASS、`run_cnn_core_v2.sh` 16/16 PASS（Quartus 待复验） |
| 5 ✅ | `cnn_core` **MAC tap II=1 流水化**：旧 6 状态串行 tap 循环改为单态 `S_MAC_RUN` + `mv[0:4]` 有效位链，每拍 issue 1 个 tap，6 级数据通路（issue/RD/MUL/MUL2/MUL3/ACC）重叠执行；`acc_q`/`w_q`/首末 tap 标志随 tap 打拍走，末 tap 写回地址在 stage5 锁存防覆盖；**不合并任何流水级**（`mac_p_r`/`v_sum_r` 均保留，避开 6→4 合并的 -18ns 路径） | `run_cnn_core_v2.sh` 16/16 PASS、`run_cnn_top.sh` 6/6 PASS、verilator `--x-initial` 16/16 PASS、iverilog 16/16 PASS（Quartus 待复验） |
| 6 ✅ | `cnn_top_core` **store 反压握手修复**：`core_o_ready` 恒 1 时 `ow_waitrequest` 期间 core 继续换数据 → 输出写被静默丢弃、DDR 残留旧数据逐图累积（上板随图片数差异放大且两次运行不一致）；改为 `core_o_ready = core_o_valid && !ow_waitrequest` + `S_REQ_OUT3` 接受拍撤销 `o_valid`；tb 注入周期性反压复现并回归 | `run_cnn_top.sh` 8/8、6/6 PASS（含反压注入）、`run_cnn_core_v2.sh` 24/24 PASS |
| 7 ✅ | **requant 数值语义对齐 cpu_ref**（`cpu_ref/` = Paddle Lite ARM 权威公式）：① round-half-away 修正（旧 `(v+half)>>s` 后对负值 -1 只在半整点正确，非半整点偏 1）；② 输出 `wrap → 饱和 [-127,127]`（`saturate_cast<int8_t>` + `-128→-127`）；③ relu6 恢复上限 `rcl6<<8`（`requant_store` 新增 cfg_sel 4 存储）。同步更新 `ref_int8.py`/`ref_cnn_top.py` 与 tb 期望 | `run_cnn_core_v2.sh` 24/24 PASS、`run_cnn_top.sh` 10/10、6/6 PASS、`ref_int8.self_test` max\|err\|=1、lint 0（Quartus 待复验） |
| 8 ⏳ | 下一步待定（候选：requant II=1 重做 / requant 拍数压缩 / 150MHz） | — |

## 时序收敛（2026-08-05，150MHz 目标）

PLL outclk_1 恢复 150MHz（`e49c95c`）后，综合报告逐条拆流水直至收敛：

| # | failing path（slack @150MHz） | 根因 | 修复 |
|---|---|---|---|
| 1 | `rq_m_store_5 → v_rq64_l`（-10.1ns） | 32×33 单级组合乘法 | `0e9949a`：拆 4×16×16 DSP 部分积 + 加法树（S_REQ_MUL2/MUL3） |
| 2 | 同路径（加法树仍紧） | 64-bit 加法树 2 级 | `7de0c2f`：S_REQ_MUL3 两组中间和 + S_REQ_MUL4(5'd16) 最终和（每级 1 个加法） |
| 3 | round/输出路径（同批） | 64-bit 桶形移位 + 加法/比较串行 | `a94f74a`：移位各占一拍（S_REQ_OUT→v_rnd_delta、S_REQ_ROUND2(5'd17) 加法、S_REQ_OUT2→v_shifted、S_REQ_OUT3(5'd18) 饱和） |
| 4 | `bias_store_3 → v_act_l`（-8.6ns） | acc_q+bias 加法 + relu/rcl6 比较单级 | `7cce245`：S_REQ_MUL 加法→v_biased_l、S_REQ_MULB(5'd19) 比较→v_act_l |

requant 现为 12 拍单操作流水（S_REQ_ADDR → MUL → MULB → MUL2 → MUL3 → MUL4 → MULC → ACT → OUT → ROUND2 → OUT2 → OUT3；S_REQ_MULC2 已随 2026-08-10 rcl6 钳位移除），
**乘后域**（bias 在乘后加、relu/rcl6 在乘后施加，新增 S_REQ_ACT 拍）；
每拍仅加法/移位/比较之一（≤~5ns）；state 扩 5-bit；事件级对拍不受拍数影响（v2 24/24 随机层 PASS）。

**部署注意**：Quartus 综合读的是 QSys 生成物 `soc_system/synthesis/submodules/` 里的 RTL 拷贝
（.gitignore 忽略、git pull 不更新）——更新 `cnn_rtl/src/` 后需重新 Generate QSys，或手动拷 5 个 .v 到 submodules/。

**最终时钟状态（当前工程，非历史）**：`ip/pll` 的 outclk_1 现配置为 **100 MHz**
（`pll_inst.v` `gui_output_clock_frequency1=100.0`；`C5TB_top.sdc` 注释 "clk_cnn 150→100MHz 收敛"）——
150 MHz 目标未最终达成，PLL 已降频保证时序收敛；上表是 150 MHz 收敛过程的修复记录，全部保留防回退。

**时序/推断教训（防回退）**：
- `preserve` 只用于输入侧地址寄存器；读输出寄存器加 preserve 破坏 M10K 推断（Error 276003）
- **case 外写 RAM 破坏推断**（A&S 慢 2-3 倍）；写点必须在 case 内
- `i_ready` 打拍破坏握手（tb 数据错位：tb/DMA 在 i_ready 高沿送、core 下一拍收）；握手信号不可随意打拍
- 推断版无法强制 OLD_DATA（有/无 `no_rw_check` 均 New data）——必须显式 altsyncram

## 相关文档

- [cnn_top_spec.md](../cnn_top_spec.md) —— 规格书（接口/寄存器/指令/布局/数值语义），**RTL 与软件栈的行为契约**
  （附 A：47 层实测参数画像；附 B：数值语义实测方法与 CPU 同源算法）
- [ssd_detection/README.md](../../ssd_detection/README.md) —— 软件侧协议（intelfpga.cc 用法与调试）
