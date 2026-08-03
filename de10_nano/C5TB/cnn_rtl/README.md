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
├── README.md                    # 本文件：差距矩阵 + 改造路线图
└── upstream/                    # 上游原样文件（MIT）
    ├── LICENSE
    ├── hardware/rtl/layers/conv_layer.vhd   # 卷积数据通路底子（726 行，独立可综合）
    │                            #  + conv_layer.md（架构文档）+ State_Machine_conv.svg
    └── verification/            # GHDL 验证框架
        ├── sim/                 # run_conv.sh / run_conv_traversal.sh / run_layers.sh
        ├── tb/conv/             # 10 个定向测试（行缓冲/MAC/量化/背压/遍历）
        ├── tb/layers/           # 真实模型向量对拍 tb_conv_layer.vhd
        ├── generate/dump_vectors.py
        └── fixtures/            # 测试向量
```

## 上游 `conv_layer.vhd` 是什么

单层流式卷积数据通路（generic 可配置），内部包含：

- **行缓冲**：3+1 行（3×3 kernel），逻辑行旋转、虚拟 padding 行（行有效标志）
- **MAC**：`G_C_PAR` 个输出通道并行 × 3×3 × C_IN，int32 累加
- **量化链**：raw acc → 加 int32 bias → ReLU → 定点乘移位 requant
  （uint64×uint32 乘 + round-half-up 右移）→ 饱和 → uint8 输出
- **控制**：FSM（IDLE→初始行填充→prime 行→计算/滑窗→流式行填充→行旋转），
  自动帧重启；valid/ready 流接口 + cfg 端口（bias/requant 参数加载）

已通过 GHDL 位精确验证（5 层真实模型向量全过）。**这是"数值语义已确定"的可信底子**，
与 cnn_top 规格书 §7 的量化链（int32 累加 → bias → relu → requant → 饱和）同构。

## 差距矩阵（上游 → cnn_top 规格）

| 维度 | 上游 conv_layer.vhd | cnn_top（[spec](../cnn_top_spec.md)） | 改造动作 |
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

## 改造路线图（每阶段有验证闸门）

> 验证方法统一采用上游方法论：**GHDL 仿真输出 vs Python 整数模型位精确对拍**。
> 本机若无 GHDL，先在开发机安装（`apt install ghdl`）或任选支持 VHDL-2008 的
> 仿真器；上板对拍按 [cnn_top_spec.md §10](../cnn_top_spec.md)。

| 阶段 | 内容 | 验证闸门 |
|---|---|---|
| **0** | 上游基线引入（本文件即完成） | 有 GHDL 环境跑 `run_conv.sh` / `run_conv_traversal.sh` / `run_layers.sh` 全绿 |
| **1** | **数值对齐**：uint8→int8；requant 改 signed 乘加（软件预转 per-channel `mult`/`bias_rq`/`shift`，RTL 保留乘移位结构）；activation 加 relu6 | 新增 `tb_conv_layer_s8.vhd` + Python 整数参考对拍，定向+随机向量 bit-exact 通过 |
| **2** | **维度运行时化**：generic → 运行时参数（in_c/h/w、out_c/h/w、pad、stride）；param 块解析器（DDR 读 `struct parameter`，转成控制信号） | 用软件栈实际 param 内容构造测试向量，逐层对拍 |
| **3** | **Avalon 接口**：从接口 + 6 寄存器（START/DDRIN/DDRW/DDROUT/PARAM/SCALE）；启动/完成自清/轮询 | 寄存器级 tb：写寄存器→启动→完成时序符合 spec §4 |
| **4** | **DMA**：4 个 Avalon 主（param/load 读、output 写 64-bit burst）；地址 = 基址寄存器 + word offset×8 | DMA tb：burst 传输 + 地址生成正确 |
| **5** | **布局适配**：输入 NHWC8、权重 tile 化、输出 NHWC8、行块调度（5/11） | 全层模拟：与 intelfpga.cc 重排后数据逐层一致 |
| **6** | **depthwise**：权重对角化 + type=4 路径 | dw 层对拍 |
| **7** | **上板替换**：打包为 QSys IP（新 hw.tcl + 新 qxp），替换 `ip/cnn_top.qxp` | 板上 `ssd_detection` 检测结果与旧 IP 一致 |

> **对拍基准（已完成）**：`tools/ref_cnn_top.py` 是 cnn_top 的**参数驱动行为模型**
> （numpy）：row_block × chn_block 双层循环、NHWC8 字偏移寻址、dw 对角权重、
> 每层 param 驱动（维度/kernel/stride/pad/act/偏移）。验证已通过：
> 80 随机层 + 真实 47 层（参数取自 [model_profile.md](model_profile.md)）
> 分块执行 == 整图直算 bit-exact（`python3 tools/ref_cnn_top.py` → ALL PASS）。
> 阶段 2~6 的 RTL 改造以它作软件对拍基准。

## 实际执行进度（Verilog 化，verilator 对拍）

> 规划表是早期 VHDL 视角；实际按下面推进，每阶段都以 `ref_cnn_top.py` 位精确对拍为闸门。

| 阶段 | 交付 | 验证 |
|---|---|---|
| 1 ✅ | `src/conv_layer_s8.v`（定点卷积数据通路，Verilog，阶段 1 中间产物）——**2026-08 清理删除**（历史见 git） | `verification/sim/run_conv_s8.sh`（iverilog/verilator）全绿 |
| 2 ✅ | `tools/ref_cnn_top.py`（numpy 行为模型，裁判） | 80 随机层 + 真实 47 层 ALL PASS |
| 3 ✅ | `src/cnn_core.v`（参数驱动单层执行器，k=3 子集）——**2026-08 清理删除**（历史见 git） | `verification/sim/run_cnn_core.sh` 24 层 bit-exact |
| 4 ✅ | `src/cnn_core_v2.v`（**NHWC8 块序流**：slice 权重、行块 acc 缓冲、单行块执行供 DMA 调度） | `verification/sim/run_cnn_core_v2.sh` 24 层 bit-exact（conv/dw × s1/s2 × act0/1/2 × row_block≥1） |
| 5 ✅ | `cnn_top.v` Avalon 顶层（从接口 + 6 寄存器 + param/scale 解析 + DMA + 行块调度）；requant 定点参数**软件侧预转**（`conv_op.cc` scale 区每通道 4 字：mult/bias_int/shift/rcl6，RTL 无浮点） | 端到端 tb：**6/6 层 bit-exact**（单块/多块/多组、38 高长层） |
| 6 ✅ | depthwise（type=4）：权重对角化 + MAC 复用 | v2 回归覆盖 dw 层 |
| 7 ✅ | 打包 QSys IP 上板替换（黑盒 `cnn_top.qxp` 已删除） | 上板 `test.sh` 全链路运行；**检测正确性验证中**（见下节调试历史） |

> **验证覆盖边界（重要）**：tb 各层向量均为**小通道参数**（`in_c`/`out_c` ≤ 32，见
> `verification/v2/layer_*/cfg.hex`），**大通道（in_c/out_c 最大 1024）未经仿真覆盖**，
> 其正确性依赖上板验证与下述 2026-08 容量修复。
>
> **2026-08 清理**：阶段 1/3 的中间产物（`conv_layer_s8.v`/`cnn_core.v` 及其 tb、
> run 脚本、向量生成器）已被 v2/top 取代，随 `STAGE1_plan.md` 一并删除，历史见 git；
> `tools/ref_int8.py` 保留（`conv_op.cc` 定点公式的对齐依据）。

**阶段 5 交付（Quartus 接入）**：`cnn_top.v`（QSys 适配层，端口与 `ip/cnn_top_hw.tcl` 一致）+ `cnn_top_core.v`（RTL 本体，tb 直接对拍）+ `cnn_core_v2.v`；`ip/cnn_top_hw.tcl` 已把 `cnn_top.qxp` 黑盒引用改为 3 个 `.v`。接入步骤（Windows Quartus）：
1. 把 `ip/` 下 3 个 `.v` 随工程提交（已复制）；Platform Designer 打开 `soc_system.qsys` → Generate HDL（顶层会例化 `cnn_top` = 新 wrapper）
2. 全量编译（Analysis & Synthesis → Fitter → Timing；黑盒换 RTL 不能增量）
3. 关注 `clk_cnn`（50MHz）时序；board 验证：寄存器读写 → 单层对拍（软件侧跑一层，比对 DDR 输出）
- 适配要点：burstcount 固定 1、byteenable 全 1；**读完成以 `readdatavalid` 为准**（2026-08 起，见调试历史；早期实现曾错误地以 `read && !waitrequest` 判完成）；lr/wr 共用 load master（黑盒无独立权重 master），多笔在途独立计数

**阶段 5 关键设计**：
- requant 定点参数**软件侧预转**（量化优先、RTL 无浮点）：`conv_op.cc` 把 float scale（ws/bias/os）转成每通道 4 个 int32（mult/bias_int/shift/rcl6）写入 scale 区，公式与舍入（away-from-zero）对齐 `ref_int8.quantize_params`（500 轮随机验证一致）；`intelfpga.cc` memcpy 4×out_c 字
- `cnn_top.v` 端到端：寄存器（START=0x00/DDRIN=0x10/DDRW=0x1C/DDROUT=0x28/PARAM=0x34/SCALE=0x40）→ param 块 27 字解析 → scale 读取 → cfg 配 cnn_core_v2 → 行块循环（base_row 重定位 + DMA 跟随 core 流握手）
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
| 5 | 有输出但检测不出目标 | **容量假设与实测模型不符**（`model_profile.md` 实测：in_c/out_c 最大 1024）：① `G_MAX_C=128` 的 requant 数组从 conv8（out_c=256）起丢弃参数 → 输出全 0；② `req_buf[0:511]` 中转装不下 4×1024 字 | `d64c2cb`：`G_MAX_C` 128→1024，requant 数组改 8 lane 独立 M10K（8 lane 并行读 8 个通道地址，单端口 RAM 每拍 1 地址 → 每 lane 一份，约 104 块 M10K）；`S_RD_SCALE` 改边读边写 core（删 `req_buf`，4096 字寄存器堆在 5CSEBA6U23 上不可行） |
| 6 | （架构）lb 全通道驻留不可行 | `G_MAX_IN_CB=4`（in_c≤32）下 lb 4cb 驻留；模型 in_c≤1024 → 需 128cb ≈ 10160 块 M10K > 器件 553 块。层 4（in_c=64）起地址越界错乱，比 G_MAX_C 更早触发 | `368568f` **输入流式化**（对齐黑盒流式部分和架构）：`G_MAX_IN_CB` 4→1，`i_group+1 → S_LOAD` 重装下一 cb（部分和留 acc）、`o_group+1 → S_LOAD` 重装新组 cb；顶层 lr/wr 单笔→**多笔在途**（≤4，桥 `MAX_PENDING_RESPONSES=4`），每 o_group 轮末重置地址（CONV 回行块首、DW +1 cb） |

**遗留**：上板检测正确性待验证（368568f 后）；性能（装载/MAC 未重叠、pr/sr 仍单笔在途）待优化；tb 覆盖仅小通道参数（见上节验证覆盖边界）。

## 相关文档

- [cnn_top_spec.md](../cnn_top_spec.md) —— 黑盒规格（接口/寄存器/指令/布局/数值语义），**改造的唯一验收依据**
- [ssd_detection/README.md](../../ssd_detection/README.md) —— 软件侧协议（intelfpga.cc 用法与调试）
