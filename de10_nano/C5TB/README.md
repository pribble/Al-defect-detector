# C5TB/ — Quartus SoC 工程（FPGA 可编程逻辑侧）

> 导航：上一级 [de10_nano/](../README.md) · 相关 [kernel/](../kernel/README.md)、[ssd_detection/](../ssd_detection/README.md)

这是 DE10-Nano（Cyclone V SoC `5CSEBA6U23I7`）的 Quartus Prime 工程，对应 SoC
中 **FPGA 可编程逻辑（FPGA fabric）** 部分。它把 HPS 外设引脚（DDR3、以太网、
SDMMC、USB、UART）接到 QSys 生成的 `soc_system` 系统上，并在其中例化自定义的
**CNN 加速器 IP（`cnn_top`）**——推理服务（[`ssd_detection/`](../ssd_detection/README.md)）
通过 Avalon 接口把卷积计算卸载到该加速器。

编译产物 `soc_system.rbf`（FPGA 配置比特流）在 HPS 启动时加载到 FPGA，
`soc_system.dtb`（设备树）交给 HPS 侧 Linux 使用。

## 目录结构

| 路径 | 说明 |
|------|------|
| `C5TB_top.qpf` / `C5TB_top.qsf` | Quartus 工程文件；顶层实体 `C5TB_top`，器件 Cyclone V 5CSEBA6U23I7 |
| `C5TB_top.v` | 顶层 Verilog：例化 `pll_inst`（PLL）与 `soc_system`（QSys 系统），连接 HPS 外设引脚 |
| `C5TB_top.sdc` | 时序约束（Quartus 18.1 生成） |
| `soc_system.qsys` / `soc_system.sopcinfo` | QSys 系统描述：HPS + 时钟 + `cnn_top_0` 加速器 |
| `cnn_top_spec.md` | **cnn_top 规格书**：接口/寄存器/指令格式/数据布局/数值语义（现行行为契约；附 A 47 层实测参数、附 B CPU 算法） |
| `cnn_rtl/` | **cnn_top 开源复现工作区**：上游卷积数据通路（MIT）+ 差距矩阵与分阶段改造路线图 |
| `soc_system_board_info.xml` / `hps_common_board_info.xml` | 生成设备树时使用的板级信息（供 sopc2dts） |
| `ip/cnn_top_hw.tcl` | 自定义 CNN 加速器 IP 硬件描述（fileset 引用 `cnn_rtl/src/` 下 `cnn_top.v`/`cnn_top_core.v`/`cnn_core.v`/`requant_store.v`/`mac8x8_dsp.v`/`mac8x8_lut.v` 六个 RTL 文件，已替换旧黑盒归档；RTL 仅此一份） |
| `cnn_rtl/src/` | cnn_top 的 RTL 唯一实现（QSys 适配层 + 核心 + 卷积执行器 + 8×8 乘法器，见 `cnn_rtl/README.md`） |
| `ip/pll/` | PLL IP：`FPGA_CLK1_50`（50 MHz 参考）分出三路：`fpga_clk_50`（outclk_0，50 MHz）/ `fpga_clk_cnn`（outclk_1，**100 MHz**，驱动 cnn_top）/ `fpga_clk_stp`（outclk_2，100 MHz） |
| `tools/` | 构建辅助脚本与**已提交的生成产物**（见下） |

### 顶层结构（C5TB_top.v）

```
FPGA_CLK1_50 ──> pll_inst ──> fpga_clk_50(50M) / fpga_clk_cnn(100M) / fpga_clk_stp(100M)
                                   │
                              soc_system (QSys)
                                   ├── HPS DDR3（32-bit）
                                   ├── HPS EMAC1（千兆以太网，含 GPIO35 中断）
                                   ├── HPS SDMMC / USB1 / UART0
                                   └── cnn_top_0（CNN 加速器，Avalon 从接口 hps2cnn_avs）
```

QSys 系统中 `cnn_top_0` 的参数（`ip/cnn_top_hw.tcl`）：4 个 Avalon 主接口的地址宽度
32-bit、数据宽度 32/64-bit（param/scale 读 32-bit，load 读/output 写 64-bit）。
黑盒时代的 `IMAGE_MAX_W=302`、`INPUT_CHANNEL_TILE=8`、`INPUT_ROW_TILE=11` 等尺寸参数
已随 RTL 化删除——各层尺寸由运行时 param 块驱动（见 `cnn_top_spec.md` §3/§5）。

### tools/ 内容

| 文件 | 说明 |
|------|------|
| `sof_to_rbf.bat` | 用 `quartus_cpf` 把 `output_files/C5TB_top.sof` 转换为 `soc_system.rbf`（开启 bitstream 压缩） |
| `gen_dtb.bat` | 用 Intel FPGA `sopc2dts`（配合两个 board xml）生成 `soc_system.dts`，再用 `dtc` 编译为 `soc_system.dtb` |
| `generate_hps_qsys_header.sh` | 用 `sopc-create-header-files` 从 sopcinfo 生成 `hps_0.h`（HPS 外设地址头文件，供裸机/驱动使用） |
| `soc_system.rbf` / `soc_system.dts` / `soc_system.dtb` | **已提交的生成产物**，可直接烧录/启动，无需重新编译 |

> `sof_to_rbf.bat` 与 `gen_dtb.bat` 是 Windows 批处理，需要 Windows 上的
> Quartus（及 embedded tools）环境；`generate_hps_qsys_header.sh` 可在 Linux 上运行。

## 已知问题 / 调试记录（重要）

| 日期 | 问题 | 根因 | 修复 |
|------|------|------|------|
| 2026-08-10 | 上板检测结果错（几百框），与 main 分支（黑盒 qxp）不符 | `cnn_core_v2.v` MAC 乘法器 **b 输入索引交叉错位**：`w_q[mac_lane_i][mac_m_i]`（= W[输入 lane][输出 m]）乘 `a[m]`（输入通道 m），实际算成 `acc[lane]=Σ_m 输入[m]·W[输入 lane][输出 m]`，正确应为 `acc[o]=Σ_i 输入[i]·W[输入 i][输出 o]`。tb 未暴露：`gen_*_vectors.py` 权重流按 ref 布局 `[mo][k][mi]`（mo 主序）生成，与软件真实布局 `[mi][k][mo]`（mi 主序）字节序不同，tb 恰好自洽 | 乘法器 b 改为 `w_q[mac_m_i][mac_lane_i]`；`gen_cnn_core_v2_vectors.py` / `gen_cnn_top_vectors.py` 权重流改软件布局字节序（`[mi][k][mo]`，每 8 字节 = mo）。修复后主干与黑盒 100% 一致 |
| 2026-08-10 | box 头（conv2d_69-80，act=0）与黑盒仍有 30-55% 差异 | requant 输出缺**负值 -1 修正**：黑盒实测语义是 `round_half_away(fv)` 后负值再 -1（floor 除法特性），RTL 只做 `(v+2^29)>>30`，所有负 logits 偏大 1-2 LSB | `S_REQ_OUT3` 输出截断前加 `v_shifted<0 → v_shifted-1`（`& 64'hFF` 截断）；同步 `ref_int8.py conv_s8` / `ref_cnn_top.py post_np`。修复后 47 层黑盒 log 对比：主干 100%、box 头 97-100%（= float 公式同精度） |
| 2026-08-10 | Quartus 时序失败（换 seed 无法解决）：`mac_c_valid_r → mac_p_r` slack -0.167ns | pad 列清零在**乘法器内部**（`mac8x8_dsp/lut` 的 `en` 选择 a），路径 = `mac_c_valid_r → mux → 乘法 → mac_p_r` 穿乘法器，单拍超限 | 乘法器去掉 `en` 端口（纯 `a×b`）；pad 列清零提前到 `S_MAC_MUL` 采样处（`mac_a_q <= mac_c_valid_r ? lb_q : 0`，0×b=0 等价）。关键路径退化为纯乘法器；verilator 对拍 16/16 + 6/6 保持 PASS |

> **修改 `cnn_rtl/src/*.v` 后必须重新 QSys Generate**（Qsys 把 HDL 复制到
> `soc_system/synthesis/` 后才参与编译），再全编译 + 重新烧录 `.rbf`。

## 构建流程

```text
1. Quartus Prime（23.1 Lite Edition；qpf 版本 17.1，SDC 为 18.1 生成，兼容正常）打开 C5TB_top.qpf
2. 全编译（Analysis & Synthesis → Fitter → Assembler）
   —— 顶层 = C5TB_top.v + soc_system/synthesis/soc_system.qip（QSys 生成，目录被 gitignore）
        + ip/pll/pll_inst.qip + ip/cnn_top_hw.tcl（fileset 引用 cnn_rtl/src 的 5 个 .v）
   —— 产物：output_files/C5TB_top.sof
3. 运行 tools/sof_to_rbf.bat   → tools/soc_system.rbf（FPGA 配置比特流）
4. 运行 tools/gen_dtb.bat      → tools/soc_system.dts / soc_system.dtb（设备树）
5. 把 soc_system.rbf / soc_system.dtb 部署到 HPS 启动分区（SD 卡 FAT 分区）：
   —— 启动时 .rbf 配置 FPGA，.dtb 由内核加载
```

> 修改了 QSys 系统（`soc_system.qsys`）或 IP 参数后，需在 QSys 中重新
> Generate，再执行上述 2-5 步。

## 源码与生成物的区分（.gitignore）

**提交到 git 的源码**：`C5TB_top.v/.qsf/.qpf/.sdc`、`soc_system.qsys/.sopcinfo`、
板级 xml、`ip/`（cnn_top_hw.tcl 与 pll 的源）、`cnn_rtl/src/`（唯一 RTL 源）、`tools/` 脚本及其生成的 rbf/dtb/dts。

**被 gitignore 的构建产物**（每次编译重新生成，不入库）：

```
C5TB/db/  C5TB/incremental_db/  C5TB/output_files/  C5TB/hps_isw_handoff/
C5TB/.qsys_edit/  C5TB/soc_system/  C5TB/*.qws  C5TB/c5_pin_model_dump.txt
C5TB/hps_sdram_p0_summary.csv
```

其中 `soc_system/` 是 QSys Generate 的输出目录（含 `synthesis/soc_system.qip`），
是编译的必要输入之一，但在干净克隆上需要先用 QSys 打开 `soc_system.qsys`
重新 Generate 才能编译工程。
