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
| `cnn_top_spec.md` | **cnn_top 黑盒规格书**：接口/寄存器/指令格式/数据布局/数值语义，纯 RTL 复现 `cnn_top` 的依据 |
| `cnn_rtl/` | **cnn_top 开源复现工作区**：上游卷积数据通路（MIT）+ 差距矩阵与分阶段改造路线图 |
| `soc_system_board_info.xml` / `hps_common_board_info.xml` | 生成设备树时使用的板级信息（供 sopc2dts） |
| `ip/cnn_top_hw.tcl` | 自定义 CNN 加速器 IP 硬件描述（引用 `cnn_top.v`/`cnn_top_core.v`/`cnn_core_v2.v`/`mac8x8_dsp.v`/`mac8x8_lut.v` 五个 RTL 文件，已替换旧黑盒归档） |
| `ip/cnn_top.v` 等五个 `.v` | cnn_top 的 RTL 实现（QSys 适配层 + 核心 + 卷积执行器 + 8×8 乘法器，见 `cnn_rtl/README.md`） |
| `ip/pll/` | PLL IP：`FPGA_CLK1_50`（50 MHz）分出 `fpga_clk_50` / `fpga_clk_cnn` / `fpga_clk_stp` 三路时钟 |
| `tools/` | 构建辅助脚本与**已提交的生成产物**（见下） |

### 顶层结构（C5TB_top.v）

```
FPGA_CLK1_50 ──> pll_inst ──> fpga_clk_50 / fpga_clk_cnn / fpga_clk_stp
                                   │
                              soc_system (QSys)
                                   ├── HPS DDR3（32-bit）
                                   ├── HPS EMAC1（千兆以太网，含 GPIO35 中断）
                                   ├── HPS SDMMC / USB1 / UART0
                                   └── cnn_top_0（CNN 加速器，Avalon 从接口 hps2cnn_avs）
```

QSys 系统中 `cnn_top_0` 的关键参数（`soc_system.qsys`）：`IMAGE_MAX_W=302`、
`INPUT_CHANNEL_TILE=8`、`INPUT_ROW_TILE=11`、32-bit AXI/Avalon 数据宽度等。

### tools/ 内容

| 文件 | 说明 |
|------|------|
| `sof_to_rbf.bat` | 用 `quartus_cpf` 把 `output_files/C5TB_top.sof` 转换为 `soc_system.rbf`（开启 bitstream 压缩） |
| `gen_dtb.bat` | 用 Intel FPGA `sopc2dts`（配合两个 board xml）生成 `soc_system.dts`，再用 `dtc` 编译为 `soc_system.dtb` |
| `generate_hps_qsys_header.sh` | 用 `sopc-create-header-files` 从 sopcinfo 生成 `hps_0.h`（HPS 外设地址头文件，供裸机/驱动使用） |
| `soc_system.rbf` / `soc_system.dts` / `soc_system.dtb` | **已提交的生成产物**，可直接烧录/启动，无需重新编译 |

> `sof_to_rbf.bat` 与 `gen_dtb.bat` 是 Windows 批处理，需要 Windows 上的
> Quartus（及 embedded tools）环境；`generate_hps_qsys_header.sh` 可在 Linux 上运行。

## 构建流程

```text
1. Quartus Prime（23.1 Lite Edition；工程创建于 18.1 Standard，SDC 为 18.1 生成，兼容正常）打开 C5TB_top.qpf
2. 全编译（Analysis & Synthesis → Fitter → Assembler）
   —— 顶层 = C5TB_top.v + soc_system/synthesis/soc_system.qip（QSys 生成，目录被 gitignore）
        + ip/pll/pll_inst.qip + ip/cnn_top（cnn_top_hw.tcl + 5 个 .v）
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
板级 xml、`ip/`（cnn_top 与 pll 的源）、`tools/` 脚本及其生成的 rbf/dtb/dts。

**被 gitignore 的构建产物**（每次编译重新生成，不入库）：

```
C5TB/db/  C5TB/incremental_db/  C5TB/output_files/  C5TB/hps_isw_handoff/
C5TB/.qsys_edit/  C5TB/soc_system/  C5TB/*.qws  C5TB/c5_pin_model_dump.txt
C5TB/hps_sdram_p0_summary.csv
```

其中 `soc_system/` 是 QSys Generate 的输出目录（含 `synthesis/soc_system.qip`），
是编译的必要输入之一，但在干净克隆上需要先用 QSys 打开 `soc_system.qsys`
重新 Generate 才能编译工程。
