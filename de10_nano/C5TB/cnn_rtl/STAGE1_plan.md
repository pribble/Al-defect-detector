# 阶段 1 方案：数值对齐（int8 + per-channel scale，Verilog 实现）

> 目标：把数值语义对齐到 [cnn_top_spec.md §7](../cnn_top_spec.md) 的 int8/per-channel 语义。
> **实现语言：Verilog**（C5TB 工程顶层 `C5TB_top.v` 为 Verilog，QSys 侧一致性好）。
> 上游 `conv_layer.vhd`（VHDL）**仅作架构参考**，不翻译原码，按同构架构用 Verilog
> 直接实现目标语义。
> 本阶段**只动数值链**，不动接口/DMA/布局（那些是阶段 2~6）。

## 1. 上游数值链路现状（已核实，行号以 `upstream/hardware/rtl/layers/conv_layer.vhd` 为准）

| 环节 | 上游实现 | 位置 |
|---|---|---|
| 输入符号 | `v_activation := signed('0' & line_buffer(...))` —— **无符号扩展**（uint8） | 行 517 |
| 权重 | `weight_buffer(cnt) <= signed(i_weight_data)` —— **已是 signed int8** ✓ | 行 354 |
| MAC | signed 9bit × signed 8bit → 累加 signed int32 ✓ | 行 522-523 |
| bias | `v_biased_acc := raw + bias_store`（signed int32 per-channel） | 行 529 |
| activation | 硬编码 ReLU（`if v_biased_acc(31)='1' then 0`）—— 无配置 | 行 531-535 |
| requant | **无符号** `unsigned(acc_pos) * rq_m_store`（uint64×uint32）→ round-half-up → 右移 → 饱和 255（uint8） | 行 537-551 |
| 量化参数 | per-channel：bias(int32)/mult(uint32)/shift(uint8)，cfg 端口三槽（`cfg_sel=01/10/11`） | 行 368-388 |

**结论**：权重符号、int32 累加、per-channel 参数结构**已匹配**；要改的是——
输入/输出改 signed、requant 改 signed、activation 可配置（relu6 必做，leakyrelu 标记 TODO）。

## 2. 目标数值语义（对齐规格书 §7，实现 I'：bias 加在 raw 域）

```
raw      = Σ (signed int8 in × signed int8 w)          // int32
raw'     = act(raw + bias_int)                          // bias 先加（标准 conv 语义），再激活
           act ∈ {none: x, relu: max(x,0), relu6: clamp(x,0,RAW_CLAMP6)}
out      = sat( round_half_up( raw' · mult >> shift ), [-128, 127] )
```

> 推导：`out_fp = act(Σ(w·x) + b) / os`，令 `s = ws·is/os`，则
> `out_int8 = round( act( raw·s + b/os ) )`。定点化为：
> `raw' = act(raw + bias_int)`，bias 加在 raw 域需满足 `bias_int·s = b/os`，即
> `bias_int = round(b/(ws·is))`（raw 域 int32），
> `out = round(raw'·mult >> shift)`。

### 关键决策：requant 用"软件预转定点乘移位"，RTL 保留乘移位结构

理由：
1. 乘移位结构简洁、已验证过类似实现，RTL 改动最小；
2. RTL 浮点乘资源大，且原 `cnn_top` 内部 float 实现细节未知（spec §11 未决点 1），
   bit-exact 一致性留到阶段 7 上板对拍摸底，本阶段先保证**自洽**可信。

### 软件预转换公式（`tools/gen_params.py`，per-channel）

```
shift      = 30（全局统一，保证 |mult| ≤ 2^31 且精度足够）
mult_ch    = round( (ws_ch · is / os) · 2^shift )        // uint32
bias_int_c = round( b_ch / (ws_ch · is) )                // int32（raw 域，= b/s 再 /os 的推导见上）
RAW_CLAMP6 = round( 6 · os / (ws_ch · is) )              // relu6 的 raw 域上限（SCALE_6=6.0 参与）
```

> 注：`scale` 数组（`ws`、`b/os`）是原 IP 协议的一部分，本方案软件转换器直接从
> 模型参数（`b`、`ws`、`is`、`os`）计算上述定点参数，不必依赖 `b/os` 字段。

## 3. Verilog 实现要点（`src/conv_layer_s8.v`，架构同构自上游 VHDL）

模块接口与上游 `conv_layer.vhd` 一致（便于后续阶段 3~6 做 Avalon 封装时对照）：

```verilog
module conv_layer_s8 #(
    parameter G_C_IN    = 3,
    parameter G_C_OUT   = 4,
    parameter G_W_IN    = 4,
    parameter G_H_IN    = 4,
    parameter G_C_PAR   = 4,     // 并行输出通道数（= G_C_OUT 时无分组）
    parameter G_KERNEL  = 3,
    parameter G_PADDING = 1,
    parameter G_STRIDE  = 1,
    parameter G_ACT     = 1,     // 0 none / 1 relu / 2 relu6（阶段 1 综合期固定）
    parameter G_RAW_CLAMP6 = 0   // relu6 的 raw 域上限（阶段 1 综合期固定，阶段 2 运行时化）
)(
    input  wire               clk, rst_n,
    input  wire               i_valid,
    output wire               i_ready,
    input  wire [7:0]         i_data,          // signed int8
    input  wire               i_weight_valid,
    output wire               o_weight_ready,
    input  wire [7:0]         i_weight_data,   // signed int8
    input  wire               cfg_we,
    input  wire [1:0]         cfg_sel,         // 01=bias_int, 10=mult, 11=shift
    input  wire [19:0]        cfg_addr,
    input  wire [31:0]        cfg_wdata,
    output wire               o_valid,
    output wire [G_C_PAR*8-1:0] o_data,        // signed int8
    output wire               o_done,
    input  wire               i_acc_ready,
    output wire               o_acc_valid,
    output wire [G_C_PAR*32-1:0] o_acc_data    // 原始 int32 累加（调试/对拍）
);
```

数值链实现要点（与上游同构 + 阶段 1 改动）：

| 环节 | 实现 |
|---|---|
| 输入符号 | `i_data` 直接按 signed int8 使用（`$signed` 扩展） |
| MAC | `v_product = $signed(v_activation) * weight_buffer[...]`（signed 17bit），`accumulators[lane] += v_product`（signed 32bit） |
| bias | `v_biased = v_raw + bias_store[out_ch]`（raw 域 int32，`cfg_sel=01` 加载） |
| activation | `G_ACT`：0 直通 / 1 `max(v_biased,0)` / 2 `clamp(v_biased, 0, G_RAW_CLAMP6)` |
| requant | `v_rq = v_act * rq_m_store[out_ch]`（signed 64），`+ (1<<(shift-1))` round-half-up，**算术右移** `>>> shift` |
| 饱和 | `clamp(v_rq, -128, 127)` 输出 signed int8 |
| 量化参数 | per-channel 三槽不变：`01`=bias_int(int32)、`10`=mult(uint32)、`11`=shift(uint8)，数值按 §2 由软件生成 |

> 与上游 VHDL 的差异（阶段 1 全部改动）：① 输入/输出改 signed int8；
> ② requant 从"relu 后无符号乘+饱和 255"改为"activation 后 signed 乘+算术右移+饱和
> [-128,127]"；③ activation 从硬编码 ReLU 改为 `G_ACT` 可配（relu6 用 `G_RAW_CLAMP6`）。
> leakyrelu 标记 TODO（阶段 2 与 activation 运行时化一起做，本项目 SSD 无 leakyrelu）。

## 4. 验证计划（阶段 1 闸门）

1. **Python 整数参考模型**（`tools/ref_int8.py`，本机可跑）：
   - 读入 int8 输入/权重、float 参数（ws/is/os/b），按 §2 公式生成 mult/bias_int/shift
   - 实现与 RTL 同构的定点计算（signed 乘加、round-half-up、算术右移、饱和 int8）
   - 随机向量自测：参考模型输出 vs 直接 float 计算输出（评估定点误差 ≤1 LSB）
2. **Verilog tb**（`verification/tb/tb_conv_layer_s8.v`）：
   - 定向用例：signed 负输入、负权重、饱和上下界（>127、<-128）、round 边界（正/负）、relu6 clamp
   - 随机向量对拍：Python 生成期望（`$readmemh` 读入），与 RTL `o_data` 比较
3. **run 脚本**（`verification/sim/run_conv_s8.sh`，iverilog + vvp）：
   - 本机无权限装 iverilog，脚本已备好，在有 iverilog/Quartus 环境执行
4. **闸门**：定向 + 随机向量全部 bit-exact 通过。

## 5. 未决点（记录，不阻塞本阶段）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 原 `cnn_top` 的 float 舍入细节 | 阶段 7 黑盒对拍摸底；若差异超预期，调 §2 公式（shift 位数/round 模式） |
| 2 | relu6 的 `RAW_CLAMP6` per-channel 还是全局 | 软件取每层最大值（保守），阶段 2 随参数运行时化细调 |
| 3 | shift 全局=30 是否精度足够 | 由 Python 参考验证误差；必要时 per-layer shift |

## 6. 阶段 1 交付物清单

- [x] `src/conv_layer_s8.v`（Verilog 数据通路，§3 语义；数值链路含 bias_int 修正推导）
- [x] `tools/gen_params.py`（float 参数 → mult/bias_int/shift，CLI/JSON）
- [x] `tools/ref_int8.py`（Python 定点同构参考；随机自测通过：max|err|≤6 LSB 且集中在饱和/relu6 clamp 边界，97%+ 完全一致）
- [x] `tools/gen_tb_vectors.py` + `verification/vec/*.hex`（对拍向量，期望由定点同构生成）
- [x] `verification/tb/tb_conv_layer_s8.v` + `verification/sim/run_conv_s8.sh`（iverilog 一键回归）
- [ ] **验证全绿**：需在有 iverilog 的环境执行 `bash verification/sim/run_conv_s8.sh`（本机无权限安装）
