# cnn_core_v2 流水拍表与时序预估

> 时钟：clk_cnn = 150MHz → **周期 6.667ns**。目标：每拍组合链（串行步）≤ 6.667ns。
> 本文档逐拍列出"做什么 / 拍内串行链 / 并行项 / 预估"；配合 Top5 报告判断剩余风险。
> 对应代码：`ip/cnn_core_v2.v`（fc050c2 起 requant RAM 为 altsyncram OLD_DATA）。

## 一、状态机全景（每拍 1 个状态，无流水重叠）

```
S_IDLE → S_LOAD(流式,每拍1字) → S_ACC_CLR(清acc,循环) → S_WEIGHT(72/8拍)
  → [S_MAC_ADDR→RD→MUL→MUL2→MUL3→ACC] × 9tap(或1tap, k=1) × N位置
  → S_REQ_ADDR→MUL→MULB→MULC→MUL2→MUL3→MUL4→OUT→ROUND2→OUT2→OUT3（requant 11 级）
  → S_DONE
```

约定：`N = out_row_tile×out_w`（输出位置数）、`L = in_row_tile×in_w`（装载字数）、
`T = 9`（k=3）/ `1`（k=1）。一拍 = 一个状态 = 一次时钟沿。

## 二、逐拍明细（串行 = 该拍组合链，按依赖顺序；并行 = 同时发生互不依赖）

### S_LOAD（每 o_group/每 i_group 一次，L 拍）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | `lb_wa_q <= lb_waddr[15:0]; lb_wd_q <= i_data`（pad 行写 0）——收拍 | 并行 |
| 2 | `lb[lb_wa_q] <= lb_wd_q`——写上一拍收的字（流水写，无条件） | 并行 |
| 3 | 行末判断：`load_col==in_w_reg-1` → `load_row==in_row_tile_reg-1` → 转移/归零 | 串行链 A |
| 4 | `load_in_row=base_row_reg+load_row`（13-bit 加）→ 2 比较 → `load_row_valid` → `i_ready`（组合输出，跨模块到 cnn_top_core 握手） | 串行链 B（并行于 A） |
| 5 | `lb_waddr = load_row*512 + load_col`（移位+加法，20-bit） | 并行 |

- 串行链 A：`load_col →(12-bit 比较)→ load_row →(12-bit 比较)→ 转移 mux → rq_row/state D`——**2 级比较 + mux ≈ 3-4ns + 布线**（窄化后，原 32-bit 比较 -3.05 已修）
- 串行链 B：`load_row →(13-bit 加法+2 比较)→ i_ready →(cnn_top_core 组合 wr_read)→ cmd_type_q D`——**跨模块 ≈ 6-8ns（Top4-5 -2.826 的根源，未修）**
- 预估：链 A 收敛 ✓；链 B 是当前 Top4-5（-2.826），SEED/布线改善或 cnn_top_core 侧处理

### S_ACC_CLR（每 o_group 首轮，N 拍循环：每拍清 1 字）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | `acc[acc_clr_wa] <= 0`——清当前 rq 位置（地址为寄存器） | 并行 |
| 2 | 推进：`rq_col/rq_row +1`，末位置判断（2×12-bit 比较）→ 转移 | 串行链 |
| 3 | `acc_clr_wa <= 推进后地址`（`rq_row*150+rq_col+1`，×150 拆 128+16+4+2 移位加法树） | 串行链（与 2 同链） |

- 串行链：`rq_row/rq_col →(×150 加法树 3 级 + 12-bit 比较)→ acc_clr_wa D / state D`——**≈ 4-5ns**（1 拍窗口：rq 每拍推进）
- 预估：临界收敛，观察点（若冒头：×150 改为移位寄存器链或提前一拍）

### S_WEIGHT（k=3：72 拍；k=1：8 拍）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | `wbuf[wf_lane][m][wf_t] <= iw_data[m*8+:8]`（8 字节切片，寄存器阵列写） | 并行 |
| 2 | `wf_cnt/wf_t/wf_lane` 计数推进 + `wf_cnt==71`（8-bit）→ 转移 + mac_* 归零 | 串行链 |
| 3 | `lb[lb_wa_q] <= lb_wd_q`（补写 S_LOAD 最后 1 字，重装轮） | 并行 |

- 串行链：`iw_valid →(写译码 wf_lane[2:0]+wf_t[3:0] → 72 组)+wf_cnt 比较`——**≈ 3-4ns** ✓
- 预估：收敛 ✓

### S_MAC_ADDR（每位置每 tap 1 拍）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | 写回搭车：`if (acc_wr_we_q) acc[acc_wa_q] <= acc_wr_q` + 清脉冲 | 并行 |
| 2 | `lb_addr_r <= lb_raddr[15:0]`（`mac_r*512 + mac_c_cl`） | 串行链 A |
| 3 | `acc_addr_mac_r <= acc_raddr_mac[15:0]`（`mac_row*150 + mac_col`，preserve） | 串行链 B（并行于 A） |
| 4 | `mac_c_valid`（`mac_c = mac_col*stride + mac_kw_q - pad`，12-bit signed 加减比较）→ 下一拍寄存 | 串行链 C（并行） |

- 串行链 A：`mac_row →(stride 移位 + mac_kh_q 加法)→ mac_r →(×512 移位 + mac_c_cl)→ lb_addr_r D`——**≈ 4-5ns**（mac_kh_q 已提前寄存，查表拆出）
- 串行链 B：`mac_row →(×150 加法树)→ acc_addr_mac_r D`——**≈ 4-5ns**（preserve 防吸收）
- 预估：3 链并行，各 ≈4-5ns + 布线，1 拍窗口（mac_row/col 在 S_MAC_ACC 末更新）——**临界观察点**

### S_MAC_RD（每位置每 tap 1 拍）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | `lb_q <= lb[lb_addr_r]`（M10K 同步读） | 并行 |
| 2 | `w_q[lane][m] <= wbuf[lane][m][mac_t]`（仅本拍采样，9:1 mux，2 拍窗口） | 并行 |
| 3 | `acc_q <= acc[acc_addr_mac_r]`（M10K 同步读） | 并行 |
| 4 | `mac_c_valid_r <= mac_c_valid` | 并行 |

- 4 路并行读，各从寄存器地址出发：`地址寄存器 Q → M10K 读 → 数据寄存器 D`——**≈ 3-4ns + 布线**（w_q 9:1 mux ≈2ns，2 拍窗口无压力）
- 预估：收敛 ✓（P3 观察）

### S_MAC_MUL（每位置每 tap 1 拍）
- `mac_a_q[lane][m] <= lb_q 的 8-bit 切片`（64 个切片，纯连线）
- 预估：≈0.5ns ✓

### S_MAC_MUL2（每位置每 tap 1 拍）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | `mac_p_r <= mac_p`（8×8 乘法：lane0-3 DSP、lane4-7 LUT，64 个并行） | 串行链 |
| 2 | `mac_first_q/mac_last_q <= (mac_t==0/8)`（4-bit 比较） | 并行 |

- 串行链：`mac_a_q/w_q →(DSP 3-4ns / LUT 4-5ns)→ mac_p_r D`——**≈ 4-5ns + w_q fanout 布线**
- 预估：临界观察点（w_q 64 位 fanout 到 32 个 DSP 的布线）

### S_MAC_MUL3（每位置每 tap 1 拍）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | `v_sum_r[lane] <= mac_p_r[lane][0..7] 相加`（8 项加法树 3 级 LUT） | 串行链 |
| 2 | `acc_wa_q <= acc_waddr_mac[15:0]`（写回地址，多拍窗口采样） | 并行 |

- 串行链：`mac_p_r →(3 级加法树)→ v_sum_r D`——**≈ 4-5ns**（8 lane 并行）
- 预估：收敛 ✓

### S_MAC_ACC（每位置每 tap 1 拍，干 4 件事）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | `acc_local <= acc_next`（8 lane 独立 32-bit 加法 + `mac_first_q` 2:1 mux） | 串行链 A |
| 2 | 末 tap：`acc_wr_q <= acc_wr_next`（写数据打拍）+ `acc_wr_we_q <= 1` | 串行链 A（同链） |
| 3 | `mac_t <= mac_t+1`（或 0）；`mac_row/mac_col` 推进（2×12-bit 比较）→ 转移 | 串行链 B（并行于 A） |
| 4 | `mac_kh_q/mac_kw_q <= 查表(mac_t+1)`（4-bit case） | 串行链 C（并行） |

- 串行链 A：`v_sum_r →(32-bit 加法 + mux)→ acc_next/acc_wr_next → acc_local/acc_wr_q D`——**≈ 3-4ns**
- 串行链 B：`mac_row/mac_col →(12-bit 比较 ×2 + 转移 mux)→ rq_row/state D`——**≈ 4-5ns**（1 拍窗口）
- 串行链 C：`mac_t →(+1 + case 查表)→ mac_kh_q D`——**≈ 2-3ns**
- 预估：A/B/C 并行，各 <6.7ns——**B 是临界观察点**；写回已打拍（不直接连 acc 写口 256-bit fanout）

### S_REQ_ADDR（每位置 1 拍）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | 写回搭车（acc_wr_we_q，同 S_MAC_ADDR） | 并行 |
| 2 | `acc_q <= acc[acc_raddr_r]`（每拍采样，地址已打拍） | 并行 |
| 3 | `o_valid <= 0` | 并行 |

- 预估：M10K 读 ≈3-4ns ✓

### S_REQ_MUL（打拍级：断 RAM 读与运算）
- `rq_bias_m/rq_mult_m/v_rcl6_l <= rq_bias_q/rq_mult_q/rq_rcl6_q`——**q_\* 现在是 altsyncram q_b 输出（寄存器，综合版）/ 推断读（仿真版）**——纯连线
- 预估：✓（altsyncram OLD_DATA 后无 wren 依赖，Top1-3 -3.394 已结构消除）

### S_REQ_MULB / S_REQ_MULC（各 1 拍）
- MULB：`v_biased_l <= acc_q + rq_bias_m`（32-bit 加法 ≈2-3ns）✓
- MULC：`v_act_l <= relu/rcl6 比较 + mux`（≈3-4ns）✓

### S_REQ_MUL2（DSP 乘法级）
- `v_p_lolo/lohi/hilo/hihi <= mul16x16_dsp(v_act_l, rq_mult_m)`（4×16×16 DSP，32 个并行）
- 串行链：`v_act_l/rq_mult_m →(DSP 3-4ns)→ v_p_* D`；`v_shift_l <= rq_shift_q`（并行）
- 预估：≈4-5ns + 布线 ✓

### S_REQ_MUL3 / S_REQ_MUL4（64-bit 加法树 2 级）
- MUL3：`v_sum_lo/v_sum_hi`（2 个 64-bit 加法并行，各 ≈3-4ns）✓
- MUL4：`v_rq64_l <= v_sum_lo + v_sum_hi`（1 个 64-bit 加法 ≈3-4ns）✓

### S_REQ_OUT / S_REQ_ROUND2 / S_REQ_OUT2（round 三拍）
- OUT：`v_rnd_delta <= 1 << (v_shift_l-1)`——**64-bit 桶形移位 ≈5ns**（观察点）
- ROUND2：`v_round_l <= v_rq64_l + v_rnd_delta`（64-bit 加法 ≈3-4ns）✓
- OUT2：`v_shifted <= v_round_l >>> v_shift_l`——**64-bit 桶形移位 ≈5ns**（观察点）
- 预估：两个桶形移位各 ≈5ns + 布线，临界；从未进 Top5，观察

### S_REQ_OUT3（输出拍，干 3 件事）
| 项 | 内容 | 类型 |
|---|---|---|
| 1 | `v_q <= 饱和`（64-bit 比较 ×2 + mux）→ `o_data <= {v_q[7..0]}` | 串行链 A |
| 2 | `rq_row/rq_col` 推进（12-bit 比较 ×2）→ `o_group/i_group` 判断 → 转移 | 串行链 B（并行） |

- 串行链 A：`v_shifted →(64-bit 比较 + mux)→ v_q → o_data D`——**≈ 4-5ns**
- 串行链 B：`rq_row/rq_col →(比较 + 转移 mux)→ rq_row/state D`——**≈ 3-4ns**
- 预估：A/B 并行 ✓

### S_DONE
- `o_done <= 1; state <= S_IDLE` ✓

## 三、剩余时序风险清单（按冒头概率，2026-08 状态）

| 优先级 | 路径 | 预估 | 对策 |
|---|---|---|---|
| 1 | **load_row → i_ready → cmd_type_q**（S_LOAD 链 B，跨模块） | -2.8 级（Top4-5） | SEED/布线；cnn_top_core 侧拆（i_ready 打拍已证破坏握手，不可行） |
| 2 | S_MAC_ADDR 的 ×150 乘加链（acc_addr_mac_r） | 临界 -1~-2 | preserve 已加；若冒头：×150 改移位组合（128+16+4+2 已由 Quartz 拆）或地址提前一拍 |
| 3 | S_MAC_MUL2 的 w_q→DSP fanout | 临界 | 观察；若冒头：w_q 按 lane 复制 |
| 4 | S_MAC_ACC 推进链（比较+mux） | 临界 | 已窄化 12-bit，观察 |
| 5 | S_REQ_OUT/OUT2 桶形移位（64-bit） | ≈5ns | 从未进 Top5，观察 |
| 6 | S_MAC_RD lb_q/acc_q M10K 读 | ≈3-4ns | 观察 |

## 四、已修复（历史记录，防回退）

1. **wren→q_\* pass-through（-3.394）**：32 个 requant RAM 显式 altsyncram `OLD_DATA`（fc050c2），读口纯同步读；仿真走 `ifdef SIMULATION` 推断版
2. **acc 写回打拍（-6.23 级）**：`acc_wr_q` + `acc_wr_we_q` 单拍脉冲，写回随转移状态（S_MAC_ADDR/S_LOAD/S_REQ_ADDR）搭车；写点在 case 内保 M10K 推断（10999 教训：case 外写破坏推断）
3. **清零地址（P1）**：`acc_clr_wa` 随推进分支寄存"推进后地址"，全覆盖拍数不变
4. **lb 写打拍（P2-2）**：`lb_wa_q/lb_wd_q` 流水写；补写扩展到 S_WEIGHT（多 i_group 重装轮最后 1 字）
5. **窄化（54363d3）**：行/列/配置寄存器 32-bit → 12-bit/4-bit，所有比较乘加变短（load_row 行末比较 -3.05 → 收敛）
6. **mac_kh_q/kw_q（P2-3）**：查表(mac_t+1) 在 S_MAC_ACC 拍提前寄存，S_MAC_ADDR 拍省查表段
7. **rq_raddr_q**：q_\* 采样地址打拍，断 o_group→32 个 RAM 地址布线
8. **早期**：lb/acc 地址打拍 preserve、w_q 条件采样（2 拍窗口）、除法器改计数器（-53ns）、mac_first/last_q 提前、multstyle 拆分（DSP/LUT）

## 五、教训

- **preserve 只用于输入侧地址寄存器**；读输出寄存器 preserve 破坏 M10K 推断（Error 276003）
- **case 外写 RAM 破坏推断**（10999，A&S 慢 2-3 倍）；写点必须在 case 内
- **i_ready 打拍破坏握手**（tb 数据错位：tb/DMA 在 i_ready 高沿送、core 下一拍收，打拍后与 load_col 推进错位）
- **推断版无法强制 OLD_DATA**（有/无 no_rw_check 均 New data）——必须显式 altsyncram
- 每次改完跑 `tb_cnn_core_v2`（seed=7/9 各 16 层，按事件对拍）
