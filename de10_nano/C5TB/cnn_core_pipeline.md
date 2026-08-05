# cnn_core_v2 流水拍表与隐患清单

> 150MHz（clk_cnn）时序收敛工作笔记。目标：每拍组合链 ≤ 6.667ns，
> 所有 RAM 地址/读输出从寄存器出发（或窗口 ≥ 2 拍）。

## 一、状态机全景

```
S_IDLE → S_LOAD(流式,每拍1字) → S_ACC_CLR(1拍) → S_WEIGHT(72/8拍)
  → [S_MAC_ADDR→RD→MUL→MUL2→MUL3→ACC] × 9tap(或1tap, k=1)
  → S_REQ_ADDR → MUL→MULB→MULC→MUL2→MUL3→MUL4→OUT→ROUND2→OUT2→OUT3
  → (循环下一组 / S_DONE)
```

## 二、每拍操作与组合链（★= 已修，▲= 隐患）

### 装载/清零/权重

| 拍 | 操作 | 组合链 | 状态 |
|---|---|---|---|
| S_LOAD | 写 `lb[lb_waddr]`（`lb_waddr = load_row*G_MAX_W+load_col`）；`i_ready = state==S_LOAD && load_row_valid`（32-bit 加法+比较） | `load_row → lb_waddr 乘加 → lb 写口地址`（每拍变，1 拍窗口）；`load_row → load_row_valid`（喂顶层握手） | ▲ P2-2（lb 写地址，8 块 M10K，fanout 较小，未冒头） |
| S_ACC_CLR | 写 `acc[acc_waddr_clr]`（组合）；`rq_row/rq_col` 推进 | `rq_row → 乘加 → acc 写口`；推进拍后 1 拍窗口 | ▲ P1（大概率下一条 -4 级） |
| S_WEIGHT | 写 `wbuf[wf_lane][m][wf_t]`；`wf_cnt/wf_lane/wf_t` 计数 | 计数器短译码 | ✓（除法器已改计数器） |

### MAC（每 tap 6 拍）

| 拍 | 操作 | 组合链 | 状态 |
|---|---|---|---|
| S_MAC_ADDR | `lb_addr_r <= lb_raddr`；`acc_addr_mac_r <= acc_raddr_mac` | `mac_t → 查表 kh/kw → mac_r/mac_c 加减 → mac_c_valid → lb_raddr 乘加`（1 拍窗口） | ▲ P2-3（lb 读地址链，未冒头） |
| S_MAC_RD | `lb_q <= lb[lb_addr_r]`；`w_q <= wbuf[lane][m][mac_t]`（仅本拍）；`mac_c_valid_r <= mac_c_valid`；`acc_q <= acc[acc_addr_mac_r]` | wbuf 9:1 mux（2 拍窗口）★；lb/acc M10K 读（1 拍，地址已寄存） | ✓ |
| S_MAC_MUL | `mac_a_q <= lb_q`（a 输入打拍） | 连线 | ✓ ★ |
| S_MAC_MUL2 | `mac_p_r <= mac_p`（DSP/LUT 乘法）；`mac_first/last_q` 标志 | `mac_a_q/w_q → 乘法`（1 拍） | ✓ |
| S_MAC_MUL3 | `v_sum_r <= 加法树`；`acc_wa_q <= acc_waddr_mac`（写回地址，多拍窗口） | 8 项加法树 | ✓ ★ |
| S_MAC_ACC | `acc_local <= acc_next`；末 tap 写 `acc[acc_wa_q]`；`mac_t/row/col` 推进 | `v_sum_r → 32-bit 加法 → acc_local/acc 写数据`（256-bit 写数据 fanout 到 128 块） | ✓；▲ P3-6（写数据 fanout 观察） |

### requant（11 级流水）

| 拍 | 操作 | 组合链 | 状态 |
|---|---|---|---|
| S_REQ_ADDR | `acc_q <= acc[acc_raddr_r]`（★ 已打拍）；`o_valid<=0` | — | ✓ |
| S_REQ_MUL | `rq_bias_m/rq_mult_m/v_rcl6_l <= q_*`（RAM 读打拍） | `q_*` 每拍无条件采样本身：`wren_reg → pass-through mux → q_* D`（1 拍窗口） | ▲ P2-4（bias/mult 4 块 M10K，临界 5-7ns，未冒头） |
| S_REQ_MULB | `v_biased_l <= acc_q + rq_bias_m` | 32-bit 加法 | ✓ ★ |
| S_REQ_MULC | `v_act_l <= relu/rcl6 比较` | 比较+mux | ✓ ★ |
| S_REQ_MUL2 | `v_p_* <= DSP 乘法`；`v_shift_l <= rq_shift_q` | `v_act_l/rq_mult_m → 4×16×16 DSP` | ✓ ★ |
| S_REQ_MUL3/4 | 64-bit 加法树两级 | 各 1 个 64-bit 加法 | ✓ |
| S_REQ_OUT/ROUND2/OUT2 | 桶形移位 / 64-bit 加法 / 桶形移位 | 各一拍 | ✓ |
| S_REQ_OUT3 | 饱和 → `o_data`；`rq_row/rq_col` 推进（829-842） | 推进拍后 acc 清零地址 1 拍窗口 | ▲ P1（同 S_ACC_CLR） |

## 三、隐患清单（按优先级）

| 编号 | 路径 | 预估 | 解法 |
|---|---|---|---|
| P1 | `rq_row → acc_waddr_clr → acc porta`（清零写，S_REQ_OUT3/S_ACC_CLR 两处推进拍后 1 拍窗口） | -4 级 | 状态机两个推进分支显式寄存"推进后地址" `acc_clr_wa`（S_ACC_CLR 清零改 `acc[acc_clr_wa]`；首轮 rq=0,0 复位即 0） |
| P2-2 | `load_row → lb_waddr → lb 写口`（S_LOAD 每拍写） | -1~-3 | lb 写地址打拍（写数据同步打拍） |
| P2-3 | `mac_t → 查表 kh/kw → mac_r/mac_c → lb_raddr → lb_addr_r`（S_MAC_ADDR 1 拍窗口） | -1~-3 | `mac_t+1` 的 kh/kw 在 S_MAC_MUL3 拍提前寄存（省查表段） |
| P2-4 | `wren_reg → pass-through → q_* D`（bias/mult/rcl6 读，每拍无条件采样） | 0~-2 | 读地址 `rq_raddr` 打拍（省 o_group 段）；wren 段仅能靠布局。条件采样无效（wren_reg 每拍更新，窗口恒 1 拍） |
| P3-5 | `lb_addr_r → lb_q`（M10K 读 1 拍） | 0~-1 | 观察（当前收敛） |
| P3-6 | `acc_wr_next 256-bit → acc 写数据`（128 块 fanout） | 0~-1 | 观察 |

## 四、已验证有效的修复模式（复用）

1. **RAM 读 → 下游运算打拍**：`rq_mult_m`（DSP 乘法）、`rq_bias_m`（加法）、`mac_a_q`（MAC DSP）、`v_rcl6_l/v_shift_l`——**断 pass-through 读路径与运算的组合穿透**。
2. **地址寄存器 preserve**：`acc_addr_mac_r`/`acc_raddr_r`/`acc_wa_q`——**阻止 Quartus 吸收进 M10K 地址寄存器**（输入侧 preserve 安全，不触发 276003）。
3. **采样窗口加宽**：`w_q` 仅 S_MAC_RD 拍采样（mac_t 更新后 2 拍窗口）。
4. **计数器替代除法**：`wf_lane/wf_t`（`wf_cnt/9` 除法器 -53ns 教训）。
5. **比较标志提前**：`mac_first_q/mac_last_q`（32-bit 比较拆出累加拍）。

## 五、教训（避免重蹈）

- `wf_cnt/9` 32-bit 组合除法器 → -53ns（Quartus 不优化恒 0 高位）。
- `acc_wa_q` 清零打拍 → 仿真 fail：`rq_row` 在 S_REQ_OUT3/S_ACC_CLR **两处**推进，打拍采样与推进同拍错位——**清零地址必须随推进分支显式寄存，不能"每拍采样"**。
- preserve 只用于输入侧地址寄存器；读输出寄存器 preserve 会破坏 RAM 推断（Error 276003）。
- 每次改完必跑 `tb_cnn_core_v2`（verilator 对拍），流水级数变化 tb 不关心（按事件对拍）。
