//=============================================================================
// cnn_core_v2 — 黑盒式行块驻留卷积执行器（阶段 4 核心）
//=============================================================================// 与阶段 3 的 cnn_core 相比（黑盒真实架构对齐）：
//   1) 输入流改为 64-bit NHWC8 块序：[C/8][H][W][8]，DDR 顺序 burst 读；
//      pad 行由本模块补 0（不拉 i_ready，DMA 自动跳过）。
//   2) 行缓冲按输入通道块组织：[in_cb][in_row_tile][W][8]（行块驻留）。
//   3) 计算序 = 输出通道组 × 输入通道组 × 窗口（对齐 tools/ref_cnn_top.py）：
//        for cb_out: acc 清零
//          for cb_in (dw 只 cb_in==cb_out): 读 slice → 全行块窗口 MAC 累加
//          全部输入组完成 → requant → 输出事件流（64-bit NHWC8 块序）
//   4) 单行块执行：行块循环/地址重定位在 cnn_top 层。本行块的输入 base
//      行号（= rb*out_row_tile*stride - pad）经 cfg 标量 15 写入。
//
// 可综合性（2025 同步读改造，解决 quartus_map OOM）：
//   - lb 声明为 64-bit 字数组 [cb][row][col]（每字 = 8 lane），S_LOAD 每拍
//     写一字；S_MAC 拆成 S_MAC_RD（同步采样 lb_q）/S_MAC_ACC（乘加累加）
//     两拍流水，使 lb 可被 Quartus 推断为 M10K（原组合读会被展开成
//     寄存器堆 + 读端口 mux，A&S 内存爆掉）。
//   - acc 声明为 256-bit 字数组 [o_row][o_col]（每字 = 8 lane × int32），
//     窗口内只在首 tap 读回 acc_q、寄存器内累加、末 tap 一次性写回；
//     requant 拆 S_REQ_ADDR（采样 acc_q）/S_REQ_OUT（组合 requant）两拍。
//   - 性能：S_MAC/S_REQUANT 每拍流变两拍（tap 率/事件率减半），功能
//     bit-exact（tb 仅按握手比对事件序列，不测周期数）。
//
// 接口（供 cnn_top 的 DMA 层驱动）：
//   cfg：sel=0 标量（addr 索引见下）、sel=1..4 requant 数组（addr=通道）
//   i_stream / iw_stream：64-bit 握手（DMA 连续读，地址 = 已发字节数）
//   o_stream：64-bit 输出事件（8 通道/拍，NHWC8 块序）
//=============================================================================

//-----------------------------------------------------------------------------
// 16×16 乘法单元（模块级 multstyle=dsp：Quartus 对"数组元素上的 multstyle
// 属性"推断不可靠，模块级属性 + 显式例化 100% 走 DSP 18×18，~3-4ns）。
// A_SIGNED=1 时 a 按有符号解释（hilo/hihi 的 a_hi），否则零扩展（lolo/lohi）。
// b 恒为无符号（m_lo/m_hi）。乘积统一 17×17 signed，赋给 32-bit 目标时
// 低 32 位即正确补码（lolo/lohi ≤ 2^32-1、hilo/hihi ∈ [-2^31, 2^31-1]）。
//
// a_signed 必须是参数而非运行时信号：Quartus 对带符号选择 mux 的乘法器
// 无法确定符号，每个例化被实现为 2 个乘法器（占满 1 个 Two Independent
// 18x18 块）——实测 DSP 从预期 40 涨到 80/112（71%），与 M10K 78% 一起
// 造成局部布线拥塞（fit 失败）。参数化后符号在综合时确定，每例化 1 个
// 18×18 且可打包，DSP 块数减半，时序不变（仍走 DSP）。
//-----------------------------------------------------------------------------
(* multstyle = "dsp" *) module mul16x16_dsp #(
    parameter A_SIGNED = 0    // 1: a 按有符号解释（hilo/hihi）；0: 零扩展（lolo/lohi）
) (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [31:0] p
);
    wire signed [16:0] sa = A_SIGNED ? $signed(a) : $signed({1'b0, a});
    wire signed [16:0] sb = $signed({1'b0, b});
    assign p = sa * sb;
endmodule

module cnn_core_v2 #(
    parameter G_MAX_IN_CB  = 1,    // 输入通道块流式驻留（每 o_group 轮重读 1 cb）
    parameter G_MAX_OUT_CB = 4,    // 最大输出通道块（模型 out_c<=32）
    parameter G_MAX_IN_ROWS= 41,   // 最大输入行块高度（模型 max in_tile=39）
    parameter G_MAX_OROWS  = 20,   // 最大输出行块高度（模型 max tile=19）
    parameter G_MAX_W      = 512,  // 输入行缓冲最大列数（2 的幂：lb 地址乘法变移位，150MHz 收敛；软件实际 ≤302）
    parameter G_MAX_OW     = 150,  // 最大输出宽度（OUTPUT_MAX_W）
    parameter G_MAX_C      = 1024  // 最大输出通道数（requant 数组容量，模型 out_c 最大 1024）
)(
    input  wire                clk,
    input  wire                rst_n,

    // ---- cfg 参数加载 ----
    input  wire                cfg_we,
    input  wire [2:0]          cfg_sel,
    input  wire [19:0]         cfg_addr,
    input  wire [31:0]         cfg_wdata,

    // ---- 启动（单行块；o_done 后由 cnn_top 重定位下一行块）----
    input  wire                start,

    // ---- 输入流（64-bit NHWC8 块序，signed int8×8）----
    input  wire                i_valid,
    output wire                i_ready,
    input  wire [63:0]         i_data,

    // ---- 权重流（64-bit，slice 序 [cb_out][cb_in][8][9][8] 连续）----
    input  wire                iw_valid,
    output wire                ow_ready,
    input  wire [63:0]         iw_data,

    // ---- 输出流（64-bit，NHWC8 块序：行→列→8 通道）----
    output reg                 o_valid,
    input  wire                o_ready,
    output reg  [63:0]         o_data,

    // ---- 单行块完成 ----
    output reg                 o_done,

    // ---- 观测输出（调试寄存器用，组合直读 core 内部状态/数据；不带 QSys 接口）----
    output wire [31:0]  dbg_ptr0,   // {state[4:0], o_group[10:0], i_group[10:0], mac_t[3:0]}
    output wire [31:0]  dbg_ptr1,   // {rq_row[15:0], rq_col[15:0]}
    output wire [31:0]  dbg_ptr2,   // {mac_row[15:0], mac_col[15:0]}
    output wire [31:0]  dbg_ptr3,   // {load_row[9:0], load_col[9:0], wf_cnt[7:0]}
    output wire [127:0] dbg_data0,  // {acc_q[31:0], v_biased_l[0], v_act_l[0], v_rq64_l[0][31:0]}
    output wire [127:0] dbg_data1,  // {v_round_l[0], v_shifted[0], v_rnd_delta[0], w_q[0][3:0]}
    output wire [31:0]  dbg_lb      // lb_q[31:0]
);

    //-----------------------------------------------------------------------
    // 运行时标量寄存器（cfg_sel=0，cfg_addr=索引，cfg_wdata=值）
    //  0:type 1:act 2:in_c 3:in_h 4:in_w 5:out_c 6:out_h 7:out_w
    //  8:k 9:pad 10:stride 11:out_row_tile 12:in_row_tile
    //  13:in_cb 14:out_cb 15:base_row（本行块输入 base 行号）
    //-----------------------------------------------------------------------
    reg [3:0]  type_reg, act_reg;
    reg [31:0] in_h_reg,  in_w_reg;
    reg [31:0] out_c_reg, out_w_reg;
    reg [31:0] k_reg, pad_reg, stride_reg;
    reg [31:0] out_row_tile_reg, in_row_tile_reg;
    reg [31:0] in_cb_reg, out_cb_reg;
    reg signed [31:0] base_row_reg;

    // requant 参数存储（cfg_sel 1/2/3/4 = bias/mult/shift/rcl6，addr=输出通道）
    // 见下方 generate 块 g_rq_param：8 lane 并行读 8 个不同通道地址，
    // 单端口 RAM 每拍只能 1 地址 → 每 lane 独立一份 M10K（8×4 组）

    integer i;
    always @(posedge clk) begin
        if (cfg_we) begin
            case (cfg_sel)
                3'd0: begin
                    case (cfg_addr)
                        20'd0: type_reg      <= cfg_wdata[3:0];
                        20'd1: act_reg       <= cfg_wdata[1:0];
                        20'd3: in_h_reg      <= cfg_wdata;
                        20'd4: in_w_reg      <= cfg_wdata;
                        20'd5: out_c_reg     <= cfg_wdata;
                        20'd7: out_w_reg     <= cfg_wdata;
                        20'd8: k_reg         <= cfg_wdata;
                        20'd9: pad_reg       <= cfg_wdata;
                        20'd10: stride_reg   <= cfg_wdata;
                        20'd11: out_row_tile_reg <= cfg_wdata;
                        20'd12: in_row_tile_reg  <= cfg_wdata;
                        20'd13: in_cb_reg     <= cfg_wdata;
                        20'd14: out_cb_reg    <= cfg_wdata;
                        20'd15: base_row_reg  <= cfg_wdata;
                        default: ;
                    endcase
                end
                default: ;
            endcase
        end
    end

    //-----------------------------------------------------------------------
    // requant 参数存储：8 lane 独立 M10K（cfg_sel 1/2/3/4 = bias/mult/shift/rcl6，
    // addr=输出通道）。8 lane 并行读 8 个不同通道地址（v_out_ch = o_group*8+ln），
    // 单端口 RAM 每拍只能 1 地址 → 每 lane 一份（8×4 组，约 104 块 M10K）。
    // M10K 同步读 1 拍延迟：bias/rcl6 于 S_REQ_ADDR 发起（S_REQ_MUL 用）、
    // mult 于 S_REQ_MUL 发起（S_REQ_MUL2 用）、shift 于 S_REQ_MUL2 发起
    // （S_REQ_OUT 用）——正好插入现有 4 拍 requant 流水，事件序列不变。
    // 读数据经 q_* 输出到模块顶层 wire 数组（generate 实例内声明的对象
    // 作用域只在实例内，主逻辑引用不到）。
    //-----------------------------------------------------------------------
    wire signed [31:0] rq_bias_q  [0:7];
    wire [31:0]        rq_mult_q  [0:7];
    wire [7:0]         rq_shift_q [0:7];
    wire signed [31:0] rq_rcl6_q  [0:7];
    // requant 参数存储：8 lane 显式展开（移出 generate）。
    // Quartus 对 generate 块内数组的 M10K 推断不可靠——实测（map.rpt）仅靠
    // RAM 恢复救回 16/32 个（g_rq_param 奇数实例），偶数实例数组照旧展开成
    // 寄存器（ALM 爆）。lb/acc 在模块顶层且全部推断成功，故此处与它们
    // 完全同构：顶层数组 + 每数组独立写 always + 读地址打拍 + 无条件采样。
    // 每 lane 读地址 = o_group*8 + lane（10-bit，o_group ≤ 127；写地址
    // cfg_addr[9:0] 截断，值域 ≤1023 与 1024 深匹配）。
    wire [9:0] rq_raddr = {o_group[6:0], 3'd0};   // 组合地址，直接驱动读端口（链短）
    // 注意：不可打拍（rq_raddr_r）——S_REQ_ADDR 拍采新组地址、S_REQ_MUL 拍
    // 就要用 q_bias；打拍 + M10K 同步读 = 2 拍延迟，bias/rcl6 错 1 组
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store_0 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store_0 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store_0 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg signed [31:0] rcl6_store_0 [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store_0[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store_0[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store_0[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd4 && cfg_addr < G_MAX_C)
            rcl6_store_0[cfg_addr[9:0]] <= cfg_wdata;
    end
    reg signed [31:0] q_bias_0;
    reg [31:0] q_mult_0;
    reg [7:0]  q_shift_0;
    reg signed [31:0] q_rcl6_0;
    always @(posedge clk) begin
        q_bias_0  <= bias_store_0[rq_raddr];
        q_rcl6_0  <= rcl6_store_0[rq_raddr];
        q_mult_0  <= rq_m_store_0[rq_raddr];
        q_shift_0 <= rq_r_store_0[rq_raddr];
    end
    assign rq_bias_q[0]  = q_bias_0;
    assign rq_mult_q[0]  = q_mult_0;
    assign rq_shift_q[0] = q_shift_0;
    assign rq_rcl6_q[0]  = q_rcl6_0;
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store_1 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store_1 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store_1 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg signed [31:0] rcl6_store_1 [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store_1[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store_1[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store_1[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd4 && cfg_addr < G_MAX_C)
            rcl6_store_1[cfg_addr[9:0]] <= cfg_wdata;
    end
    reg signed [31:0] q_bias_1;
    reg [31:0] q_mult_1;
    reg [7:0]  q_shift_1;
    reg signed [31:0] q_rcl6_1;
    always @(posedge clk) begin
        q_bias_1  <= bias_store_1[rq_raddr + 9'd1];
        q_rcl6_1  <= rcl6_store_1[rq_raddr + 9'd1];
        q_mult_1  <= rq_m_store_1[rq_raddr + 9'd1];
        q_shift_1 <= rq_r_store_1[rq_raddr + 9'd1];
    end
    assign rq_bias_q[1]  = q_bias_1;
    assign rq_mult_q[1]  = q_mult_1;
    assign rq_shift_q[1] = q_shift_1;
    assign rq_rcl6_q[1]  = q_rcl6_1;
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store_2 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store_2 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store_2 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg signed [31:0] rcl6_store_2 [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store_2[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store_2[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store_2[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd4 && cfg_addr < G_MAX_C)
            rcl6_store_2[cfg_addr[9:0]] <= cfg_wdata;
    end
    reg signed [31:0] q_bias_2;
    reg [31:0] q_mult_2;
    reg [7:0]  q_shift_2;
    reg signed [31:0] q_rcl6_2;
    always @(posedge clk) begin
        q_bias_2  <= bias_store_2[rq_raddr + 9'd2];
        q_rcl6_2  <= rcl6_store_2[rq_raddr + 9'd2];
        q_mult_2  <= rq_m_store_2[rq_raddr + 9'd2];
        q_shift_2 <= rq_r_store_2[rq_raddr + 9'd2];
    end
    assign rq_bias_q[2]  = q_bias_2;
    assign rq_mult_q[2]  = q_mult_2;
    assign rq_shift_q[2] = q_shift_2;
    assign rq_rcl6_q[2]  = q_rcl6_2;
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store_3 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store_3 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store_3 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg signed [31:0] rcl6_store_3 [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store_3[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store_3[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store_3[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd4 && cfg_addr < G_MAX_C)
            rcl6_store_3[cfg_addr[9:0]] <= cfg_wdata;
    end
    reg signed [31:0] q_bias_3;
    reg [31:0] q_mult_3;
    reg [7:0]  q_shift_3;
    reg signed [31:0] q_rcl6_3;
    always @(posedge clk) begin
        q_bias_3  <= bias_store_3[rq_raddr + 9'd3];
        q_rcl6_3  <= rcl6_store_3[rq_raddr + 9'd3];
        q_mult_3  <= rq_m_store_3[rq_raddr + 9'd3];
        q_shift_3 <= rq_r_store_3[rq_raddr + 9'd3];
    end
    assign rq_bias_q[3]  = q_bias_3;
    assign rq_mult_q[3]  = q_mult_3;
    assign rq_shift_q[3] = q_shift_3;
    assign rq_rcl6_q[3]  = q_rcl6_3;
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store_4 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store_4 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store_4 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg signed [31:0] rcl6_store_4 [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store_4[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store_4[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store_4[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd4 && cfg_addr < G_MAX_C)
            rcl6_store_4[cfg_addr[9:0]] <= cfg_wdata;
    end
    reg signed [31:0] q_bias_4;
    reg [31:0] q_mult_4;
    reg [7:0]  q_shift_4;
    reg signed [31:0] q_rcl6_4;
    always @(posedge clk) begin
        q_bias_4  <= bias_store_4[rq_raddr + 9'd4];
        q_rcl6_4  <= rcl6_store_4[rq_raddr + 9'd4];
        q_mult_4  <= rq_m_store_4[rq_raddr + 9'd4];
        q_shift_4 <= rq_r_store_4[rq_raddr + 9'd4];
    end
    assign rq_bias_q[4]  = q_bias_4;
    assign rq_mult_q[4]  = q_mult_4;
    assign rq_shift_q[4] = q_shift_4;
    assign rq_rcl6_q[4]  = q_rcl6_4;
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store_5 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store_5 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store_5 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg signed [31:0] rcl6_store_5 [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store_5[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store_5[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store_5[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd4 && cfg_addr < G_MAX_C)
            rcl6_store_5[cfg_addr[9:0]] <= cfg_wdata;
    end
    reg signed [31:0] q_bias_5;
    reg [31:0] q_mult_5;
    reg [7:0]  q_shift_5;
    reg signed [31:0] q_rcl6_5;
    always @(posedge clk) begin
        q_bias_5  <= bias_store_5[rq_raddr + 9'd5];
        q_rcl6_5  <= rcl6_store_5[rq_raddr + 9'd5];
        q_mult_5  <= rq_m_store_5[rq_raddr + 9'd5];
        q_shift_5 <= rq_r_store_5[rq_raddr + 9'd5];
    end
    assign rq_bias_q[5]  = q_bias_5;
    assign rq_mult_q[5]  = q_mult_5;
    assign rq_shift_q[5] = q_shift_5;
    assign rq_rcl6_q[5]  = q_rcl6_5;
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store_6 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store_6 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store_6 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg signed [31:0] rcl6_store_6 [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store_6[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store_6[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store_6[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd4 && cfg_addr < G_MAX_C)
            rcl6_store_6[cfg_addr[9:0]] <= cfg_wdata;
    end
    reg signed [31:0] q_bias_6;
    reg [31:0] q_mult_6;
    reg [7:0]  q_shift_6;
    reg signed [31:0] q_rcl6_6;
    always @(posedge clk) begin
        q_bias_6  <= bias_store_6[rq_raddr + 9'd6];
        q_rcl6_6  <= rcl6_store_6[rq_raddr + 9'd6];
        q_mult_6  <= rq_m_store_6[rq_raddr + 9'd6];
        q_shift_6 <= rq_r_store_6[rq_raddr + 9'd6];
    end
    assign rq_bias_q[6]  = q_bias_6;
    assign rq_mult_q[6]  = q_mult_6;
    assign rq_shift_q[6] = q_shift_6;
    assign rq_rcl6_q[6]  = q_rcl6_6;
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store_7 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store_7 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store_7 [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg signed [31:0] rcl6_store_7 [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store_7[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store_7[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store_7[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd4 && cfg_addr < G_MAX_C)
            rcl6_store_7[cfg_addr[9:0]] <= cfg_wdata;
    end
    reg signed [31:0] q_bias_7;
    reg [31:0] q_mult_7;
    reg [7:0]  q_shift_7;
    reg signed [31:0] q_rcl6_7;
    always @(posedge clk) begin
        q_bias_7  <= bias_store_7[rq_raddr + 9'd7];
        q_rcl6_7  <= rcl6_store_7[rq_raddr + 9'd7];
        q_mult_7  <= rq_m_store_7[rq_raddr + 9'd7];
        q_shift_7 <= rq_r_store_7[rq_raddr + 9'd7];
    end
    assign rq_bias_q[7]  = q_bias_7;
    assign rq_mult_q[7]  = q_mult_7;
    assign rq_shift_q[7] = q_shift_7;
    assign rq_rcl6_q[7]  = q_rcl6_7;

    //-----------------------------------------------------------------------
    // 存储：行缓冲 lb（64-bit 字 = 8 lane，col 0..G_MAX_W-1）
    //       acc（256-bit 字 = 8 lane × int32）
    //       wbuf [lane*72 + t*8 + m]（权重 slice，小数组保持寄存器）
    // 强制 ramstyle = M10K 且展平为单维数组（Quartus 对多维 unpacked 数组
    // 的 RAM 推断不可靠：lb[cb][row][col] 三维 + acc 256-bit 超宽会被展开成
    // 寄存器堆 + 读端口 mux，A&S 内存爆到 40GB+；单维数组 + 常量乘法拼接
    // 地址是官方推荐的可靠推断写法，功能/时序完全不变）。
    //-----------------------------------------------------------------------
    (* ramstyle = "M10K, no_rw_check" *) reg [63:0] lb [0:G_MAX_IN_ROWS*G_MAX_W-1];
    (* ramstyle = "M10K, no_rw_check" *) reg signed [255:0] acc [0:G_MAX_OROWS*G_MAX_OW-1];

    // 单维索引拼接（常量乘法综合时折叠成移位/加法，不产生逻辑）
    wire [31:0] lb_waddr = load_row*G_MAX_W + load_col;
    wire [31:0] lb_raddr = mac_r*G_MAX_W + mac_c_cl;
    wire [31:0] acc_waddr_clr = rq_row*G_MAX_OW + rq_col;
    wire [31:0] acc_waddr_mac = mac_row*G_MAX_OW + mac_col;
    wire [31:0] acc_raddr_clr = rq_row*G_MAX_OW + rq_col;
    wire [31:0] acc_raddr_mac = mac_row*G_MAX_OW + mac_col;
    reg signed [7:0] wbuf [0:7][0:7][0:8];   // [lane][m][t]：w_q 读 = 9:1 mux（原 [lane*72+t*8+m] 平铺是 72:1，mac_t→w_q 路径 18.95ns）

    // MAC 8×8 乘法器：拆成独立子模块（模块级 multstyle 强制 DSP/LUT 分配），
    // lane 0-3 走 DSP（32 个 × ≤2 block = ≤64）、lane 4-7 走 LUT（32 个 ≈2K ALUT）。
    // 乘法器与加法树物理隔离，避免 Quartus 重新融合成 mult_hlmac（每乘加 2 block）。
    wire signed [15:0] mac_p [0:7][0:7];  // signed：负积需符号扩展进加法树

    // w_q：权重读寄存器（S_MAC_RD 拍与 lb_q 同步采样，拆掉 mac_t→wbuf 读 mux）
    reg [7:0] w_q [0:7][0:7];
    // mac_p_r：乘法结果寄存器（S_MAC_MUL 拍采样），把 lb M10K 输出→乘法→
    // 加法树→v_sum_r 的组合链再拆一段（4 拍/tap：RD/MUL/MUL2/ACC）
    reg signed [15:0] mac_p_r [0:7][0:7];
    // mac_c_valid_r：S_MAC_RD 拍寄存的 tap 列有效位。必须在使用（generate 实例
    // .en 端口）之前声明为 reg，否则 Verilog 隐式声明为 wire，实例 en 悬空（z）
    reg mac_c_valid_r;
    genvar mac_lane_i, mac_m_i;
    generate
        for (mac_lane_i = 0; mac_lane_i < 8; mac_lane_i = mac_lane_i + 1) begin : mac_lane_g
            for (mac_m_i = 0; mac_m_i < 8; mac_m_i = mac_m_i + 1) begin : mac_mul_g
                if (mac_lane_i < 4) begin : u_dsp
                    mac8x8_dsp u_mac (
                        .en(mac_c_valid_r),
                        .a (lb_q[8*mac_m_i +: 8]),
                        .b (w_q[mac_lane_i][mac_m_i]),
                        .p (mac_p[mac_lane_i][mac_m_i])
                    );
                end else begin : u_lut
                    mac8x8_lut u_mac (
                        .en(mac_c_valid_r),
                        .a (lb_q[8*mac_m_i +: 8]),
                        .b (w_q[mac_lane_i][mac_m_i]),
                        .p (mac_p[mac_lane_i][mac_m_i])
                    );
                end
            end
        end
    endgenerate

    // 同步读采样寄存器（RAM 读输出，读地址 = 组合函数，晚一拍有效）
    reg [63:0]          lb_q;
    reg signed [255:0]  acc_q;
    reg signed [255:0]  acc_local;   // 窗口内累加（首 tap 用 acc_q 初始化）
    // 读地址寄存（S_MAC_ADDR 拍采样）：lb_raddr/acc_raddr_mac 是 32-bit 常量乘法
    // 组合链（×41*302 / ×302），直通 M10K 地址端口会超 50MHz 周期（setup 违例
    // -0.839ns 主因）；打一拍后组合链终点变为普通寄存器，M10K 地址由寄存器驱动。
    // 最大地址 = 4*41*302-1 = 49527（lb）、19*150+149 = 2999（acc），16-bit 够。
    reg [15:0]          lb_addr_r;
    reg [15:0]          acc_addr_mac_r;

    //-----------------------------------------------------------------------
    // 状态机
    //-----------------------------------------------------------------------
    localparam S_IDLE     = 4'd0;
    localparam S_LOAD     = 4'd1;   // 装载输入行块
    localparam S_ACC_CLR  = 4'd2;   // acc 清零（每输出组）
    localparam S_WEIGHT   = 4'd3;   // 装载权重 slice（72 拍）
    localparam S_MAC_ADDR = 4'd13;  // 窗口 tap 地址拍（寄存 lb/acc 读地址，断 32-bit 乘法组合链）
    localparam S_MAC_RD   = 4'd4;   // 窗口 tap 请求拍（同步采样 lb_q/acc_q）
    localparam S_MAC_MUL  = 4'd11;  // 窗口 tap 乘拍（乘法器 → mac_p_r）
    localparam S_MAC_MUL2 = 4'd12;  // 窗口 tap 加法树拍（mac_p_r → v_sum_r）
    localparam S_MAC_ACC  = 4'd5;   // 窗口 tap 累加拍（v_sum_r → acc_local）
    localparam S_REQ_ADDR = 4'd6;   // requant 请求拍（采样 acc_q）
    localparam S_REQ_MUL  = 4'd7;   // requant 乘加拍级 1（acc_q+bias，寄存 v_biased_l）
    localparam S_REQ_MULB = 5'd19;  // requant 乘加拍级 2（bias 加法 → v_biased_l）
    localparam S_REQ_MULC = 5'd20;  // requant 乘加拍级 3（relu/rcl6 比较 → v_act_l）
    localparam S_REQ_MUL2 = 4'd8;   // requant 乘法拍级 1（4×16×16 DSP 部分积，寄存 v_p_*/v_shift_l）
    localparam S_REQ_MUL3 = 4'd15;  // requant 乘法拍级 2（两组中间和 → v_sum_lo/v_sum_hi）
    localparam S_REQ_MUL4 = 5'd16;  // requant 乘法拍级 3（中间和相加 → v_rq64_l，每级仅 1 个 64-bit 加法）
    localparam S_REQ_OUT  = 4'd9;   // requant round 拍级 1（round 桶形移位 → v_rnd_delta）
    localparam S_REQ_ROUND2 = 5'd17; // requant round 拍级 2（v_rq64_l + v_rnd_delta → v_round_l）
    localparam S_REQ_OUT2 = 4'd14;  // requant 输出移位拍（v_round_l >>> shift → v_shifted）
    localparam S_REQ_OUT3 = 5'd18;  // requant 输出拍（饱和 → o_data，o_valid 拉高）
    localparam S_DONE     = 4'd10;

    reg [4:0] state;
    reg [31:0] load_row, load_col;
    reg        load_first;        // 本轮装载是 o_group 首 cb（完成后需清 acc）
    reg [31:0] wf_cnt;
    reg [2:0]  wf_lane;   // wbuf 写 lane 计数（k=3：wf_cnt/9；k=1：wf_cnt），替代除法器
    reg [3:0]  wf_t;      // wbuf 写 tap 计数（k=3：wf_cnt%9；k=1：恒 0）
    reg [31:0] o_group, i_group;
    reg [31:0] mac_row, mac_col, mac_t;
    reg        mac_first_q, mac_last_q;   // S_MAC_MUL2 拍寄存首/末 tap 标志（拆 mac_t 32-bit 比较链）
    reg [31:0] rq_row, rq_col;

    // 行缓冲行 r 对应输入行 base_row_reg + r；有效 = 输入行 ∈ [0, in_h)
    wire signed [31:0] load_in_row = base_row_reg + load_row;
    wire load_row_valid = (load_in_row >= 0) && (load_in_row < in_h_reg);

    //-----------------------------------------------------------------------
    // 主状态机
    //-----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            o_done <= 1'b0;
            o_valid <= 1'b0;
            load_row <= 0; load_col <= 0;
            load_first <= 1'b1;
            wf_cnt <= 0; wf_lane <= 0; wf_t <= 0; o_group <= 0; i_group <= 0;
            mac_row <= 0; mac_col <= 0; mac_t <= 0;
            rq_row <= 0; rq_col <= 0;
            mac_c_valid_r <= 1'b0;
            acc_local <= 256'sd0;   // 复位归并到主状态机（单一驱动，避免 Quartus 10028）
            // 地址寄存器复位同样归并到主状态机（S_MAC_ADDR 分支在同一块）：
            // lb_q/acc_q 每拍无条件采样，无复位时 x 地址越界读会污染 acc_q
            lb_addr_r <= 16'd0;
            acc_addr_mac_r <= 16'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    o_done <= 1'b0;
                    if (start) begin
                        load_row <= 0; load_col <= 0;
            load_first <= 1'b1;
                        o_group <= 0; i_group <= 0;
                        state <= S_LOAD;
                    end
                end

                //---- 装载输入行块（流式：每轮 1 个通道块，cb 选择由顶层 DMA 轮控制）----
                S_LOAD: begin
                    // 多 o_group 时从 S_REQ_OUT3 直接进入：必须拉低 o_valid，
                    // 否则残留 1 → 输出持续被收集（o_ready=1 恒等）→ 失控
                    // （单组层走 S_DONE 拉低故未暴露）
                    o_valid <= 1'b0;
                    if (load_row_valid) begin
                        if (i_valid) begin
                            lb[lb_waddr] <= i_data;
                            if (load_col == in_w_reg - 1) begin
                                load_col <= 0;
                                if (load_row == in_row_tile_reg - 1) begin
                                    load_row <= 0;
                                    // 1 cb 完成：o_group 首 cb（load_first）→ 清 acc 开新组；
                                    // i_group 循环 → 直接装权重 slice（acc 保留部分和）
                                    if (load_first) begin
                                        load_first <= 0;
                                        rq_row <= 0; rq_col <= 0;
                                        state <= S_ACC_CLR;
                                    end else
                                        state <= S_WEIGHT;
                                end else
                                    load_row <= load_row + 1;
                            end else
                                load_col <= load_col + 1;
                        end
                    end else begin
                        // pad 行：写 0 并跳过（不消费输入；BRAM 上电不定须清零）
                        lb[lb_waddr] <= 64'h0;
                        if (load_col == in_w_reg - 1) begin
                            load_col <= 0;
                            if (load_row == in_row_tile_reg - 1) begin
                                load_row <= 0;
                                if (load_first) begin
                                    load_first <= 0;
                                    rq_row <= 0; rq_col <= 0;
                                    state <= S_ACC_CLR;
                                end else
                                    state <= S_WEIGHT;
                            end else
                                load_row <= load_row + 1;
                        end else
                            load_col <= load_col + 1;
                    end
                end

                //---- acc 清零（每输出组开始，逐 256-bit 字）----
                S_ACC_CLR: begin
                    o_valid <= 1'b0;
                    acc[acc_waddr_clr] <= 256'sd0;
                    if (rq_row == out_row_tile_reg - 1 && rq_col == out_w_reg - 1) begin
                        rq_row <= 0; rq_col <= 0;
                        i_group <= 0;
                        state <= S_WEIGHT;
                    end else if (rq_col == out_w_reg - 1) begin
                        rq_col <= 0;
                        rq_row <= rq_row + 1;
                    end else
                        rq_col <= rq_col + 1;
                end

                //---- 装载权重 slice（k=3：72 拍 [lane][9tap][8]；k=1：8 拍，lane 主序跳 72B 到 t=0 组）----
                // 软件布局（conv2d_weight_reorganize）：k=3 slice = 8×9×8B，k=1 slice = 8×1×8B
                S_WEIGHT: begin
                    if (iw_valid) begin
                        // wbuf[lane][m][t] 写：wf_lane/wf_t 同步计数（wf_cnt/9、%9 除法器
                        // 时序爆炸 -53ns，改用计数器：k=3 t 主序递增；k=1 lane 主序）
                        if (k_reg == 1) begin
                            for (m = 0; m < 8; m = m + 1)
                                wbuf[wf_lane][m][0] <= iw_data[m*8 +: 8];
                            wf_lane <= wf_lane + 1;
                        end else begin
                            for (m = 0; m < 8; m = m + 1)
                                wbuf[wf_lane][m][wf_t] <= iw_data[m*8 +: 8];
                            if (wf_t == 4'd8) begin
                                wf_t <= 4'd0;
                                wf_lane <= wf_lane + 1;
                            end else
                                wf_t <= wf_t + 1;
                        end
                        if (wf_cnt == (k_reg == 1 ? 8*1 - 1 : 8*9 - 1)) begin
                            wf_cnt <= 0;
                            wf_lane <= 3'd0;
                            wf_t <= 4'd0;
                            mac_row <= 0; mac_col <= 0; mac_t <= 0;
                            state <= S_MAC_ADDR;
                        end else
                            wf_cnt <= wf_cnt + 1;
                    end
                end

                //---- 窗口 MAC（地址拍）：组合计算 lb/acc 读地址并寄存，
                //     断开 32-bit 常量乘法链直通 M10K 地址端口的长路径 ----
                S_MAC_ADDR: begin
                    lb_addr_r      <= lb_raddr[15:0];
                    acc_addr_mac_r <= acc_raddr_mac[15:0];
                    state <= S_MAC_RD;
                end

                //---- 窗口 MAC（请求拍）：地址已寄存，上升沿同步采样 lb_q/acc_q ----
                S_MAC_RD: begin
                    mac_c_valid_r <= mac_c_valid;   // 组合 valid 寄存（断 k_reg→乘加树链）
                    state <= S_MAC_MUL;
                end

                //---- 窗口 MAC（乘拍）：乘加树 → v_sum_r 寄存（拆开累加，缩短组合链）----
                S_MAC_MUL: begin
                    // 乘拍：采样乘法器输出（mac_p → mac_p_r）
                    for (lane = 0; lane < 8; lane = lane + 1)
                        for (m = 0; m < 8; m = m + 1)
                            mac_p_r[lane][m] <= mac_p[lane][m];
                    state <= S_MAC_MUL2;
                end
                S_MAC_MUL2: begin
                    // 加法树拍：mac_p_r（寄存器）→ 8 项加法树 → v_sum_r
                    for (lane = 0; lane < 8; lane = lane + 1)
                        v_sum[lane] =
                            mac_p_r[lane][0] + mac_p_r[lane][1] + mac_p_r[lane][2] + mac_p_r[lane][3] +
                            mac_p_r[lane][4] + mac_p_r[lane][5] + mac_p_r[lane][6] + mac_p_r[lane][7];
                    for (lane = 0; lane < 8; lane = lane + 1)
                        v_sum_r[lane] <= v_sum[lane];
                    // 首/末 tap 标志提前寄存（32-bit 比较拆出 S_MAC_ACC 拍）
                    mac_first_q <= (mac_t == 32'd0);
                    mac_last_q  <= (mac_t == (k_reg == 1 ? 32'd0 : 32'd8));
                    state <= S_MAC_ACC;
                end

                //---- 窗口 MAC（累加拍）：用上拍寄存的 v_sum_r 累加到 acc_local；
                //     首 tap（mac_t==0）以 acc_q 初始化；末 tap（k=3：mac_t==8；k=1：mac_t==0）写回 acc ----
                S_MAC_ACC: begin
                    // 8 lane 独立 32-bit 累加（acc_local 打包为 256-bit，但每个 lane
                    // 是独立 int32，无跨 lane 进位；整体 256-bit 加法器进位链过长，
                    // 拆成 lane 级加法消除，语义不变）
                    // 注意：必须先用组合变量拼成整字再整字写 acc_local/acc——
                    // 直接 part-select 写（acc[...][32*lane +: 32] <= ...）会阻止
                    // Quartus 把 acc 推断为 M10K，导致 A&S 内存反弹（10+GB）。
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if (mac_first_q)
                            acc_next[32*lane +: 32] = acc_q[32*lane +: 32] + v_sum_r[lane];
                        else
                            acc_next[32*lane +: 32] = acc_local[32*lane +: 32] + v_sum_r[lane];
                    end
                    acc_local <= acc_next;
                    if (mac_last_q) begin
                        // 末 tap：写回最终值（acc_local 尚缺本 tap 部分和；
                        // k=1 单 tap 时直接用 acc_q 累加，避免旧 acc_local 串扰）
                        for (lane = 0; lane < 8; lane = lane + 1)
                            acc_wr_next[32*lane +: 32] = (k_reg == 1) ?
                                (acc_q[32*lane +: 32] + v_sum_r[lane]) :
                                (acc_local[32*lane +: 32] + v_sum_r[lane]);
                        acc[acc_waddr_mac] <= acc_wr_next;
                        mac_t <= 0;
                        if (mac_col == out_w_reg - 1) begin
                            mac_col <= 0;
                            if (mac_row == out_row_tile_reg - 1) begin
                                mac_row <= 0;
                                if (type_reg == 4 || i_group == in_cb_reg - 1) begin
                                    rq_row <= 0; rq_col <= 0;
                                    state <= S_REQ_ADDR;
                                end else begin
                                    i_group <= i_group + 1;
                                    state <= S_LOAD;   // 流式：重装下一输入 cb（lb 单块驻留）
                                end
                            end else begin
                                mac_row <= mac_row + 1;
                                state <= S_MAC_ADDR;
                            end
                        end else begin
                            mac_col <= mac_col + 1;
                            state <= S_MAC_ADDR;
                        end
                    end else begin
                        mac_t <= mac_t + 1;
                        state <= S_MAC_ADDR;
                    end
                end

                //---- requant 请求拍：采样 acc[rq_row][rq_col] ----
                S_REQ_ADDR: begin
                    o_valid <= 1'b0;
                    state <= S_REQ_MUL;
                end

                //---- requant 乘加拍级 1：acc_q + bias（32-bit 加法单独一拍）----
                S_REQ_MUL: begin
                    state <= S_REQ_MULB;
                end

                //---- requant 乘加拍级 2：relu/rcl6 比较 → v_act_l（比较+mux 单独一拍）----
                S_REQ_MULB: begin
                    state <= S_REQ_MULC;
                end
                S_REQ_MULC: begin
                    state <= S_REQ_MUL2;
                end

                //---- requant 乘法拍级 1：16×16 部分积（4 个 DSP 乘法，见独立 always）----
                S_REQ_MUL2: begin
                    state <= S_REQ_MUL3;
                end

                //---- requant 乘法拍级 2：两组中间和（lolo+lohi、hilo+hihi 并行 64-bit 加法）----
                S_REQ_MUL3: begin
                    state <= S_REQ_MUL4;
                end

                //---- requant 乘法拍级 3：中间和相加 → v_rq64_l（见独立 always）----
                S_REQ_MUL4: begin
                    state <= S_REQ_OUT;
                end

                //---- requant round 拍级 1：round 桶形移位（1<<(shift-1)，单独一拍）----
                S_REQ_OUT: begin
                    state <= S_REQ_ROUND2;
                end

                //---- requant round 拍级 2：v_rq64_l + v_rnd_delta（64-bit 加法单独一拍）----
                S_REQ_ROUND2: begin
                    state <= S_REQ_OUT2;
                end

                //---- requant 输出移位拍：v_round_l >>> shift（桶形移位单独一拍）----
                S_REQ_OUT2: begin
                    state <= S_REQ_OUT3;
                end

                //---- requant 输出拍（饱和 → o_data；行→列→8 通道，块序）----
                S_REQ_OUT3: begin
                    o_valid <= 1'b1;
                    if (o_ready) begin
                        if (rq_col == out_w_reg - 1) begin
                            rq_col <= 0;
                            if (rq_row == out_row_tile_reg - 1) begin
                                rq_row <= 0;
                                // o_valid 由下一状态（S_ACC_CLR/S_DONE）拉低，
                                // 保证最后一事件 o_data 已被输出握手收走
                                if (o_group == out_cb_reg - 1) begin
                                    state <= S_DONE;
                                end else begin
                                    o_group <= o_group + 1;
                                    rq_row <= 0; rq_col <= 0;
                                    load_first <= 1'b1;
                                    state <= S_LOAD;   // 流式：重装新 o_group 的输入 cb
                                end
                            end else begin
                                rq_row <= rq_row + 1;
                                state <= S_REQ_ADDR;
                            end
                        end else begin
                            rq_col <= rq_col + 1;
                            state <= S_REQ_ADDR;
                        end
                    end
                end

                S_DONE: begin
                    o_valid <= 1'b0;
                    o_done <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    //-----------------------------------------------------------------------
    // 握手
    //-----------------------------------------------------------------------
    assign i_ready = (state == S_LOAD) && load_row_valid;
    assign ow_ready = (state == S_WEIGHT);

    //-----------------------------------------------------------------------
    // 组合乘加树（MAC）：lb_q[cb][r][c] × wbuf[lane][t][m] 累加
    //-----------------------------------------------------------------------
    // 窗口 tap 位置：k_reg ∈ {1,3}（model_profile 实测）、mac_t ∈ [0,8]，
    // 查表替代 32-bit 除法器（lpm_divide，组合链超长导致 setup 违例 -82ns）
    // （lb 已流式单 cb 驻留，不再有 mac_cb 通道块维度）
    reg [3:0] mac_kh, mac_kw;
    always @(*) begin
        if (k_reg == 3) begin
            case (mac_t[3:0])
                4'd0: begin mac_kh = 0; mac_kw = 0; end
                4'd1: begin mac_kh = 0; mac_kw = 1; end
                4'd2: begin mac_kh = 0; mac_kw = 2; end
                4'd3: begin mac_kh = 1; mac_kw = 0; end
                4'd4: begin mac_kh = 1; mac_kw = 1; end
                4'd5: begin mac_kh = 1; mac_kw = 2; end
                4'd6: begin mac_kh = 2; mac_kw = 0; end
                4'd7: begin mac_kh = 2; mac_kw = 1; end
                default: begin mac_kh = 2; mac_kw = 2; end
            endcase
        end else begin
            mac_kh = mac_t[3:0];   // k=1：kh=mac_t, kw=0（mac_t%1=0）
            mac_kw = 4'd0;
        end
    end
    // 行（lb 索引）：窗口第 kh 行 = o_row*stride + kh + pad（装载时行 0 = base 行）；
    // stride_reg ∈ {1,2}，移位替代乘法器（mac_row*stride_reg 会被综合成 32-bit 乘法）
    wire [31:0] mac_r = ((stride_reg == 2) ? {mac_row[30:0], 1'b0} : mac_row) + mac_kh;
    // 列（输入列）：w*stride + kw - pad，越界补 0
    wire signed [31:0] mac_c = $signed((stride_reg == 2) ? {mac_col[30:0], 1'b0} : mac_col)
                            + $signed(mac_kw) - $signed(pad_reg);
    wire mac_c_valid = (mac_c >= 0) && (mac_c < in_w_reg);
    wire [31:0] mac_c_cl = mac_c[31:0];

    // S_MAC_RD 拍寄存 tap 列有效位（拆 k_reg→乘加树组合链，setup 违例 -6.18ns 主因）

    // 同步采样：RAM 读端口（组合地址在上升沿被采样，数据晚一拍到 lb_q/acc_q）
    // acc 读复用同一端口（S_MAC 读 {mac_row,mac_col}，requant 读 {rq_row,rq_col}，
    // 两阶段状态互斥）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lb_q <= 64'h0;
            acc_q <= 256'sd0;
        end else begin
            lb_q <= lb[lb_addr_r];
            // 权重采样（S_MAC_RD 拍）：mac_t 在 S_MAC_ACC 末更新，隔 S_MAC_ADDR 拍到
            // S_MAC_RD 沿共 2 拍窗口，拆 mac_t→wbuf 72:1 mux 组合链（原每拍采样
            // 窗口仅 1 拍，w_q 路径 slack -5.189；S_MAC_ADDR/RD 拍不用 w_q，
            // 乘法仍在 S_MAC_MUL 拍用新值，拍序不变）
            if (state == S_MAC_RD) begin
                for (lane = 0; lane < 8; lane = lane + 1)
                    for (m = 0; m < 8; m = m + 1)
                        w_q[lane][m] <= wbuf[lane][m][mac_t];
            end
            if (state == S_REQ_ADDR || state == S_REQ_MUL || state == S_REQ_MUL2)
                acc_q <= acc[acc_raddr_clr];
            else
                acc_q <= acc[acc_addr_mac_r];
        end
    end

    integer lane, m;
    (* multstyle = "logic" *) reg signed [31:0] v_sum [0:7];   // 8×8 MAC 树用 LUT（省 DSP，~4ns 无时序风险）
    reg signed [31:0] v_sum_r [0:7];   // S_MAC_MUL 拍寄存的乘加树结果（拆流水）

    // 组合中间变量：8 lane 独立 32-bit 加法后拼成整字（避免 part-select 写 RAM
    // 破坏 M10K 推断；acc_local/acc 保持整字写，Quartus 才可推断 block RAM）
    reg [255:0] acc_next;      // acc_local 的下一拍值（每 lane 独立 int32 累加）
    reg [255:0] acc_wr_next;   // 末 tap 写回 acc 的整字（每 lane = acc_local + 本 tap 部分和）

    //-----------------------------------------------------------------------
    // requant 流水（拆流水，避免单拍组合链过深）：
    //   S_REQ_MUL  拍：寄存 rq_bias_m/rq_mult_m（RAM 读打拍，断 pass-through 读路径）
    //   S_REQ_MULB 拍：acc_q + rq_bias_m → v_biased_l（32-bit 加法单独一拍，~3ns）
    //   S_REQ_MULC 拍：relu/rcl6 比较 → v_act_l（比较+mux 单独一拍，~3ns）
    //   S_REQ_MUL2 拍：4×16×16 部分积（DSP）→ 寄存 v_p_*/v_shift_l
    //   S_REQ_MUL3 拍：部分积移位相加 → v_rq64_l（64-bit 加法树）
    //   S_REQ_OUT  拍：round 加法 → v_round_l（用上拍寄存的积）
    // 拆流水后功能/事件序列不变（tb 按事件对拍，不测周期数）。
    //-----------------------------------------------------------------------
    integer ln;
    reg signed [31:0] v_act;
    reg signed [63:0] v_rq64;
    reg signed [31:0] v_act_l [0:7];   // 每 lane 的 act 后值（流水级 2）
    reg signed [31:0] v_biased_l [0:7]; // 每 lane 的 bias 后值（流水级 1，拆加法/比较链）
    reg signed [31:0] v_rcl6_l [0:7];   // 每 lane 的 rcl6 上限（S_REQ_MUL 拍寄存 RAM 读，断组合读链）
    // 乘法拆三级（150MHz 单级 32×33 组合乘法 slack -10.1ns；64-bit 加法树 2 级仍紧）：
    //   级 1 = 4 个 16×16 DSP 乘法（v_act_l = a_hi<<16 + a_lo，rq_mult_q = m_hi<<16 + m_lo）
    //   级 2 = 两组 64-bit 并行加法（v_sum_lo/v_sum_hi），每级仅 1 个加法器
    //   级 3 = 中间和相加 → v_rq64_l
    (* multstyle = "dsp" *) reg [31:0] v_p_lolo [0:7];   // a_lo × m_lo（无符号 16×16，DSP 18×18）
    (* multstyle = "dsp" *) reg [31:0] v_p_lohi [0:7];   // a_lo × m_hi（无符号 16×16）
    (* multstyle = "dsp" *) reg signed [31:0] v_p_hilo [0:7];  // a_hi × m_lo（signed 16 × unsigned 16）
    (* multstyle = "dsp" *) reg signed [31:0] v_p_hihi [0:7];  // a_hi × m_hi（signed 16 × unsigned 16）
    reg [31:0] rq_mult_m [0:7];   // S_REQ_MUL 拍寄存的乘法器 b 输入（拆 RAM 读与 DSP 乘法组合链）
    reg signed [31:0] rq_bias_m [0:7];  // S_REQ_MUL 拍寄存的 bias RAM 读（拆 RAM 读与 32-bit 加法组合链）

    // 16×16 乘法用显式 DSP 例化（模块级 multstyle 最可靠；数组属性曾被 Quartus 忽略）
    wire [31:0] mul_lolo [0:7], mul_lohi [0:7], mul_hilo [0:7], mul_hihi [0:7];
    genvar gi;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : g_mul16
            mul16x16_dsp #(.A_SIGNED(0)) u_lolo (.a(v_act_l[gi][15:0]), .b(rq_mult_m[gi][15:0]), .p(mul_lolo[gi]));
            mul16x16_dsp #(.A_SIGNED(0)) u_lohi (.a(v_act_l[gi][15:0]), .b(rq_mult_m[gi][31:16]), .p(mul_lohi[gi]));
            mul16x16_dsp #(.A_SIGNED(1)) u_hilo (.a(v_act_l[gi][31:16]), .b(rq_mult_m[gi][15:0]), .p(mul_hilo[gi]));
            mul16x16_dsp #(.A_SIGNED(1)) u_hihi (.a(v_act_l[gi][31:16]), .b(rq_mult_m[gi][31:16]), .p(mul_hihi[gi]));
        end
    endgenerate
    reg signed [63:0] v_sum_lo [0:7];  // lolo + lohi<<16（无符号两项）
    reg signed [63:0] v_sum_hi [0:7];  // hilo<<16 + hihi<<32（符号扩展两项）
    reg signed [63:0] v_rnd_delta [0:7];  // round 桶形移位结果（1<<(shift-1)）
    reg signed [63:0] v_shifted [0:7];    // 算术右移结果（>>> shift）
    (* multstyle = "dsp" *) reg signed [63:0] v_rq64_l [0:7];  // 每 lane 的 64-bit 积（流水级 2，保 DSP）
    reg [7:0] v_shift_l [0:7];         // 每 lane 的 shift 值（S_REQ_MUL2 拍寄存 RAM 读，断 M10K q 路径）
    reg signed [63:0] v_round_l [0:7]; // 每 lane 的 round 后值（流水级 3）
    reg [31:0] v_out_ch;
    reg [7:0] v_q [0:7];

    // 乘加拍级 1（S_REQ_MUL）：bias/mult RAM 读打拍 → rq_bias_m/rq_mult_m
    // （bias 加法与 DSP 乘法各拆独立一拍，断 M10K pass-through 读路径组合穿透）
    always @(posedge clk) begin
        if (state == S_REQ_MUL) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                rq_bias_m[ln] <= rq_bias_q[ln];
                v_rcl6_l[ln] <= rq_rcl6_q[ln];
            end
        end
    end

    // 乘法器 b 输入打拍（S_REQ_MUL 拍寄存 rq_mult_q）：拆开 RAM 读路径
    // （含 read-during-write pass-through mux）与 DSP 乘法，避免组合穿透直达 v_p_*
    always @(posedge clk) begin
        if (state == S_REQ_MUL) begin
            for (ln = 0; ln < 8; ln = ln + 1)
                rq_mult_m[ln] <= rq_mult_q[ln];
        end
    end

    // 乘加拍级 2（S_REQ_MULB）：acc_q + rq_bias_m → v_biased_l（32-bit 加法单独一拍，~3ns）
    always @(posedge clk) begin
        if (state == S_REQ_MULB) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_biased_l[ln] <= 32'sd0;
                else
                    v_biased_l[ln] <= acc_q[32*ln +: 32] + rq_bias_m[ln];
            end
        end
    end

    // 乘加拍级 3（S_REQ_MULC）：relu/rcl6 比较 → v_act_l（比较+mux 单独一拍，~3ns）
    always @(posedge clk) begin
        if (state == S_REQ_MULC) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_act_l[ln] <= 32'sd0;
                else begin
                    v_act = v_biased_l[ln];
                    if (act_reg == 2'd1) begin
                        v_act = v_biased_l[ln][31] ? 32'sd0 : v_biased_l[ln];
                    end else if (act_reg == 2'd2) begin
                        v_act = v_biased_l[ln][31] ? 32'sd0 : v_biased_l[ln];
                        if (v_act > v_rcl6_l[ln]) v_act = v_rcl6_l[ln];
                    end
                    v_act_l[ln] <= v_act;
                end
            end
        end
    end

    // 乘法级 1（S_REQ_MUL2）：v_act_l × rq_mult_q 拆 4 个 16×16 部分积（DSP 18×18，~4ns）；
    // 同时把 RAM 读出的 shift 值寄存为 v_shift_l——断开 M10K q 输出到输出拍组合链的长路径
    always @(posedge clk) begin
        if (state == S_REQ_MUL2) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    begin
                        v_p_lolo[ln] <= 32'd0;
                        v_p_lohi[ln] <= 32'd0;
                        v_p_hilo[ln] <= 32'sd0;
                        v_p_hihi[ln] <= 32'sd0;
                    end
                else begin
                    v_p_lolo[ln] <= mul_lolo[ln];
                    v_p_lohi[ln] <= mul_lohi[ln];
                    v_p_hilo[ln] <= mul_hilo[ln];
                    v_p_hihi[ln] <= mul_hihi[ln];
                end
                v_shift_l[ln] <= rq_shift_q[ln];
            end
        end
    end

    // 乘法级 2（S_REQ_MUL3）：4 项部分积分为两组中间和（并行 64-bit 加法，各 ~3ns）
    always @(posedge clk) begin
        if (state == S_REQ_MUL3) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg) begin
                    v_sum_lo[ln] <= 64'sd0;
                    v_sum_hi[ln] <= 64'sd0;
                end else begin
                    v_sum_lo[ln] <= {32'd0, v_p_lolo[ln]}
                                  + {16'd0, v_p_lohi[ln], 16'd0};
                    v_sum_hi[ln] <= {{16{v_p_hilo[ln][31]}}, v_p_hilo[ln], 16'd0}
                                  + {v_p_hihi[ln], 32'd0};
                end
            end
        end
    end

    // 乘法级 3（S_REQ_MUL4）：两组中间和相加 → v_rq64_l（单个 64-bit 加法，~3ns）
    always @(posedge clk) begin
        if (state == S_REQ_MUL4) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_rq64_l[ln] <= 64'sd0;
                else
                    v_rq64_l[ln] <= v_sum_lo[ln] + v_sum_hi[ln];
            end
        end
    end

    // round 拍级 1（S_REQ_OUT）：round 桶形移位单独一拍（1<<(shift-1)，~5ns）
    always @(posedge clk) begin
        if (state == S_REQ_OUT) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_rnd_delta[ln] <= 64'sd0;
                else if (v_shift_l[ln] > 8'd0)
                    v_rnd_delta[ln] <= 64'sd1 << (v_shift_l[ln] - 8'd1);
                else
                    v_rnd_delta[ln] <= 64'sd0;
            end
        end
    end

    // round 拍级 2（S_REQ_ROUND2）：v_rq64_l + v_rnd_delta → v_round_l（64-bit 加法单独一拍，~3ns）
    always @(posedge clk) begin
        if (state == S_REQ_ROUND2) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_round_l[ln] <= 64'sd0;
                else
                    v_round_l[ln] <= v_rq64_l[ln] + v_rnd_delta[ln];
            end
        end
    end

    // 输出移位拍（S_REQ_OUT2）：v_round_l >>> shift（64-bit 桶形移位单独一拍，~5ns）
    always @(posedge clk) begin
        if (state == S_REQ_OUT2) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_shifted[ln] <= 64'sd0;
                else
                    v_shifted[ln] <= v_round_l[ln] >>> v_shift_l[ln];
            end
        end
    end

    // 输出拍（S_REQ_OUT3）：移位值饱和 → o_data（64-bit 比较 + mux，~3ns）
    always @(posedge clk) begin
        if (state == S_REQ_OUT3) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg) begin
                    v_q[ln] = 8'd0;
                end else begin
                v_rq64 = v_shifted[ln];
                if (v_rq64 > 64'sd127)      v_q[ln] = 8'sd127;
                else if (v_rq64 < -64'sd128) v_q[ln] = 8'h80;   // -128（8'sd128 字面量溢出）
                else                         v_q[ln] = v_rq64[7:0];
                end
            end
            o_data <= {v_q[7], v_q[6], v_q[5], v_q[4],
                       v_q[3], v_q[2], v_q[1], v_q[0]};
        end
    end

    // ---- 观测输出（组合直读；Quartus 不支持跨模块层次引用，故以端口引出）----
    assign dbg_ptr0  = {state[4:0], o_group[10:0], i_group[10:0], mac_t[3:0]};
    assign dbg_ptr1  = {rq_row[15:0], rq_col[15:0]};
    assign dbg_ptr2  = {mac_row[15:0], mac_col[15:0]};
    assign dbg_ptr3  = {load_row[9:0], load_col[9:0], wf_cnt[7:0]};
    assign dbg_data0 = {acc_q[31:0], v_biased_l[0], v_act_l[0], v_rq64_l[0][31:0]};
    assign dbg_data1 = {v_round_l[0][31:0], v_shifted[0][31:0], v_rnd_delta[0][31:0],
                        {w_q[0][3], w_q[0][2], w_q[0][1], w_q[0][0]}};
    assign dbg_lb    = lb_q[31:0];

endmodule
