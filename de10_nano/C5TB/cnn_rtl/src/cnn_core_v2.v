//=============================================================================
// cnn_core_v2 — 黑盒式行块驻留卷积执行器（阶段 4 核心）
//=============================================================================
// 与阶段 3 的 cnn_core 相比（黑盒真实架构对齐）：
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
// 接口（供 cnn_top 的 DMA 层驱动）：
//   cfg：sel=0 标量（addr 索引见下）、sel=1..4 requant 数组（addr=通道）
//   i_stream / iw_stream：64-bit 握手（DMA 连续读，地址 = 已发字节数）
//   o_stream：64-bit 输出事件（8 通道/拍，NHWC8 块序）
//=============================================================================

module cnn_core_v2 #(
    parameter G_MAX_IN_CB  = 4,    // 最大输入通道块（模型 in_c<=32）
    parameter G_MAX_OUT_CB = 4,    // 最大输出通道块（模型 out_c<=32）
    parameter G_MAX_IN_ROWS= 41,   // 最大输入行块高度（模型 max in_tile=39）
    parameter G_MAX_OROWS  = 20,   // 最大输出行块高度（模型 max tile=19）
    parameter G_MAX_W      = 302,  // 最大输入宽度（IMAGE_MAX_W）
    parameter G_MAX_OW     = 150,  // 最大输出宽度（OUTPUT_MAX_W）
    parameter G_MAX_C      = 128   // 最大输出通道数（requant 数组容量）
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
    output reg                 o_done
);

    //-----------------------------------------------------------------------
    // 运行时标量寄存器（cfg_sel=0，cfg_addr=索引，cfg_wdata=值）
    //  0:type 1:act 2:in_c 3:in_h 4:in_w 5:out_c 6:out_h 7:out_w
    //  8:k 9:pad 10:stride 11:out_row_tile 12:in_row_tile
    //  13:in_cb 14:out_cb 15:base_row（本行块输入 base 行号）
    //-----------------------------------------------------------------------
    reg [3:0]  type_reg, act_reg;
    reg [31:0] in_c_reg,  in_h_reg,  in_w_reg;
    reg [31:0] out_c_reg, out_h_reg, out_w_reg;
    reg [31:0] k_reg, pad_reg, stride_reg;
    reg [31:0] out_row_tile_reg, in_row_tile_reg;
    reg [31:0] in_cb_reg, out_cb_reg;
    reg signed [31:0] base_row_reg;
    reg [31:0] row_block_reg;      // 供 tb/调度层读取（cnn_core 本身单行块）

    // requant 数组（cfg_sel 1/2/3/4 = bias/mult/shift/rcl6，addr=输出通道）
    reg signed [31:0] bias_store [0:G_MAX_C-1];
    reg [31:0] rq_m_store [0:G_MAX_C-1];
    reg [7:0]  rq_r_store [0:G_MAX_C-1];
    reg signed [31:0] rcl6_store [0:G_MAX_C-1];

    integer i;
    always @(posedge clk) begin
        if (cfg_we) begin
            case (cfg_sel)
                3'd0: begin
                    case (cfg_addr)
                        20'd0: type_reg      <= cfg_wdata[3:0];
                        20'd1: act_reg       <= cfg_wdata[1:0];
                        20'd2: in_c_reg      <= cfg_wdata;
                        20'd3: in_h_reg      <= cfg_wdata;
                        20'd4: in_w_reg      <= cfg_wdata;
                        20'd5: out_c_reg     <= cfg_wdata;
                        20'd6: out_h_reg     <= cfg_wdata;
                        20'd7: out_w_reg     <= cfg_wdata;
                        20'd8: k_reg         <= cfg_wdata;
                        20'd9: pad_reg       <= cfg_wdata;
                        20'd10: stride_reg   <= cfg_wdata;
                        20'd11: out_row_tile_reg <= cfg_wdata;
                        20'd12: in_row_tile_reg  <= cfg_wdata;
                        20'd13: in_cb_reg     <= cfg_wdata;
                        20'd14: out_cb_reg    <= cfg_wdata;
                        20'd15: base_row_reg  <= cfg_wdata;
                        20'd16: row_block_reg <= cfg_wdata;
                        default: ;
                    endcase
                end
                3'd1: if (cfg_addr < G_MAX_C) bias_store[cfg_addr]  <= cfg_wdata;
                3'd2: if (cfg_addr < G_MAX_C) rq_m_store[cfg_addr]  <= cfg_wdata;
                3'd3: if (cfg_addr < G_MAX_C) rq_r_store[cfg_addr]  <= cfg_wdata[7:0];
                3'd4: if (cfg_addr < G_MAX_C) rcl6_store[cfg_addr]  <= cfg_wdata;
                default: ;
            endcase
        end
    end

    //-----------------------------------------------------------------------
    // 存储：行缓冲 [cb][row][w*8+m]；权重 slice wbuf [lane*72 + t*8 + m]
    //-----------------------------------------------------------------------
    reg [7:0] lb [0:G_MAX_IN_CB-1][0:G_MAX_IN_ROWS-1][0:G_MAX_W*8-1];
    reg signed [7:0] wbuf [0:8*9*8-1];

    // acc 行块缓冲 [o_row][w][lane]（int32）
    reg signed [31:0] acc [0:G_MAX_OROWS-1][0:G_MAX_OW-1][0:7];

    //-----------------------------------------------------------------------
    // 状态机
    //-----------------------------------------------------------------------
    localparam S_IDLE     = 3'd0;
    localparam S_LOAD     = 3'd1;   // 装载输入行块
    localparam S_ACC_CLR  = 3'd2;   // acc 清零（每输出组）
    localparam S_WEIGHT   = 3'd3;   // 装载权重 slice（72 拍）
    localparam S_MAC      = 3'd4;   // 窗口 MAC（9 tap/窗口 × 全行块窗口）
    localparam S_REQUANT  = 3'd5;   // requant + 输出事件
    localparam S_DONE     = 3'd6;

    reg [2:0] state;
    reg [31:0] load_cb, load_row, load_col;
    reg [31:0] wf_cnt;
    reg [31:0] o_group, i_group;
    reg [31:0] mac_row, mac_col, mac_t;
    reg [31:0] rq_row, rq_col, rq_ln;

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
            load_cb <= 0; load_row <= 0; load_col <= 0;
            wf_cnt <= 0; o_group <= 0; i_group <= 0;
            mac_row <= 0; mac_col <= 0; mac_t <= 0;
            rq_row <= 0; rq_col <= 0; rq_ln <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    o_done <= 1'b0;
                    if (start) begin
                        load_cb <= 0; load_row <= 0; load_col <= 0;
                        o_group <= 0; i_group <= 0;
                        state <= S_LOAD;
                    end
                end

                //---- 装载输入行块（NHWC8 块序：cb 外层，行、列内层）----
                S_LOAD: begin
                    if (load_row_valid) begin
                        if (i_valid) begin
                            lb[load_cb][load_row][load_col*8 + 0] <= i_data[7:0];
                            lb[load_cb][load_row][load_col*8 + 1] <= i_data[15:8];
                            lb[load_cb][load_row][load_col*8 + 2] <= i_data[23:16];
                            lb[load_cb][load_row][load_col*8 + 3] <= i_data[31:24];
                            lb[load_cb][load_row][load_col*8 + 4] <= i_data[39:32];
                            lb[load_cb][load_row][load_col*8 + 5] <= i_data[47:40];
                            lb[load_cb][load_row][load_col*8 + 6] <= i_data[55:48];
                            lb[load_cb][load_row][load_col*8 + 7] <= i_data[63:56];
                            if (load_col == in_w_reg - 1) begin
                                load_col <= 0;
                                if (load_row == in_row_tile_reg - 1) begin
                                    load_row <= 0;
                                    if (load_cb == in_cb_reg - 1) begin
                                        load_cb <= 0;
                                        o_group <= 0;
                                        rq_row <= 0; rq_col <= 0; rq_ln <= 0;
                                        state <= S_ACC_CLR;
                                    end else
                                        load_cb <= load_cb + 1;
                                end else
                                    load_row <= load_row + 1;
                            end else
                                load_col <= load_col + 1;
                        end
                    end else begin
                        // pad 行：写 0 并跳过（不消费输入；BRAM 上电不定须清零）
                        lb[load_cb][load_row][load_col*8 + 0] <= 8'h00;
                        lb[load_cb][load_row][load_col*8 + 1] <= 8'h00;
                        lb[load_cb][load_row][load_col*8 + 2] <= 8'h00;
                        lb[load_cb][load_row][load_col*8 + 3] <= 8'h00;
                        lb[load_cb][load_row][load_col*8 + 4] <= 8'h00;
                        lb[load_cb][load_row][load_col*8 + 5] <= 8'h00;
                        lb[load_cb][load_row][load_col*8 + 6] <= 8'h00;
                        lb[load_cb][load_row][load_col*8 + 7] <= 8'h00;
                        if (load_col == in_w_reg - 1) begin
                            load_col <= 0;
                            if (load_row == in_row_tile_reg - 1) begin
                                load_row <= 0;
                                if (load_cb == in_cb_reg - 1) begin
                                    load_cb <= 0;
                                    o_group <= 0;
                                    rq_row <= 0; rq_col <= 0; rq_ln <= 0;
                                    state <= S_ACC_CLR;
                                end else
                                    load_cb <= load_cb + 1;
                            end else
                                load_row <= load_row + 1;
                        end else
                            load_col <= load_col + 1;
                    end
                end

                //---- acc 清零（每输出组开始）----
                S_ACC_CLR: begin
                    o_valid <= 1'b0;
                    acc[rq_row][rq_col][rq_ln] <= 32'sd0;
                    if (rq_row == out_row_tile_reg - 1 && rq_col == out_w_reg - 1
                        && rq_ln == 7) begin
                        rq_row <= 0; rq_col <= 0; rq_ln <= 0;
                        i_group <= 0;
                        state <= S_WEIGHT;
                    end else if (rq_ln == 7) begin
                        rq_ln <= 0;
                        if (rq_col == out_w_reg - 1) begin
                            rq_col <= 0;
                            rq_row <= rq_row + 1;
                        end else
                            rq_col <= rq_col + 1;
                    end else
                        rq_ln <= rq_ln + 1;
                end

                //---- 装载权重 slice（72 拍，64-bit/拍）----
                S_WEIGHT: begin
                    if (iw_valid) begin
                        wbuf[wf_cnt*8 + 0] <= iw_data[7:0];
                        wbuf[wf_cnt*8 + 1] <= iw_data[15:8];
                        wbuf[wf_cnt*8 + 2] <= iw_data[23:16];
                        wbuf[wf_cnt*8 + 3] <= iw_data[31:24];
                        wbuf[wf_cnt*8 + 4] <= iw_data[39:32];
                        wbuf[wf_cnt*8 + 5] <= iw_data[47:40];
                        wbuf[wf_cnt*8 + 6] <= iw_data[55:48];
                        wbuf[wf_cnt*8 + 7] <= iw_data[63:56];
                        if (wf_cnt == 8*9 - 1) begin
                            wf_cnt <= 0;
                            mac_row <= 0; mac_col <= 0; mac_t <= 0;
                            state <= S_MAC;
                        end else
                            wf_cnt <= wf_cnt + 1;
                    end
                end

                //---- 窗口 MAC：全行块窗口 × 当前 (o_group, i_group) slice ----
                S_MAC: begin
                    if (mac_t == 8) begin
                        mac_t <= 0;
                        if (mac_col == out_w_reg - 1) begin
                            mac_col <= 0;
                            if (mac_row == out_row_tile_reg - 1) begin
                                mac_row <= 0;
                                if (type_reg == 4 || i_group == in_cb_reg - 1) begin
                                    rq_row <= 0; rq_col <= 0;
                                    state <= S_REQUANT;
                                end else begin
                                    i_group <= i_group + 1;
                                    state <= S_WEIGHT;
                                end
                            end else
                                mac_row <= mac_row + 1;
                        end else
                            mac_col <= mac_col + 1;
                    end else
                        mac_t <= mac_t + 1;
                end

                //---- requant + 输出事件（行→列→8 通道，块序）----
                S_REQUANT: begin
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
                                    rq_row <= 0; rq_col <= 0; rq_ln <= 0;
                                    state <= S_ACC_CLR;
                                end
                            end else
                                rq_row <= rq_row + 1;
                        end else
                            rq_col <= rq_col + 1;
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
    // 组合乘加树（MAC）：lb[cb][r][c] × wbuf[lane][t][m] 累加
    //-----------------------------------------------------------------------
    wire [31:0] mac_cb   = (type_reg == 4) ? o_group : i_group;
    wire [31:0] mac_kh   = mac_t / k_reg;
    wire [31:0] mac_kw   = mac_t % k_reg;
    // 行（lb 索引）：窗口第 kh 行 = o_row*stride + kh + pad（装载时行 0 = base 行）
    wire [31:0] mac_r    = mac_row * stride_reg + mac_kh;
    // 列（输入列）：w*stride + kw - pad，越界补 0
    wire signed [31:0] mac_c = $signed(mac_col * stride_reg + mac_kw) - $signed(pad_reg);
    wire mac_c_valid = (mac_c >= 0) && (mac_c < in_w_reg);
    wire [31:0] mac_c_cl = mac_c[31:0];

    wire [7:0] lb_m [0:7];
    assign lb_m[0] = mac_c_valid ? lb[mac_cb][mac_r][mac_c_cl*8 + 0] : 8'h00;
    assign lb_m[1] = mac_c_valid ? lb[mac_cb][mac_r][mac_c_cl*8 + 1] : 8'h00;
    assign lb_m[2] = mac_c_valid ? lb[mac_cb][mac_r][mac_c_cl*8 + 2] : 8'h00;
    assign lb_m[3] = mac_c_valid ? lb[mac_cb][mac_r][mac_c_cl*8 + 3] : 8'h00;
    assign lb_m[4] = mac_c_valid ? lb[mac_cb][mac_r][mac_c_cl*8 + 4] : 8'h00;
    assign lb_m[5] = mac_c_valid ? lb[mac_cb][mac_r][mac_c_cl*8 + 5] : 8'h00;
    assign lb_m[6] = mac_c_valid ? lb[mac_cb][mac_r][mac_c_cl*8 + 6] : 8'h00;
    assign lb_m[7] = mac_c_valid ? lb[mac_cb][mac_r][mac_c_cl*8 + 7] : 8'h00;

    integer lane, m;
    reg signed [31:0] v_sum [0:7];

    always @(posedge clk) begin
        if (state == S_MAC) begin
            for (lane = 0; lane < 8; lane = lane + 1) begin
                v_sum[lane] = 0;
                for (m = 0; m < 8; m = m + 1) begin
                    v_sum[lane] = v_sum[lane] +
                        $signed(lb_m[m]) *
                        wbuf[lane*72 + mac_t*8 + m];
                end
            end
            for (lane = 0; lane < 8; lane = lane + 1) begin
                acc[mac_row][mac_col][lane] <=
                    acc[mac_row][mac_col][lane] + v_sum[lane];
            end
        end
    end

    //-----------------------------------------------------------------------
    // requant + 输出数据
    //-----------------------------------------------------------------------
    integer ln;
    reg signed [31:0] v_raw, v_biased, v_act;
    reg signed [63:0] v_rq64;
    reg [31:0] v_out_ch;
    reg [7:0] v_q [0:7];

    always @(posedge clk) begin
        if (state == S_REQUANT) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg) begin
                    v_q[ln] = 8'd0;
                end else begin
                v_raw = acc[rq_row][rq_col][ln];
                v_biased = v_raw + bias_store[v_out_ch];
                v_act = v_biased;
                if (act_reg == 2'd1) begin
                    v_act = v_biased[31] ? 32'sd0 : v_biased;
                end else if (act_reg == 2'd2) begin
                    v_act = v_biased[31] ? 32'sd0 : v_biased;
                    if (v_act > rcl6_store[v_out_ch]) v_act = rcl6_store[v_out_ch];
                end
                v_rq64 = $signed(v_act) * $signed({1'b0, rq_m_store[v_out_ch]});
                if (rq_r_store[v_out_ch] > 0)
                    v_rq64 = v_rq64 + (64'sd1 << (rq_r_store[v_out_ch] - 1));
                v_rq64 = v_rq64 >>> rq_r_store[v_out_ch];
                if (v_rq64 > 64'sd127)      v_q[ln] = 8'sd127;
                else if (v_rq64 < -64'sd128) v_q[ln] = 8'h80;   // -128（8'sd128 字面量溢出）
                else                         v_q[ln] = v_rq64[7:0];
                end
            end
            o_data <= {v_q[7], v_q[6], v_q[5], v_q[4],
                       v_q[3], v_q[2], v_q[1], v_q[0]};
        end
    end

endmodule
