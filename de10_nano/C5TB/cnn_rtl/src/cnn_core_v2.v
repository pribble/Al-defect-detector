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
    // 存储：行缓冲 lb（64-bit 字 = 8 lane，col 0..G_MAX_W-1）
    //       acc（256-bit 字 = 8 lane × int32）
    //       wbuf [lane*72 + t*8 + m]（权重 slice，小数组保持寄存器）
    // 强制 ramstyle = M10K 且展平为单维数组（Quartus 对多维 unpacked 数组
    // 的 RAM 推断不可靠：lb[cb][row][col] 三维 + acc 256-bit 超宽会被展开成
    // 寄存器堆 + 读端口 mux，A&S 内存爆到 40GB+；单维数组 + 常量乘法拼接
    // 地址是官方推荐的可靠推断写法，功能/时序完全不变）。
    //-----------------------------------------------------------------------
    (* ramstyle = "M10K" *) reg [63:0] lb [0:G_MAX_IN_CB*G_MAX_IN_ROWS*G_MAX_W-1];
    (* ramstyle = "M10K" *) reg signed [255:0] acc [0:G_MAX_OROWS*G_MAX_OW-1];

    // 单维索引拼接（常量乘法综合时折叠成移位/加法，不产生逻辑）
    wire [31:0] lb_waddr = load_cb*G_MAX_IN_ROWS*G_MAX_W + load_row*G_MAX_W + load_col;
    wire [31:0] lb_raddr = mac_cb*G_MAX_IN_ROWS*G_MAX_W + mac_r*G_MAX_W + mac_c_cl;
    wire [31:0] acc_waddr_clr = rq_row*G_MAX_OW + rq_col;
    wire [31:0] acc_waddr_mac = mac_row*G_MAX_OW + mac_col;
    wire [31:0] acc_raddr_clr = rq_row*G_MAX_OW + rq_col;
    wire [31:0] acc_raddr_mac = mac_row*G_MAX_OW + mac_col;
    reg signed [7:0] wbuf [0:8*9*8-1];

    // 同步读采样寄存器（RAM 读输出，读地址 = 组合函数，晚一拍有效）
    reg [63:0]          lb_q;
    reg signed [255:0]  acc_q;
    reg signed [255:0]  acc_local;   // 窗口内累加（首 tap 用 acc_q 初始化）

    //-----------------------------------------------------------------------
    // 状态机
    //-----------------------------------------------------------------------
    localparam S_IDLE     = 4'd0;
    localparam S_LOAD     = 4'd1;   // 装载输入行块
    localparam S_ACC_CLR  = 4'd2;   // acc 清零（每输出组）
    localparam S_WEIGHT   = 4'd3;   // 装载权重 slice（72 拍）
    localparam S_MAC_RD   = 4'd4;   // 窗口 tap 请求拍（同步采样 lb_q/acc_q）
    localparam S_MAC_MUL  = 4'd11;  // 窗口 tap 乘拍（乘加树 → v_sum_r）
    localparam S_MAC_ACC  = 4'd5;   // 窗口 tap 累加拍（v_sum_r → acc_local）
    localparam S_REQ_ADDR = 4'd6;   // requant 请求拍（采样 acc_q）
    localparam S_REQ_MUL  = 4'd7;   // requant 乘加拍（bias+act，寄存 v_act_l）
    localparam S_REQ_MUL2 = 4'd8;   // requant 乘法拍（32×32 mult，寄存 v_rq64_l）
    localparam S_REQ_OUT  = 4'd9;   // requant 输出拍（round+shift+饱和 → o_data）
    localparam S_DONE     = 4'd10;

    reg [3:0] state;
    reg [31:0] load_cb, load_row, load_col;
    reg [31:0] wf_cnt;
    reg [31:0] o_group, i_group;
    reg [31:0] mac_row, mac_col, mac_t;
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
            load_cb <= 0; load_row <= 0; load_col <= 0;
            wf_cnt <= 0; o_group <= 0; i_group <= 0;
            mac_row <= 0; mac_col <= 0; mac_t <= 0;
            rq_row <= 0; rq_col <= 0;
            acc_local <= 256'sd0;   // 复位归并到主状态机（单一驱动，避免 Quartus 10028）
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
                            lb[lb_waddr] <= i_data;
                            if (load_col == in_w_reg - 1) begin
                                load_col <= 0;
                                if (load_row == in_row_tile_reg - 1) begin
                                    load_row <= 0;
                                    if (load_cb == in_cb_reg - 1) begin
                                        load_cb <= 0;
                                        o_group <= 0;
                                        rq_row <= 0; rq_col <= 0;
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
                        lb[lb_waddr] <= 64'h0;
                        if (load_col == in_w_reg - 1) begin
                            load_col <= 0;
                            if (load_row == in_row_tile_reg - 1) begin
                                load_row <= 0;
                                if (load_cb == in_cb_reg - 1) begin
                                    load_cb <= 0;
                                    o_group <= 0;
                                    rq_row <= 0; rq_col <= 0;
                                    state <= S_ACC_CLR;
                                end else
                                    load_cb <= load_cb + 1;
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
                            state <= S_MAC_RD;
                        end else
                            wf_cnt <= wf_cnt + 1;
                    end
                end

                //---- 窗口 MAC（请求拍）：组合地址稳定，上升沿同步采样 lb_q/acc_q ----
                S_MAC_RD: begin
                    state <= S_MAC_MUL;
                end

                //---- 窗口 MAC（乘拍）：乘加树 → v_sum_r 寄存（拆开累加，缩短组合链）----
                S_MAC_MUL: begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        v_sum[lane] = 0;
                        for (m = 0; m < 8; m = m + 1) begin
                            v_sum[lane] = v_sum[lane] +
                                $signed(lb_q[8*m +: 8]) *
                                wbuf[lane*72 + mac_t*8 + m];
                        end
                    end
                    for (lane = 0; lane < 8; lane = lane + 1)
                        v_sum_r[lane] <= v_sum[lane];
                    state <= S_MAC_ACC;
                end

                //---- 窗口 MAC（累加拍）：用上拍寄存的 v_sum_r 累加到 acc_local；
                //     首 tap（mac_t==0）以 acc_q 初始化；末 tap（mac_t==8）写回 acc ----
                S_MAC_ACC: begin
                    // 8 lane 独立 32-bit 累加（acc_local 打包为 256-bit，但每个 lane
                    // 是独立 int32，无跨 lane 进位；整体 256-bit 加法器进位链过长，
                    // 拆成 lane 级加法消除，语义不变）
                    // 注意：必须先用组合变量拼成整字再整字写 acc_local/acc——
                    // 直接 part-select 写（acc[...][32*lane +: 32] <= ...）会阻止
                    // Quartus 把 acc 推断为 M10K，导致 A&S 内存反弹（10+GB）。
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if (mac_t == 0)
                            acc_next[32*lane +: 32] = acc_q[32*lane +: 32] + v_sum_r[lane];
                        else
                            acc_next[32*lane +: 32] = acc_local[32*lane +: 32] + v_sum_r[lane];
                    end
                    acc_local <= acc_next;
                    if (mac_t == 8) begin
                        // 末 tap：写回最终值（acc_local 尚缺本 tap 部分和）
                        for (lane = 0; lane < 8; lane = lane + 1)
                            acc_wr_next[32*lane +: 32] = acc_local[32*lane +: 32] + v_sum_r[lane];
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
                                    state <= S_WEIGHT;
                                end
                            end else
                                mac_row <= mac_row + 1;
                        end else
                            mac_col <= mac_col + 1;
                    end else
                        mac_t <= mac_t + 1;
                end

                //---- requant 请求拍：采样 acc[rq_row][rq_col] ----
                S_REQ_ADDR: begin
                    o_valid <= 1'b0;
                    state <= S_REQ_MUL;
                end

                //---- requant 乘加拍：bias+act（快），结果寄存 v_act_l ----
                S_REQ_MUL: begin
                    state <= S_REQ_MUL2;
                end

                //---- requant 乘法拍：32×32 mult，结果寄存 v_rq64_l ----
                S_REQ_MUL2: begin
                    state <= S_REQ_OUT;
                end

                //---- requant 输出拍（round+移位+饱和，行→列→8 通道，块序）----
                S_REQ_OUT: begin
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
                                    state <= S_ACC_CLR;
                                end
                            end else
                                rq_row <= rq_row + 1;
                        end else
                            rq_col <= rq_col + 1;
                        state <= S_REQ_ADDR;
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
    wire [31:0] mac_cb   = (type_reg == 4) ? o_group : i_group;
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

    // 同步采样：RAM 读端口（组合地址在上升沿被采样，数据晚一拍到 lb_q/acc_q）
    // acc 读复用同一端口（S_MAC 读 {mac_row,mac_col}，requant 读 {rq_row,rq_col}，
    // 两阶段状态互斥）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lb_q <= 64'h0;
            acc_q <= 256'sd0;
        end else begin
            lb_q <= lb[lb_raddr];
            if (state == S_REQ_ADDR || state == S_REQ_MUL || state == S_REQ_MUL2)
                acc_q <= acc[acc_raddr_clr];
            else
                acc_q <= acc[acc_raddr_mac];
        end
    end

    integer lane, m;
    reg signed [31:0] v_sum [0:7];
    reg signed [31:0] v_sum_r [0:7];   // S_MAC_MUL 拍寄存的乘加树结果（拆流水）

    // 组合中间变量：8 lane 独立 32-bit 加法后拼成整字（避免 part-select 写 RAM
    // 破坏 M10K 推断；acc_local/acc 保持整字写，Quartus 才可推断 block RAM）
    reg [255:0] acc_next;      // acc_local 的下一拍值（每 lane 独立 int32 累加）
    reg [255:0] acc_wr_next;   // 末 tap 写回 acc 的整字（每 lane = acc_local + 本 tap 部分和）

    //-----------------------------------------------------------------------
    // requant 流水（拆三段，避免单拍组合链过深）：
    //   S_REQ_MUL  拍：bias+act → v_act_l（寄存 32-bit）
    //   S_REQ_MUL2 拍：32×32 mult → v_rq64_l（寄存 64-bit 积）
    //   S_REQ_OUT  拍：round+shift+饱和 → o_data（用上拍寄存的积）
    // 拆流水后功能/事件序列不变（tb 按事件对拍，不测周期数）。
    //-----------------------------------------------------------------------
    integer ln;
    reg signed [31:0] v_raw, v_biased, v_act;
    reg signed [63:0] v_rq64;
    reg signed [31:0] v_act_l [0:7];   // 每 lane 的 act 后值（流水级 1）
    reg signed [63:0] v_rq64_l [0:7];  // 每 lane 的 64-bit 积（流水级 2）
    reg [31:0] v_out_ch;
    reg [7:0] v_q [0:7];

    // 乘加拍：acc_q（S_REQ_ADDR 采样的值）→ bias → act，寄存 v_act_l
    always @(posedge clk) begin
        if (state == S_REQ_MUL) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg) begin
                    v_act_l[ln] <= 32'sd0;
                end else begin
                v_raw = acc_q[32*ln +: 32];
                v_biased = v_raw + bias_store[v_out_ch];
                v_act = v_biased;
                if (act_reg == 2'd1) begin
                    v_act = v_biased[31] ? 32'sd0 : v_biased;
                end else if (act_reg == 2'd2) begin
                    v_act = v_biased[31] ? 32'sd0 : v_biased;
                    if (v_act > rcl6_store[v_out_ch]) v_act = rcl6_store[v_out_ch];
                end
                v_act_l[ln] <= v_act;
                end
            end
        end
    end

    // 乘法拍：v_act_l × rq_m_store（32×32），寄存 v_rq64_l
    always @(posedge clk) begin
        if (state == S_REQ_MUL2) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_rq64_l[ln] <= 64'sd0;
                else
                    v_rq64_l[ln] <= $signed(v_act_l[ln]) * $signed({1'b0, rq_m_store[v_out_ch]});
            end
        end
    end

    // 输出拍：round + shift + 饱和 → o_data
    always @(posedge clk) begin
        if (state == S_REQ_OUT) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg) begin
                    v_q[ln] = 8'd0;
                end else begin
                v_rq64 = v_rq64_l[ln];
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
