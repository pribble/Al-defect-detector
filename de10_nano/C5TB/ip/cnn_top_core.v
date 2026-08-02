//=============================================================================
// cnn_top — cnn_top 黑盒替代顶层（最小接口，对接软件侧，阶段 5）
//=============================================================================
// 协议对齐 cnn_top_spec.md。requant 定点参数由软件预转写入 scale 区
//（每通道 4 个 int32：mult/bias_int/shift/rcl6），RTL 无浮点。
//
// 从接口 hps2cnn_avs（8-bit 字地址，零等待）：
//   START=0x00（写 1 启动，完成自清）、DDRIN=0x10、DDRW=0x1C、
//   DDROUT=0x28、PARAM=0x34、SCALE=0x40
//
// 执行（START 置位后）：
//   1) param 块（27 字 struct parameter）→ 解析
//   2) scale 区（4×out_c 字）→ requant 参数
//   3) 配 cnn_core_v2 → 行块循环 rb=0..row_block-1：
//        base_row = rb*tile*stride - pad；core start；DMA 跟随流喂/收
//   4) START 自清
//
// 主接口（简单读/写，burstcount=1）：pr/sr 32b，lr/wr 64b 读，ow 64b 写。
//=============================================================================

(* multstyle = "logic" *) module cnn_top_core (   // 模块级属性：子模块 cnn_core_v2 的 MAC 8×8 走 LUT（数组信号属性不生效）
    input  wire                clk,
    input  wire                rst_n,

    // ---- hps2cnn_avs ----
    input  wire [7:0]          as_address,
    input  wire                as_write,
    input  wire                as_read,
    input  wire [31:0]         as_writedata,
    output reg  [31:0]         as_readdata,
    output wire                as_waitrequest,

    // ---- param 读（32-bit）----
    output reg  [31:0]         pr_address,
    output reg                 pr_read,
    input  wire [31:0]         pr_readdata,
    input  wire                pr_waitrequest,

    // ---- scale 读（32-bit）----
    output reg  [31:0]         sr_address,
    output reg                 sr_read,
    input  wire [31:0]         sr_readdata,
    input  wire                sr_waitrequest,

    // ---- 输入读（64-bit）----
    output reg  [31:0]         lr_address,
    output wire                lr_read,
    input  wire [63:0]         lr_readdata,
    input  wire                lr_waitrequest,

    // ---- 权重读（64-bit）----
    output reg  [31:0]         wr_address,
    output wire                wr_read,
    input  wire [63:0]         wr_readdata,
    input  wire                wr_waitrequest,

    // ---- 输出写（64-bit）----
    output reg  [31:0]         ow_address,
    output wire                ow_write,
    output wire [63:0]         ow_writedata,
    input  wire                ow_waitrequest
);

    //-----------------------------------------------------------------------
    // 寄存器
    //-----------------------------------------------------------------------
    reg [31:0] reg_ddrin, reg_ddrw, reg_ddrout, reg_param, reg_scale;
    reg [31:0] reg_start;
    reg        start_clear_pending;

    assign as_waitrequest = 1'b0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ddrin <= 0; reg_ddrw <= 0; reg_ddrout <= 0;
            reg_param <= 0; reg_scale <= 0; reg_start <= 0;
        end else begin
            if (as_write) begin
                case (as_address)
                    8'h00: reg_start   <= as_writedata;
                    8'h10: reg_ddrin   <= as_writedata;
                    8'h1C: reg_ddrw   <= as_writedata;
                    8'h28: reg_ddrout <= as_writedata;
                    8'h34: reg_param  <= as_writedata;
                    8'h40: reg_scale  <= as_writedata;
                    default: ;
                endcase
            end
            if (start_clear_pending)
                reg_start <= 0;
        end
    end

    always @(*) begin
        case (as_address)
            8'h00: as_readdata = reg_start;
            8'h10: as_readdata = reg_ddrin;
            8'h1C: as_readdata = reg_ddrw;
            8'h28: as_readdata = reg_ddrout;
            8'h34: as_readdata = reg_param;
            8'h40: as_readdata = reg_scale;
            default: as_readdata = 32'h0;
        endcase
    end

    //-----------------------------------------------------------------------
    // cnn_core_v2
    //-----------------------------------------------------------------------
    reg         core_cfg_we, core_start;
    wire        core_done;
    reg  [2:0]  core_cfg_sel;
    reg  [19:0] core_cfg_addr;
    reg  [31:0] core_cfg_wdata;
    wire        core_i_valid, core_i_ready;
    wire [63:0] core_i_data;
    wire        core_iw_valid, core_ow_ready;
    wire [63:0] core_iw_data;
    wire        core_o_valid, core_o_ready;
    wire [63:0] core_o_data;

    cnn_core_v2 core (
        .clk(clk), .rst_n(rst_n),
        .cfg_we(core_cfg_we), .cfg_sel(core_cfg_sel),
        .cfg_addr(core_cfg_addr), .cfg_wdata(core_cfg_wdata),
        .start(core_start), .o_done(core_done),
        .i_valid(core_i_valid), .i_ready(core_i_ready), .i_data(core_i_data),
        .iw_valid(core_iw_valid), .ow_ready(core_ow_ready), .iw_data(core_iw_data),
        .o_valid(core_o_valid), .o_ready(core_o_ready), .o_data(core_o_data)
    );

    //-----------------------------------------------------------------------
    // param 解析缓存 + 派生量
    //-----------------------------------------------------------------------
    reg [31:0] param_buf [0:26];
    reg [31:0] req_buf [0:511];

    reg [15:0] p_in_c, p_in_h, p_in_w, p_out_c, p_out_h, p_out_w;   // ≤302
    reg [7:0]  p_k, p_pad, p_stride, p_act, p_type;
    reg [7:0]  p_out_row_tile, p_in_row_tile, p_in_cb, p_out_cb, p_row_block;   // ≤19/4

    //-----------------------------------------------------------------------
    // 派生量预计算（消除 S_RUN 每拍 32-bit 变量乘法链——setup 违例 -30.8ns 主因）
    // 层级常量（每层一次，S_PREP_L 拍）：in_cb_stride / out_cb_stride /
    //   out_rb_stride / rb_tile_stride / w_rb_beats
    // 行块常量（每行块一次，S_PREP 拍）：rb_base / r0 / r1 / load_rows /
    //   in_rb_base / in_seg_words / in_seg_tail / out_rb_base / out_seg_words
    // S_RUN 地址全改增量（段内 +8，段尾 +in_cb_stride - in_seg_tail），无乘法。
    // 与原组合公式数学等价（见 ref_cnn_top.py / 旧注释），事件序列不变。
    //-----------------------------------------------------------------------
    reg signed [31:0] rb_base_r;        // 当前行块 base 行号（= rb*tile*stride - pad，增量维护）
    reg [31:0] in_hw_r;                 // p_in_h * p_in_w
    reg [31:0] out_hw_r;                // p_out_h * p_out_w
    reg [31:0] in_cb_stride_r;          // p_in_h*p_in_w*8（输入每 cb 块字节偏移）
    reg [31:0] out_cb_stride_r;         // p_out_h*p_out_w*8（输出每 cb 块字节偏移）
    reg [31:0] out_rb_stride_r;         // p_out_row_tile*p_out_w*8（输出每行块字节偏移）
    reg signed [31:0] rb_tile_stride_r; // p_out_row_tile * p_stride（rb_base 增量）
    reg [31:0] w_cb_r;                  // p_out_cb * (type==4 ? 1 : p_in_cb)
    reg [31:0] w_rb_beats_r;            // w_cb_r * 72（每行块权重拍数）
    reg [31:0] in_row_w8_r;             // p_in_w * 8
    reg signed [15:0] r0_in_r;          // max(rb_base_r, 0)，≤in_h
    reg signed [15:0] r1_in_r;          // min(rb_base_r + in_row_tile, in_h)
    reg [7:0]  load_rows_r;             // max(r1_in_r - r0_in_r, 0)，≤in_row_tile
    reg [31:0] in_rb_base_r;            // r0_in_r * p_in_w * 8（输入行块内偏移）
    reg [31:0] in_seg_words_r;          // load_rows_r * p_in_w（每输入块段拍数）
    reg [31:0] in_seg_tail_r;           // (in_seg_words_r - 1) * 8（段尾地址回退）
    reg [31:0] out_rb_base_r;           // rb * out_rb_stride_r（输出行块内偏移）
    reg [15:0] out_row_prod_r;          // rb * p_out_row_tile
    reg [15:0] rb_out_rows_r;           // min(out_row_prod_r + tile, out_h) - out_row_prod_r 裁剪
    reg [31:0] out_seg_words_r;         // rb_out_rows_r * p_out_w（每输出块段拍数）
    reg [31:0] out_seg_tail_r;          // (out_seg_words_r - 1) * 8（段尾地址回退）

    //-----------------------------------------------------------------------
    // 执行状态机
    //-----------------------------------------------------------------------
    localparam S_IDLE      = 4'd0;
    localparam S_RD_PARAM  = 4'd1;
    localparam S_RD_SCALE  = 4'd2;
    localparam S_CFG       = 4'd3;
    localparam S_WR_CFG    = 4'd4;
    localparam S_PREP_L    = 4'd11;   // 层级派生量预计算（每层一次）
    localparam S_WR_BASE   = 4'd5;
    localparam S_PREP      = 4'd10;   // 行块派生量预计算（每行块一次）
    localparam S_START     = 4'd6;
    localparam S_RUN       = 4'd7;
    localparam S_NEXT_RB   = 4'd8;
    localparam S_CLEAR     = 4'd9;

    reg [3:0] state;
    reg [31:0] rd_cnt;          // param/scale 读计数
    reg [31:0] cfg_idx;         // cfg 写入索引
    reg [15:0] rb;              // 行块索引（≤row_block）

    // DMA 计数器
    reg [7:0]  dma_icb;                 // 输入：块（≤4）
    reg [15:0] dma_ibeat;             // 输入：段内拍（≤in_seg_words）
    reg [15:0] dma_wbeat;             // 权重：总拍（64-bit，≤w_rb_beats）
    reg [7:0]  dma_ocb;                 // 输出：块（≤4）
    reg [15:0] dma_obeat;             // 输出：段内拍（≤out_seg_words）
    reg        input_done, weight_done, output_done;   // 本行块完成标志

    // 主接口握手完成
    wire pr_got = pr_read && !pr_waitrequest;
    wire sr_got = sr_read && !sr_waitrequest;
    wire lr_got = lr_read && !lr_waitrequest;
    wire wr_got = wr_read && !wr_waitrequest;
    wire ow_got = ow_write && !ow_waitrequest;

    // core 流映射
    assign core_i_valid  = lr_got;
    assign core_i_data   = lr_readdata;
    assign core_iw_valid = wr_got;
    assign core_iw_data  = wr_readdata;
    assign lr_read       = core_i_ready;   // 组合：无 1 拍残留（边界精确）
    assign wr_read       = core_ow_ready;
    assign core_o_ready  = !output_done;   // core 输出不阻塞（与 v2 tb 的 o_ready=1 一致）
    assign ow_writedata  = core_o_data;
    assign ow_write      = core_o_valid;

    //-----------------------------------------------------------------------
    // 主状态机
    //-----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            rb <= 0; rd_cnt <= 0; cfg_idx <= 0;
            pr_read <= 0; sr_read <= 0;
            core_cfg_we <= 0; core_start <= 0;
            start_clear_pending <= 0;
            dma_icb <= 0; dma_ibeat <= 0; dma_wbeat <= 0;
            dma_ocb <= 0; dma_obeat <= 0;
            input_done <= 0; weight_done <= 0; output_done <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    start_clear_pending <= 0;
                    if (reg_start[0]) begin
                        rd_cnt <= 0;
                        pr_address <= reg_param;
                        sr_address <= reg_scale;
                        state <= S_RD_PARAM;
                    end
                end

                //---- 读 param 块（27 字）----
                S_RD_PARAM: begin
                    pr_read <= 1'b1;
                    if (pr_got) begin
                        param_buf[rd_cnt] <= pr_readdata;
                        pr_address <= pr_address + 4;
                        if (rd_cnt == 26) begin
                            pr_read <= 0;
                            rd_cnt <= 0;
                            state <= S_CFG;
                        end else
                            rd_cnt <= rd_cnt + 1;
                    end
                end

                //---- 读 scale（4×out_c 字）----
                S_RD_SCALE: begin
                    sr_read <= 1'b1;
                    if (sr_got) begin
                        req_buf[rd_cnt] <= sr_readdata;
                        sr_address <= sr_address + 4;
                        if (rd_cnt == 4 * p_out_c - 1) begin
                            sr_read <= 0;
                            rd_cnt <= 0;
                            state <= S_WR_CFG;
                        end else
                            rd_cnt <= rd_cnt + 1;
                    end
                end

                //---- 解析（param_buf → p_*）----
                S_CFG: begin
                    p_in_c  <= param_buf[4];
                    p_in_h  <= param_buf[5];
                    p_in_w  <= param_buf[6];
                    p_out_c <= param_buf[7];
                    p_out_h <= param_buf[8];
                    p_out_w <= param_buf[9];
                    p_k     <= param_buf[10];
                    p_pad   <= param_buf[11];
                    p_stride<= param_buf[13];
                    p_act   <= param_buf[14];
                    p_type  <= param_buf[15];
                    p_out_row_tile <= param_buf[17];
                    p_in_row_tile  <= param_buf[18];
                    p_row_block <= param_buf[19];
                    p_in_cb  <= (param_buf[4]  + 7) >> 3;
                    p_out_cb <= (param_buf[7] + 7) >> 3;
                    cfg_idx <= 0; rb <= 0;
                    state <= S_PREP_L;   // 先预计算层级常量，再读 scale（需 p_out_c）
                end

                //---- 层级派生量预计算（每层一次，2 拍；消除 S_RUN 每拍 32×32 乘法链）----
                S_PREP_L: begin
                    if (rd_cnt == 0) begin
                        rd_cnt <= 1;
                        rb_base_r <= 32'sd0 - $signed(p_pad);   // rb=0：0*tile*stride - pad
                        in_hw_r  <= p_in_h * p_in_w;
                        out_hw_r <= p_out_h * p_out_w;
                        rb_tile_stride_r <= p_out_row_tile * p_stride;
                        w_cb_r   <= p_out_cb * (p_type == 4 ? 1 : p_in_cb);
                        in_row_w8_r <= p_in_w << 3;
                    end else begin
                        rd_cnt <= 0;
                        in_cb_stride_r  <= in_hw_r << 3;
                        out_cb_stride_r <= out_hw_r << 3;
                        out_rb_stride_r <= (p_out_row_tile * p_out_w) << 3;
                        w_rb_beats_r    <= w_cb_r * 72;
                        state <= S_RD_SCALE;
                    end
                end

                //---- 写 cfg（16 标量 + 4×out_c requant + base_row）----
                S_WR_CFG: begin
                    core_cfg_we <= 1'b1;
                    if (cfg_idx < 16) begin
                        core_cfg_sel <= 0;
                        case (cfg_idx)
                            0:  begin core_cfg_addr <= 0;  core_cfg_wdata <= p_type; end
                            1:  begin core_cfg_addr <= 1;  core_cfg_wdata <= p_act; end
                            2:  begin core_cfg_addr <= 2;  core_cfg_wdata <= p_in_c; end
                            3:  begin core_cfg_addr <= 3;  core_cfg_wdata <= p_in_h; end
                            4:  begin core_cfg_addr <= 4;  core_cfg_wdata <= p_in_w; end
                            5:  begin core_cfg_addr <= 5;  core_cfg_wdata <= p_out_c; end
                            6:  begin core_cfg_addr <= 6;  core_cfg_wdata <= p_out_h; end
                            7:  begin core_cfg_addr <= 7;  core_cfg_wdata <= p_out_w; end
                            8:  begin core_cfg_addr <= 8;  core_cfg_wdata <= p_k; end
                            9:  begin core_cfg_addr <= 9;  core_cfg_wdata <= p_pad; end
                            10: begin core_cfg_addr <= 10; core_cfg_wdata <= p_stride; end
                            11: begin core_cfg_addr <= 11; core_cfg_wdata <= p_out_row_tile; end
                            12: begin core_cfg_addr <= 12; core_cfg_wdata <= p_in_row_tile; end
                            13: begin core_cfg_addr <= 13; core_cfg_wdata <= p_in_cb; end
                            14: begin core_cfg_addr <= 14; core_cfg_wdata <= p_out_cb; end
                            15: begin core_cfg_addr <= 15; core_cfg_wdata <= rb_base_r[31:0]; end
                            default: begin core_cfg_addr <= 0; core_cfg_wdata <= 0; end
                        endcase
                    end else if (cfg_idx < 16 + p_out_c) begin
                        core_cfg_sel <= 3'd1;   // bias_int
                        core_cfg_addr <= cfg_idx - 16;
                        core_cfg_wdata <= req_buf[p_out_c + (cfg_idx - 16)];
                    end else if (cfg_idx < 16 + 2 * p_out_c) begin
                        core_cfg_sel <= 3'd2;   // mult
                        core_cfg_addr <= cfg_idx - 16 - p_out_c;
                        core_cfg_wdata <= req_buf[cfg_idx - 16 - p_out_c];
                    end else if (cfg_idx < 16 + 3 * p_out_c) begin
                        core_cfg_sel <= 3'd3;   // shift
                        core_cfg_addr <= cfg_idx - 16 - 2 * p_out_c;
                        core_cfg_wdata <= req_buf[2 * p_out_c + (cfg_idx - 16 - 2 * p_out_c)];
                    end else begin
                        core_cfg_sel <= 3'd4;   // rcl6
                        core_cfg_addr <= cfg_idx - 16 - 3 * p_out_c;
                        core_cfg_wdata <= req_buf[3 * p_out_c + (cfg_idx - 16 - 3 * p_out_c)];
                    end
                    if (cfg_idx == 16 + 4 * p_out_c - 1) begin
                        // 最后一个 cfg（rcl6[out_c-1]）写入后由 S_WR_BASE 拉低 cfg_we
                        cfg_idx <= 0;
                        dma_icb <= 0; dma_ibeat <= 0; dma_wbeat <= 0;
                        dma_ocb <= 0; dma_obeat <= 0;
                        input_done <= 0; weight_done <= 0; output_done <= 0;
                        state <= S_WR_BASE;
                    end else
                        cfg_idx <= cfg_idx + 1;
                end

                //---- 行块 base_row（每行块）----
                S_WR_BASE: begin
                    core_cfg_we <= 1'b1;
                    core_cfg_sel <= 0; core_cfg_addr <= 15;
                    core_cfg_wdata <= rb_base_r[31:0];
                    state <= S_PREP;
                end

                //---- 行块派生量预计算（每行块一次，5 拍；消除 S_RUN 每拍乘法）----
                S_PREP: begin
                    if (rd_cnt == 0) begin
                        rd_cnt <= 1;
                        r0_in_r <= (rb_base_r > 32'sd0) ? rb_base_r : 32'sd0;
                        r1_in_r <= (rb_base_r + $signed(p_in_row_tile) < $signed(p_in_h)) ?
                                   (rb_base_r + $signed(p_in_row_tile)) : $signed(p_in_h);
                        out_rb_base_r  <= rb * out_rb_stride_r;
                        out_row_prod_r <= rb * p_out_row_tile;
                    end else if (rd_cnt == 1) begin
                        rd_cnt <= 2;
                        load_rows_r  <= (r1_in_r > r0_in_r) ? (r1_in_r - r0_in_r) : 32'sd0;
                        in_rb_base_r <= r0_in_r * in_row_w8_r;
                    end else if (rd_cnt == 2) begin
                        rd_cnt <= 3;
                        in_seg_words_r <= load_rows_r * p_in_w;
                        rb_out_rows_r <= (out_row_prod_r + p_out_row_tile < p_out_h) ?
                                         p_out_row_tile : (p_out_h - out_row_prod_r);
                    end else if (rd_cnt == 3) begin
                        rd_cnt <= 4;
                        in_seg_tail_r  <= (in_seg_words_r - 1) * 8;
                        out_seg_words_r <= rb_out_rows_r * p_out_w;
                    end else begin
                        rd_cnt <= 0;
                        out_seg_tail_r <= (out_seg_words_r - 1) * 8;
                        state <= S_START;
                    end
                end

                S_START: begin
                    core_cfg_we <= 0;
                    lr_address <= reg_ddrin + in_rb_base_r;
                    wr_address <= reg_ddrw;
                    ow_address <= reg_ddrout + out_rb_base_r;
                    core_start <= 1'b1;
                    state <= S_RUN;
                end

                //---- 运行：DMA 跟随 core 流（地址增量推进，无每拍乘法）----
                S_RUN: begin
                    core_start <= 0;

                    if (lr_got &&
                        !(dma_icb == p_in_cb - 1 && dma_ibeat == in_seg_words_r - 1)) begin
                        if (dma_ibeat == in_seg_words_r - 1) begin
                            lr_address <= lr_address + in_cb_stride_r - in_seg_tail_r;
                            dma_ibeat <= 0;
                            dma_icb <= dma_icb + 1;
                        end else begin
                            lr_address <= lr_address + 8;
                            dma_ibeat <= dma_ibeat + 1;
                        end
                    end
                    if (wr_got && dma_wbeat < w_rb_beats_r - 1) begin
                        dma_wbeat <= dma_wbeat + 1;
                        wr_address <= wr_address + 8;
                    end
                    if (ow_got &&
                        !(dma_ocb == p_out_cb - 1 && dma_obeat == out_seg_words_r - 1)) begin
                        if (dma_obeat == out_seg_words_r - 1) begin
                            ow_address <= ow_address + out_cb_stride_r - out_seg_tail_r;
                            dma_obeat <= 0;
                            dma_ocb <= dma_ocb + 1;
                        end else begin
                            ow_address <= ow_address + 8;
                            dma_obeat <= dma_obeat + 1;
                        end
                    end

                    if (core_done) begin
                        state <= S_NEXT_RB;
                    end
                end

                S_NEXT_RB: begin
                    if (rb == p_row_block - 1)
                        state <= S_CLEAR;
                    else begin
                        rb <= rb + 1;
                        rb_base_r <= rb_base_r + rb_tile_stride_r;   // 增量：base += tile*stride
                        dma_icb <= 0; dma_ibeat <= 0; dma_wbeat <= 0;
                        dma_ocb <= 0; dma_obeat <= 0;
                        input_done <= 0; weight_done <= 0; output_done <= 0;
                        state <= S_WR_BASE;
                    end
                end

                S_CLEAR: begin
                    start_clear_pending <= 1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
