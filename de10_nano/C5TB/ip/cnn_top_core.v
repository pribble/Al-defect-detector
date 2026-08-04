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

module cnn_top_core (
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
    input  wire                pr_readdatavalid,
    input  wire                pr_waitrequest,

    // ---- scale 读（32-bit）----
    output reg  [31:0]         sr_address,
    output reg                 sr_read,
    input  wire [31:0]         sr_readdata,
    input  wire                sr_readdatavalid,
    input  wire                sr_waitrequest,

    // ---- 输入读（64-bit）----
    output reg  [31:0]         lr_address,
    output wire                lr_read,
    input  wire [63:0]         lr_readdata,
    input  wire                lr_readdatavalid,
    input  wire                lr_waitrequest,

    // ---- 权重读（64-bit）----
    output reg  [31:0]         wr_address,
    output wire                wr_read,
    input  wire [63:0]         wr_readdata,
    input  wire                wr_readdatavalid,
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
            // 调试只读寄存器（复现核专用，非黑盒协议；软件 start_fpga
            // 轮询超时期间读取，定位死锁现场）：
            //   0x44 = {3'b0, start_clear_pending, core_done, 1'b0,
            //          state[3:0], lr_pending[2:0], wr_pending[2:0],
            //          dma_icb[7:0], dma_ibeat[7:0]}
            //   0x48 = {dma_wbeat[19:0], dma_ibeat[15:8],
            //          lr_round_end, lr_round_reset, core_i_ready, core_ow_ready}
            8'h11: as_readdata = {3'b0, start_clear_pending, core_done, 1'b0,
                                  state, lr_pending, wr_pending,
                                  dma_icb, dma_ibeat[7:0]};
            8'h12: as_readdata = {dma_wbeat, dma_ibeat[15:8],
                                  lr_round_end, lr_round_reset,
                                  core_i_ready, core_ow_ready};
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

    reg [15:0] p_in_c, p_in_h, p_in_w, p_out_c, p_out_h, p_out_w;   // ≤302
    reg [31:0] p_input_offset, p_weight_offset, p_output_offset;   // param[0/1/3]，字偏移（×8 = 字节）
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
    reg [7:0]  lr_last_cb_r;            // lr 轮末 cb：CONV = p_in_cb-1，DW = 0（单 cb）
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
    reg [19:0] dma_wbeat;             // 权重：总拍（64-bit，≤w_rb_beats ≤1.18M）
    reg [7:0]  dma_ocb;                 // 输出：块（≤4）
    reg [15:0] dma_obeat;             // 输出：段内拍（≤out_seg_words）

    // 主接口读握手（mm_bridge_sdram0 为流水读桥：waitrequest 拉低仅表示命令
    // 被接受，数据由 readdatavalid 延迟返回）：
    //   每笔读 = read 拉高 → 命令接受（read && !waitrequest）→ 拉低 read
    //            → 等 readdatavalid（数据与 valid 同拍）→ 完成
    //   lr/wr 多笔在途（≤4，桥 MAX_PENDING_RESPONSES=4）：命令在 core 消费
    //   节奏下提前发出（地址/计数在命令接受拍推进），数据按序返回。
    //   流式化后每 o_group 轮重读输入，单笔在途每 8B 等一轮 DDR 延迟会
    //   把 fpga_time 拖到秒级，必须多笔在途隐藏延迟。
    //   pr/sr（param/scale 块读）保持单笔：pr 27 字开销可忽略，sr 后续优化。
    // 写完成 = write && !waitrequest，协议不变。
    reg pr_pending, sr_pending;
    reg [2:0] lr_pending, wr_pending;   // 在途笔数 0..4

    wire pr_got = pr_pending && pr_readdatavalid;
    wire sr_got = sr_pending && sr_readdatavalid;
    wire lr_got = lr_readdatavalid;
    wire wr_got = wr_readdatavalid;
    wire ow_got = ow_write && !ow_waitrequest;

    // core 流映射
    assign core_i_valid  = lr_got;
    assign core_i_data   = lr_readdata;
    assign core_iw_valid = wr_got;
    assign core_iw_data  = wr_readdata;
    // lr 轮边界（每 o_group 一轮：CONV = 全部 in_cb 个 cb；DW = 单 cb）
    wire lr_round_end   = (dma_icb == lr_last_cb_r && dma_ibeat == in_seg_words_r - 1);
    wire lr_round_reset = (lr_pending == 0 && lr_round_end && core_i_ready);
    assign lr_read = core_i_ready && (lr_pending < 3'd4) && !lr_round_end && !lr_round_reset;
    assign wr_read = core_ow_ready && (wr_pending < 3'd4) && (dma_wbeat < w_rb_beats_r);
    assign core_o_ready  = 1'b1;   // core 输出不阻塞（与 v2 tb 的 o_ready=1 一致）
    assign ow_writedata  = core_o_data;
    assign ow_write      = core_o_valid;

    // lr/wr 多笔在途计数：数据返回 -1、命令接受 +1（同拍净 0）；
    // core_done 拍清零（行块末丢弃在途残留，下个行块从头重读）——
    // 清零并入本自动机，避免与主状态机多 always 驱动（Quartus 10028）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lr_pending <= 0;
            wr_pending <= 0;
        end else if (core_done) begin
            lr_pending <= 0;
            wr_pending <= 0;
        end else begin
            // lr/wr 共享 load master：任一方的 readdatavalid 都会到达两路。
            // 不在途一侧必须"非零保护"（pending==0 时不减），否则被对方
            // 数据返回下溢回绕（0-1=7），pending 永久不归零 → lr_round_reset
            // 永远不触发 → 死锁（板上实测：wbeat=0 但 wr_p=1、lr_p=3 卡死）。
            // 误减边界：本侧在途>0 时对方数据返回会误减 1，但对方返回与
            // 本侧命令接受同拍 +1/-1 抵消（单 readdatavalid/拍），净计数准确。
            if (lr_readdatavalid && lr_pending != 3'd0) lr_pending <= lr_pending - 3'd1;
            if (lr_read && !lr_waitrequest)             lr_pending <= lr_pending + 3'd1;
            if (wr_readdatavalid && wr_pending != 3'd0) wr_pending <= wr_pending - 3'd1;
            if (wr_read && !wr_waitrequest)             wr_pending <= wr_pending + 3'd1;
        end
    end

    //-----------------------------------------------------------------------
    // 主状态机
    //-----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            rb <= 0; rd_cnt <= 0; cfg_idx <= 0;
            pr_read <= 0; sr_read <= 0;
            pr_pending <= 0; sr_pending <= 0;
            core_cfg_we <= 0; core_start <= 0;
            start_clear_pending <= 0;
            dma_icb <= 0; dma_ibeat <= 0; dma_wbeat <= 0;
            dma_ocb <= 0; dma_obeat <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    start_clear_pending <= 0;
                    // 自清拍（start_clear_pending=1）不重启：reg_start 在本拍沿
                    // 才被清 0，沿前判定会误触发新一轮执行
                    if (reg_start[0] && !start_clear_pending) begin
                        rd_cnt <= 0;
                        pr_address <= reg_param;
                        sr_address <= reg_scale;
                        pr_pending <= 0; sr_pending <= 0;
                        state <= S_RD_PARAM;
                    end
                end

                //---- 读 param 块（27 字）：每笔 read 拉高 → 接受后拉低 → 等 readdatavalid ----
                S_RD_PARAM: begin
                    if (pr_got) begin
                        param_buf[rd_cnt] <= pr_readdata;
                        pr_address <= pr_address + 4;
                        pr_pending <= 0;
                        if (rd_cnt == 26) begin
                            pr_read <= 0;
                            rd_cnt <= 0;
                            state <= S_CFG;
                        end else begin
                            rd_cnt <= rd_cnt + 1;
                        end
                    end else if (!pr_pending) begin
                        // 无在途：发新请求（read 拉高，地址已稳定）
                        pr_read <= 1'b1;
                        pr_pending <= 1'b1;
                    end else if (pr_read && !pr_waitrequest) begin
                        // 命令被桥接受：拉低 read，只发这一笔，等 readdatavalid
                        pr_read <= 1'b0;
                    end
                end

                //---- 读 scale（4×out_c 字），边读边写 core requant 数组 ----
                // 数据返回拍直接把 sr_readdata 写进 core（无需 req_buf 中转：
                // 4096 字寄存器堆爆资源；布局与软件一致：mult/bias/shift/rcl6
                // 各 out_c 字），S_WR_CFG 只写标量 0..15。数组/标量写入顺序
                // 无依赖（core 按地址索引）。
                S_RD_SCALE: begin
                    core_cfg_we <= 1'b0;
                    if (sr_got) begin
                        core_cfg_we    <= 1'b1;
                        core_cfg_wdata <= sr_readdata;
                        if (rd_cnt < p_out_c) begin
                            core_cfg_sel  <= 3'd2;                 // mult
                            core_cfg_addr <= rd_cnt;
                        end else if (rd_cnt < 2 * p_out_c) begin
                            core_cfg_sel  <= 3'd1;                 // bias_int
                            core_cfg_addr <= rd_cnt - p_out_c;
                        end else if (rd_cnt < 3 * p_out_c) begin
                            core_cfg_sel  <= 3'd3;                 // shift
                            core_cfg_addr <= rd_cnt - 2 * p_out_c;
                        end else begin
                            core_cfg_sel  <= 3'd4;                 // rcl6
                            core_cfg_addr <= rd_cnt - 3 * p_out_c;
                        end
                        sr_address <= sr_address + 4;
                        sr_pending <= 0;
                        if (rd_cnt == 4 * p_out_c - 1) begin
                            sr_read <= 0;
                            rd_cnt <= 0;
                            state <= S_WR_CFG;
                        end else begin
                            rd_cnt <= rd_cnt + 1;
                        end
                    end else if (!sr_pending) begin
                        sr_read <= 1'b1;
                        sr_pending <= 1'b1;
                    end else if (sr_read && !sr_waitrequest) begin
                        sr_read <= 1'b0;
                    end
                end

                //---- 解析（param_buf → p_*）----
                // param[0..3] 为 input/weight/scale/output 四区字偏移（软件
                // conv_op.cc 每层分配并写入；FPGA 必须把 input/weight/output
                // 偏移加进 DDR 基址，否则第 1 层起读写错位（层 0 偏移恰好全 0
                // 掩盖了问题）。scale 区每层 memcpy 到同一 cb_scale，偏移恒 0。
                S_CFG: begin
                    p_input_offset  <= param_buf[0];
                    p_weight_offset <= param_buf[1];
                    p_output_offset <= param_buf[3];
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
                        lr_last_cb_r <= (p_type == 4) ? 8'd0 : (p_in_cb - 1);
                    end else begin
                        rd_cnt <= 0;
                        in_cb_stride_r  <= in_hw_r << 3;
                        out_cb_stride_r <= out_hw_r << 3;
                        out_rb_stride_r <= (p_out_row_tile * p_out_w) << 3;
                        w_rb_beats_r    <= w_cb_r * (p_k == 1 ? 8 : 72);   // k=1：slice=8 拍；k=3：72 拍
                        state <= S_RD_SCALE;
                    end
                end

                //---- 写 cfg（16 标量 + 4×out_c requant + base_row）----
                //---- 写标量 cfg（0..15）；requant 数组已在 S_RD_SCALE 边读边写 ----
                S_WR_CFG: begin
                    core_cfg_we <= 1'b1;
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
                    if (cfg_idx == 15) begin
                        cfg_idx <= 0;
                        dma_icb <= 0; dma_ibeat <= 0; dma_wbeat <= 0;
                        dma_ocb <= 0; dma_obeat <= 0;
                        state <= S_WR_BASE;
                    end else begin
                        cfg_idx <= cfg_idx + 1;
                    end
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
                    lr_address <= reg_ddrin + (p_input_offset << 3) + in_rb_base_r;
                    wr_address <= reg_ddrw + (p_weight_offset << 3);
                    ow_address <= reg_ddrout + (p_output_offset << 3) + out_rb_base_r;
                    core_start <= 1'b1;
                    state <= S_RUN;
                end

                //---- 运行：DMA 跟随 core 流（lr/wr 多笔在途，地址按命令接受拍推进）----
                S_RUN: begin
                    core_start <= 0;

                    // lr：命令接受拍推进地址/计数（cb 段尾跳转）；每 o_group 轮末
                    // （数据全返回后 core 重新拉 i_ready）重置：CONV 回行块首重读
                    // 全部 cb，DW 地址 +1 cb（每 o_group 只读自己的 cb）
                    if (lr_round_reset) begin
                        dma_icb <= 0; dma_ibeat <= 0;
                        lr_address <= (p_type == 4) ? (lr_address + in_cb_stride_r)
                                                    : (reg_ddrin + (p_input_offset << 3) + in_rb_base_r);
                    end else if (lr_read && !lr_waitrequest) begin
                        if (dma_ibeat == in_seg_words_r - 1) begin
                            lr_address <= lr_address + in_cb_stride_r - in_seg_tail_r;
                            dma_ibeat <= 0;
                            dma_icb <= dma_icb + 1;
                        end else begin
                            lr_address <= lr_address + 8;
                            dma_ibeat <= dma_ibeat + 1;
                        end
                    end
                    // wr：命令接受拍推进（连续，无跳转；行块末 core_done 清 pending）
                    if (wr_read && !wr_waitrequest) begin
                        if (dma_wbeat < w_rb_beats_r - 1) begin
                            dma_wbeat <= dma_wbeat + 1;
                            wr_address <= wr_address + 8;
                        end
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
