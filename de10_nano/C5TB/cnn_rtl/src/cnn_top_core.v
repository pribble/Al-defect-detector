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
//   1) param 块（20 字，0..19；边读边解析，无需缓存）
//   2) scale 区（4×out_c 字）→ requant 参数
//   3) 配 cnn_core → 行块循环 rb=0..row_block-1：
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
    output wire                pr_read,
    input  wire [31:0]         pr_readdata,
    input  wire                pr_readdatavalid,
    input  wire                pr_waitrequest,

    // ---- scale 读（32-bit）----
    output reg  [31:0]         sr_address,
    output wire                sr_read,
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
    input  wire                ow_waitrequest,

    // ---- load master 突发长度（lr/wr 共享；由 burst 控制器驱动）----
    output wire [4:0]          load_burstcount
);

    //-----------------------------------------------------------------------
    // 寄存器
    //-----------------------------------------------------------------------
    reg [31:0] reg_ddrin, reg_ddrw, reg_ddrout, reg_param, reg_scale;
    reg reg_start;

    assign as_waitrequest = 0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ddrin <= 0; reg_ddrw <= 0; reg_ddrout <= 0;
            reg_param <= 0; reg_scale <= 0;
        end else begin
            if (as_write) begin
                case (as_address)
                    8'h10: reg_ddrin   <= as_writedata;
                    8'h1C: reg_ddrw   <= as_writedata;
                    8'h28: reg_ddrout <= as_writedata;
                    8'h34: reg_param  <= as_writedata;
                    8'h40: reg_scale  <= as_writedata;
                    default: ;
                endcase
            end
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
    // cnn_core
    //-----------------------------------------------------------------------
    reg         core_cfg_we, core_start;
    wire        core_done;
    reg  [2:0]  core_cfg_sel;
    reg  [19:0] core_cfg_addr;
    reg  [31:0] core_cfg_wdata;
    wire        core_i_valid, core_i_ready, core_i_pf_ready;
    wire [63:0] core_i_data;
    wire        core_iw_valid, core_ow_ready;
    wire [63:0] core_iw_data;
    wire        core_o_valid, core_o_ready;
    wire [63:0] core_o_data;

    cnn_core core (
        .clk(clk), .rst_n(rst_n),
        .cfg_we(core_cfg_we), .cfg_sel(core_cfg_sel),
        .cfg_addr(core_cfg_addr), .cfg_wdata(core_cfg_wdata),
        .start(core_start), .o_done(core_done),
        .i_valid(core_i_valid), .i_ready(core_i_ready), .i_pf_ready(core_i_pf_ready), .i_data(core_i_data),
        .iw_valid(core_iw_valid), .ow_ready(core_ow_ready), .iw_data(core_iw_data),
        .o_valid(core_o_valid), .o_ready(core_o_ready), .o_data(core_o_data)
    );

    //-----------------------------------------------------------------------
    // param 解析 + 派生量
    //-----------------------------------------------------------------------

    reg [15:0] p_in_c, p_in_h, p_in_w, p_out_c, p_out_h, p_out_w;   // ≤302
    reg [31:0] p_input_offset, p_weight_offset, p_output_offset;   // param[0/1/3]，字偏移（×8 = 字节）
    reg [7:0]  p_pad, p_stride, p_act, p_type;
    reg [1:0]  p_k;   // 窄化（k ∈ {1,3}，比较/选择变短，拆 p_k→w_rb_beats_r 乘法链）
    reg [7:0]  p_out_row_tile, p_in_row_tile, p_in_cb, p_out_cb, p_row_block;   // ≤19/4

    // 4*out_c（scale 读取总字数；p_out_c ≤ 1024，移位替代乘法）
    wire [15:0] sr_total_w = p_out_c << 2;

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
    reg [15:0] w_cb_r;                  // p_out_cb * (type==4 ? 1 : p_in_cb)（窄化：实际 ≤1024，乘法树变短）
    reg [23:0] w_rb_beats_r;            // w_cb_r * 72（每行块权重拍数，窄化：≤4.7M）
    reg [23:0] w_rb_beats_last_q;       // w_rb_beats_r - 1（S_PREP_L 拍 2 预计算，拆 S_RUN 的 24-bit 减法链）
    reg [7:0]  wr_slice_q;              // S_PREP_L 拍 1 寄存 (p_k==1 ? 8 : 72)（拆 p_k 比较与乘法链）
    reg [7:0]  lr_last_cb_r;            // lr 轮末 cb：CONV = p_in_cb-1，DW = 0（单 cb）
    reg [31:0] in_row_w8_r;             // p_in_w * 8
    reg signed [15:0] r0_in_r;          // max(rb_base_r, 0)，≤in_h
    reg signed [15:0] r1_in_r;          // min(rb_base_r + in_row_tile, in_h)
    reg [15:0] load_rows_r;             // max(r1_in_r - r0_in_r, 0)，≤in_row_tile
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
    localparam S_IDLE     = 4'd0;
    localparam S_RD_PARAM = 4'd1;   // 读 param 块（0..19，边读边解析）
    localparam S_PREP_L   = 4'd2;   // 层级派生量预计算（每层一次）
    localparam S_RD_SCALE = 4'd3;   // 读 scale（4×out_c 字），边读边写 requant 数组
    localparam S_WR_CFG   = 4'd4;   // 写标量 cfg（0..15）
    localparam S_WR_BASE  = 4'd5;   // 每行块写 base_row
    localparam S_PREP     = 4'd6;   // 行块派生量预计算（每行块一次）
    localparam S_WR_TILE  = 4'd7;   // 每行块写 cfg 11（本行块实际输出行数）
    localparam S_START    = 4'd8;   // 拉高 core start
    localparam S_RUN      = 4'd9;   // 运行：DMA 跟随 core 流
    localparam S_NEXT_RB  = 4'd10;  // 下一行块 / 完成

    reg [3:0] state;
    reg [31:0] rd_cnt;          // param/scale 读返回计数（解析用）
    reg [31:0] cfg_idx;         // cfg 写入索引
    reg [15:0] rb;              // 行块索引（≤row_block）

    // pr/sr 多笔在途：命令计数与在途笔数分离，read 保持拉高直到
    // 命令发完或在途笔数达到桥上限（4 = MAX_PENDING_RESPONSES）。
    reg [4:0]  pr_cmd_cnt;      // pr 已发命令数（0..20）
    reg [15:0] sr_cmd_cnt;      // sr 已发命令数（0..4*out_c）
    reg [7:0]  pr_pending;      // pr 在途笔数（≤4）
    reg [7:0]  sr_pending;      // sr 在途笔数（≤4）

    assign pr_read = (state == S_RD_PARAM) && (pr_cmd_cnt < 5'd20) && (pr_pending < 8'd4);
    assign sr_read = (state == S_RD_SCALE) && (sr_cmd_cnt < sr_total_w) && (sr_pending < 8'd4);

    // DMA 计数器
    reg [7:0]  dma_icb;                 // 输入：块（≤4）
    reg [15:0] dma_ibeat;             // 输入：段内拍（≤in_seg_words）
    reg [19:0] dma_wbeat;             // 权重：总拍（64-bit，≤w_rb_beats ≤1.18M）
    reg [7:0]  dma_ocb;                 // 输出：块（≤4）
    reg [15:0] dma_obeat;             // 输出：段内拍（≤out_seg_words）

    reg [7:0]  lr_pending, wr_pending;   // 在途笔数 0..4（8-bit：3-bit 会回绕，
    // 回绕后 lr_p<4 恒真 → lr_read 失去在途限制 → 命令洪泛 → 死锁（板上
    // 实测 lr_cmd=52772 远超 lr_rdv=24528）
    // lr/wr 共享 load master：cnn_top.v 把 lr/wr 的 readdatavalid 都接到
    // load_avm_readdatavalid，任一方的数据返回都会广播到两路。命令串行
    // （S_LOAD/S_WEIGHT 互斥，lr_read 与 wr_read 不同时拉高），但返回会
    // 交错：S_LOAD 末几笔在途命令的返回在 S_WEIGHT 期间到达。裸接时 wr
    // 返回会冒充 i_valid、lr 返回会冒充 iw_valid → 装载计数错乱 → core
    // 状态机破坏 → core_done 不来 → 死锁（板上实测 lr_rdv=lr_cmd+wr_cmd）。
    // 正解：命令类型 FIFO（深度 4 = 桥 MAX_PENDING_RESPONSES，Avalon
    // 返回保序），返回拍按队首类型路由回本侧。
    reg [3:0]  cmd_type_q;             // 4 深：0=lr，1=wr
    reg [2:0]  cmd_beats_q [0:3];      // 每命令剩余返回拍数（burstcount 1..4）
    reg [1:0]  cmd_head, cmd_tail;
    reg [2:0]  cmd_cnt;
    wire cmd_accept = (lr_read && !lr_waitrequest) || (wr_read && !wr_waitrequest);
    wire cmd_rdv_beat = (lr_readdatavalid || wr_readdatavalid) && (cmd_cnt != 3'd0);
    wire cmd_is_wr  = cmd_type_q[cmd_head];
    wire cmd_last_beat = cmd_rdv_beat && (cmd_beats_q[cmd_head] == 3'd1);
    wire cmd_nlast_beat = cmd_rdv_beat && (cmd_beats_q[cmd_head] != 3'd1);
    wire cmd_winner_wr = wr_read;      // lr/wr 不同时拉高，无冲突

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_head <= 0; cmd_tail <= 0; cmd_cnt <= 0;
        end else if (core_done) begin
            // 行块结束：丢弃在途残留（与 lr/wr_pending 清零一致），
            // 残留返回到达时 cmd_cnt==0 → 不 pop → 被丢弃
            cmd_head <= 0; cmd_tail <= 0; cmd_cnt <= 0;
        end else begin
            // 4 种组合显式处理，避免"同拍入队 + 队首非末拍递减"漏更新
            // （漏更新会让 beats_left 比真实返回多 1，命令完成判据错位）。
            if (cmd_accept && (cmd_cnt < 3'd4)) begin
                // 同拍入队
                cmd_type_q[cmd_tail]  <= cmd_winner_wr;
                cmd_beats_q[cmd_tail] <= load_burstcount[2:0];
                cmd_tail <= cmd_tail + 2'd1;
                if (cmd_last_beat) begin
                    // 同拍出队：net 0，head 同步推进
                    cmd_head <= cmd_head + 2'd1;
                end else begin
                    cmd_cnt <= cmd_cnt + 1;
                    if (cmd_nlast_beat)
                        cmd_beats_q[cmd_head] <= cmd_beats_q[cmd_head] - 3'd1;
                end
            end else if (cmd_last_beat) begin
                cmd_head <= cmd_head + 2'd1;
                cmd_cnt <= cmd_cnt - 1;
            end else if (cmd_nlast_beat) begin
                cmd_beats_q[cmd_head] <= cmd_beats_q[cmd_head] - 3'd1;
            end
        end
    end

    wire lr_got = (lr_readdatavalid || wr_readdatavalid) && (cmd_cnt != 3'd0) && !cmd_is_wr;
    wire wr_got = (lr_readdatavalid || wr_readdatavalid) && (cmd_cnt != 3'd0) &&  cmd_is_wr;
    wire ow_got = ow_write && !ow_waitrequest;

    //-----------------------------------------------------------------------
    // lr/wr 输入 FIFO（step A：解耦 readdatavalid 返回与 core 消费）
    //   FWFT：core_i_valid = !empty，数据 = 队头；pop = core 握手成功。
    //   step A 仍按 core 就绪才发读命令（FIFO 深度在途 ≤4），行为与直连等价；
    //   step B 改为按 FIFO 空间发命令后，这里就是真正的解耦缓冲。
    //-----------------------------------------------------------------------
    localparam LR_FIFO_DEPTH = 32;
    localparam LR_FIFO_AW    = 5;
    localparam [LR_FIFO_AW:0] LR_FIFO_DEPTH_C = LR_FIFO_DEPTH;   // 定宽常量（full 比较用）
    reg [63:0] lr_fifo [0:LR_FIFO_DEPTH-1];
    reg [63:0] wr_fifo [0:LR_FIFO_DEPTH-1];
    reg [LR_FIFO_AW-1:0] lr_wptr, lr_rptr, wr_wptr, wr_rptr;
    reg [LR_FIFO_AW:0]   lr_fifo_cnt, wr_fifo_cnt;   // 0..32
    reg [5:0]            lr_inflight, wr_inflight;   // 本侧在途未返回拍数（burst 拍）

    wire lr_fifo_empty = (lr_fifo_cnt == 0);
    wire lr_fifo_full  = (lr_fifo_cnt == LR_FIFO_DEPTH_C);
    wire wr_fifo_empty = (wr_fifo_cnt == 0);
    wire wr_fifo_full  = (wr_fifo_cnt == LR_FIFO_DEPTH_C);

    // 队头（FWFT 组合读；空时数据无效，core 由 valid 门控不采样）
    wire [63:0] lr_fifo_head = lr_fifo[lr_rptr];
    wire [63:0] wr_fifo_head = wr_fifo[wr_rptr];

    // 入队：返回拍按 cmd FIFO 队首类型路由；满保护（step A 不应发生，防御）
    wire lr_fifo_push = lr_got && !lr_fifo_full;
    wire wr_fifo_push = wr_got && !wr_fifo_full;
    // 出队：core 侧握手（i_ready 或 i_pf_ready / ow_ready）
    wire lr_fifo_pop  = !lr_fifo_empty && (core_i_ready || core_i_pf_ready);
    wire wr_fifo_pop  = !wr_fifo_empty && core_ow_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lr_wptr <= 0; lr_rptr <= 0; lr_fifo_cnt <= 0;
        end else if (core_done) begin
            // 行块结束：丢弃 FIFO 残留（在途返回由 cmd FIFO 清空拦截，不会写入）
            lr_wptr <= 0; lr_rptr <= 0; lr_fifo_cnt <= 0;
        end else begin
            if (lr_fifo_push)
                lr_fifo[lr_wptr] <= lr_readdata;
            // 单语句净 0：同拍 push+pop 计数不变（与 pending 历史 bug 同类防护）
            lr_fifo_cnt <= lr_fifo_cnt + (lr_fifo_push ? 1'b1 : 1'b0)
                                      - (lr_fifo_pop  ? 1'b1 : 1'b0);
            if (lr_fifo_push) lr_wptr <= lr_wptr + 1'b1;
            if (lr_fifo_pop)  lr_rptr <= lr_rptr + 1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_wptr <= 0; wr_rptr <= 0; wr_fifo_cnt <= 0;
        end else if (core_done) begin
            wr_wptr <= 0; wr_rptr <= 0; wr_fifo_cnt <= 0;
        end else begin
            if (wr_fifo_push)
                wr_fifo[wr_wptr] <= wr_readdata;
            wr_fifo_cnt <= wr_fifo_cnt + (wr_fifo_push ? 1'b1 : 1'b0)
                                        - (wr_fifo_pop  ? 1'b1 : 1'b0);
            if (wr_fifo_push) wr_wptr <= wr_wptr + 1'b1;
            if (wr_fifo_pop)  wr_rptr <= wr_rptr + 1'b1;
        end
    end

    // core 流映射（FIFO 队头）
    assign core_i_valid  = !lr_fifo_empty;
    assign core_i_data   = lr_fifo_head;
    assign core_iw_valid = !wr_fifo_empty;
    assign core_iw_data  = wr_fifo_head;

    //-----------------------------------------------------------------------
    // load master burst 控制器（step B）
    //   lr/wr 各按 FIFO 空间与剩余量发 ≤4 拍突发；共享 load master，
    //   wr 优先；busy 锁存保证 waitrequest 期间同一命令稳定在总线上。
    //   返回拍按 cmd FIFO 队首类型路由进 lr/wr FIFO，core 只从 FIFO 消费，
    //   因此不再需要旧 round_end 状态机（返回错位问题被 FIFO 吸收）。
    //-----------------------------------------------------------------------
    reg [7:0]  lr_rounds_done;   // 已发完的 lr 轮数（0..out_cb）
    // 容量预算：FIFO 现有 + 在途未返回拍（两者之和才是未来可能占用的 FIFO 深度）
    wire [5:0] lr_avail = 6'd32 - (lr_fifo_cnt + lr_inflight);
    wire [5:0] wr_avail = 6'd32 - (wr_fifo_cnt + wr_inflight);

    // lr：段内剩余（burst 不跨段；段尾地址跳变）
    wire [31:0] lr_seg_rem = in_seg_words_r - {16'd0, dma_ibeat};
    wire [4:0] lr_burst_len_c =
        (lr_seg_rem >= 32'd4 && lr_avail >= 6'd4) ? 5'd4 :
        (lr_seg_rem >= 32'd3 && lr_avail >= 6'd3) ? 5'd3 :
        (lr_seg_rem >= 32'd2 && lr_avail >= 6'd2) ? 5'd2 :
        (lr_seg_rem >= 32'd1 && lr_avail >= 6'd1) ? 5'd1 : 5'd0;

    // wr：行块内总剩余（权重流连续，无段边界）
    wire [23:0] wr_rem = w_rb_beats_r - dma_wbeat;
    wire [4:0] wr_burst_len_c =
        (wr_rem >= 24'd4 && wr_avail >= 6'd4) ? 5'd4 :
        (wr_rem >= 24'd3 && wr_avail >= 6'd3) ? 5'd3 :
        (wr_rem >= 24'd2 && wr_avail >= 6'd2) ? 5'd2 :
        (wr_rem >= 24'd1 && wr_avail >= 6'd1) ? 5'd1 : 5'd0;

    wire lr_issue_req = (state == S_RUN) && (cmd_cnt < 3'd4) &&
                        (lr_rounds_done < p_out_cb) && (lr_burst_len_c != 5'd0);
    wire wr_issue_req = (state == S_RUN) && (cmd_cnt < 3'd4) &&
                        (dma_wbeat < w_rb_beats_r) && (wr_burst_len_c != 5'd0);

    // 共享总线仲裁：wr 优先；busy 期间保持同一命令直至 waitrequest 释放
    reg        load_busy;
    reg        load_busy_is_wr;
    reg [4:0]  load_burst_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            load_busy <= 0; load_busy_is_wr <= 0; load_burst_q <= 5'd0;
        end else if (core_done) begin
            load_busy <= 0; load_busy_is_wr <= 0;
        end else if (!load_busy) begin
            if (wr_issue_req) begin
                load_burst_q <= wr_burst_len_c;
                load_busy_is_wr <= 1;
                load_busy <= lr_waitrequest;   // 未被接受则下拍保持
            end else if (lr_issue_req) begin
                load_burst_q <= lr_burst_len_c;
                load_busy_is_wr <= 0;
                load_busy <= lr_waitrequest;
            end
        end else begin
            if (!lr_waitrequest) load_busy <= 0;   // 命令被接受
        end
    end

    assign wr_read = load_busy ? load_busy_is_wr : wr_issue_req;
    assign lr_read = load_busy ? !load_busy_is_wr : (lr_issue_req && !wr_issue_req);
    // 总线突发长度：busy 保持期间用锁存值；请求拍用候选值
    assign load_burstcount = load_busy ? load_burst_q :
                             (wr_read ? wr_burst_len_c : lr_burst_len_c);

    // lr 接受拍地址/计数推进（突发；段尾跳变，CONV 轮末回段首）
    wire [31:0] lr_accept_beats = {27'd0, load_burstcount};
    wire lr_accept_hits_seg_end = ({16'd0, dma_ibeat} + lr_accept_beats == in_seg_words_r);
    wire [31:0] lr_round_base = reg_ddrin + (p_input_offset << 3) + in_rb_base_r;
    wire [31:0] lr_next_addr =
        lr_accept_hits_seg_end ?
            ((p_type != 4 && dma_icb == lr_last_cb_r) ? lr_round_base :
             lr_address + (lr_accept_beats << 3) + in_cb_stride_r - (in_seg_words_r << 3))
        : lr_address + (lr_accept_beats << 3);

    wire lr_cmd_complete = cmd_last_beat && !cmd_is_wr;
    wire wr_cmd_complete = cmd_last_beat &&  cmd_is_wr;

    assign core_o_ready  = 1;   // core 输出不阻塞（与 v2 tb 的 o_ready=1 一致）
    assign ow_writedata  = core_o_data;
    assign ow_write      = core_o_valid;

    // lr/wr 多笔在途计数：数据返回 -1、命令接受 +1（同拍净 0）；
    // core_done 拍清零（行块末丢弃在途残留，下个行块从头重读）——
    // 清零并入本自动机，避免与主状态机多 always 驱动（Quartus 10028）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lr_pending <= 0;
            wr_pending <= 0;
            lr_inflight <= 0;
            wr_inflight <= 0;
        end else if (core_done) begin
            lr_pending <= 0;
            wr_pending <= 0;
            lr_inflight <= 0;
            wr_inflight <= 0;
        end else begin
            // 共享 load master 的返回已由 cmd FIFO 按命令序路由
            // （见 lr_got/wr_got 定义）：本侧只消费本侧命令的返回。
            // 命令级计数：接受 +1，队首命令最后一拍返回 -1。
            // 单语句净 0（同拍返回+接受，无虚高）+ 非零保护（残留
            // 丢弃双保险，无下溢）：
            //   pending + 接受 - (本侧命令完成 && pending≠0)
            lr_pending <= lr_pending + (lr_read && !lr_waitrequest)
                        - (lr_cmd_complete && lr_pending != 8'd0);
            wr_pending <= wr_pending + (wr_read && !wr_waitrequest)
                        - (wr_cmd_complete && wr_pending != 8'd0);
            // 拍级在途：接受 +burst 拍，每拍返回 -1（防下溢）
            lr_inflight <= lr_inflight
                         + ((lr_read && !lr_waitrequest) ? {1'b0, load_burstcount} : 6'd0)
                         - (lr_got && lr_inflight != 6'd0 ? 6'd1 : 6'd0);
            wr_inflight <= wr_inflight
                         + ((wr_read && !wr_waitrequest) ? {1'b0, load_burstcount} : 6'd0)
                         - (wr_got && wr_inflight != 6'd0 ? 6'd1 : 6'd0);
        end
    end

    // pr/sr 多笔在途计数（数据返回 -1、命令接受 +1；同拍净 0）。
    // pr/sr 各走独立 master，返回保序，无 lr/wr 的共享广播问题。
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pr_pending <= 0;
            sr_pending <= 0;
        end else begin
            pr_pending <= pr_pending + (pr_read && !pr_waitrequest)
                        - (pr_readdatavalid && pr_pending != 8'd0);
            sr_pending <= sr_pending + (sr_read && !sr_waitrequest)
                        - (sr_readdatavalid && sr_pending != 8'd0);
        end
    end

    //-----------------------------------------------------------------------
    // 主状态机
    //-----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; reg_start <= 0;
            rb <= 0; rd_cnt <= 0; cfg_idx <= 0;
            pr_cmd_cnt <= 0; sr_cmd_cnt <= 0;
            core_cfg_we <= 0; core_start <= 0;
            dma_icb <= 0; dma_ibeat <= 0; dma_wbeat <= 0;
            dma_ocb <= 0; dma_obeat <= 0;
            lr_rounds_done <= 0;
            wr_slice_q <= 8'd0;
            w_rb_beats_last_q <= 24'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (as_write && (as_address == 8'h00) && as_writedata[0]) begin
                        reg_start <= 1;
                        rd_cnt <= 0;
                        pr_cmd_cnt <= 0;
                        pr_address <= reg_param;
                        sr_address <= reg_scale;
                        state <= S_RD_PARAM;
                    end
                end

                //---- 读 param 块（20 字，0..19）：多笔在途流水读 ----
                // 跳过的索引：2=scale_offset（软件每层 memcpy 到同一 cb_scale，恒 0）、
                // 12=out_pad、16=output_channel_block_num、20+=软件记账（input_scale/lr/...）
                S_RD_PARAM: begin
                    // 多笔在途：命令接受拍推进地址/命令计数；数据返回拍解析
                    if (pr_read && !pr_waitrequest) begin
                        pr_address <= pr_address + 4;
                        pr_cmd_cnt <= pr_cmd_cnt + 1;
                    end
                    if (pr_readdatavalid) begin
                        case (rd_cnt)
                            0: p_input_offset  <= pr_readdata;
                            1: p_weight_offset <= pr_readdata;
                            3: p_output_offset <= pr_readdata;
                            4: p_in_c  <= pr_readdata;
                            5: p_in_h  <= pr_readdata;
                            6: p_in_w  <= pr_readdata;
                            7: p_out_c <= pr_readdata;
                            8: p_out_h <= pr_readdata;
                            9: p_out_w <= pr_readdata;
                            10: p_k     <= pr_readdata;
                            11: p_pad   <= pr_readdata;
                            13: p_stride<= pr_readdata;
                            14: p_act   <= pr_readdata;
                            15: p_type  <= pr_readdata;
                            17: p_out_row_tile <= pr_readdata;
                            18: p_in_row_tile  <= pr_readdata;
                            19: p_row_block <= pr_readdata;
                            default: ;
                        endcase
                        if (rd_cnt == 19) begin
                            rd_cnt <= 0;
                            pr_cmd_cnt <= 0;
                            p_in_cb <= (p_in_c  + 7) >> 3;
                            p_out_cb <= (p_out_c + 7) >> 3;
                            cfg_idx <= 0;
                            rb <= 0;
                            state <= S_PREP_L;
                        end else begin
                            rd_cnt <= rd_cnt + 1;
                        end
                    end
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
                        // p_k 比较/选择提前一拍（拆 p_k→w_rb_beats_r 的 32-bit 乘法链）
                        wr_slice_q <= (p_k == 2'd1) ? 8'd8 : 8'd72;
                    end else begin
                        rd_cnt <= 0;
                        in_cb_stride_r  <= in_hw_r << 3;
                        out_cb_stride_r <= out_hw_r << 3;
                        out_rb_stride_r <= (p_out_row_tile * p_out_w) << 3;
                        w_rb_beats_r    <= w_cb_r * wr_slice_q;   // 16×8 乘法（原 32-bit×mux 链）
                        w_rb_beats_last_q <= w_cb_r * wr_slice_q - 24'd1;   // 末值-1 提前（拆 S_RUN 减法链）

                        sr_cmd_cnt <= 0;
                        state <= S_RD_SCALE;
                    end
                end

                //---- 读 scale（4×out_c 字），边读边写 core requant 数组 ----
                // 数据返回拍直接把 sr_readdata 写进 core（无需 req_buf 中转：
                // 4096 字寄存器堆爆资源；布局与软件一致：mult/bias/shift/rcl6
                // 各 out_c 字），S_WR_CFG 只写标量 0..15。数组/标量写入顺序
                // 无依赖（core 按地址索引）。
                S_RD_SCALE: begin
                    core_cfg_we <= sr_readdatavalid;
                    // 多笔在途：命令接受拍推进地址/命令计数；数据返回拍解析
                    if (sr_read && !sr_waitrequest) begin
                        sr_address <= sr_address + 4;
                        sr_cmd_cnt <= sr_cmd_cnt + 1;
                    end
                    if (sr_readdatavalid) begin
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
                        if (rd_cnt == sr_total_w - 1) begin
                            rd_cnt <= 0;
                            sr_cmd_cnt <= 0;
                            state <= S_WR_CFG;
                        end else begin
                            rd_cnt <= rd_cnt + 1;
                        end
                    end
                end

                //---- 写 cfg（16 标量 + 4×out_c requant + base_row）----
                //---- 写标量 cfg（0..15）；requant 数组已在 S_RD_SCALE 边读边写 ----
                S_WR_CFG: begin
                    core_cfg_we <= 1;
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
                        lr_rounds_done <= 0;
                        state <= S_WR_BASE;
                    end else begin
                        cfg_idx <= cfg_idx + 1;
                    end
                end

                //---- 行块 base_row（每行块）----
                S_WR_BASE: begin
                    core_cfg_we <= 1;
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
                        load_rows_r  <= (r1_in_r > r0_in_r) ? (r1_in_r - r0_in_r) : 16'sd0;
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
                        state <= S_WR_TILE;
                    end
                end

                //---- 每行块写 core cfg 11（out_row_tile）= 本行块实际输出行数 ----
                // 核心修复：core 的 requant/MAC 行循环按 cfg 11 固定输出 tile 行，
                // 最后行块（out_h 非 tile 整数倍）实际行数 < tile，若不改写 cfg 11，
                // core 多算的行会写穿 DMA 段尾（ow 地址停在段尾 → 数据覆盖错乱）。
                // rb_out_rows_r 在 S_PREP rd_cnt==2 拍已算好（= min(tile, out_h - rb*tile)）。
                //（与 ip/ 版一致：c36fb6d 只改了 ip/，此处同步保证仿真=上板行为）
                S_WR_TILE: begin
                    core_cfg_we <= 1;
                    core_cfg_sel <= 0;
                    core_cfg_addr <= 20'd11;
                    core_cfg_wdata <= {16'd0, rb_out_rows_r};   // cfg 11：本行块实际输出行数
                    state <= S_START;
                end

                S_START: begin
                    core_cfg_we <= 0;
                    lr_address <= reg_ddrin + (p_input_offset << 3) + in_rb_base_r;
                    wr_address <= reg_ddrw + (p_weight_offset << 3);
                    ow_address <= reg_ddrout + (p_output_offset << 3) + out_rb_base_r;
                    core_start <= 1;
                    state <= S_RUN;
                end

                //---- 运行：DMA 跟随 core 流（lr/wr 多笔在途，地址按命令接受拍推进）----
                S_RUN: begin
                    core_start <= 0;

                    // lr：命令接受拍推进（突发；段尾跳变，CONV 轮末回段首）
                    if (lr_read && !lr_waitrequest) begin
                        lr_address <= lr_next_addr;
                        if (lr_accept_hits_seg_end) begin
                            dma_ibeat <= 0;
                            if (dma_icb == lr_last_cb_r) begin
                                dma_icb <= 0;
                                lr_rounds_done <= lr_rounds_done + 1;
                            end else
                                dma_icb <= dma_icb + 1;
                        end else
                            dma_ibeat <= dma_ibeat + lr_accept_beats[15:0];
                    end
                    // wr：命令接受拍推进（突发；行块内连续）
                    if (wr_read && !wr_waitrequest) begin
                        dma_wbeat <= dma_wbeat + {14'd0, load_burstcount};
                        wr_address <= wr_address + (load_burstcount << 3);
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
                    if (rb == p_row_block - 1) begin
                        reg_start <= 0;
                        state <= S_IDLE;
                    end else begin
                        rb <= rb + 1;
                        rb_base_r <= rb_base_r + rb_tile_stride_r;   // 增量：base += tile*stride
                        dma_icb <= 0; dma_ibeat <= 0; dma_wbeat <= 0;
                        dma_ocb <= 0; dma_obeat <= 0;
                        lr_rounds_done <= 0;
                        state <= S_WR_BASE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
