//=============================================================================
// cnn_top_core — 调试版（纯输入观察器）
//=============================================================================
// 2026-08 上板排查"检测结果错误"的临时调试版：不例化 cnn_core_v2（无任何
// 计算逻辑），只保留：
//   1) avalon 从端口（寄存器读写，协议不变——软件无需改）
//   2) param 区读回（pr，20 字 → param_buf）
//   3) scale 区读回（sr，ch0 的 mult/bias/shift/rcl6 按 out_c 偏移 → scale_buf）
//   4) 输入区读回（lr，前 4 字 → in_buf）
//   5) as_readdata 暴露上述读回值（0x180/0x1D0/0x1E0，与正式版地址一致）
// start 置位后状态机顺序读 param → scale → 输入，完成后 reg_start 自清。
// 查完输入后 git 回退此提交恢复正式版（含 cnn_core_v2 计算）。
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

    // ---- 权重读（64-bit）----（调试版不读，固定拉低）
    output wire [31:0]         wr_address,
    output wire                wr_read,
    input  wire [63:0]         wr_readdata,
    input  wire                wr_readdatavalid,
    input  wire                wr_waitrequest,

    // ---- 输出写（64-bit）----（调试版不写，固定拉低）
    output wire [31:0]         ow_address,
    output wire                ow_write,
    output wire [63:0]         ow_writedata,
    input  wire                ow_waitrequest
);

    //-----------------------------------------------------------------------
    // 寄存器（协议与正式版一致：0x00 start、0x10 输入基址、0x34 param 基址、
    // 0x40 scale 基址）
    //-----------------------------------------------------------------------
    reg [31:0] reg_ddrin, reg_ddrw, reg_ddrout, reg_param, reg_scale;
    reg [31:0] reg_start;

    assign as_waitrequest = 1'b0;
    assign wr_read = 1'b0;      assign wr_address = 32'd0;
    assign ow_write = 1'b0;     assign ow_address = 32'd0;
    assign ow_writedata = 64'd0;

    //-----------------------------------------------------------------------
    // 读回缓冲区（暴露）
    //-----------------------------------------------------------------------
    reg [31:0] param_buf [0:19];     // param 区 20 字（RTL 实际读回的）
    reg [31:0] scale_buf [0:3];      // ch0: mult/bias/shift/rcl6（按 out_c 偏移）
    reg [63:0] in_buf    [0:3];      // 输入区前 4 字（64-bit）

    //-----------------------------------------------------------------------
    // 主状态机：IDLE → RD_PARAM(20) → RD_SCALE(4) → RD_IN(4) → DONE
    //-----------------------------------------------------------------------
    localparam S_IDLE = 3'd0, S_RD_PARAM = 3'd1, S_RD_SCALE = 3'd2,
               S_RD_IN = 3'd3, S_DONE = 3'd4;
    reg [2:0] state;
    reg [7:0] cnt;
    reg [31:0] rd_addr;
    reg pr_busy, sr_busy, lr_busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ddrin <= 0; reg_ddrw <= 0; reg_ddrout <= 0;
            reg_param <= 0; reg_scale <= 0; reg_start <= 0;
        end else if (as_write) begin
            case (as_address)
                8'h00: reg_start   <= as_writedata;
                8'h10: reg_ddrin   <= as_writedata;
                8'h1C: reg_ddrw    <= as_writedata;
                8'h28: reg_ddrout  <= as_writedata;
                8'h34: reg_param   <= as_writedata;
                8'h40: reg_scale   <= as_writedata;
                default: ;
            endcase
        end else if (state == S_DONE) begin
            reg_start[0] <= 1'b0;   // 完成自清（软件轮询 start 位 0）
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; cnt <= 0; rd_addr <= 0;
            pr_busy <= 0; sr_busy <= 0; lr_busy <= 0;
            pr_read <= 0; sr_read <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (reg_start[0]) begin
                        state <= S_RD_PARAM;
                        cnt <= 0;
                        rd_addr <= reg_param;
                        pr_busy <= 1; pr_read <= 1; pr_address <= reg_param;
                    end
                end

                //---- param 读（20 笔 32-bit，单笔在途：接受即拉低，等 readdatavalid）----
                S_RD_PARAM: begin
                    if (pr_busy && pr_read && !pr_waitrequest) begin
                        pr_read <= 1'b0;   // 命令接受即拉低（流水桥防重复发命令）
                    end else if (pr_busy && !pr_read && pr_readdatavalid) begin
                        param_buf[cnt] <= pr_readdata;
                        pr_busy <= 1'b0;
                        if (cnt == 8'd19) begin
                            state <= S_RD_SCALE;
                            cnt <= 0;
                            // ch0 偏移：mult=scale[0]、bias=scale[out_c]、
                            // shift=scale[2*out_c]、rcl6=scale[3*out_c]（out_c=param_buf[7]）
                            rd_addr <= reg_scale;
                            sr_busy <= 1; sr_read <= 1; sr_address <= reg_scale;
                        end else begin
                            cnt <= cnt + 1;
                            rd_addr <= rd_addr + 4;
                            pr_busy <= 1; pr_read <= 1; pr_address <= rd_addr + 4;
                        end
                    end
                end

                //---- scale 读（4 笔，偏移 0/out_c/2out_c/3out_c 字）----
                S_RD_SCALE: begin
                    if (sr_busy && sr_read && !sr_waitrequest) begin
                        sr_read <= 1'b0;
                    end else if (sr_busy && !sr_read && sr_readdatavalid) begin
                        scale_buf[cnt] <= sr_readdata;
                        sr_busy <= 1'b0;
                        if (cnt == 8'd3) begin
                            state <= S_RD_IN;
                            cnt <= 0;
                            // 本层输入 = 输入区基址 + input_offset×8（param_buf[0] 为字偏移）
                            rd_addr <= reg_ddrin + param_buf[0] * 8;
                            lr_busy <= 1;
                        end else begin
                            cnt <= cnt + 1;
                            rd_addr <= rd_addr + param_buf[7] * 4;
                            sr_busy <= 1; sr_read <= 1;
                            sr_address <= rd_addr + param_buf[7] * 4;
                        end
                    end
                end

                //---- 输入读（4 笔 64-bit）----
                S_RD_IN: begin
                    if (lr_busy && lr_read && !lr_waitrequest) begin
                        // lr_read 是组合（正式版由状态机驱动），这里用 reg 驱动
                    end
                    if (lr_busy && !lr_read && lr_readdatavalid) begin
                        in_buf[cnt] <= lr_readdata;
                        lr_busy <= 1'b0;
                        if (cnt == 8'd3) begin
                            state <= S_DONE;
                        end else begin
                            cnt <= cnt + 1;
                            rd_addr <= rd_addr + 8;
                            lr_busy <= 1;
                        end
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

    // lr_read 驱动（单笔：busy 期间拉高，接受后拉低）
    reg lr_read_r;
    assign lr_read = lr_read_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lr_read_r <= 1'b0;
        end else if (state == S_RD_IN && lr_busy && !lr_read_r) begin
            lr_read_r <= 1'b1;
            lr_address <= rd_addr;
        end else if (lr_read_r && !lr_waitrequest) begin
            lr_read_r <= 1'b0;
        end
    end

    //-----------------------------------------------------------------------
    // as_readdata（地址与正式版一致：0x180 param_buf、0x1D0 scale、0x1E0 in）
    //-----------------------------------------------------------------------
    always @(*) begin
        case (as_address)
            8'h00: as_readdata = reg_start;
            8'h10: as_readdata = reg_ddrin;
            8'h1C: as_readdata = reg_ddrw;
            8'h28: as_readdata = reg_ddrout;
            8'h34: as_readdata = reg_param;
            8'h40: as_readdata = reg_scale;
            8'h11: as_readdata = {state, cnt, pr_busy, sr_busy, lr_busy, 20'd0};
            8'h60: as_readdata = param_buf[0];
            8'h61: as_readdata = param_buf[1];
            8'h62: as_readdata = param_buf[2];
            8'h63: as_readdata = param_buf[3];
            8'h64: as_readdata = param_buf[4];
            8'h65: as_readdata = param_buf[5];
            8'h66: as_readdata = param_buf[6];
            8'h67: as_readdata = param_buf[7];
            8'h68: as_readdata = param_buf[8];
            8'h69: as_readdata = param_buf[9];
            8'h6A: as_readdata = param_buf[10];
            8'h6B: as_readdata = param_buf[11];
            8'h6C: as_readdata = param_buf[12];
            8'h6D: as_readdata = param_buf[13];
            8'h6E: as_readdata = param_buf[14];
            8'h6F: as_readdata = param_buf[15];
            8'h70: as_readdata = param_buf[16];
            8'h71: as_readdata = param_buf[17];
            8'h72: as_readdata = param_buf[18];
            8'h73: as_readdata = param_buf[19];
            8'h74: as_readdata = scale_buf[0];   // mult
            8'h75: as_readdata = scale_buf[1];   // bias
            8'h76: as_readdata = scale_buf[2];   // shift
            8'h77: as_readdata = scale_buf[3];   // rcl6
            8'h78: as_readdata = in_buf[0][31:0];
            8'h79: as_readdata = in_buf[0][63:32];
            8'h7A: as_readdata = in_buf[1][31:0];
            8'h7B: as_readdata = in_buf[1][63:32];
            default: as_readdata = 32'd0;
        endcase
    end

endmodule
