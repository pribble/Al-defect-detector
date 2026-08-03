//=============================================================================
// cnn_top_core — 验证版（echo）：收到 START 立即自清，不做任何计算/读操作
//=============================================================================
// 用途：隔离 wait ip fail 的问题层次。
//   - 若此版能过 start_fpga（START 自清、不报超时）：hps2cnn_avs 从接口、
//     QSys 连接、时钟/复位、FPGA 配置均正常 → 问题在正式核内部逻辑
//     （状态机/主接口握手），恢复正式版继续查。
//   - 若此版仍 wait ip fail：问题在更底层（从接口地址/数据、时钟、复位、
//     FPGA 未配置等），用 devmem 读寄存器确认。
//
// 与正式版接口完全一致（cnn_top.v 例化连接不变），仅内部逻辑替换。
// 恢复正式版：git checkout 78cbe65 -- de10_nano/C5TB/ip/cnn_top_core.v
//   （并同步 cnn_rtl/src/cnn_top_core.v）
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
    output wire [31:0]         pr_address,
    output wire                pr_read,
    input  wire [31:0]         pr_readdata,
    input  wire                pr_readdatavalid,
    input  wire                pr_waitrequest,

    // ---- scale 读（32-bit）----
    output wire [31:0]         sr_address,
    output wire                sr_read,
    input  wire [31:0]         sr_readdata,
    input  wire                sr_readdatavalid,
    input  wire                sr_waitrequest,

    // ---- 输入读（64-bit）----
    output wire [31:0]         lr_address,
    output wire                lr_read,
    input  wire [63:0]         lr_readdata,
    input  wire                lr_readdatavalid,
    input  wire                lr_waitrequest,

    // ---- 权重读（64-bit）----
    output wire [31:0]         wr_address,
    output wire                wr_read,
    input  wire [63:0]         wr_readdata,
    input  wire                wr_readdatavalid,
    input  wire                wr_waitrequest,

    // ---- 输出写（64-bit）----
    output wire [31:0]         ow_address,
    output wire                ow_write,
    output wire [63:0]         ow_writedata,
    input  wire                ow_waitrequest
);

    //-----------------------------------------------------------------------
    // 寄存器（与正式版一致，软件 fpga_init 会写 6 个寄存器）
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
    // 验证版：收到 START 即自清（下一拍清 0，软件轮询立即返回），
    // 不执行任何计算，主接口全部不驱动
    //-----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            start_clear_pending <= 0;
        else begin
            start_clear_pending <= 0;
            if (reg_start[0] && !start_clear_pending)
                start_clear_pending <= 1;
        end
    end

    assign pr_address    = 32'd0;
    assign pr_read       = 1'b0;
    assign sr_address    = 32'd0;
    assign sr_read       = 1'b0;
    assign lr_address    = 32'd0;
    assign lr_read       = 1'b0;
    assign wr_address    = 32'd0;
    assign wr_read       = 1'b0;
    assign ow_address    = 32'd0;
    assign ow_write      = 1'b0;
    assign ow_writedata  = 64'd0;

endmodule
