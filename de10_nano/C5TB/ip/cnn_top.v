//=============================================================================
// cnn_top — QSys 适配层（阶段 5 交付，替代 cnn_top.qxp 黑盒）
//=============================================================================
// 端口与 ip/cnn_top_hw.tcl 完全一致（QSys 例化 cnn_top_0 的接线不变）：
//   - hps2cnn_avs   Avalon 从接口（ARM 寄存器，8-bit 地址，waitrequest）
//   - clock_1/reset 50MHz（soc_system.qsys 的 clk_cnn）
//   - 4 个 Avalon 主接口：param/scale 32b 读、load 64b 读、output 64b 写
// 适配说明：
//   - 内部例化 cnn_top_core（RTL 本体，tb_cnn_top 直接对拍它）
//   - burstcount 固定 1（不做突发）；byteenable 全 1（32b/64b）
//   - readdatavalid 组合 = read && !waitrequest（读完成拍有效，与 readdata 同拍）
//   - 41 个 HDL_PARAMETER 参数照单声明（QSys 例化传参），值忽略（RTL 常量已对齐）
//=============================================================================

module cnn_top #(
    // ---- QSys HDL_PARAMETER（与 hw.tcl 同名同序，值由 QSys 传入，忽略）----
    parameter OUTPUT_CHANNEL_TILE  = 8,
    parameter INPUT_CHANNEL_TILE   = 8,
    parameter INPUT_WIDTH          = 8,
    parameter WEIGHT_WIDTH         = 8,
    parameter INPUT_ROW_TILE       = 11,
    parameter IMAGE_MAX_W          = 302,
    parameter OUTPUT_ROW_TILE      = 5,
    parameter OUTPUT_MAX_W         = 150,
    parameter IN_C_WIDTH           = 11,
    parameter IN_H_WIDTH           = 9,
    parameter IN_W_WIDTH           = 9,
    parameter OUTPUT_C_WIDTH       = 11,
    parameter OUTPUT_H_WIDTH       = 9,
    parameter OUTPUT_W_WIDTH       = 9,
    parameter KERNEL_WIDTH         = 2,
    parameter KERNEL_SIZE_WIDTH    = 4,
    parameter INPUT_PAD_WIDTH      = 2,
    parameter OUTPUT_PAD_WIDTH     = 2,
    parameter CNN_TYPE_WIDTH       = 3,
    parameter CNN_STRIDE_WIDTH     = 2,
    parameter CNN_RELU_WIDTH       = 3,
    parameter KERNAL_SIZE          = 9,
    parameter CFG_PARAM_WIDTH      = 32,
    parameter OUTPUT_ADDR_WIDTH    = 10,
    parameter OUTPUT_WIDTH         = 32,
    parameter SCALE_6              = 32'h40C00000,
    parameter REORGANIZE_BUFF_DEEP = 1024,
    parameter REORG_RAM_ADDR_WIDTH = 10,
    parameter REORG_RAM_DATA_WIDTH = 64,
    parameter REORG_RAM_SEL_WIDTH  = 3,
    parameter INPUT_RAM_DATA_WIDTH = 64,
    parameter CFG_M_AXI_ADDR_WIDTH   = 32,
    parameter CFG_M_AXI_DATA_WIDTH   = 32,
    parameter LOAD_M_AXI_ADDR_WIDTH  = 32,
    parameter LOAD_M_AXI_DATA_WIDTH  = 64,
    parameter REORG_M_AXI_ADDR_WIDTH = 32,
    parameter REORG_M_AXI_DATA_WIDTH = 64,
    parameter SCALE_M_AXI_ADDR_WIDTH = 32,
    parameter SCALE_M_AXI_DATA_WIDTH = 32,
    parameter OUTPUT_M_AXI_ADDR_WIDTH = 32,
    parameter OUTPUT_M_AXI_DATA_WIDTH = 64
)(
    // ---- clock / reset ----
    input  wire                sysclk,      // clock_1.clk（clk_cnn，50MHz）
    input  wire                rst_n,       // reset.reset_n

    // ---- hps2cnn_avs（Avalon 从）----
    input  wire [7:0]          as_address,
    input  wire                as_write,
    input  wire                as_read,
    input  wire [31:0]         as_writedata,
    output wire [31:0]         as_readdata,
    output wire                as_data_waitquest,

    // ---- param_read_avalon（32b 读）----
    output wire [CFG_M_AXI_ADDR_WIDTH-1:0]   param_avm_address,
    output wire [4:0]          param_avm_burstcount,
    output wire [CFG_M_AXI_DATA_WIDTH/8-1:0] param_avm_byteenable,
    output wire                param_avm_read,
    input  wire [CFG_M_AXI_DATA_WIDTH-1:0]   param_avm_readdata,
    input  wire                param_avm_readdatavalid,
    input  wire                param_avm_waitrequest,

    // ---- load_read_avalon（64b 读）----
    output wire [LOAD_M_AXI_ADDR_WIDTH-1:0]  load_avm_address,
    output wire [4:0]          load_avm_burstcount,
    output wire [LOAD_M_AXI_DATA_WIDTH/8-1:0] load_avm_byteenable,
    output wire                load_avm_read,
    input  wire [LOAD_M_AXI_DATA_WIDTH-1:0]  load_avm_readdata,
    input  wire                load_avm_readdatavalid,
    input  wire                load_avm_waitrequest,

    // ---- output_read_avalon（64b 写）----
    output wire [OUTPUT_M_AXI_ADDR_WIDTH-1:0] output_avm_address,
    output wire [4:0]          output_avm_burstcount,
    output wire [OUTPUT_M_AXI_DATA_WIDTH/8-1:0] output_avm_byteenable,
    input  wire                output_avm_waitrequest,
    output wire                output_avm_write,
    output wire [OUTPUT_M_AXI_DATA_WIDTH-1:0] output_avm_writedata,

    // ---- scale_avm_avalon（32b 读）----
    output wire [SCALE_M_AXI_ADDR_WIDTH-1:0] scale_avm_address,
    output wire [4:0]          scale_avm_burstcount,
    output wire [SCALE_M_AXI_DATA_WIDTH/8-1:0] scale_avm_byteenable,
    output wire                scale_avm_read,
    input  wire [SCALE_M_AXI_DATA_WIDTH-1:0]  scale_avm_readdata,
    input  wire                scale_avm_readdatavalid,
    input  wire                scale_avm_waitrequest
);

    // ---- core 本体 ----
    wire [31:0] pr_address, sr_address, lr_address, wr_address, ow_address;
    wire        pr_read, sr_read, lr_read, wr_read, ow_write;
    wire [63:0] lr_readdata, wr_readdata, ow_writedata;
    wire        pr_waitrequest, sr_waitrequest, lr_waitrequest, wr_waitrequest;
    wire        ow_waitrequest;

    // 权重读复用 load master（黑盒无独立权重 master；core 的 S_LOAD/S_WEIGHT
    // 串行，lr_read 与 wr_read 不同时拉高——load_avm_readdata 同时回灌两路）
    assign lr_readdata = load_avm_readdata;
    assign wr_readdata = load_avm_readdata;
    assign lr_waitrequest = load_avm_waitrequest;
    assign wr_waitrequest = load_avm_waitrequest;

    cnn_top_core core (
        .clk           (sysclk),
        .rst_n         (rst_n),
        .as_address    (as_address),
        .as_write      (as_write),
        .as_read       (as_read),
        .as_writedata  (as_writedata),
        .as_readdata   (as_readdata),
        .as_waitrequest(as_data_waitquest),
        .pr_address    (pr_address),
        .pr_read       (pr_read),
        .pr_readdata   (param_avm_readdata),
        .pr_waitrequest(param_avm_waitrequest),
        .sr_address    (sr_address),
        .sr_read       (sr_read),
        .sr_readdata   (scale_avm_readdata),
        .sr_waitrequest(scale_avm_waitrequest),
        .lr_address    (lr_address),
        .lr_read       (lr_read),
        .lr_readdata   (lr_readdata),
        .lr_waitrequest(lr_waitrequest),
        .wr_address    (wr_address),
        .wr_read       (wr_read),
        .wr_readdata   (wr_readdata),
        .wr_waitrequest(wr_waitrequest),
        .ow_address    (ow_address),
        .ow_write      (ow_write),
        .ow_writedata  (ow_writedata),
        .ow_waitrequest(ow_waitrequest)
    );

    // ---- Avalon 主接口适配 ----
    assign param_avm_address    = pr_address;
    assign param_avm_burstcount = 5'd1;
    assign param_avm_byteenable = {(CFG_M_AXI_DATA_WIDTH/8){1'b1}};
    assign param_avm_read       = pr_read;

    assign load_avm_address     = lr_read ? lr_address : wr_address;
    assign load_avm_burstcount  = 5'd1;
    assign load_avm_byteenable  = {(LOAD_M_AXI_DATA_WIDTH/8){1'b1}};
    assign load_avm_read        = lr_read || wr_read;

    assign output_avm_address   = ow_address;
    assign output_avm_burstcount = 5'd1;
    assign output_avm_byteenable = {(OUTPUT_M_AXI_DATA_WIDTH/8){1'b1}};
    assign output_avm_write     = ow_write;
    assign output_avm_writedata = ow_writedata;

    assign scale_avm_address    = sr_address;
    assign scale_avm_burstcount = 5'd1;
    assign scale_avm_byteenable = {(SCALE_M_AXI_DATA_WIDTH/8){1'b1}};
    assign scale_avm_read       = sr_read;

endmodule
