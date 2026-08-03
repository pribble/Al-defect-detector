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
//   - 读完成 = read && readdatavalid（mm_bridge_sdram0 为流水读桥，数据由
//     readdatavalid 延迟返回；waitrequest 仅表示命令是否被接受）
//   - 41 个 HDL_PARAMETER 参数照单声明（QSys 例化传参），值忽略（RTL 常量已对齐）
//=============================================================================

module cnn_top #(
    // ---- QSys HDL_PARAMETER（与 hw.tcl 同名同序，值由 QSys 传入，忽略）----
    parameter CFG_M_AXI_ADDR_WIDTH   = 32,
    parameter CFG_M_AXI_DATA_WIDTH   = 32,
    parameter LOAD_M_AXI_ADDR_WIDTH  = 32,
    parameter LOAD_M_AXI_DATA_WIDTH  = 64,
    parameter SCALE_M_AXI_ADDR_WIDTH = 32,
    parameter SCALE_M_AXI_DATA_WIDTH = 32,
    parameter OUTPUT_M_AXI_ADDR_WIDTH = 32,
    parameter OUTPUT_M_AXI_DATA_WIDTH = 64
)(
    // ---- clock / reset ----
    input  wire                sysclk,      // 工作时钟：QSys clk_cnn（PLL fpga_clk_cnn，50MHz），驱动全部逻辑与 4 个 Avalon 主接口
    input  wire                rst_n,       // 低有效异步复位：QSys clk_cnn.clk_reset

    // ---- hps2cnn_avs（Avalon 从接口，HPS→FPGA 寄存器访问）----
    // HPS 软件（foo_set/foo_get，mmap /dev/mem 0xFF200000）通过此口读写 6 个
    // 控制寄存器：START(0x00)/DDRIN(0x10)/DDRW(0x1C)/DDROUT(0x28)/PARAM(0x34)/SCALE(0x40)
    input  wire [7:0]          as_address,      // 寄存器字地址（WORDS 单位，256 个 32-bit 寄存器空间）
    input  wire                as_write,        // 写请求：HPS 写 START=1（启动）、写 5 个 DDR 基址寄存器
    input  wire                as_read,         // 读请求：HPS 轮询 START 自清（bit0=0 即完成）
    input  wire [31:0]         as_writedata,    // 写数据
    output wire [31:0]         as_readdata,     // 读数据（组合 mux，读地址稳定即有效）
    output wire                as_data_waitquest, // 反压（恒 0：从接口从不等待，HPS 无感知）

    // ---- param_read_avalon（Avalon 主接口 #1：读 param 指令块，32-bit）----
    // S_RD_PARAM 状态用：从 cb_param（reg_param 基址）逐字读 struct parameter
    // （27 个字），单笔在途（burstcount 恒 1）
    output wire [CFG_M_AXI_ADDR_WIDTH-1:0]   param_avm_address,  // 读地址（DDR 物理地址，= reg_param + 4×rd_cnt）
    output wire [4:0]          param_avm_burstcount,  // 突发长度（恒 5'd1 = 单拍）
    output wire [CFG_M_AXI_DATA_WIDTH/8-1:0] param_avm_byteenable,  // 字节使能（恒全 1）
    output wire                param_avm_read,       // 读请求（read 拉高 → 命令接受（!waitrequest）→ 拉低，只发一笔）
    input  wire [CFG_M_AXI_DATA_WIDTH-1:0]   param_avm_readdata,   // 读数据（与 readdatavalid 同拍有效）
    input  wire                param_avm_readdatavalid,  // 数据返回标志（流水桥：延迟于命令，必须以此判完成）
    input  wire                param_avm_waitrequest,   // 桥反压（=1 时命令不被接受）

    // ---- load_read_avalon（Avalon 主接口 #2：读输入特征图 + 权重，64-bit）----
    // 最忙的口，lr/wr 两路分时复用（core 的 S_LOAD 与 S_WEIGHT 互斥）：
    //   lr（S_LOAD）：读输入 NHWC8 行块，每拍 8 字节 = 8 通道 int8 → lb 行缓冲
    //   wr（S_WEIGHT）：读权重 slice（k=3 吃 72 拍 / k=1 吃 8 拍）→ wbuf
    // 多笔在途（≤4，桥 MAX_PENDING_RESPONSES=4），readdata 同时回灌两路
    output wire [LOAD_M_AXI_ADDR_WIDTH-1:0]  load_avm_address,  // 读地址（lr 时 = reg_ddrin+input_offset×8+行块偏移；wr 时 = reg_ddrw+weight_offset×8）
    output wire [4:0]          load_avm_burstcount,  // 突发长度（恒 5'd1）
    output wire [LOAD_M_AXI_DATA_WIDTH/8-1:0] load_avm_byteenable,  // 字节使能（恒全 1）
    output wire                load_avm_read,       // 读请求（lr_read || wr_read）
    input  wire [LOAD_M_AXI_DATA_WIDTH-1:0]  load_avm_readdata,   // 读数据（64-bit = 8 个 int8 通道）
    input  wire                load_avm_readdatavalid,  // 数据返回（lr/wr 共用；不在途一侧的 pending 会下溢，靠回绕自愈）
    input  wire                load_avm_waitrequest,   // 桥反压

    // ---- output_read_avalon（Avalon 主接口 #3：写输出特征图，64-bit）----
    // 名字含 read 是黑盒时代遗留（实际是写口）。ow 把 core requant 后的输出
    // 事件（每拍 1 像素 × 8 通道 int8，NHWC8 块序）写回 DDR
    // （reg_ddrout + output_offset×8 + 行块偏移）
    output wire [OUTPUT_M_AXI_ADDR_WIDTH-1:0] output_avm_address,  // 写地址
    output wire [4:0]          output_avm_burstcount,  // 突发长度（恒 5'd1）
    output wire [OUTPUT_M_AXI_DATA_WIDTH/8-1:0] output_avm_byteenable,  // 字节使能（恒全 1）
    input  wire                output_avm_waitrequest,  // 桥反压（写完成 = write && !waitrequest）
    output wire                output_avm_write,       // 写请求（= core_o_valid，输出事件有效即写）
    output wire [OUTPUT_M_AXI_DATA_WIDTH-1:0] output_avm_writedata,  // 写数据（= core_o_data，8 通道 int8）

    // ---- scale_avm_avalon（Avalon 主接口 #4：读 requant 定点参数，32-bit）----
    // S_RD_SCALE 状态用：从 cb_scale（reg_scale 基址）读每通道 4 个 int32
    // （mult/bias_int/shift/rcl6，共 4×out_c 字），边读边写 core requant 数组。
    // 单笔在途；接口名 avm 重复为黑盒时代遗留
    output wire [SCALE_M_AXI_ADDR_WIDTH-1:0] scale_avm_address,  // 读地址（= reg_scale + 4×rd_cnt）
    output wire [4:0]          scale_avm_burstcount,  // 突发长度（恒 5'd1）
    output wire [SCALE_M_AXI_DATA_WIDTH/8-1:0] scale_avm_byteenable,  // 字节使能（恒全 1）
    output wire                scale_avm_read,       // 读请求（单笔在途：拉高 → 接受 → 拉低 → 等 readdatavalid）
    input  wire [SCALE_M_AXI_DATA_WIDTH-1:0]  scale_avm_readdata,   // 读数据（32-bit）
    input  wire                scale_avm_readdatavalid,  // 数据返回标志（以此判完成）
    input  wire                scale_avm_waitrequest   // 桥反压
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
        .pr_readdatavalid(param_avm_readdatavalid),
        .pr_waitrequest(param_avm_waitrequest),
        .sr_address    (sr_address),
        .sr_read       (sr_read),
        .sr_readdata   (scale_avm_readdata),
        .sr_readdatavalid(scale_avm_readdatavalid),
        .sr_waitrequest(scale_avm_waitrequest),
        .lr_address    (lr_address),
        .lr_read       (lr_read),
        .lr_readdata   (lr_readdata),
        .lr_readdatavalid(load_avm_readdatavalid),
        .lr_waitrequest(lr_waitrequest),
        .wr_address    (wr_address),
        .wr_read       (wr_read),
        .wr_readdata   (wr_readdata),
        .wr_readdatavalid(load_avm_readdatavalid),
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
