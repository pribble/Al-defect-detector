//=============================================================================
// requant_store — 单 lane requant 参数存储（cnn_core 内 8 份实例化）
//=============================================================================
// 从 cnn_core_v2 拆出（原 8×3 组内联展开约 800 行 → 本模块 ×8 实例）。
// 每 lane 独立 M10K：cfg_sel 1/2/3 = bias/mult/shift，addr = 输出通道。
// 8 lane 并行读 8 个不同通道地址（v_out_ch = o_group*8+ln），单端口 RAM
// 每拍只能 1 地址 → 每 lane 一份（8×3 组，约 78 块 M10K）。
// M10K 同步读 1 拍延迟：bias 于 S_REQ_ADDR 发起（S_REQ_MUL 用）、
// mult 于 S_REQ_MUL 发起（S_REQ_MUL2 用）、shift 于 S_REQ_MUL2 发起
// （S_REQ_OUT 用）——正好插入现有 requant 流水，事件序列不变。
//
// 写端口（A 口）：cfg_we + cfg_sel 选择存储，cfg_addr[9:0] 通道地址，
//   值域 ≤1023 与 1024 深匹配（cfg_addr[19:0] < G_MAX_C 判定禁写越界）。
// 读端口（B 口）：raddr = rq_raddr_q + lane，同步读 1 拍延迟。
//
// SIMULATION：reg 数组 + ramstyle 提示（verilator 行为模型）
// 综合：显式 altsyncram，read_during_write_mode_mixed_ports = OLD_DATA
//   （推断版被 Quartus 选 New data：B 口 pass-through mux 在 wren_reg 传播
//   路径上，slack -3.39；OLD_DATA 读口纯同步读，无 wren 依赖）
//=============================================================================

module requant_store #(
    parameter G_MAX_C = 1024   // 最大输出通道数（数组容量）
)(
    input  wire        clk,

    // ---- 配置写（A 口，sel 1/2/3 = bias/mult/shift）----
    input  wire        cfg_we,
    input  wire [2:0]  cfg_sel,
    input  wire [19:0] cfg_addr,
    input  wire [31:0] cfg_wdata,

    // ---- 运行读（B 口，本 lane 通道地址）----
    input  wire [9:0]  raddr,

    // ---- 输出（同步读，1 拍延迟）----
    output wire signed [31:0] q_bias,
    output wire [31:0] q_mult,
    output wire [7:0]  q_shift
);

`ifdef SIMULATION
    (* ramstyle = "M10K" *) reg signed [31:0] bias_store [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [31:0] rq_m_store [0:G_MAX_C-1];
    (* ramstyle = "M10K" *) reg [7:0]  rq_r_store [0:G_MAX_C-1];
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C)
            bias_store[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C)
            rq_m_store[cfg_addr[9:0]] <= cfg_wdata;
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C)
            rq_r_store[cfg_addr[9:0]] <= cfg_wdata[7:0];
    end
    reg signed [31:0] q_bias_r;
    reg [31:0] q_mult_r;
    reg [7:0]  q_shift_r;
    always @(posedge clk) begin
        q_bias_r  <= bias_store[raddr];
        q_mult_r  <= rq_m_store[raddr];
        q_shift_r <= rq_r_store[raddr];
    end
    assign q_bias  = q_bias_r;
    assign q_mult  = q_mult_r;
    assign q_shift = q_shift_r;
`else
    // 综合版：显式 altsyncram，read_during_write_mode_mixed_ports = OLD_DATA
    altsyncram #(
        .operation_mode("DUAL_PORT"),
        .width_a(32), .widthad_a(10), .numwords_a(G_MAX_C),
        .width_b(32), .widthad_b(10), .numwords_b(G_MAX_C),
        .read_during_write_mode_mixed_ports("OLD_DATA"),
        .outdata_reg_b("CLOCK0"),
        .address_aclr_a("NONE"),
        .address_aclr_b("NONE"),
        .outdata_aclr_b("NONE"),
        .clock_enable_input_a("BYPASS"),
        .clock_enable_input_b("BYPASS"),
        .clock_enable_output_b("BYPASS"),
        .power_up_uninitialized("FALSE"),
        .intended_device_family("Cyclone V")
    ) u_rq_bias (
        .clock0(clk),
        .clock1(clk),
        .address_a(cfg_addr[9:0]),
        .wren_a(cfg_we && cfg_sel == 3'd1 && cfg_addr < G_MAX_C),
        .data_a(cfg_wdata),
        .address_b(raddr),
        .q_b(q_bias)
    );
    altsyncram #(
        .operation_mode("DUAL_PORT"),
        .width_a(32), .widthad_a(10), .numwords_a(G_MAX_C),
        .width_b(32), .widthad_b(10), .numwords_b(G_MAX_C),
        .read_during_write_mode_mixed_ports("OLD_DATA"),
        .outdata_reg_b("CLOCK0"),
        .address_aclr_a("NONE"),
        .address_aclr_b("NONE"),
        .outdata_aclr_b("NONE"),
        .clock_enable_input_a("BYPASS"),
        .clock_enable_input_b("BYPASS"),
        .clock_enable_output_b("BYPASS"),
        .power_up_uninitialized("FALSE"),
        .intended_device_family("Cyclone V")
    ) u_rq_mult (
        .clock0(clk),
        .clock1(clk),
        .address_a(cfg_addr[9:0]),
        .wren_a(cfg_we && cfg_sel == 3'd2 && cfg_addr < G_MAX_C),
        .data_a(cfg_wdata),
        .address_b(raddr),
        .q_b(q_mult)
    );
    altsyncram #(
        .operation_mode("DUAL_PORT"),
        .width_a(8), .widthad_a(10), .numwords_a(G_MAX_C),
        .width_b(8), .widthad_b(10), .numwords_b(G_MAX_C),
        .read_during_write_mode_mixed_ports("OLD_DATA"),
        .outdata_reg_b("CLOCK0"),
        .address_aclr_a("NONE"),
        .address_aclr_b("NONE"),
        .outdata_aclr_b("NONE"),
        .clock_enable_input_a("BYPASS"),
        .clock_enable_input_b("BYPASS"),
        .clock_enable_output_b("BYPASS"),
        .power_up_uninitialized("FALSE"),
        .intended_device_family("Cyclone V")
    ) u_rq_shift (
        .clock0(clk),
        .clock1(clk),
        .address_a(cfg_addr[9:0]),
        .wren_a(cfg_we && cfg_sel == 3'd3 && cfg_addr < G_MAX_C),
        .data_a(cfg_wdata[7:0]),
        .address_b(raddr),
        .q_b(q_shift)
    );
`endif

endmodule
