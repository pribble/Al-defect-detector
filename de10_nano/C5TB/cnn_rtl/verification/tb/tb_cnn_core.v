//=============================================================================
// tb_cnn_core.v — cnn_core.v 参数驱动对拍 testbench（阶段 3）
//------------------------------------------------------------------------------
// 向量：tools/gen_cnn_core_vectors.py → verification/vec_core/layer_XX/
//   cfg.hex    每行 48bit {sel[1:0], addr[19:0], wdata[31:0]}
//   in.hex     输入流（行→列→通道）
//   w.hex      权重流（RTL 执行轨迹的 slice 序列，循环发送）
//   expect.hex 期望输出（有效通道字节序列，(h,g,w,m有效) 事件序）
// 期望生成 = ref_cnn_top.run_layer_tiled（cnn_top 行为模型）。
// 运行：bash verification/sim/run_cnn_core.sh
//=============================================================================
`timescale 1ns/1ps

module tb_cnn_core;

    localparam G_C_PAR = 8;
    localparam G_MAX_C = 128;
    localparam G_MAX_LINE = 1024;
    localparam G_MAX_ROWS = 4;

    localparam MAX_CFG = 4096;
    localparam MAX_IN  = 65536;
    localparam MAX_W   = 1048576;
    localparam MAX_EXP = 262144;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    reg        cfg_we = 0;
    reg [1:0]  cfg_sel = 0;
    reg [19:0] cfg_addr = 0;
    reg [31:0] cfg_wdata = 0;
    reg        start = 0;
    reg        i_valid = 0;
    wire       i_ready;
    reg [7:0]  i_data = 0;
    reg        i_weight_valid = 0;
    wire       o_weight_ready;
    reg [7:0]  i_weight_data = 0;
    wire       o_valid;
    wire [63:0] o_data;
    wire       o_done;
    reg        i_acc_ready = 1;

    cnn_core #(
        .G_C_PAR(G_C_PAR), .G_MAX_C(G_MAX_C),
        .G_MAX_LINE(G_MAX_LINE), .G_MAX_ROWS(G_MAX_ROWS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr),
        .cfg_wdata(cfg_wdata),
        .start(start),
        .i_valid(i_valid), .i_ready(i_ready), .i_data(i_data),
        .i_weight_valid(i_weight_valid), .o_weight_ready(o_weight_ready),
        .i_weight_data(i_weight_data),
        .o_valid(o_valid), .o_data(o_data), .o_done(o_done),
        .i_acc_ready(i_acc_ready), .o_acc_valid(), .o_acc_data()
    );

    // ---- 向量存储 ----
    reg [53:0] mem_cfg [0:MAX_CFG-1];   // 54bit: {sel[1:0], addr[19:0], wdata[31:0]}
    reg [7:0]  mem_in  [0:MAX_IN-1];
    reg [7:0]  mem_w   [0:MAX_W-1];
    reg [7:0]  mem_exp [0:MAX_EXP-1];

    integer cfg_n, in_n, w_n, exp_n;
    initial begin
        $readmemh("vec_core/cfg.hex", mem_cfg);
        $readmemh("vec_core/in.hex", mem_in);
        $readmemh("vec_core/w.hex", mem_w);
        $readmemh("vec_core/expect.hex", mem_exp);
        // 数组全 0 时无法探测长度；用哨兵扫描（非零计数不可靠），改用
        // 文件大小由生成器写入 info——此处用固定最大并靠 watchdoog。
        cfg_n = 0; in_n = 0; w_n = 0; exp_n = 0;
    end

    // ---- 参数（cfg 解析）----
    integer out_c, out_h, out_w, chn_block;
    integer errors = 0, exp_idx = 0, events = 0;

    // ---- 仿真周期 + watchdog ----
    reg [31:0] sim_cycles = 0;
    always @(posedge clk) begin
        sim_cycles <= sim_cycles + 1;
        if (sim_cycles == 32'd5000000) begin
            $display("TIMEOUT: state=%0d in_idx=%0d w_idx=%0d exp_idx=%0d events=%0d",
                     dut.state, in_idx, w_idx, exp_idx, events);
            $display("  calc=%0d calc_done=%0d wt_fill=%0d wt_grp=%0d stream=%0d rot_done=%0d has_next=%0d drain=%0d out_row=%0d",
                     dut.calculation_active, dut.calculation_done,
                     dut.weight_fill_active, dut.weight_group_ready,
                     dut.stream_line_fill_done, dut.line_rotation_done,
                     dut.line_rotation_has_next, dut.drain_input_active,
                     dut.output_row_count);
            $display("  i_ready=%0d i_valid=%0d init_active=%0d init_done=%0d prime_active=%0d first_wnd=%0d iw_valid=%0d ow_ready=%0d in_n=%0d w_n=%0d",
                     i_ready, i_valid, dut.initial_fill_active,
                     dut.initial_line_fill_done, dut.prime_k_line_active,
                     dut.first_window_ready, i_weight_valid, o_weight_ready,
                     in_n_count, w_n_count);
            $display("FAIL: timeout");
            $finish;
        end
    end

    // ---- 输出检查（事件序 (h,g,w)，每事件 8 字节取有效）----
    integer g_now, co_now, l;
    always @(posedge clk) begin
        if (o_valid && i_acc_ready) begin
            g_now = (events / out_w) % chn_block;
            co_now = out_c - g_now * 8;
            if (co_now > 8) co_now = 8;
            for (l = 0; l < co_now; l = l + 1) begin
                if (o_data[l*8 +: 8] !== mem_exp[exp_idx]) begin
                    $display("MISMATCH @event=%0d(g=%0d,h=%0d,w=%0d) lane=%0d got=%0d exp=%0d",
                             events, g_now, events / (out_w*chn_block),
                             events % out_w, l,
                             $signed(o_data[l*8 +: 8]), $signed(mem_exp[exp_idx]));
                    errors = errors + 1;
                end
                exp_idx = exp_idx + 1;
            end
            events = events + 1;
        end
    end

    // ---- 输入流驱动（i_ready 时给；完整输入后停）----
    integer in_idx = 0;
    always @(posedge clk) begin
        if (i_ready) begin
            if (in_idx < in_n_count) begin
                i_valid <= 1'b1;
                i_data  <= mem_in[in_idx];
                in_idx  <= in_idx + 1;
            end else begin
                i_valid <= 1'b0;
            end
        end
    end

    // ---- 权重流驱动（循环发送轨迹序列）----
    integer w_idx = 0;
    always @(posedge clk) begin
        if (o_weight_ready) begin
            if (w_n_count > 0) begin
                i_weight_valid <= 1'b1;
                i_weight_data  <= mem_w[w_idx];
                if (w_idx == w_n_count - 1)
                    w_idx <= 0;
                else
                    w_idx <= w_idx + 1;
            end else begin
                i_weight_valid <= 1'b0;
            end
        end
    end

    // ---- 输入/权重长度：由 cfg 与文件加载时探测（见 initial）----
    integer in_n_count = 0, w_n_count = 0;

    // ---- 主流程：复位 → 配置 → 等 o_done ----
    integer c;
    integer ncfg;
    initial begin
        // 配置期间保持复位（RTL 复位后自动启动，须先配完参数再释放）
        #30;
        // 配置（从 cfg.hex 读）
        ncfg = 0;
        // 文件长度探测：扫 cfg.hex 直到 48'hFFFFFFFFFFFFFFFF 哨兵不存在，
        // 改用固定上限 + 首个无效行判断：生成器保证行数 < MAX_CFG
        // 直接读全部 MAX_CFG 行直到出现重复/0 行不现实，此处由生成器写
        // 行数到 cfg.hex 尾部哨兵：sel=3 addr=0xFFFFF data=行数
        for (c = 0; c < MAX_CFG; c = c + 1) begin
            if (mem_cfg[c][1:0] == 2'b11 && mem_cfg[c][21:2] == 20'hFFFFF) begin
                ncfg = c + 2;  // 配置行数 = in 哨兵索引 + 2（含两条哨兵）
                c = MAX_CFG;
            end
        end
        $display("cfg lines: %0d", ncfg);
        for (c = 0; c < ncfg; c = c + 1) begin
            @(posedge clk);
            cfg_we   <= 1'b1;
            cfg_sel  <= mem_cfg[c][1:0];
            cfg_addr <= mem_cfg[c][21:2];
            cfg_wdata<= mem_cfg[c][53:22];
            // 记录 out 维度 + 输入/权重长度哨兵
            if (mem_cfg[c][1:0] == 2'b00) begin
                case (mem_cfg[c][21:2])
                    20'd4: out_c   <= mem_cfg[c][53:22];
                    20'd5: out_h   <= mem_cfg[c][53:22];
                    20'd6: out_w   <= mem_cfg[c][53:22];
                    default: ;
                endcase
            end else if (mem_cfg[c][1:0] == 2'b11) begin
                case (mem_cfg[c][21:2])
                    20'hFFFFF: in_n_count <= mem_cfg[c][53:22];
                    20'hFFFFE: w_n_count  <= mem_cfg[c][53:22];
                    default: ;
                endcase
            end
        end
        @(posedge clk);
        cfg_we <= 1'b0;
        chn_block = (out_c + 7) / 8;
        $display("out=%dx%dx%d chn_block=%0d", out_c, out_h, out_w, chn_block);
        #10 rst_n = 1;   // 释放复位
        @(posedge clk);
        start <= 1'b1;   // 启动推理

        wait (o_done == 1'b1);
        #50;
        if (errors == 0)
            $display("PASS: %0d events, %0d bytes bit-exact", events, exp_idx);
        else
            $display("FAIL: %0d mismatches (%0d events)", errors, events);
        $finish;
    end

endmodule
