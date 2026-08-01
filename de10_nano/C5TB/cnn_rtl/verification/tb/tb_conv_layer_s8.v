//=============================================================================
// tb_conv_layer_s8.v — conv_layer_s8 对拍 testbench
//------------------------------------------------------------------------------
// 数据来源：tools/gen_tb_vectors.py 生成的 verification/vec/*.hex
// 期望输出由 Python 定点同构参考（ref_int8.conv_s8）生成，要求与 RTL bit-exact。
// 运行：verification/sim/run_conv_s8.sh
//=============================================================================
`timescale 1ns/1ps

module tb_conv_layer_s8;

    // ---- 被测模块参数（与 gen_tb_vectors.py 一致）----
    localparam G_C_IN    = 3;
    localparam G_C_OUT   = 4;
    localparam G_W_IN    = 6;
    localparam G_H_IN    = 6;
    localparam G_C_PAR   = 4;
    localparam G_KERNEL  = 3;
    localparam G_PADDING = 1;
    localparam G_STRIDE  = 2;
    localparam G_ACT     = 1;      // relu（与向量一致）
    localparam G_RAW_CLAMP6 = 0;

    localparam C_LINE_SIZE        = G_W_IN * G_C_IN;            // 18
    localparam C_INITIAL_FILL_SIZE = (G_KERNEL - G_PADDING - 1) * C_LINE_SIZE; // 18
    localparam C_PRIME_FILL_SIZE  = G_KERNEL * G_C_IN;          // 9
    localparam C_STREAM_FILL_SIZE = (G_W_IN - G_KERNEL) * G_C_IN; // 9
    localparam C_WEIGHT_FILL_SIZE = G_C_PAR * 9;                // 36
    localparam N_IN    = G_C_IN * G_W_IN * G_H_IN;              // 108
    localparam N_W     = G_C_IN * G_C_OUT * 9;                  // 108
    localparam N_OUT   = G_C_OUT * 3 * 3;                       // 36（stride=2, 6×6→3×3）

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    reg        i_valid = 0;
    wire       i_ready;
    reg [7:0]  i_data = 0;
    reg        i_weight_valid = 0;
    wire       o_weight_ready;
    reg [7:0]  i_weight_data = 0;
    reg        cfg_we = 0;
    reg [1:0]  cfg_sel = 0;
    reg [19:0] cfg_addr = 0;
    reg [31:0] cfg_wdata = 0;
    wire       o_valid;
    wire [G_C_PAR*8-1:0] o_data;
    wire       o_done;
    reg        i_acc_ready = 1;

    conv_layer_s8 #(
        .G_C_IN(G_C_IN), .G_C_OUT(G_C_OUT), .G_W_IN(G_W_IN), .G_H_IN(G_H_IN),
        .G_C_PAR(G_C_PAR), .G_KERNEL(G_KERNEL), .G_PADDING(G_PADDING),
        .G_STRIDE(G_STRIDE), .G_ACT(G_ACT), .G_RAW_CLAMP6(G_RAW_CLAMP6)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .i_valid(i_valid), .i_ready(i_ready), .i_data(i_data),
        .i_weight_valid(i_weight_valid), .o_weight_ready(o_weight_ready),
        .i_weight_data(i_weight_data),
        .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
        .o_valid(o_valid), .o_data(o_data), .o_done(o_done),
        .i_acc_ready(i_acc_ready), .o_acc_valid(), .o_acc_data()
    );

    // ---- 向量存储 ----
    reg [7:0]  mem_in [0:N_IN-1];
    reg [7:0]  mem_w  [0:N_W-1];
    reg [31:0] mem_bias [0:G_C_OUT-1];
    reg [31:0] mem_mult [0:G_C_OUT-1];
    reg [7:0]  mem_shift[0:G_C_OUT-1];
    reg [7:0]  mem_exp [0:N_OUT-1];

    integer in_idx, w_idx, exp_idx;
    integer errors;
    reg [7:0] got_byte;

    // ---- 仿真周期计数 + watchdog（防死循环无限等待）----
    reg [31:0] sim_cycles = 0;
    always @(posedge clk) begin
        sim_cycles <= sim_cycles + 1;
        if (sim_cycles == 32'd100000) begin
            $display("TIMEOUT: 100000 cycles without o_done, aborting");
            $display("  state=%0d init_active=%0d init_done=%0d prime=%0d first_wnd=%0d wt_fill=%0d wt_grp=%0d calc=%0d calc_done=%0d stream=%0d stream_done=%0d rot_done=%0d has_next=%0d drain=%0d out_row=%0d in_idx=%0d w_idx=%0d exp_idx=%0d",
                     dut.state, dut.initial_fill_active, dut.initial_line_fill_done,
                     dut.prime_k_line_active, dut.first_window_ready,
                     dut.weight_fill_active, dut.weight_group_ready,
                     dut.calculation_active, dut.calculation_done,
                     dut.stream_line_fill_active, dut.stream_line_fill_done,
                     dut.line_rotation_done, dut.line_rotation_has_next,
                     dut.drain_input_active, dut.output_row_count,
                     in_idx, w_idx, exp_idx);
            $finish;
        end
    end

    initial begin
        $readmemh("../vec/in.hex",     mem_in);
        $readmemh("../vec/w.hex",      mem_w);
        $readmemh("../vec/bias.hex",   mem_bias);
        $readmemh("../vec/mult.hex",   mem_mult);
        $readmemh("../vec/shift.hex",  mem_shift);
        $readmemh("../vec/expect.hex", mem_exp);
        $display("vectors loaded: in=%0d w=%0d expect=%0d", N_IN, N_W, N_OUT);
    end

    // ---- 输出检查 ----
    always @(posedge clk) begin
        if (o_valid && i_acc_ready) begin
            for (integer l = 0; l < G_C_PAR; l = l + 1) begin
                got_byte = o_data[l*8 +: 8];
                if (got_byte !== mem_exp[exp_idx]) begin
                    $display("MISMATCH @out_idx=%0d lane=%0d: got=%0d expected=%0d",
                             exp_idx, l, $signed(got_byte), $signed(mem_exp[exp_idx]));
                    errors = errors + 1;
                end
                exp_idx = exp_idx + 1;
            end
        end
    end

    // ---- 输入流驱动（行→列→通道）----
    always @(posedge clk) begin
        if (i_ready) begin
            if (in_idx < N_IN) begin
                i_valid <= 1'b1;
                i_data  <= mem_in[in_idx];
                in_idx  <= in_idx + 1;
            end else begin
                i_valid <= 1'b0;
            end
        end
    end

    // ---- 权重流驱动（o_weight_ready 时给下一个权重字节）----
    // 硬件按输入通道重复加载 slice（每窗口每通道一次）：权重循环发送
    always @(posedge clk) begin
        if (o_weight_ready) begin
            i_weight_valid <= 1'b1;
            i_weight_data  <= mem_w[w_idx];
            if (w_idx == N_W - 1)
                w_idx <= 0;
            else
                w_idx <= w_idx + 1;
        end
    end

    // ---- 配置写入 + 复位 + 结束 ----
    integer c;
    initial begin
        errors = 0;
        in_idx = 0; w_idx = 0; exp_idx = 0;
        #30 rst_n = 1;
        // 写 per-channel 参数
        for (c = 0; c < G_C_OUT; c = c + 1) begin
            @(posedge clk);
            cfg_we <= 1'b1; cfg_sel <= 2'b01; cfg_addr <= c; cfg_wdata <= mem_bias[c];
            @(posedge clk);
            cfg_we <= 1'b1; cfg_sel <= 2'b10; cfg_addr <= c; cfg_wdata <= mem_mult[c];
            @(posedge clk);
            cfg_we <= 1'b1; cfg_sel <= 2'b11; cfg_addr <= c; cfg_wdata <= {24'd0, mem_shift[c]};
        end
        @(posedge clk);
        cfg_we <= 1'b0;

        // 等待完成
        wait (o_done == 1'b1);
        #100;
        if (errors == 0)
            $display("PASS: all %0d outputs bit-exact match", N_OUT);
        else
            $display("FAIL: %0d mismatches", errors);
        $finish;
    end

endmodule
