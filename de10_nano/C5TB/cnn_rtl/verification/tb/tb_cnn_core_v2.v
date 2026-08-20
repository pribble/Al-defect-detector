//=============================================================================
// tb_cnn_core_v2 — 行块驻留对拍（vs ref_cnn_top）
//=============================================================================
// 向量：cfg.hex（54-bit：sel|addr|wdata）、in.hex/w.hex/expect.hex（64-bit）
// 流程：解析 cfg 标量 → 循环行块（写 base_row → start → 驱动 in/w 段 →
//       收输出 → 比对 expect 段）→ PASS/FAIL
//=============================================================================
`timescale 1ns/1ps

module tb_cnn_core_v2;

    localparam MAX_CFG = 4096;
    localparam MAX_IN  = 262144;
    localparam MAX_W   = 1048576;
    localparam MAX_EXP = 65536;

    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n = 0;

    // cfg
    reg        cfg_we = 0;
    reg [2:0]  cfg_sel = 0;
    reg [19:0] cfg_addr = 0;
    reg [31:0] cfg_wdata = 0;
    // start
    reg start = 0;
    // 输入流
    reg i_valid = 0;
    wire i_ready;
    wire i_pf_ready;
    reg [63:0] i_data = 0;
    // 权重流
    reg iw_valid = 0;
    wire ow_ready;
    reg [63:0] iw_data = 0;
    // 输出流
    wire o_valid;
    reg o_ready = 1;
    wire [63:0] o_data;
    wire o_done;

    cnn_core dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr),
        .cfg_wdata(cfg_wdata),
        .start(start),
        .i_valid(i_valid), .i_ready(i_ready), .i_pf_ready(i_pf_ready), .i_data(i_data),
        .iw_valid(iw_valid), .ow_ready(ow_ready), .iw_data(iw_data),
        .o_valid(o_valid), .o_ready(o_ready), .o_data(o_data),
        .o_done(o_done)
    );

    // 向量存储
    reg [63:0] mem_cfg [0:MAX_CFG-1];
    reg [63:0] mem_in  [0:MAX_IN-1];
    reg [63:0] mem_w   [0:MAX_W-1];
    reg [63:0] mem_exp [0:MAX_EXP-1];
    reg [63:0] out_buf [0:MAX_EXP-1];

    integer ncfg, nin, nw, nexp;
    integer c, i;
    integer errors = 0;

    // 解析出的标量
    integer in_c, in_h, in_w, out_c, out_h, out_w;
    integer kk, pp, ss, tile, in_tile, in_cb, out_cb, row_block;
    integer typ, act;

    // 行块驱动
    integer rb, in_ptr, exp_ptr;
    integer in_rb_cnt, w_rb_cnt, o_cnt;
    integer in_seg, w_seg, exp_seg;
    reg rb_active = 0;

    // watchdog
    reg [31:0] sim_cycles = 0;
    always @(posedge clk) begin
        sim_cycles <= sim_cycles + 1;
        if (sim_cycles == 32'd8000000) begin
            $display("TIMEOUT: rb=%0d state=%0d in_rb=%0d w_rb=%0d o_cnt=%0d in_ptr=%0d exp_ptr=%0d", rb, dut.state, in_rb_cnt, w_rb_cnt, o_cnt, in_ptr, exp_ptr);
            $finish;
        end
    end

    //-----------------------------------------------------------------------
    // 输入驱动（行块段）
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rb_active && (i_ready || i_pf_ready)) begin
            if (in_rb_cnt < in_seg) begin
                i_valid <= 1'b1;
                i_data  <= mem_in[in_ptr + in_rb_cnt];
                in_rb_cnt <= in_rb_cnt + 1;
            end else
                i_valid <= 1'b0;
        end else if (!rb_active)
            i_valid <= 1'b0;
    end

    // 权重驱动（每行块重放 w 段）
    always @(posedge clk) begin
        if (rb_active && ow_ready) begin
            if (w_rb_cnt < w_seg) begin
                iw_valid <= 1'b1;
                iw_data  <= mem_w[w_rb_cnt];   // w 段从 0 开始（每行块同）
                w_rb_cnt <= w_rb_cnt + 1;
            end else
                iw_valid <= 1'b0;
        end else if (!rb_active)
            iw_valid <= 1'b0;
    end

    // 输出收集
    always @(posedge clk) begin
        if (o_valid && o_ready) begin
            out_buf[o_cnt] <= o_data;
            o_cnt <= o_cnt + 1;
        end
    end

    //-----------------------------------------------------------------------
    // 主流程
    //-----------------------------------------------------------------------
    task write_cfg(input [2:0] sel, input [19:0] addr, input [31:0] wdata);
        begin
            @(posedge clk);
            cfg_we <= 1'b1; cfg_sel <= sel; cfg_addr <= addr; cfg_wdata <= wdata;
            @(posedge clk);
            cfg_we <= 1'b0;
        end
    endtask

    initial begin
        // 读向量
        $readmemh("vec_core_v2/cfg.hex",    mem_cfg);
        $readmemh("vec_core_v2/in.hex",     mem_in);
        $readmemh("vec_core_v2/w.hex",      mem_w);
        $readmemh("vec_core_v2/expect.hex", mem_exp);

        // 哨兵扫描
        ncfg = 0;
        for (i = 0; i < MAX_CFG; i = i + 1)
            if (mem_cfg[i] == {64{1'b1}}) ncfg = i;
        nin = 0;
        for (i = 0; i < MAX_IN; i = i + 1)
            if (mem_in[i] == 64'hFFFFFFFFFFFFFFFF) nin = i;
        nw = 0;
        for (i = 0; i < MAX_W; i = i + 1)
            if (mem_w[i] == 64'hFFFFFFFFFFFFFFFF) nw = i;
        nexp = 0;
        for (i = 0; i < MAX_EXP; i = i + 1)
            if (mem_exp[i] == 64'hFFFFFFFFFFFFFFFF) nexp = i;

        // 解析标量
        in_c = 0; in_h = 0; in_w = 0; out_c = 0; out_h = 0; out_w = 0;
        kk = 0; pp = 0; ss = 0; tile = 0; in_tile = 0; in_cb = 0;
        out_cb = 0; row_block = 0; typ = 0; act = 0;
        for (c = 0; c < ncfg; c = c + 1) begin
            if (mem_cfg[c][54:52] == 3'd0) begin
                case (mem_cfg[c][51:32])
                    20'd0: typ = mem_cfg[c][31:0];
                    20'd1: act      = mem_cfg[c][31:0];
                    20'd2: in_c     = mem_cfg[c][31:0];
                    20'd3: in_h     = mem_cfg[c][31:0];
                    20'd4: in_w     = mem_cfg[c][31:0];
                    20'd5: out_c    = mem_cfg[c][31:0];
                    20'd6: out_h    = mem_cfg[c][31:0];
                    20'd7: out_w    = mem_cfg[c][31:0];
                    20'd8: kk       = mem_cfg[c][31:0];
                    20'd9: pp       = mem_cfg[c][31:0];
                    20'd10: ss      = mem_cfg[c][31:0];
                    20'd11: tile    = mem_cfg[c][31:0];
                    20'd12: in_tile = mem_cfg[c][31:0];
                    20'd13: in_cb   = mem_cfg[c][31:0];
                    20'd14: out_cb  = mem_cfg[c][31:0];
                    20'd16: row_block = mem_cfg[c][31:0];
                    default: ;
                endcase
            end
        end
        $display("out=%dx%dx%d chn_block=%d row_block=%d tile=%d in_tile=%d type=%d act=%d k=%d s=%d", out_c, out_h, out_w, out_cb, row_block, tile, in_tile, typ, act, kk, ss);

        // 复位
        #10 rst_n = 1;
        @(posedge clk);

        // 写入全部 cfg（标量 + requant 数组）
        for (c = 0; c < ncfg; c = c + 1) begin
            cfg_we <= 1'b1;
            cfg_sel <= mem_cfg[c][54:52];
            cfg_addr <= mem_cfg[c][51:32];
            cfg_wdata <= mem_cfg[c][31:0];
            @(posedge clk);
        end
        cfg_we <= 1'b0;
        @(posedge clk);

        in_ptr = 0;
        exp_ptr = 0;
        w_seg = out_cb * (typ == 4 ? 1 : in_cb) * 72;
        errors = 0;

        for (rb = 0; rb < row_block; rb = rb + 1) begin
            // 本行块装载的输入有效行数
            begin : blk
                integer base, r0, r1;
                base = rb * tile * ss - pp;
                r0 = (base > 0) ? base : 0;
                r1 = (base + in_tile < in_h) ? base + in_tile : in_h;
                in_seg = (r1 > r0 ? r1 - r0 : 0) * in_cb * in_w * out_cb;
            end
            begin : blk2
                integer r_out0, r_out1;
                r_out0 = rb * tile;
                r_out1 = (r_out0 + tile < out_h) ? r_out0 + tile : out_h;
                exp_seg = out_cb * (r_out1 - r_out0) * out_w;
            end

            in_rb_cnt = 0;
            w_rb_cnt = 0;
            o_cnt = 0;

            // 写 base_row + start
            write_cfg(2'd0, 20'd15, (rb * tile * ss - pp) & 32'hFFFFFFFF);
            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            rb_active = 1;
            // 等待完成
            begin : waitloop
                integer wc;
                wc = 0;
                while (!o_done) begin
                    @(posedge clk);
                    wc = wc + 1;
                    if (wc > 1000000) begin
                        $display("  rowblock %0d stuck: state=%0d in_rb=%0d w_rb=%0d o_cnt=%0d i_ready=%0d load_valid=%0d base_row=%0d load_row=%0d load_col=%0d", rb, dut.state, in_rb_cnt, w_rb_cnt, o_cnt, dut.i_ready, dut.load_row_valid, dut.base_row_reg, dut.load_row, dut.load_col);
                        disable waitloop;
                    end
                end
            end
            rb_active = 0;
            @(posedge clk);
            @(posedge clk);

            // 比对输出段
            if (o_cnt != exp_seg) begin
                $display("  rb %0d: o_cnt=%0d != exp_seg=%0d", rb, o_cnt,
                         exp_seg);
                $display("  DBG cfg: out_w=%0d tile=%0d og=%0d rr=%0d rc=%0d state=%0d done=%0d",
                         dut.out_w_reg, dut.out_row_tile_reg, dut.o_group,
                         dut.rq_row, dut.rq_col, dut.state, dut.o_done);
                errors = errors + 1;
            end
            for (i = 0; i < exp_seg && i < o_cnt; i = i + 1) begin
                if (out_buf[i] !== mem_exp[exp_ptr + i]) begin
                    $display("  rb %0d event %0d: got=%016x exp=%016x",
                             rb, i, out_buf[i], mem_exp[exp_ptr + i]);
                    errors = errors + 1;
                    if (errors > 20) begin
                        $display("  ... more mismatches suppressed");
                        i = exp_seg;  // 跳出本段比对
                    end
                end
            end

            in_ptr = in_ptr + in_seg;
            exp_ptr = exp_ptr + exp_seg;
        end

        if (errors == 0)
            $display("PASS: %0d rowblocks, %0d events bit-exact",
                     row_block, exp_ptr);
        else begin
            $display("FAIL: %0d errors", errors);
            $display("GOT_DUMP");
            for (i = 0; i < exp_ptr && i < o_cnt; i = i + 1)
                $display("%016x", out_buf[i]);
            $display("END_DUMP");
        end
        $finish;
    end

endmodule
