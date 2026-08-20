//=============================================================================
// tb_cnn_top — 端到端对拍（阶段 5；dut = cnn_top_core，QSys 由 cnn_top wrapper 适配）
//=============================================================================
// 模拟：DDR 内存模型（param/scale/输入/权重/输出）+ Avalon 从（ARM 侧）
//       + 主接口响应。流程：填内存 → 写 6 寄存器 → START → 轮询自清 →
//       输出区与 expect.hex 比对。
//=============================================================================
`timescale 1ns/1ps

module tb_cnn_top;

    localparam MEM_WORDS = 1 << 20;   // 8MB
    // 基址间距按最大层需求（in 最大 38×38×2cb×8B≈0x5A40；w≈0x0D80；out≈0x8760）
    // DDR 段布局（2026-08 扩大：随机层 in_h 到 75 → in_bytes 最大 16×75×75=90000B，
    // 原 DDRIN 8KB 段溢出污染 DDRW 权重段——layer_02 FAIL 根因）
    localparam PARAM_BASE  = 32'h000000;
    localparam SCALE_BASE  = 32'h001000;
    localparam DDRIN_BASE  = 32'h020000;   // 128KB（≥ in_bytes 最大 90000B）
    localparam DDRW_BASE   = 32'h040000;   // 128KB
    localparam DDROUT_BASE = 32'h080000;   // 256KB（out 最大 24×75×75=135000B）

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    // 从接口
    reg  [7:0]  as_address = 0;
    reg         as_write = 0, as_read = 0;
    reg  [31:0] as_writedata = 0;
    wire [31:0] as_readdata;
    wire        as_waitrequest;

    // 主接口
    wire [31:0] pr_address;
    wire        pr_read;
    wire [31:0] pr_readdata;
    wire        pr_readdatavalid;
    wire        pr_waitrequest;
    wire [31:0] sr_address;
    wire        sr_read;
    wire [31:0] sr_readdata;
    wire        sr_readdatavalid;
    wire        sr_waitrequest;
    wire [31:0] lr_address;
    wire        lr_read;
    wire [63:0] lr_readdata;
    wire        lr_readdatavalid;
    wire        lr_waitrequest;
    wire [31:0] wr_address;
    wire        wr_read;
    wire [63:0] wr_readdata;
    wire        wr_readdatavalid;
    wire        wr_waitrequest;
    wire [31:0] ow_address;
    wire        ow_write;
    wire [63:0] ow_writedata;
    reg         ow_waitrequest = 0;
    wire [4:0]  load_burstcount;

    cnn_top_core dut (
        .clk(clk), .rst_n(rst_n),
        .as_address(as_address), .as_write(as_write), .as_read(as_read),
        .as_writedata(as_writedata), .as_readdata(as_readdata),
        .as_waitrequest(as_waitrequest),
        .pr_address(pr_address), .pr_read(pr_read), .pr_readdata(pr_readdata),
        .pr_readdatavalid(pr_readdatavalid), .pr_waitrequest(pr_waitrequest),
        .sr_address(sr_address), .sr_read(sr_read), .sr_readdata(sr_readdata),
        .sr_readdatavalid(sr_readdatavalid), .sr_waitrequest(sr_waitrequest),
        .lr_address(lr_address), .lr_read(lr_read), .lr_readdata(lr_readdata),
        .lr_readdatavalid(lr_rdv), .lr_waitrequest(lr_waitrequest),
        .wr_address(wr_address), .wr_read(wr_read), .wr_readdata(wr_readdata),
        .wr_readdatavalid(wr_rdv), .wr_waitrequest(wr_waitrequest),
        .ow_address(ow_address), .ow_write(ow_write), .ow_writedata(ow_writedata),
        .ow_waitrequest(ow_waitrequest),
        .load_burstcount(load_burstcount)
    );

    // DDR 内存
    reg [63:0] mem [0:MEM_WORDS-1];

    //=====================================================================
    // 流水读模型：模拟 mm_bridge_sdram0（altera_avalon_mm_bridge，
    // MAX_PENDING_RESPONSES=4）→ HPS DDR 的真实读行为——
    // read 请求被接受（waitrequest=0）后，数据延迟 RD_LAT 拍由
    // readdatavalid 返回（readdata 与 readdatavalid 同拍有效）。
    // pr/sr 用单笔在途模型（请求拍锁存地址），足以暴露 read&&!waitrequest
    // 误判；lr/wr 共享 load master，用 4 深多笔在途 + readdatavalid 广播
    // 模型（与 cnn_top.v 真实接法一致），可复现 lr/wr 返回交错污染：
    // S_LOAD 末几笔在途命令的返回在 S_WEIGHT 期间到达，广播后必须由
    // cnn_top_core 内 cmd FIFO 按命令序路由回本侧，否则装载计数错乱。
    //=====================================================================
    localparam RD_LAT = 3;   // 模拟 DDR 流水读延迟（拍）

    // pr/sr 独立 4 深多笔在途模型（与 lr/wr 共享 load master 同构；
    // pop 在 lat==1 拍输出数据，保持与旧单笔模型相同的 RD_LAT=3 有效延迟）
    reg [31:0] pr_addr_q [0:3];
    reg [7:0]  pr_lat_q  [0:3];
    reg [1:0]  pr_head, pr_tail;
    reg [2:0]  pr_occ;
    integer    pr_i;
    wire pr_accept = pr_read && !pr_waitrequest;
    wire pr_pop    = (pr_occ != 3'd0) && (pr_lat_q[pr_head] == 8'd1);

    always @(posedge clk) begin
        if (!rst_n) begin
            pr_head <= 0; pr_tail <= 0; pr_occ <= 0;
            for (pr_i = 0; pr_i < 4; pr_i = pr_i + 1)
                pr_lat_q[pr_i] <= 8'd0;
        end else begin
            for (pr_i = 0; pr_i < 4; pr_i = pr_i + 1)
                if (pr_lat_q[pr_i] != 8'd0)
                    pr_lat_q[pr_i] <= pr_lat_q[pr_i] - 8'd1;
            if (pr_accept && (pr_occ < 3'd4)) begin
                pr_addr_q[pr_tail] <= pr_address;
                pr_lat_q[pr_tail]  <= RD_LAT;
                pr_tail <= pr_tail + 2'd1;
                if (pr_pop) begin
                    pr_head <= pr_head + 2'd1;   // 同拍出+入：occ 净 0
                end else
                    pr_occ <= pr_occ + 1;
            end else if (pr_pop) begin
                pr_head <= pr_head + 2'd1;
                pr_occ <= pr_occ - 1;
            end
        end
    end

    assign pr_readdata = pr_addr_q[pr_head][2] ? mem[pr_addr_q[pr_head] >> 3][63:32]
                                               : mem[pr_addr_q[pr_head] >> 3][31:0];
    assign pr_readdatavalid = pr_pop;
    assign pr_waitrequest = (pr_occ >= 3'd4);

    // sr 同构模型
    reg [31:0] sr_addr_q [0:3];
    reg [7:0]  sr_lat_q  [0:3];
    reg [1:0]  sr_head, sr_tail;
    reg [2:0]  sr_occ;
    integer    sr_i;
    wire sr_accept = sr_read && !sr_waitrequest;
    wire sr_pop    = (sr_occ != 3'd0) && (sr_lat_q[sr_head] == 8'd1);

    always @(posedge clk) begin
        if (!rst_n) begin
            sr_head <= 0; sr_tail <= 0; sr_occ <= 0;
            for (sr_i = 0; sr_i < 4; sr_i = sr_i + 1)
                sr_lat_q[sr_i] <= 8'd0;
        end else begin
            for (sr_i = 0; sr_i < 4; sr_i = sr_i + 1)
                if (sr_lat_q[sr_i] != 8'd0)
                    sr_lat_q[sr_i] <= sr_lat_q[sr_i] - 8'd1;
            if (sr_accept && (sr_occ < 3'd4)) begin
                sr_addr_q[sr_tail] <= sr_address;
                sr_lat_q[sr_tail]  <= RD_LAT;
                sr_tail <= sr_tail + 2'd1;
                if (sr_pop) begin
                    sr_head <= sr_head + 2'd1;   // 同拍出+入：occ 净 0
                end else
                    sr_occ <= sr_occ + 1;
            end else if (sr_pop) begin
                sr_head <= sr_head + 2'd1;
                sr_occ <= sr_occ - 1;
            end
        end
    end

    assign sr_readdata = sr_addr_q[sr_head][2] ? mem[sr_addr_q[sr_head] >> 3][63:32]
                                               : mem[sr_addr_q[sr_head] >> 3][31:0];
    assign sr_readdatavalid = sr_pop;
    assign sr_waitrequest = (sr_occ >= 3'd4);

    //---- 共享 load master（lr/wr 复用，cnn_top.v 真实接法）----
    // 多笔在途（深度 4 = MAX_PENDING_RESPONSES，返回保序），每笔命令接受
    // 拍入队 {addr, lat, beats_left}；RD_LAT 拍后按序逐拍返回 beats_left 拍，
    // 每拍地址 +8；最后一拍返回后出队。桥满（occ==4）时 waitrequest 反压。
    reg [31:0] ld_addr_q [0:3];
    reg [7:0]  ld_lat_q  [0:3];
    reg [2:0]  ld_beats_q [0:3];
    reg [1:0]  ld_head, ld_tail;
    reg [2:0]  ld_occ;
    integer ld_i;
    wire ld_accept = (lr_read || wr_read) && !lr_waitrequest;
    wire ld_pop    = (ld_occ != 3'd0) && (ld_lat_q[ld_head] == 8'd0) &&
                     (ld_beats_q[ld_head] == 3'd1);

    always @(posedge clk) begin
        if (!rst_n) begin
            ld_head <= 0; ld_tail <= 0; ld_occ <= 0;
            for (ld_i = 0; ld_i < 4; ld_i = ld_i + 1) begin
                ld_lat_q[ld_i] <= 8'd0;
                ld_beats_q[ld_i] <= 3'd0;
            end
        end else begin
            for (ld_i = 0; ld_i < 4; ld_i = ld_i + 1)
                if (ld_lat_q[ld_i] != 8'd0)
                    ld_lat_q[ld_i] <= ld_lat_q[ld_i] - 8'd1;
            // 队首就绪且剩余拍 >0：本拍返回一字，地址 +8、剩余 -1
            if (ld_occ != 3'd0 && ld_lat_q[ld_head] == 8'd0 &&
                ld_beats_q[ld_head] != 3'd0) begin
                ld_addr_q[ld_head]  <= ld_addr_q[ld_head] + 8;
                ld_beats_q[ld_head] <= ld_beats_q[ld_head] - 3'd1;
            end
            // 命令入队/出队（单语句净 0，同拍 head/tail 同步推进）
            if (ld_accept && (ld_occ < 3'd4)) begin
                ld_addr_q[ld_tail]  <= lr_read ? lr_address : wr_address;
                ld_lat_q[ld_tail]   <= RD_LAT;
                ld_beats_q[ld_tail] <= load_burstcount[2:0];
                ld_tail <= ld_tail + 2'd1;
                if (ld_pop) begin
                    ld_head <= ld_head + 2'd1;   // 同拍出队：head 推进、occ 不变
                end else
                    ld_occ <= ld_occ + 1;
            end else if (ld_pop) begin
                ld_head <= ld_head + 2'd1;
                ld_occ <= ld_occ - 1;
            end
        end
    end

    wire lr_rdv, wr_rdv;
    assign lr_readdata = mem[ld_addr_q[ld_head] >> 3];
    assign wr_readdata = mem[ld_addr_q[ld_head] >> 3];
    assign lr_rdv = (ld_occ != 3'd0) && (ld_lat_q[ld_head] == 8'd0) &&
                    (ld_beats_q[ld_head] != 3'd0);
    assign wr_rdv = lr_rdv;
    // 桥满反压（MAX_PENDING_RESPONSES=4）
    assign lr_waitrequest = (ld_occ >= 3'd4);
    assign wr_waitrequest = (ld_occ >= 3'd4);

    // 输出写
    always @(posedge clk) begin
        if (ow_write && !ow_waitrequest)
            mem[ow_address >> 3] <= ow_writedata;
    end

    // 向量存储
    reg [31:0] param_mem [0:511];
    reg [31:0] scale_mem [0:511];
    reg [63:0] in_mem  [0:262143];
    reg [63:0] w_mem   [0:262143];
    reg [63:0] exp_mem [0:262143];
    integer nin, nw, nexp, nscale;

    integer i, errors = 0;
    integer out_c, out_h, out_w;

    reg [31:0] sim_cycles = 0;
    always @(posedge clk) begin
        sim_cycles <= sim_cycles + 1;
        if (sim_cycles == 32'd100000000) begin
            $display("TIMEOUT: state=%0d rb=%0d pr=%0d", dut.state, dut.rb,
                     dut.pr_read);
            $finish;
        end
    end
    // 断言：lr/wr 分时（load master 复用前提）
    always @(posedge clk) begin
        if (dut.lr_read && dut.wr_read)
            $display("  ASSERT FAIL: lr_read && wr_read 同时拉高（load master 复用冲突）");
    end

    // 诊断：lr_got/wr_got 计数 vs 命令数（验证 FIFO 路由无丢失/无串扰）
    integer lr_got_cnt = 0, wr_got_cnt = 0;
    integer dbg_lr = 0, dbg_wr = 0, dbg_sr = 0, dbg_lb = 0, dbg_ov = 0;
    always @(posedge clk) begin
        if (dut.lr_got) lr_got_cnt = lr_got_cnt + 1;
        if (dut.wr_got) wr_got_cnt = wr_got_cnt + 1;
    end

    task arm_write(input [7:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            as_address <= addr; as_writedata <= data; as_write <= 1'b1;
            @(posedge clk);
            as_write <= 1'b0;
        end
    endtask

    task arm_read(input [7:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            as_address <= addr; as_read <= 1'b1;
            @(posedge clk);
            as_read <= 1'b0;
            data = as_readdata;
        end
    endtask

    // 填内存：把 32-bit 序列按半字摆进 64-bit mem
    task fill32(input [31:0] base, input integer n, input reg [31:0] arr [0:511]);
        integer j;
        begin
            for (j = 0; j < n; j = j + 1) begin
                if (j[0])
                    mem[(base + j*4) >> 3][63:32] = arr[j];
                else
                    mem[(base + j*4) >> 3][31:0] = arr[j];
            end
        end
    endtask

    initial begin
        $readmemh("vec_cnn_top/param.hex", param_mem);
        $readmemh("vec_cnn_top/scale.hex", scale_mem);
        $readmemh("vec_cnn_top/in.hex", in_mem);
        $readmemh("vec_cnn_top/w.hex", w_mem);
        $readmemh("vec_cnn_top/expect.hex", exp_mem);

        // 哨兵
        nin = 0; nw = 0; nexp = 0; nscale = 0;
        for (i = 0; i < 262143; i = i + 1)
            if (in_mem[i] == 64'hFFFFFFFFFFFFFFFF) nin = i;
        for (i = 0; i < 262143; i = i + 1)
            if (w_mem[i] == 64'hFFFFFFFFFFFFFFFF) nw = i;
        for (i = 0; i < 262143; i = i + 1)
            if (exp_mem[i] == 64'hFFFFFFFFFFFFFFFF) nexp = i;
        nscale = 4 * param_mem[7];

        out_c = param_mem[7];
        out_h = param_mem[8];
        out_w = param_mem[9];
        $display("layer: type=%0d in_c=%0d in_h=%0d out_c=%0d out_h=%0d out_w=%0d k=%0d s=%0d act=%0d tile=%0d rb=%0d", param_mem[15], param_mem[4], param_mem[5], out_c, out_h, out_w, param_mem[10], param_mem[13], param_mem[14], param_mem[17], param_mem[19]);

        #10 rst_n = 1;
        @(posedge clk);

        // 填内存
        fill32(PARAM_BASE, 27, param_mem);
        fill32(SCALE_BASE, nscale, scale_mem);
        for (i = 0; i < nin; i = i + 1)
            mem[(DDRIN_BASE + i*8) >> 3] = in_mem[i];
        for (i = 0; i < nw; i = i + 1)
            mem[(DDRW_BASE + i*8) >> 3] = w_mem[i];

        // 写寄存器（软件流程：先配 DMA 基址，再 START）
        // 注意：core 的 ow_address = reg_ddrout + p_output_offset*8
        // （param[3] = out_off_words，输出区偏移），故 reg_ddrout 需预减，
        // 使实际写入落在 DDROUT_BASE 比对区
        arm_write(8'h10, DDRIN_BASE);
        arm_write(8'h1C, DDRW_BASE);
        arm_write(8'h28, DDROUT_BASE - param_mem[3]*8);
        arm_write(8'h34, PARAM_BASE);
        arm_write(8'h40, SCALE_BASE);
        arm_write(8'h00, 32'h1);   // START

        // 轮询完成（START 自清）
        begin : poll
            integer wc;
            reg [31:0] st;
            wc = 0;
            st = 1;
            while (st[0]) begin
                arm_read(8'h00, st);
                wc = wc + 1;
                if (wc > 5000000) begin
                    $display("  START poll timeout: state=%0d rb=%0d core_st=%0d lr_p=%0d wr_p=%0d cnt=%0d is_wr=%0d lr_rd=%0d rdv=%0d lr_got=%0d i_rdy=%0d i_vld=%0d icb=%0d ibeat=%0d wbeat=%0d wre=%0d iwr=%0d wf=%0d og=%0d inseg=%0d loadr=%0d lastcb=%0d",
                             dut.state, dut.rb, dut.core.state,
                             dut.lr_pending, dut.wr_pending, dut.cmd_cnt,
                             dut.cmd_is_wr, dut.lr_read, dut.lr_readdatavalid,
                             dut.lr_got, dut.core_i_ready, dut.core_i_valid,
                             dut.dma_icb, dut.dma_ibeat, dut.dma_wbeat,
                             dut.load_busy, dut.core_ow_ready,
                             dut.core.wf_cnt, dut.core.o_group,
                             dut.in_seg_words_r, dut.load_rows_r,
                             dut.lr_last_cb_r);
                    disable poll;
                end
            end
        end

        // 比对输出区（全图 NHWC8）
        begin : cmp
            integer cnt;
            cnt = 0;
            for (i = 0; i < nexp; i = i + 1) begin
                if (exp_mem[i] == 64'hFFFFFFFFFFFFFFFF) continue;  // 哨兵
                if (mem[(DDROUT_BASE + i*8) >> 3] !== exp_mem[i]) begin
                    $display("  out event %0d: got=%016x exp=%016x", i,
                             mem[(DDROUT_BASE + i*8) >> 3], exp_mem[i]);
                    errors = errors + 1;
                    if (errors > 10) begin
                        $display("  ... more mismatches suppressed");
                        disable cmp;
                    end
                end
                cnt = cnt + 1;
            end
            $display("  compared %0d events", cnt);
        end

        if (errors == 0)
            $display("PASS: %0d events bit-exact", nexp);
        else
            $display("FAIL: %0d errors", errors);
        $display("DBG: lr_got=%0d wr_got=%0d occ=%0d cnt=%0d",
                 lr_got_cnt, wr_got_cnt, ld_occ, dut.cmd_cnt);
        $display("DBG: cfg in_cb=%0d in_h=%0d in_w=%0d out_cb=%0d out_w=%0d k=%0d stride=%0d type=%0d tile=%0d in_tile=%0d act=%0d pad=%0d og=%0d ig=%0d",
                 dut.core.in_cb_reg, dut.core.in_h_reg, dut.core.in_w_reg,
                 dut.core.out_cb_reg, dut.core.out_w_reg,
                 dut.core.k_reg, dut.core.stride_reg, dut.core.type_reg,
                 dut.core.out_row_tile_reg, dut.core.in_row_tile_reg,
                 dut.core.act_reg, dut.core.pad_reg,
                 dut.core.o_group, dut.core.i_group);
        $display("DBG: dma in_seg=%0d in_tail=%0d out_seg=%0d w_rb=%0d cb_stride=%0d lr_last_cb=%0d load_rows=%0d rb_base=%0d in_rb_base=%0d",
                 dut.in_seg_words_r, dut.in_seg_tail_r, dut.out_seg_words_r,
                 dut.w_rb_beats_r, dut.in_cb_stride_r, dut.lr_last_cb_r,
                 dut.load_rows_r, dut.rb_base_r, dut.in_rb_base_r);
        $finish;
    end

endmodule
