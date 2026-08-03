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
    localparam PARAM_BASE  = 32'h000000;
    localparam SCALE_BASE  = 32'h001000;
    localparam DDRIN_BASE  = 32'h002000;
    localparam DDRW_BASE   = 32'h010000;
    localparam DDROUT_BASE = 32'h020000;

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
    reg  [31:0] pr_readdata;
    reg         pr_readdatavalid;
    reg         pr_waitrequest = 0;
    wire [31:0] sr_address;
    wire        sr_read;
    reg  [31:0] sr_readdata;
    reg         sr_readdatavalid;
    reg         sr_waitrequest = 0;
    wire [31:0] lr_address;
    wire        lr_read;
    reg  [63:0] lr_readdata;
    reg         lr_readdatavalid;
    reg         lr_waitrequest = 0;
    wire [31:0] wr_address;
    wire        wr_read;
    reg  [63:0] wr_readdata;
    reg         wr_readdatavalid;
    reg         wr_waitrequest = 0;
    wire [31:0] ow_address;
    wire        ow_write;
    wire [63:0] ow_writedata;
    reg         ow_waitrequest = 0;

    cnn_top_core dut (
        .clk(clk), .rst_n(rst_n),
        .as_address(as_address), .as_write(as_write), .as_read(as_read),
        .as_writedata(as_writedata), .as_readdata(as_readdata),
        .as_waitrequest(as_waitrequest),
        .pr_address(pr_address), .pr_read(pr_read), .pr_readdata(pr_readdata),
        .pr_readdatavalid(pr_rdv), .pr_waitrequest(pr_waitrequest),
        .sr_address(sr_address), .sr_read(sr_read), .sr_readdata(sr_readdata),
        .sr_readdatavalid(sr_rdv), .sr_waitrequest(sr_waitrequest),
        .lr_address(lr_address), .lr_read(lr_read), .lr_readdata(lr_readdata),
        .lr_readdatavalid(lr_rdv), .lr_waitrequest(lr_waitrequest),
        .wr_address(wr_address), .wr_read(wr_read), .wr_readdata(wr_readdata),
        .wr_readdatavalid(wr_rdv), .wr_waitrequest(wr_waitrequest),
        .ow_address(ow_address), .ow_write(ow_write), .ow_writedata(ow_writedata),
        .ow_waitrequest(ow_waitrequest)
    );

    // DDR 内存
    reg [63:0] mem [0:MEM_WORDS-1];

    //=====================================================================
    // 流水读模型：模拟 mm_bridge_sdram0（altera_avalon_mm_bridge，
    // MAX_PENDING_RESPONSES=4）→ HPS DDR 的真实读行为——
    // read 请求被接受（waitrequest=0）后，数据延迟 RD_LAT 拍由
    // readdatavalid 返回（readdata 与 readdatavalid 同拍有效）。
    // 单笔在途模型（请求拍锁存地址），足以暴露 read&&!waitrequest 误判。
    //=====================================================================
    localparam RD_LAT = 3;   // 模拟 DDR 流水读延迟（拍）

    reg [31:0] pr_addr_q;  reg [7:0] pr_lat;   reg pr_rdv;
    reg [31:0] sr_addr_q;  reg [7:0] sr_lat;   reg sr_rdv;
    reg [31:0] lr_addr_q;  reg [7:0] lr_lat;   reg lr_rdv;
    reg [31:0] wr_addr_q;  reg [7:0] wr_lat;   reg wr_rdv;

    task read_model(reg read_sig, input [31:0] req_addr,
                    inout [31:0] addr_q, inout [7:0] lat_cnt,
                    inout reg rdv, input [63:0] mem_ref, ...);
```
不行，task 里不能方便地用 always。直接写 4 个 always 块。为了简洁，用 generate 或宏？直接展开写，清晰第一。

pr/sr（32-bit）：
```verilog
always @(posedge clk) begin
    if (!rst_n) begin
        pr_addr_q <= 0; pr_lat <= 0; pr_rdv <= 0; pr_readdata <= 0;
    end else if (pr_read && pr_lat == 0) begin
        pr_addr_q <= pr_address;
        pr_lat <= RD_LAT;
        pr_rdv <= 0;
    end else if (pr_lat != 0) begin
        pr_lat <= pr_lat - 1;
        if (pr_lat == 1) begin
            pr_readdata <= pr_addr_q[2] ? mem[pr_addr_q >> 3][63:32]
                                        : mem[pr_addr_q >> 3][31:0];
            pr_rdv <= 1;
        end else
            pr_rdv <= 0;
    end else
        pr_rdv <= 0;
end
```
注意 `pr_readdata` 是 reg 且现在是时序赋值 ✓（之前是组合 always @(*) 赋值，需删除旧的）。

同样 sr、lr（64-bit）、wr（64-bit）。

然后 dut 实例化加：
```verilog
.pr_readdata(pr_readdata), .pr_readdatavalid(pr_rdv), ...
```
等等——端口名。dut 的端口叫 pr_readdatavalid。tb 里我用 pr_rdv 变量接。实例化：
```verilog
.pr_readdatavalid(pr_rdv),
.sr_readdatavalid(sr_rdv),
.lr_readdatavalid(lr_rdv),
.wr_readdatavalid(wr_rdv),
```

把旧的组合读 always 块替换成这 4 个 always。<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="edit_file">
<｜｜DSML｜｜parameter name="new_string" string="true">    // DDR 内存
    reg [63:0] mem [0:MEM_WORDS-1];

    //=====================================================================
    // 流水读模型：模拟 mm_bridge_sdram0（altera_avalon_mm_bridge，
    // MAX_PENDING_RESPONSES=4/PIPELINE_RESPONSE=1）→ HPS DDR 的真实读行为：
    // read 请求被接受（waitrequest=0）后，数据延迟 RD_LAT 拍由
    // readdatavalid 返回（readdata 与 readdatavalid 同拍有效）。
    // 单笔在途模型（请求拍锁存地址）即可暴露 read&&!waitrequest 的误判。
    //=====================================================================
    localparam RD_LAT = 3;   // 模拟 DDR 流水读延迟（拍）

    reg [31:0] pr_addr_q;  reg [7:0] pr_lat;   reg pr_rdv;
    reg [31:0] sr_addr_q;  reg [7:0] sr_lat;   reg sr_rdv;
    reg [31:0] lr_addr_q;  reg [7:0] lr_lat;   reg lr_rdv;
    reg [31:0] wr_addr_q;  reg [7:0] wr_lat;   reg wr_rdv;

    always @(posedge clk) begin
        if (!rst_n) begin
            pr_addr_q <= 0; pr_lat <= 0; pr_rdv <= 0; pr_readdata <= 0;
        end else if (pr_read && pr_lat == 0) begin
            pr_addr_q <= pr_address;
            pr_lat <= RD_LAT;
            pr_rdv <= 0;
        end else if (pr_lat != 0) begin
            pr_lat <= pr_lat - 1;
            if (pr_lat == 1) begin
                pr_readdata <= pr_addr_q[2] ? mem[pr_addr_q >> 3][63:32]
                                            : mem[pr_addr_q >> 3][31:0];
                pr_rdv <= 1;
            end else
                pr_rdv <= 0;
        end else
            pr_rdv <= 0;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            sr_addr_q <= 0; sr_lat <= 0; sr_rdv <= 0; sr_readdata <= 0;
        end else if (sr_read && sr_lat == 0) begin
            sr_addr_q <= sr_address;
            sr_lat <= RD_LAT;
            sr_rdv <= 0;
        end else if (sr_lat != 0) begin
            sr_lat <= sr_lat - 1;
            if (sr_lat == 1) begin
                sr_readdata <= sr_addr_q[2] ? mem[sr_addr_q >> 3][63:32]
                                            : mem[sr_addr_q >> 3][31:0];
                sr_rdv <= 1;
            end else
                sr_rdv <= 0;
        end else
            sr_rdv <= 0;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            lr_addr_q <= 0; lr_lat <= 0; lr_rdv <= 0; lr_readdata <= 0;
        end else if (lr_read && lr_lat == 0) begin
            lr_addr_q <= lr_address;
            lr_lat <= RD_LAT;
            lr_rdv <= 0;
        end else if (lr_lat != 0) begin
            lr_lat <= lr_lat - 1;
            if (lr_lat == 1) begin
                lr_readdata <= mem[lr_addr_q >> 3];
                lr_rdv <= 1;
            end else
                lr_rdv <= 0;
        end else
            lr_rdv <= 0;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            wr_addr_q <= 0; wr_lat <= 0; wr_rdv <= 0; wr_readdata <= 0;
        end else if (wr_read && wr_lat == 0) begin
            wr_addr_q <= wr_address;
            wr_lat <= RD_LAT;
            wr_rdv <= 0;
        end else if (wr_lat != 0) begin
            wr_lat <= wr_lat - 1;
            if (wr_lat == 1) begin
                wr_readdata <= mem[wr_addr_q >> 3];
                wr_rdv <= 1;
            end else
                wr_rdv <= 0;
        end else
            wr_rdv <= 0;
    end

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
        arm_write(8'h10, DDRIN_BASE);
        arm_write(8'h1C, DDRW_BASE);
        arm_write(8'h28, DDROUT_BASE);
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
                    $display("  START poll timeout: state=%0d rb=%0d core_st=%0d",
                             dut.state, dut.rb, dut.core.state);
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
        $finish;
    end

endmodule
