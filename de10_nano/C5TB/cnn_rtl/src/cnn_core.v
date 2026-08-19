//=============================================================================
// cnn_core — 黑盒式行块驻留卷积执行器（阶段 4 核心，曾用名 cnn_core_v2）
//=============================================================================
// 与阶段 3 的 cnn_core 相比（黑盒真实架构对齐）：
//   1) 输入流改为 64-bit NHWC8 块序：[C/8][H][W][8]，DDR 顺序 burst 读；
//      pad 行由本模块补 0（不拉 i_ready，DMA 自动跳过）。
//   2) 行缓冲按输入通道块组织：[in_cb][in_row_tile][W][8]（行块驻留）。
//   3) 计算序 = 输出通道组 × 输入通道组 × 窗口（对齐 tools/ref_cnn_top.py）：
//        for cb_out: acc 清零
//          for cb_in (dw 只 cb_in==cb_out): 读 slice → 全行块窗口 MAC 累加
//          全部输入组完成 → requant → 输出事件流（64-bit NHWC8 块序）
//   4) 单行块执行：行块循环/地址重定位在 cnn_top 层。本行块的输入 base
//      行号（= rb*out_row_tile*stride - pad）经 cfg 标量 15 写入。
//
// 可综合性（2025 同步读改造，解决 quartus_map OOM）：
//   - lb 声明为 64-bit 字数组 [cb][row][col]（每字 = 8 lane），S_LOAD 每拍
//     写一字；S_MAC 拆成 S_MAC_RD（同步采样 lb_q）/S_MAC_ACC（乘加累加）
//     两拍流水，使 lb 可被 Quartus 推断为 M10K（原组合读会被展开成
//     寄存器堆 + 读端口 mux，A&S 内存爆掉）。
//   - acc 声明为 256-bit 字数组 [o_row][o_col]（每字 = 8 lane × int32），
//     窗口内只在首 tap 读回 acc_q、寄存器内累加、末 tap 一次性写回；
//     requant 拆 S_REQ_ADDR（采样 acc_q）/S_REQ_OUT（组合 requant）两拍。
//   - 性能：S_MAC/S_REQUANT 每拍流变两拍（tap 率/事件率减半），功能
//     bit-exact（tb 仅按握手比对事件序列，不测周期数）。
//
// 接口（供 cnn_top 的 DMA 层驱动）：
//   cfg：sel=0 标量（addr 索引见下）、sel=1..4 requant 数组（addr=通道）
//   i_stream / iw_stream：64-bit 握手（DMA 连续读，地址 = 已发字节数）
//   o_stream：64-bit 输出事件（8 通道/拍，NHWC8 块序）
//=============================================================================

//-----------------------------------------------------------------------------
// 16×16 乘法单元（模块级 multstyle=dsp：Quartus 对"数组元素上的 multstyle
// 属性"推断不可靠，模块级属性 + 显式例化 100% 走 DSP 18×18，~3-4ns）。
// A_SIGNED=1 时 a 按有符号解释（hilo/hihi 的 a_hi），否则零扩展（lolo/lohi）。
// b 恒为无符号（m_lo/m_hi）。乘积统一 17×17 signed，赋给 32-bit 目标时
// 低 32 位即正确补码（lolo/lohi ≤ 2^32-1、hilo/hihi ∈ [-2^31, 2^31-1]）。
//
// a_signed 必须是参数而非运行时信号：Quartus 对带符号选择 mux 的乘法器
// 无法确定符号，每个例化被实现为 2 个乘法器（占满 1 个 Two Independent
// 18x18 块）——实测 DSP 从预期 40 涨到 80/112（71%），与 M10K 78% 一起
// 造成局部布线拥塞（fit 失败）。参数化后符号在综合时确定，每例化 1 个
// 18×18 且可打包，DSP 块数减半，时序不变（仍走 DSP）。
//-----------------------------------------------------------------------------
(* multstyle = "dsp" *) module mul16x16_dsp #(
    parameter A_SIGNED = 0    // 1: a 按有符号解释（hilo/hihi）；0: 零扩展（lolo/lohi）
) (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [31:0] p
);
    wire signed [16:0] sa = A_SIGNED ? $signed(a) : $signed({1'b0, a});
    wire signed [16:0] sb = $signed({1'b0, b});
    assign p = sa * sb;
endmodule

module cnn_core #(
    parameter G_MAX_IN_ROWS= 41,   // 最大输入行块高度（模型 max in_tile=39）
    parameter G_MAX_OROWS  = 20,   // 最大输出行块高度（模型 max tile=19）
    parameter G_MAX_W      = 512,  // 输入行缓冲最大列数（2 的幂：lb 地址乘法变移位，150MHz 收敛；软件实际 ≤302）
    parameter G_MAX_OW     = 150,  // 最大输出宽度（OUTPUT_MAX_W）
    parameter G_MAX_C      = 1024  // 最大输出通道数（requant 数组容量，模型 out_c 最大 1024）
)(
    input  wire                clk,
    input  wire                rst_n,

    // ---- cfg 参数加载 ----
    input  wire                cfg_we,
    input  wire [2:0]          cfg_sel,
    input  wire [19:0]         cfg_addr,
    input  wire [31:0]         cfg_wdata,

    // ---- 启动（单行块；o_done 后由 cnn_top 重定位下一行块）----
    input  wire                start,

    // ---- 输入流（64-bit NHWC8 块序，signed int8×8）----
    input  wire                i_valid,
    output wire                i_ready,
    input  wire [63:0]         i_data,

    // ---- 权重流（64-bit，slice 序 [cb_out][cb_in][8][9][8] 连续）----
    input  wire                iw_valid,
    output wire                ow_ready,
    input  wire [63:0]         iw_data,

    // ---- 输出流（64-bit，NHWC8 块序：行→列→8 通道）----
    output reg                 o_valid,
    input  wire                o_ready,
    output reg  [63:0]         o_data,

    // ---- 单行块完成 ----
    output reg                 o_done
);

    //-----------------------------------------------------------------------
    // 运行时标量寄存器（cfg_sel=0，cfg_addr=索引，cfg_wdata=值）
    //  0:type 1:act 2:in_c 3:in_h 4:in_w 5:out_c 6:out_h 7:out_w
    //  8:k 9:pad 10:stride 11:out_row_tile 12:in_row_tile
    //  13:in_cb 14:out_cb 15:base_row（本行块输入 base 行号）
    //-----------------------------------------------------------------------
    reg [3:0]  type_reg, act_reg;
    reg [11:0] in_h_reg,  in_w_reg;
    reg [11:0] out_c_reg, out_w_reg;
    reg [3:0]  k_reg, pad_reg, stride_reg;
    reg [11:0] out_row_tile_reg, in_row_tile_reg;
    reg [11:0] in_cb_reg, out_cb_reg;
    reg signed [12:0] base_row_reg;

    // requant 参数存储（cfg_sel 1/2/3 = bias/mult/shift，addr=输出通道）
    // 见下方 8 lane 显式展开块：每 lane 并行读 8 个不同通道地址，
    // 单端口 RAM 每拍只能 1 地址 → 每 lane 独立一份 M10K（8×3 组）

    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 0) begin
            case (cfg_addr)
                20'd0: type_reg      <= cfg_wdata[3:0];
                20'd1: act_reg       <= cfg_wdata[1:0];
                20'd3: in_h_reg      <= cfg_wdata[11:0];
                20'd4: in_w_reg      <= cfg_wdata[11:0];
                20'd5: out_c_reg     <= cfg_wdata[11:0];
                20'd7: out_w_reg     <= cfg_wdata[11:0];
                20'd8: k_reg         <= cfg_wdata[3:0];
                20'd9: pad_reg       <= cfg_wdata[3:0];
                20'd10: stride_reg   <= cfg_wdata[3:0];
                20'd11: out_row_tile_reg <= cfg_wdata[11:0];
                20'd12: in_row_tile_reg  <= cfg_wdata[11:0];
                20'd13: in_cb_reg     <= cfg_wdata[11:0];
                20'd14: out_cb_reg    <= cfg_wdata[11:0];
                20'd15: base_row_reg  <= cfg_wdata[12:0];
                default: ;
            endcase
        end
    end

    //-----------------------------------------------------------------------
    // requant 参数存储（requant_store.v，8 lane 独立 M10K；cfg_sel 1/2/3 =
    // bias/mult/shift，addr=输出通道）：8 lane 并行读 8 个不同通道地址
    // （v_out_ch = o_group*8+ln），单端口 RAM 每拍仅 1 地址 → 每 lane 一份
    // （8×3 组 ≈78 块 M10K）。存储逻辑（数组 + 读写 always）在子模块内部，
    // 顶层 generate 只做实例化——Quartus 对 generate 块内数组的 M10K 推断
    // 不可靠（曾实测仅 RAM 恢复救回 16/32 个，其余展开成寄存器爆 ALM）。
    // M10K 同步读 1 拍延迟：bias 于 S_REQ_ADDR 发起、mult 于 S_REQ_MUL 发起、
    // shift 于 S_REQ_MUL2 发起——正好插入现有 requant 流水，事件序列不变。
    //-----------------------------------------------------------------------
    wire signed [31:0] rq_bias_q  [0:7];
    wire [31:0]        rq_mult_q  [0:7];
    wire [7:0]         rq_shift_q [0:7];
    // 每 lane 读地址 = o_group*8 + lane（10-bit，o_group ≤ 127，和 ≤1023 不溢出；
    // 写地址 cfg_addr[9:0] 截断，值域 ≤1023 与 1024 深匹配）。rq_raddr_q 每拍
    // 采样：o_group 在 S_REQ_OUT3 末更新，到 requant 前隔多拍早已稳定为新组。
    reg [9:0] rq_raddr_q;   // rq_raddr 打拍（断 o_group→32 个 requant RAM 地址布线段）
    wire [9:0] rq_raddr = {o_group[6:0], 3'd0};   // 组合地址，直接驱动读端口（链短）
    genvar rq_i;
    generate
        for (rq_i = 0; rq_i < 8; rq_i = rq_i + 1) begin : g_rq_store
            localparam [9:0] RQ_OFF = rq_i;
            requant_store #(.G_MAX_C(G_MAX_C)) u_rq_store (
                .clk(clk),
                .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
                .raddr(rq_raddr_q + RQ_OFF),
                .q_bias(rq_bias_q[rq_i]), .q_mult(rq_mult_q[rq_i]), .q_shift(rq_shift_q[rq_i])
            );
        end
    endgenerate
    //-----------------------------------------------------------------------
    // 存储：行缓冲 lb（64-bit 字 = 8 lane，col 0..G_MAX_W-1）
    //       acc（256-bit 字 = 8 lane × int32）
    //       wbuf [lane*72 + t*8 + m]（权重 slice，小数组保持寄存器）
    // 强制 ramstyle = M10K 且展平为单维数组（Quartus 对多维 unpacked 数组
    // 的 RAM 推断不可靠：lb[cb][row][col] 三维 + acc 256-bit 超宽会被展开成
    // 寄存器堆 + 读端口 mux，A&S 内存爆到 40GB+；单维数组 + 常量乘法拼接
    // 地址是官方推荐的可靠推断写法，功能/时序完全不变）。
    //-----------------------------------------------------------------------
    (* ramstyle = "M10K, no_rw_check" *) reg [63:0] lb [0:G_MAX_IN_ROWS*G_MAX_W-1];
    (* ramstyle = "M10K, no_rw_check" *) reg signed [255:0] acc [0:G_MAX_OROWS*G_MAX_OW-1];

    // 单维索引拼接（常量乘法综合时折叠成移位/加法，不产生逻辑）
    wire [20:0] lb_waddr = load_row*G_MAX_W + load_col;
    wire [20:0] lb_raddr = mac_r*G_MAX_W + mac_c_cl;
    wire [15:0] acc_waddr_mac = mac_row*G_MAX_OW + mac_col;
    wire [15:0] acc_raddr_clr = rq_row*G_MAX_OW + rq_col;
    wire [15:0] acc_raddr_mac = mac_row*G_MAX_OW + mac_col;
    reg signed [7:0] wbuf [0:7][0:7][0:8];   // [lane][m][t]：w_q 读 = 9:1 mux（原 [lane*72+t*8+m] 平铺是 72:1，mac_t→w_q 路径 18.95ns）

    // MAC 8×8 乘法器：拆成独立子模块（模块级 multstyle 强制 DSP/LUT 分配），
    // lane 0-3 走 DSP（32 个 × ≤2 block = ≤64）、lane 4-7 走 LUT（32 个 ≈2K ALUT）。
    // 乘法器与加法树物理隔离，避免 Quartus 重新融合成 mult_hlmac（每乘加 2 block）。
    wire signed [15:0] mac_p [0:7][0:7];  // signed：负积需符号扩展进加法树

    // w_q：权重读寄存器（S_MAC_RD 拍与 lb_q 同步采样，拆掉 mac_t→wbuf 读 mux）
    reg [7:0] w_q [0:7][0:7];
    // mac_p_r：乘法结果寄存器（S_MAC_MUL 拍采样），把 lb M10K 输出→乘法→
    // 加法树→v_sum_r 的组合链再拆一段（4 拍/tap：RD/MUL/MUL2/ACC）
    reg signed [15:0] mac_p_r [0:7][0:7];
    // mac_c_valid_r：S_MAC_RD 拍寄存的 tap 列有效位。必须在使用（generate 实例
    // .en 端口）之前声明为 reg，否则 Verilog 隐式声明为 wire，实例 en 悬空（z）
    reg mac_c_valid_r;
    genvar mac_lane_i, mac_m_i;
    generate
        for (mac_lane_i = 0; mac_lane_i < 8; mac_lane_i = mac_lane_i + 1) begin : mac_lane_g
            for (mac_m_i = 0; mac_m_i < 8; mac_m_i = mac_m_i + 1) begin : mac_mul_g
                if (mac_lane_i < 4) begin : u_dsp
                    mac8x8_dsp u_mac (
                        .a (mac_a_q[8*mac_m_i +: 8]),
                        .b (w_q[mac_m_i][mac_lane_i]),
                        .p (mac_p[mac_lane_i][mac_m_i])
                    );
                end else begin : u_lut
                    mac8x8_lut u_mac (
                        .a (mac_a_q[8*mac_m_i +: 8]),
                        .b (w_q[mac_m_i][mac_lane_i]),
                        .p (mac_p[mac_lane_i][mac_m_i])
                    );
                end
            end
        end
    endgenerate

    // 同步读采样寄存器（RAM 读输出，读地址 = 组合函数，晚一拍有效）
    reg [63:0]          lb_q;
    reg [63:0]          mac_a_q;    // 乘法器 a 输入打拍（S_MAC_MUL 拍寄存 lb_q，拆 lb M10K pass-through 读路径与 DSP 乘法）
    reg signed [255:0]  acc_q;
    reg signed [255:0]  acc_local;   // 窗口内累加（首 tap 用 acc_q 初始化）
    // 读地址寄存（S_MAC_ADDR 拍采样）：lb_raddr/acc_raddr_mac 是 32-bit 常量乘法
    // 组合链（×41*302 / ×302），直通 M10K 地址端口会超 50MHz 周期（setup 违例
    // -0.839ns 主因）；打一拍后组合链终点变为普通寄存器，M10K 地址由寄存器驱动。
    // 最大地址 = 4*41*302-1 = 49527（lb）、19*150+149 = 2999（acc），16-bit 够。
    reg [15:0]          lb_addr_r;
    (* preserve *) reg [15:0] lb_wa_q;    // lb 写地址打拍（S_LOAD 收拍寄存、下一拍写；断 load_row→lb 写口组合链 P2-2）
    reg [63:0]          lb_wd_q;          // lb 写数据打拍（与地址对齐）
    (* preserve *) reg [15:0]       acc_addr_mac_r;   // preserve：阻止吸收进 M10K 地址寄存器（mac_row→porta_address_reg 组合链 -4.223）
    (* preserve *) reg [15:0]       acc_raddr_r;      // requant 读地址打拍（preserve：拆 rq_row→porta_address_reg 组合链 -4.181）
    (* preserve *) reg [15:0]       acc_wa_q;         // acc 写口统一地址（每拍采样：S_MAC_ACC 拍取写回地址、其他拍取清零地址，写口单一寄存器源）
    (* preserve *) reg [15:0]       acc_clr_wa;       // acc 清零地址（S_ACC_CLR 沿前 = 上拍寄存的推进后地址，断 rq_row→porta 组合链 P1）

    // lb 写（流水打拍）：S_LOAD 收拍寄存地址/数据、下一拍写入；退出 S_LOAD 后
    // 由下一状态补写最后位置（S_ACC_CLR：首轮；S_WEIGHT：多 i_group 重装轮
    // load_first=0 直接进 S_WEIGHT 无 S_ACC_CLR，最后 1 字必须在此补写；
    // 后续拍重复写同位置，数据相同无害）
    always @(posedge clk) begin
        if (state == S_LOAD || state == S_ACC_CLR || state == S_WEIGHT)
            lb[lb_wa_q] <= lb_wd_q;
    end

    integer lane, m;
    (* multstyle = "logic" *) reg signed [31:0] v_sum [0:7];   // 8×8 MAC 树用 LUT（省 DSP，~4ns 无时序风险）
    reg signed [31:0] v_sum_r [0:7];   // S_MAC_MUL 拍寄存的乘加树结果（拆流水）

    // 组合中间变量：8 lane 独立 32-bit 加法后拼成整字（避免 part-select 写 RAM
    // 破坏 M10K 推断；acc_local/acc 保持整字写，Quartus 才可推断 block RAM）
    reg [255:0] acc_next;      // acc_local 的下一拍值（每 lane 独立 int32 累加）
    reg [255:0] acc_wr_next;   // 末 tap 写回 acc 的整字（每 lane = acc_local + 本 tap 部分和）
    reg [255:0] acc_wr_q;      // 末 tap 写回数据打拍（断 v_sum_r→acc 写口 256-bit 组合链）
    reg         acc_wr_we_q;   // 写回脉冲：末 tap 沿后置 1，写拍沿清 0（单拍）

    //-----------------------------------------------------------------------
    // 状态机
    //-----------------------------------------------------------------------
    localparam S_IDLE       = 5'd0;
    localparam S_LOAD       = 5'd1;    // 装载输入行块
    localparam S_ACC_CLR    = 5'd2;    // acc 清零（每输出组）
    localparam S_WEIGHT     = 5'd3;    // 装载权重 slice（72 拍）
    localparam S_MAC_ADDR   = 5'd4;    // 窗口 tap 地址拍（寄存 lb/acc 读地址，断 32-bit 乘法组合链）
    localparam S_MAC_RD     = 5'd5;    // 窗口 tap 请求拍（同步采样 lb_q/acc_q）
    localparam S_MAC_MUL    = 5'd6;    // 窗口 tap 乘拍（a 输入打拍 → mac_a_q）
    localparam S_MAC_MUL2   = 5'd7;    // 窗口 tap 乘拍（乘法器 → mac_p_r）
    localparam S_MAC_MUL3   = 5'd8;    // 窗口 tap 加法树拍（mac_p_r → v_sum_r）
    localparam S_MAC_ACC    = 5'd9;    // 窗口 tap 累加拍（v_sum_r → acc_local）
    localparam S_REQ_ADDR   = 5'd10;   // requant 请求拍（采样 acc_q）
    localparam S_REQ_MUL    = 5'd11;   // requant 拍级 1（bias/mult RAM 读打拍）
    localparam S_REQ_MULB   = 5'd12;   // requant 拍级 2（acc_q → v_act_l）
    localparam S_REQ_MUL2   = 5'd13;   // requant 拍级 3（4×16×16 部分积 → v_p_*）
    localparam S_REQ_MUL3   = 5'd14;   // requant 拍级 4（两组中间和 → v_sum_lo/hi）
    localparam S_REQ_MUL4   = 5'd15;   // requant 拍级 5（中间和相加 → v_rq64_l）
    localparam S_REQ_MULC   = 5'd16;   // requant 拍级 6（乘后 bias 加法）
    localparam S_REQ_ACT    = 5'd17;   // requant 拍级 7（relu mux → v_rq64_l）
    localparam S_REQ_OUT    = 5'd18;   // requant 拍级 8（round 桶形移位 → v_rnd_delta）
    localparam S_REQ_ROUND2 = 5'd19;   // requant 拍级 9（v_rq64_l + v_rnd_delta → v_round_l）
    localparam S_REQ_OUT2   = 5'd20;   // requant 拍级 10（v_round_l >>> shift → v_shifted）
    localparam S_REQ_OUT3   = 5'd21;   // requant 拍级 11（wrap → o_data，o_valid 拉高）
    localparam S_DONE       = 5'd22;

    reg [4:0] state;
    reg [11:0] load_row, load_col;
    reg        load_first;        // 本轮装载是 o_group 首 cb（完成后需清 acc）
    reg [7:0]  wf_cnt;
    reg [2:0]  wf_lane;   // wbuf 写 lane 计数（k=3：wf_cnt/9；k=1：wf_cnt），替代除法器
    reg [3:0]  wf_t;      // wbuf 写 tap 计数（k=3：wf_cnt%9；k=1：恒 0）
    reg [11:0] o_group, i_group;
    reg [11:0] mac_row, mac_col;
    reg [3:0]  mac_t;
    reg        mac_first_q, mac_last_q;   // S_MAC_MUL2 拍寄存首/末 tap 标志（拆 mac_t 32-bit 比较链）
    reg [15:0] rq_row, rq_col;      // ≤ out_row_tile×out_w（≤19×150），16-bit 足够

    // 行缓冲行 r 对应输入行 base_row_reg + r；有效 = 输入行 ∈ [0, in_h)
    wire signed [12:0] load_in_row = base_row_reg + load_row;
    wire load_row_valid = (load_in_row >= 0) && (load_in_row < in_h_reg);

    //-----------------------------------------------------------------------
    // 主状态机
    //-----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            o_done <= 0;
            o_valid <= 0;
            load_row <= 0; load_col <= 0;
            load_first <= 1;
            wf_cnt <= 0; wf_lane <= 0; wf_t <= 0; o_group <= 0; i_group <= 0;
            mac_row <= 0; mac_col <= 0; mac_t <= 0;
            rq_row <= 0; rq_col <= 0;
            mac_c_valid_r <= 0;
            acc_local <= 256'sd0;   // 复位归并到主状态机（单一驱动，避免 Quartus 10028）
            acc_wr_q <= 256'sd0;
            acc_wr_we_q <= 0;
            // 地址寄存器复位同样归并到主状态机（S_MAC_ADDR 分支在同一块）：
            // lb_q/acc_q 每拍无条件采样，无复位时 x 地址越界读会污染 acc_q
            lb_addr_r <= 16'd0;
            lb_wa_q <= 16'd0;
            lb_wd_q <= 64'h0;
            acc_addr_mac_r <= 16'd0;
            acc_clr_wa <= 16'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    o_done <= 0;
                    if (start) begin
                        load_row <= 0; load_col <= 0;
                        load_first <= 1;
                        o_group <= 0; i_group <= 0;
                        state <= S_LOAD;
                    end
                end

                //---- 装载输入行块（流式：每轮 1 个通道块，cb 选择由顶层 DMA 轮控制）----
                S_LOAD: begin
                    // 多 o_group 时从 S_REQ_OUT3 直接进入：必须拉低 o_valid，
                    // 否则残留 1 → 输出持续被收集（o_ready=1 恒等）→ 失控
                    // （单组层走 S_DONE 拉低故未暴露）
                    o_valid <= 0;
                    // 末 tap 写回（延后 1 拍搭车，case 内写点保 M10K 推断；写拍沿清脉冲）
                    if (acc_wr_we_q) begin
                        acc[acc_wa_q] <= acc_wr_q;
                        acc_wr_we_q <= 0;
                    end
                    // 有效行等输入；pad 行（load_row_valid=0）不消费输入，写 0 跳过
                    //（BRAM 上电不定须清零）
                    if (!load_row_valid || i_valid) begin
                        lb_wa_q <= lb_waddr[15:0];
                        lb_wd_q <= load_row_valid ? i_data : 64'h0;
                        if (load_col == in_w_reg - 1) begin
                            load_col <= 0;
                            if (load_row == in_row_tile_reg - 1) begin
                                load_row <= 0;
                                // 1 cb 完成：o_group 首 cb（load_first）→ 清 acc 开新组；
                                // i_group 循环 → 直接装权重 slice（acc 保留部分和）
                                if (load_first) begin
                                    load_first <= 0;
                                    rq_row <= 0; rq_col <= 0;
                                    acc_clr_wa <= 16'd0;
                                    state <= S_ACC_CLR;
                                end else
                                    state <= S_WEIGHT;
                            end else
                                load_row <= load_row + 1;
                        end else
                            load_col <= load_col + 1;
                    end
                end

                //---- acc 清零（每输出组开始，逐 256-bit 字）----
                S_ACC_CLR: begin
                    o_valid <= 0;
                    // 清零写（沿前 acc_clr_wa = 上拍寄存的"推进后地址" = 当前 rq 位置）
                    acc[acc_clr_wa] <= 256'sd0;
                    // 推进 + 同步寄存"推进后地址"（断 rq_row→acc 写口组合链，P1）
                    if (rq_row == out_row_tile_reg - 1 && rq_col == out_w_reg - 1) begin
                        rq_row <= 0; rq_col <= 0;
                        acc_clr_wa <= 16'd0;
                        i_group <= 0;
                        state <= S_WEIGHT;
                    end else if (rq_col == out_w_reg - 1) begin
                        rq_col <= 0;
                        rq_row <= rq_row + 1;
                        // 递推推进后地址（替代 rq_row*G_MAX_OW 乘加树，-2.596 路径）
                        acc_clr_wa <= acc_clr_wa + G_MAX_OW - out_w_reg + 1;
                    end else begin
                        rq_col <= rq_col + 1;
                        acc_clr_wa <= acc_clr_wa + 1;
                    end
                end

                //---- 装载权重 slice（k=3：72 拍 [lane][9tap][8]；k=1：8 拍，lane 主序跳 72B 到 t=0 组）----
                // 软件布局（conv2d_weight_reorganize）：k=3 slice = 8×9×8B，k=1 slice = 8×1×8B
                S_WEIGHT: begin
                    if (iw_valid) begin
                        // wbuf[lane][m][t] 写：wf_lane/wf_t 同步计数（wf_cnt/9、%9 除法器
                        // 时序爆炸 -53ns，改用计数器：k=3 t 主序递增；k=1 lane 主序，t 恒 0）
                        for (m = 0; m < 8; m = m + 1)
                            wbuf[wf_lane][m][(k_reg == 1) ? 3'd0 : wf_t] <= iw_data[m*8 +: 8];
                        if (k_reg == 1)
                            wf_lane <= wf_lane + 1;
                        else begin
                            if (wf_t == 4'd8) begin
                                wf_t <= 4'd0;
                                wf_lane <= wf_lane + 1;
                            end else
                                wf_t <= wf_t + 1;
                        end
                        if (wf_cnt == (k_reg == 1 ? 8*1 - 1 : 8*9 - 1)) begin
                            wf_cnt <= 0;
                            wf_lane <= 3'd0;
                            wf_t <= 4'd0;
                            mac_row <= 0; mac_col <= 0; mac_t <= 0;
                            mac_kh_q <= 4'd0; mac_kw_q <= 4'd0;
                            state <= S_MAC_ADDR;
                        end else
                            wf_cnt <= wf_cnt + 1;
                    end
                end

                //---- 窗口 MAC（地址拍）：组合计算 lb/acc 读地址并寄存，
                //     断开 32-bit 常量乘法链直通 M10K 地址端口的长路径 ----
                S_MAC_ADDR: begin
                    // 末 tap 写回（延后 1 拍搭车，case 内写点保 M10K 推断；写拍沿清脉冲）
                    if (acc_wr_we_q) begin
                        acc[acc_wa_q] <= acc_wr_q;
                        acc_wr_we_q <= 0;
                    end
                    lb_addr_r      <= lb_raddr[15:0];
                    acc_addr_mac_r <= acc_raddr_mac[15:0];
                    state <= S_MAC_RD;
                end

                //---- 窗口 MAC（请求拍）：地址已寄存，上升沿同步采样 lb_q/acc_q ----
                S_MAC_RD: begin
                    mac_c_valid_r <= mac_c_valid;   // 组合 valid 寄存（断 k_reg→乘加树链）
                    state <= S_MAC_MUL;
                end

                //---- 窗口 MAC（乘拍）：乘加树 → v_sum_r 寄存（拆开累加，缩短组合链）----
                S_MAC_MUL: begin
                    // a 输入打拍：lb_q（S_MAC_RD 沿的新值）→ mac_a_q，拆 lb M10K
                    // pass-through 读路径与 DSP 乘法；b 输入 w_q 已在 S_MAC_RD 沿采样
                    mac_a_q <= mac_c_valid_r ? lb_q : 64'sd0;
                    state <= S_MAC_MUL2;
                end
                S_MAC_MUL2: begin
                    // 乘拍：mac_a_q × w_q → mac_p → mac_p_r（原 S_MAC_MUL 内容移来）
                    for (lane = 0; lane < 8; lane = lane + 1)
                        for (m = 0; m < 8; m = m + 1)
                            mac_p_r[lane][m] <= mac_p[lane][m];
                    // 首/末 tap 标志提前寄存（32-bit 比较拆出 S_MAC_ACC 拍）
                    mac_first_q <= (mac_t == 4'd0);
                    mac_last_q  <= (mac_t == (k_reg == 1 ? 4'd0 : 4'd8));
                    state <= S_MAC_MUL3;
                end
                S_MAC_MUL3: begin
                    // 加法树拍：mac_p_r（寄存器）→ 8 项加法树 → v_sum_r
                    for (lane = 0; lane < 8; lane = lane + 1)
                        v_sum[lane] =
                            mac_p_r[lane][0] + mac_p_r[lane][1] + mac_p_r[lane][2] + mac_p_r[lane][3] +
                            mac_p_r[lane][4] + mac_p_r[lane][5] + mac_p_r[lane][6] + mac_p_r[lane][7];
                    for (lane = 0; lane < 8; lane = lane + 1)
                        v_sum_r[lane] <= v_sum[lane];
                    state <= S_MAC_ACC;
                end

                //---- 窗口 MAC（累加拍）：用上拍寄存的 v_sum_r 累加到 acc_local；
                //     首 tap（mac_t==0）以 acc_q 初始化；末 tap（k=3：mac_t==8；k=1：mac_t==0）写回 acc ----
                S_MAC_ACC: begin
                    // 8 lane 独立 32-bit 累加（acc_local 打包为 256-bit，但每个 lane
                    // 是独立 int32，无跨 lane 进位；整体 256-bit 加法器进位链过长，
                    // 拆成 lane 级加法消除，语义不变）
                    // 注意：必须先用组合变量拼成整字再整字写 acc_local/acc——
                    // 直接 part-select 写（acc[...][32*lane +: 32] <= ...）会阻止
                    // Quartus 把 acc 推断为 M10K，导致 A&S 内存反弹（10+GB）。
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if (mac_first_q)
                            acc_next[32*lane +: 32] = acc_q[32*lane +: 32] + v_sum_r[lane];
                        else
                            acc_next[32*lane +: 32] = acc_local[32*lane +: 32] + v_sum_r[lane];
                    end
                    acc_local <= acc_next;
                    if (mac_last_q) begin
                        // 末 tap：写回最终值（acc_local 尚缺本 tap 部分和；
                        // k=1 单 tap 时直接用 acc_q 累加，避免旧 acc_local 串扰）
                        for (lane = 0; lane < 8; lane = lane + 1)
                            acc_wr_next[32*lane +: 32] = (k_reg == 1) ?
                                (acc_q[32*lane +: 32] + v_sum_r[lane]) :
                                (acc_local[32*lane +: 32] + v_sum_r[lane]);
                        acc_wr_q <= acc_wr_next;   // 写数据打拍（下一拍随转移状态写回）
                        acc_wr_we_q <= 1;       // 写回脉冲（写拍沿清 0）
                        mac_t <= 0;
                        mac_kh_q <= 4'd0;
                        mac_kw_q <= 4'd0;
                        if (mac_col == out_w_reg - 1) begin
                            mac_col <= 0;
                            if (mac_row == out_row_tile_reg - 1) begin
                                mac_row <= 0;
                                if (type_reg == 4 || i_group == in_cb_reg - 1) begin
                                    rq_row <= 0; rq_col <= 0;
                                    acc_clr_wa <= 16'd0;   // requant 从 (0,0) 起，清零地址同步归零
                                    state <= S_REQ_ADDR;
                                end else begin
                                    i_group <= i_group + 1;
                                    state <= S_LOAD;   // 流式：重装下一输入 cb（lb 单块驻留）
                                end
                            end else begin
                                mac_row <= mac_row + 1;
                                state <= S_MAC_ADDR;
                            end
                        end else begin
                            mac_col <= mac_col + 1;
                            state <= S_MAC_ADDR;
                        end
                    end else begin
                        mac_t <= mac_t + 1;
                        mac_kh_q <= (k_reg == 3) ? mac_kh_next : 4'd0;
                        mac_kw_q <= (k_reg == 3) ? mac_kw_next : 4'd0;
                        state <= S_MAC_ADDR;
                    end
                end

                //---- requant 请求拍：采样 acc[rq_row][rq_col] ----
                S_REQ_ADDR: begin
                    // 末 tap 写回（延后 1 拍搭车，case 内写点保 M10K 推断；写拍沿清脉冲）
                    if (acc_wr_we_q) begin
                        acc[acc_wa_q] <= acc_wr_q;
                        acc_wr_we_q <= 0;
                    end
                    o_valid <= 0;
                    state <= S_REQ_MUL;
                end

                //---- requant 乘加拍级 1：acc_q + bias（32-bit 加法单独一拍）----
                S_REQ_MUL: begin
                    state <= S_REQ_MULB;
                end

                //---- requant 乘加拍级 2：relu/rcl6 比较 → v_act_l（比较+mux 单独一拍）----
                S_REQ_MULB: begin
                    state <= S_REQ_MUL2;   // 乘后域：acc 打拍后直接乘法（bias 移到乘后）
                end
                S_REQ_MUL4: begin
                    state <= S_REQ_MULC;   // 乘后 bias 加法拍
                end
                S_REQ_MULC: begin
                    state <= S_REQ_ACT;    // relu mux 拍
                end
                S_REQ_ACT: begin
                    state <= S_REQ_OUT;
                end

                //---- requant 乘法拍级 1：16×16 部分积（4 个 DSP 乘法，见独立 always）----
                S_REQ_MUL2: begin
                    state <= S_REQ_MUL3;
                end

                //---- requant 乘法拍级 2：两组中间和（lolo+lohi、hilo+hihi 并行 64-bit 加法）----
                S_REQ_MUL3: begin
                    state <= S_REQ_MUL4;
                end

                //---- requant 乘法拍级 3：中间和相加 → v_rq64_l（见独立 always）----
                // 注意：S_REQ_MUL4 的转移在上面（MUL4 → MULC 乘后 bias）

                //---- requant round 拍级 1：round 桶形移位（1<<(shift-1)，单独一拍）----
                S_REQ_OUT: begin
                    state <= S_REQ_ROUND2;
                end

                //---- requant round 拍级 2：v_rq64_l + v_rnd_delta（64-bit 加法单独一拍）----
                S_REQ_ROUND2: begin
                    state <= S_REQ_OUT2;
                end

                //---- requant 输出移位拍：v_round_l >>> shift（桶形移位单独一拍）----
                S_REQ_OUT2: begin
                    state <= S_REQ_OUT3;
                end

                //---- requant 输出拍（饱和 → o_data；行→列→8 通道，块序）----
                S_REQ_OUT3: begin
                    o_valid <= 1;
                    if (o_ready) begin
                        if (rq_col == out_w_reg - 1) begin
                            rq_col <= 0;
                            if (rq_row == out_row_tile_reg - 1) begin
                                rq_row <= 0;
                                // o_valid 由下一状态（S_ACC_CLR/S_DONE）拉低，
                                // 保证最后一事件 o_data 已被输出握手收走
                                if (o_group == out_cb_reg - 1) begin
                                    acc_clr_wa <= 16'd0;
                                    state <= S_DONE;
                                end else begin
                                    o_group <= o_group + 1;
                                    rq_row <= 0; rq_col <= 0;
                                    acc_clr_wa <= 16'd0;
                                    load_first <= 1;
                                    state <= S_LOAD;   // 流式：重装新 o_group 的输入 cb
                                end
                            end else begin
                                rq_row <= rq_row + 1;
                                // 递推清零地址（跟随 rq，替代乘加树）
                                acc_clr_wa <= acc_clr_wa + G_MAX_OW - out_w_reg + 1;
                                state <= S_REQ_ADDR;
                            end
                        end else begin
                            rq_col <= rq_col + 1;
                            acc_clr_wa <= acc_clr_wa + 1;
                            state <= S_REQ_ADDR;
                        end
                    end
                end

                S_DONE: begin
                    o_valid <= 0;
                    o_done <= 1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    //-----------------------------------------------------------------------
    // 握手
    //-----------------------------------------------------------------------
    assign i_ready = (state == S_LOAD) && load_row_valid;
    assign ow_ready = (state == S_WEIGHT);

    //-----------------------------------------------------------------------
    // 组合乘加树（MAC）：lb_q[cb][r][c] × wbuf[lane][t][m] 累加
    //-----------------------------------------------------------------------
    // 窗口 tap 位置：k_reg ∈ {1,3}（model_profile 实测）、mac_t ∈ [0,8]，
    // 查表替代 32-bit 除法器（lpm_divide，组合链超长导致 setup 违例 -82ns）
    // （lb 已流式单 cb 驻留，不再有 mac_cb 通道块维度）
    // 下一 tap 的 kh/kw 提前寄存（S_MAC_ACC 拍采样查表(mac_t+1)，S_MAC_ADDR 拍用，
    // 拆 mac_t→查表→lb_raddr 组合链 P2-3；S_WEIGHT 末/末 tap 分支重置为 kh(0)=0）
    reg [3:0] mac_kh_q, mac_kw_q;
    wire [3:0] mac_t_next = mac_t + 4'd1;
    reg [3:0] mac_kh_next, mac_kw_next;
    always @(*) begin
        if (k_reg == 3) begin
            case (mac_t_next[3:0])
                4'd0: begin mac_kh_next = 4'd0; mac_kw_next = 4'd0; end
                4'd1: begin mac_kh_next = 4'd0; mac_kw_next = 4'd1; end
                4'd2: begin mac_kh_next = 4'd0; mac_kw_next = 4'd2; end
                4'd3: begin mac_kh_next = 4'd1; mac_kw_next = 4'd0; end
                4'd4: begin mac_kh_next = 4'd1; mac_kw_next = 4'd1; end
                4'd5: begin mac_kh_next = 4'd1; mac_kw_next = 4'd2; end
                4'd6: begin mac_kh_next = 4'd2; mac_kw_next = 4'd0; end
                4'd7: begin mac_kh_next = 4'd2; mac_kw_next = 4'd1; end
                default: begin mac_kh_next = 4'd2; mac_kw_next = 4'd2; end
            endcase
        end else begin
            mac_kh_next = 4'd0; mac_kw_next = 4'd0;
        end
    end
    // 行（lb 索引）：窗口第 kh 行 = o_row*stride + kh + pad（装载时行 0 = base 行）；
    // stride_reg ∈ {1,2}，移位替代乘法器（mac_row*stride_reg 会被综合成 32-bit 乘法）
    wire [11:0] mac_r = ((stride_reg == 2) ? {mac_row[10:0], 1'b0} : mac_row) + mac_kh_q;
    // 列（输入列）：w*stride + kw - pad，越界补 0
    wire signed [12:0] mac_c = $signed((stride_reg == 2) ? {mac_col[10:0], 1'b0} : mac_col)
                            + $signed(mac_kw_q) - $signed(pad_reg);
    wire mac_c_valid = (mac_c >= 0) && (mac_c < in_w_reg);
    wire [11:0] mac_c_cl = mac_c[11:0];

    // S_MAC_RD 拍寄存 tap 列有效位（拆 k_reg→乘加树组合链，setup 违例 -6.18ns 主因）

    // 同步采样：RAM 读端口（组合地址在上升沿被采样，数据晚一拍到 lb_q/acc_q）
    // acc 读复用同一端口（S_MAC 读 {mac_row,mac_col}，requant 读 {rq_row,rq_col}，
    // 两阶段状态互斥）
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lb_q <= 64'h0;
            acc_q <= 256'sd0;
        end else begin
            lb_q <= lb[lb_addr_r];
            // 权重采样（S_MAC_RD 拍）：mac_t 在 S_MAC_ACC 末更新，隔 S_MAC_ADDR 拍到
            // S_MAC_RD 沿共 2 拍窗口，拆 mac_t→wbuf 72:1 mux 组合链（原每拍采样
            // 窗口仅 1 拍，w_q 路径 slack -5.189；S_MAC_ADDR/RD 拍不用 w_q，
            // 乘法仍在 S_MAC_MUL 拍用新值，拍序不变）
            if (state == S_MAC_RD) begin
                for (lane = 0; lane < 8; lane = lane + 1)
                    for (m = 0; m < 8; m = m + 1)
                        w_q[lane][m] <= wbuf[lane][m][mac_t];
            end
            acc_raddr_r <= acc_raddr_clr[15:0];   // requant 读地址打拍（每拍采样，rq_row 推进后首拍锁存新值）
            rq_raddr_q <= rq_raddr;                 // requant 参数读地址打拍（每拍，断 o_group 布线）
            // acc 写口统一地址：S_MAC_MUL3 拍取写回地址（当轮 mac_row），S_MAC_ACC 拍写回
            // 用沿前值；其他拍取清零地址（rq_row 稳定）——写口永不直接连组合
            if (state == S_MAC_MUL3)
                acc_wa_q <= acc_waddr_mac[15:0];
            // acc_q 采样：requant 首拍（S_REQ_ADDR）时 acc_raddr_r 尚是旧地址
            //（S_REQ_OUT3 末拍才推进 rq_row/rq_col，acc_raddr_r 同步采样的是沿前
            // 旧值），S_REQ_MUL 拍才采到新地址——两拍连续采样取后者（S_REQ_MULB
            // 用）；S_REQ_MUL2 起地址不变，重复读无意义，保持即可。
            // 单读端口：两个互斥地址在 RAM 口前 mux。原 if/else 全覆盖写法
            // Quartus 可合并成 1 端口，改 if/else if（带保持）后被推断成 2 个读
            // 端口 = 2 份 acc RAM（≈75 个 M10K），超 5CSEMA5F31 容量报 276003；
            // 显式单表达式强制 1 端口，功能/时序不变。
            if (state == S_MAC_RD || state == S_REQ_ADDR || state == S_REQ_MUL)
                acc_q <= acc[(state == S_MAC_RD) ? acc_addr_mac_r : acc_raddr_r];
        end
    end

    //-----------------------------------------------------------------------
    // requant 流水（拆流水，避免单拍组合链过深）：
    //   S_REQ_MUL  拍：寄存 rq_bias_m/rq_mult_m（RAM 读打拍，断 pass-through 读路径）
    //   S_REQ_MULB 拍：acc_q → v_act_l（乘法输入打拍；bias 移到乘后 MULC 拍）
    //   S_REQ_MULC 拍：relu/rcl6 比较 → v_act_l（比较+mux 单独一拍，~3ns）
    //   S_REQ_MUL2 拍：4×16×16 部分积（DSP）→ 寄存 v_p_*/v_shift_l
    //   S_REQ_MUL3 拍：部分积移位相加 → v_rq64_l（64-bit 加法树）
    //   S_REQ_OUT  拍：round 加法 → v_round_l（用上拍寄存的积）
    // 拆流水后功能/事件序列不变（tb 按事件对拍，不测周期数）。
    //-----------------------------------------------------------------------
    integer ln;
    reg signed [63:0] v_rq64;
    reg signed [31:0] v_act_l [0:7];   // 每 lane 的 act 后值（流水级 2）
    // 乘法拆三级（150MHz 单级 32×33 组合乘法 slack -10.1ns；64-bit 加法树 2 级仍紧）：
    //   级 1 = 4 个 16×16 DSP 乘法（v_act_l = a_hi<<16 + a_lo，rq_mult_q = m_hi<<16 + m_lo）
    //   级 2 = 两组 64-bit 并行加法（v_sum_lo/v_sum_hi），每级仅 1 个加法器
    //   级 3 = 中间和相加 → v_rq64_l
    (* multstyle = "dsp" *) reg [31:0] v_p_lolo [0:7];   // a_lo × m_lo（无符号 16×16，DSP 18×18）
    (* multstyle = "dsp" *) reg [31:0] v_p_lohi [0:7];   // a_lo × m_hi（无符号 16×16）
    (* multstyle = "dsp" *) reg signed [31:0] v_p_hilo [0:7];  // a_hi × m_lo（signed 16 × unsigned 16）
    (* multstyle = "dsp" *) reg signed [31:0] v_p_hihi [0:7];  // a_hi × m_hi（signed 16 × unsigned 16）
    reg [31:0] rq_mult_m [0:7];   // S_REQ_MUL 拍寄存的乘法器 b 输入（拆 RAM 读与 DSP 乘法组合链）
    reg signed [31:0] rq_bias_m [0:7];  // S_REQ_MUL 拍寄存的 bias RAM 读（拆 RAM 读与 32-bit 加法组合链）

    // 16×16 乘法用显式 DSP 例化（模块级 multstyle 最可靠；数组属性曾被 Quartus 忽略）
    wire [31:0] mul_lolo [0:7], mul_lohi [0:7], mul_hilo [0:7], mul_hihi [0:7];
    genvar gi;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : g_mul16
            mul16x16_dsp #(.A_SIGNED(0)) u_lolo (.a(v_act_l[gi][15:0]), .b(rq_mult_m[gi][15:0]), .p(mul_lolo[gi]));
            mul16x16_dsp #(.A_SIGNED(0)) u_lohi (.a(v_act_l[gi][15:0]), .b(rq_mult_m[gi][31:16]), .p(mul_lohi[gi]));
            mul16x16_dsp #(.A_SIGNED(1)) u_hilo (.a(v_act_l[gi][31:16]), .b(rq_mult_m[gi][15:0]), .p(mul_hilo[gi]));
            mul16x16_dsp #(.A_SIGNED(1)) u_hihi (.a(v_act_l[gi][31:16]), .b(rq_mult_m[gi][31:16]), .p(mul_hihi[gi]));
        end
    endgenerate
    reg signed [63:0] v_sum_lo [0:7];  // lolo + lohi<<16（无符号两项）
    reg signed [63:0] v_sum_hi [0:7];  // hilo<<16 + hihi<<32（符号扩展两项）
    reg signed [63:0] v_rnd_delta [0:7];  // round 桶形移位结果（1<<(shift-1)）
    reg signed [63:0] v_shifted [0:7];    // 算术右移结果（>>> shift）
    (* multstyle = "dsp" *) reg signed [63:0] v_rq64_l [0:7];  // 每 lane 的 64-bit 积（流水级 2，保 DSP）
    reg [7:0] v_shift_l [0:7];         // 每 lane 的 shift 值（S_REQ_MUL2 拍寄存 RAM 读，断 M10K q 路径）
    reg signed [63:0] v_round_l [0:7]; // 每 lane 的 round 后值（流水级 3）
    reg [11:0] v_out_ch;
    reg [7:0] v_q [0:7];

    // 乘加拍级 1（S_REQ_MUL）：bias/mult RAM 读打拍 → rq_bias_m/rq_mult_m
    // （bias 加法与 DSP 乘法各拆独立一拍，断 M10K pass-through 读路径组合穿透）
    always @(posedge clk) begin
        if (state == S_REQ_MUL) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                rq_bias_m[ln] <= rq_bias_q[ln];
            end
        end
    end

    // 乘法器 b 输入打拍（S_REQ_MUL 拍寄存 rq_mult_q）：拆开 RAM 读路径
    // （含 read-during-write pass-through mux）与 DSP 乘法，避免组合穿透直达 v_p_*
    always @(posedge clk) begin
        if (state == S_REQ_MUL) begin
            for (ln = 0; ln < 8; ln = ln + 1)
                rq_mult_m[ln] <= rq_mult_q[ln];
        end
    end

    // 乘加拍级 2（S_REQ_MULB）：acc_q → v_act_l（乘法输入打拍；bias 已移到乘后域，
    // 与 CPU cvt_kernel 语义 round(acc·ws·is/os + bias/os) 对齐）
    always @(posedge clk) begin
        if (state == S_REQ_MULB) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_act_l[ln] <= 32'sd0;
                else
                    v_act_l[ln] <= acc_q[32*ln +: 32];
            end
        end
    end

    // 乘法级 3（S_REQ_MUL4）→ 乘后 bias（S_REQ_MULC）→ act mux（S_REQ_ACT）：
    // v_rq64_l 单一 always 驱动（Quartus 10028 多驱动——verilator 容忍多块赋值、
    // Quartus 报 constant driver 冲突；合并后每 state 分支仍只含单拍操作）
    always @(posedge clk) begin
        if (state == S_REQ_MUL4) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_rq64_l[ln] <= 64'sd0;
                else
                    v_rq64_l[ln] <= v_sum_lo[ln] + v_sum_hi[ln];
            end
        end else if (state == S_REQ_MULC) begin
            // 乘后 bias 加法：v_rq64_l += bias_mul<<8（q22 左移 8 对齐 q30 域）
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_rq64_l[ln] <= 64'sd0;
                else
                    v_rq64_l[ln] <= v_rq64_l[ln] +
                                    {{32{rq_bias_m[ln][31]}}, rq_bias_m[ln], 8'd0};
            end
        end else if (state == S_REQ_ACT) begin
            // relu/rcl6 mux（乘后域，64-bit）
            // 2026-08-10：relu6 去掉 rcl6 钳位（黑盒实测无 min6，见
            // BLACKBOX_NUMERICS.md；CPU 语义的 min(6/os) 与黑盒不符，且
            // box 头输入直接来自 relu6_1/relu6_3 输出，钳位改变检测头
            // 输入 → 上板几百框误检）。act==1 与 act==2 现行为相同。
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg) begin
                    v_rq64_l[ln] <= 64'sd0;
                end else if (act_reg == 2'd1 || act_reg == 2'd2) begin
                    v_rq64_l[ln] <= v_rq64_l[ln][63] ? 64'sd0 : v_rq64_l[ln];
                end
            end
        end
    end

    // 乘加拍级 3c（S_REQ_ACT）的 relu mux 已并入乘法级 3 的单一 always
    //（v_rq64_l 单驱动，Quartus 10028）

    // 乘法级 1（S_REQ_MUL2）：v_act_l × rq_mult_q 拆 4 个 16×16 部分积（DSP 18×18，~4ns）；
    // 同时把 RAM 读出的 shift 值寄存为 v_shift_l——断开 M10K q 输出到输出拍组合链的长路径
    always @(posedge clk) begin
        if (state == S_REQ_MUL2) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg) begin
                    v_p_lolo[ln] <= 32'd0;
                    v_p_lohi[ln] <= 32'd0;
                    v_p_hilo[ln] <= 32'sd0;
                    v_p_hihi[ln] <= 32'sd0;
                end else begin
                    v_p_lolo[ln] <= mul_lolo[ln];
                    v_p_lohi[ln] <= mul_lohi[ln];
                    v_p_hilo[ln] <= mul_hilo[ln];
                    v_p_hihi[ln] <= mul_hihi[ln];
                end
                v_shift_l[ln] <= rq_shift_q[ln];
            end
        end
    end

    // 乘法级 2（S_REQ_MUL3）：4 项部分积分为两组中间和（并行 64-bit 加法，各 ~3ns）
    always @(posedge clk) begin
        if (state == S_REQ_MUL3) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg) begin
                    v_sum_lo[ln] <= 64'sd0;
                    v_sum_hi[ln] <= 64'sd0;
                end else begin
                    v_sum_lo[ln] <= {32'd0, v_p_lolo[ln]}
                                  + {16'd0, v_p_lohi[ln], 16'd0};
                    v_sum_hi[ln] <= {{16{v_p_hilo[ln][31]}}, v_p_hilo[ln], 16'd0}
                                  + {v_p_hihi[ln], 32'd0};
                end
            end
        end
    end

    //（v_rq64_l 的赋值已并入乘法级 3 的单一 always——见上）
    // 乘加拍级 3（S_REQ_MUL4 分支）：两组中间和相加 → v_rq64_l（单个 64-bit 加法，~3ns）
    // 注意：MUL4/MULC/ACT 三个 state 分支共用此 always（v_rq64_l 单驱动）

    // round 拍级 1（S_REQ_OUT）：round 桶形移位单独一拍（1<<(shift-1)，~5ns）
    always @(posedge clk) begin
        if (state == S_REQ_OUT) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_rnd_delta[ln] <= 64'sd0;
                else if (v_shift_l[ln] > 8'd0)
                    v_rnd_delta[ln] <= 64'sd1 << (v_shift_l[ln] - 8'd1);
                else
                    v_rnd_delta[ln] <= 64'sd0;
            end
        end
    end

    // round 拍级 2（S_REQ_ROUND2）：v_rq64_l + v_rnd_delta → v_round_l（64-bit 加法单独一拍，~3ns）
    always @(posedge clk) begin
        if (state == S_REQ_ROUND2) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_round_l[ln] <= 64'sd0;
                else
                    v_round_l[ln] <= v_rq64_l[ln] + v_rnd_delta[ln];
            end
        end
    end

    // 输出移位拍（S_REQ_OUT2）：v_round_l >>> shift（64-bit 桶形移位单独一拍，~5ns）
    always @(posedge clk) begin
        if (state == S_REQ_OUT2) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_shifted[ln] <= 64'sd0;
                else
                    v_shifted[ln] <= v_round_l[ln] >>> v_shift_l[ln];
            end
        end
    end

    // 输出拍（S_REQ_OUT3）：移位值 8 位截断（wrap）→ o_data
    // 黑盒语义（BLACKBOX_NUMERICS.md 实测）：y = r & 0xFF，非饱和！
    // 饱和会把 box 头（act=0）超出 int8 范围的 logits 全部钳成 ±127，
    // 抹平 softmax 区分度 → 全图高 score 误检框（上板实测几百框）。
    // wrap 保留字节差异（超出部分翻转为对端符号，与黑盒位模式一致）。
    always @(posedge clk) begin
        if (state == S_REQ_OUT3) begin
            for (ln = 0; ln < 8; ln = ln + 1) begin
                v_out_ch = o_group * 8 + ln;
                if (v_out_ch >= out_c_reg)
                    v_q[ln] = 8'd0;
                else
                    // 黑盒实测（BLACKBOX_NUMERICS.md）：round 后负值 -1（floor 除法特性）。
                    // box 头（act=0）负 logits 无此修正会偏大 1-2 LSB（2026-08-10）
                    v_q[ln] = (v_shifted[ln] - {63'd0, v_shifted[ln][63]}) & 64'hFF;
            end
            o_data <= {v_q[7], v_q[6], v_q[5], v_q[4],
                       v_q[3], v_q[2], v_q[1], v_q[0]};
        end
    end

endmodule
