//=============================================================================
// cnn_core.v — 参数驱动单层卷积执行器（cnn_top 复现阶段 3）
//------------------------------------------------------------------------------
// 基于阶段 1 已验证的 conv_layer_s8.v 增量改造：所有综合期 parameter
// （维度/kernel/stride/pad/act/clamp6）改为运行时寄存器（cfg 加载），
// 支持 conv(1) 与 dw(4) 两种类型；G_C_PAR 固定 8（硬件 8×8 MAC 语义）。
//
// 输入/输出流布局（与 ref_cnn_top.py 对拍基准一致）：
//   - 激活流：行→列→通道（(h,w,c)，c 内层，8 通道一组连续 = NHWC8 顺序）
//   - 权重流：slice = [lane(8) × k²]，按输入通道 ci 循环（每 slice 一个 ci）
//   - 输出：每拍 8 通道打包（块末不足 8 通道时无效 lane 输出 0，对拍只取有效）
//
// cfg 寄存器（cfg_we/cfg_sel/cfg_addr/cfg_wdata）：
//   sel 0：维度参数（addr = 编号）：
//     0={type[3:0],act[1:0]} 1=in_c 2=in_h 3=in_w 4=out_c 5=out_h 6=out_w
//     7=kernel 8=pad 9=stride 10=out_row_tile 11=in_row_tile
//   sel 1：bias_int（per out channel，int32）
//   sel 2：mult（per out channel，uint32）
//   sel 3：shift（per out channel，uint8）
//   sel 4：raw_clamp6（per out channel，int32，relu6 用）
//
// 本阶段不含行块循环（row_block 固定 1，输出一次算完；小尺寸验证）；
// 行块调度、Avalon/DMA 属阶段 3b/4+。
//=============================================================================

module cnn_core #(
    parameter G_C_PAR   = 8,     // 并行输出通道（固定 8，硬件语义）
    parameter G_MAX_C   = 128,   // 最大通道数（cfg 数组/权重容量）
    parameter G_MAX_LINE= 1024,  // 最大行字节数（in_w*in_c 上限）
    parameter G_MAX_ROWS= 4      // 行缓冲行数（kernel 最大 3 + 1）
)(
    input  wire                clk,
    input  wire                rst_n,

    // ---- cfg 参数加载 ----
    input  wire                cfg_we,
    input  wire [1:0]          cfg_sel,
    input  wire [19:0]         cfg_addr,
    input  wire [31:0]         cfg_wdata,

    // ---- 启动（写 1 启动一层；o_done 后须重新拉高）----
    input  wire                start,

    // ---- 激活流（signed int8，行→列→通道）----
    input  wire                i_valid,
    output wire                i_ready,
    input  wire [7:0]          i_data,

    // ---- 权重流（signed int8，slice = 8×k²，按输入通道循环）----
    input  wire                i_weight_valid,
    output wire                o_weight_ready,
    input  wire [7:0]          i_weight_data,

    // ---- 量化输出（8 通道打包）----
    output wire                o_valid,
    output wire [G_C_PAR*8-1:0] o_data,
    output wire                o_done,

    // ---- 原始 int32 累加输出（调试）----
    input  wire                i_acc_ready,
    output wire                o_acc_valid,
    output wire [G_C_PAR*32-1:0] o_acc_data
);

    //-----------------------------------------------------------------------
    // 运行时参数寄存器（cfg 加载）
    //-----------------------------------------------------------------------
    reg [3:0]  type_reg;          // 1=conv, 4=dw
    reg [1:0]  act_reg;           // 0 none / 1 relu / 2 relu6
    reg [31:0] in_c_reg,  in_h_reg,  in_w_reg;
    reg [31:0] out_c_reg, out_h_reg, out_w_reg;
    reg [31:0] k_reg, pad_reg, stride_reg;
    reg [31:0] out_row_tile_reg, in_row_tile_reg;
    reg [31:0] row_block_reg;     // 阶段 3 固定为 1（预留）

    reg signed [31:0] bias_store [0:G_MAX_C-1];
    reg [31:0] rq_m_store [0:G_MAX_C-1];
    reg [7:0]  rq_r_store [0:G_MAX_C-1];
    reg signed [31:0] rcl6_store [0:G_MAX_C-1];

    always @(posedge clk) begin
        if (cfg_we) begin
            case (cfg_sel)
                2'b00: begin
                    if (cfg_addr < 13) begin
                        case (cfg_addr)
                            0: begin type_reg <= cfg_wdata[3:0]; act_reg <= cfg_wdata[5:4]; end
                            1: in_c_reg   <= cfg_wdata;
                            2: in_h_reg   <= cfg_wdata;
                            3: in_w_reg   <= cfg_wdata;
                            4: out_c_reg  <= cfg_wdata;
                            5: out_h_reg  <= cfg_wdata;
                            6: out_w_reg  <= cfg_wdata;
                            7: k_reg      <= cfg_wdata;
                            8: pad_reg    <= cfg_wdata;
                            9: stride_reg <= cfg_wdata;
                            10: out_row_tile_reg <= cfg_wdata;
                            11: in_row_tile_reg  <= cfg_wdata;
                            12: row_block_reg    <= cfg_wdata;
                            default: ;
                        endcase
                    end else if (cfg_addr >= 16 && cfg_addr < 16 + G_MAX_C) begin
                        // rcl6（per out channel）：复用 sel=00，addr 16+channel
                        rcl6_store[cfg_addr - 16] <= $signed(cfg_wdata);
                    end
                end
                2'b01: if (cfg_addr < G_MAX_C) bias_store[cfg_addr] <= $signed(cfg_wdata);
                2'b10: if (cfg_addr < G_MAX_C) rq_m_store[cfg_addr] <= cfg_wdata;
                2'b11: if (cfg_addr < G_MAX_C) rq_r_store[cfg_addr] <= cfg_wdata[7:0];
                default: ;
            endcase
        end
    end

    // 派生（组合）
    wire [31:0] c_line_size       = in_w_reg * in_c_reg;
    wire [31:0] c_initial_fill    = (k_reg >= pad_reg + 1) ? (k_reg - pad_reg - 1) * c_line_size : 0;
    wire [31:0] c_prime_fill      = k_reg * in_c_reg;
    wire [31:0] c_kernel_size     = k_reg * k_reg;
    wire [31:0] c_weight_fill     = G_C_PAR * c_kernel_size;
    wire [31:0] c_stream_fill     = (in_w_reg > k_reg) ? (in_w_reg - k_reg) * in_c_reg : 0;
    wire [31:0] c_output_width    = out_w_reg;
    wire [31:0] c_output_height   = out_h_reg;
    wire [31:0] c_output_groups   = (out_c_reg + G_C_PAR - 1) / G_C_PAR;

    //-----------------------------------------------------------------------
    // 状态机（conv_layer_s8 同构）
    //-----------------------------------------------------------------------
    localparam [2:0] S_IDLE              = 3'd0,
                     S_INITIAL_LINE_FILL = 3'd1,
                     S_PRIME_K_LINE      = 3'd2,
                     S_CALC_AND_SLIDING_WINDOW = 3'd3,
                     S_STREAM_LINE_FILLING = 3'd4,
                     S_LINE_ROTATION     = 3'd5;
    reg [2:0] state = S_IDLE;

    //-----------------------------------------------------------------------
    // 存储体
    //-----------------------------------------------------------------------
    reg signed [7:0] weight_buffer [0:G_C_PAR*9-1];        // 当前 slice（k²≤9）
    reg [7:0] line_buffer [0:G_MAX_ROWS-1][0:G_MAX_LINE-1];// 行缓冲（运行时行大小）
    reg signed [31:0] accumulators [0:G_C_PAR-1];
    reg signed [31:0] result_buffer [0:G_C_PAR-1];
    reg [7:0] quantized_result_buffer [0:G_C_PAR-1];
    reg [7:0] row_map [0:2];                               // 逻辑行→物理行（k≤3）
    reg row_valid [0:2];
    reg [7:0] spare_row;
    integer logical_top_row;

    // 控制/计数
    reg initial_fill_active, initial_line_fill_done;
    reg [31:0] initial_fill_count;
    reg prime_k_line_active, first_window_ready;
    reg [31:0] prime_fill_count;
    reg weight_fill_active, weight_group_ready;
    reg [31:0] weight_fill_count;
    reg stream_line_fill_active, stream_line_fill_done;
    reg [31:0] stream_fill_count;
    reg [31:0] stream_bytes_available;
    reg calculation_active, calculation_done, calculation_waiting_for_weights;
    reg [31:0] calculation_kernel_count, calculation_channel_count;
    reg weight_refill_request;
    reg next_window_pending;
    reg [31:0] window_column_count;
    reg [31:0] output_group_count;
    reg result_pending;
    reg line_rotation_done, line_rotation_has_next;
    reg [31:0] output_row_count;
    reg [31:0] vertical_advance_remaining;
    reg drain_input_active;

    //-----------------------------------------------------------------------
    // 组合逻辑
    //-----------------------------------------------------------------------
    wire start_line_fill        = (state == S_IDLE);
    wire start_weight_fill      = (state == S_IDLE) || weight_refill_request ||
                                  (line_rotation_done && line_rotation_has_next &&
                                   !drain_input_active && vertical_advance_remaining == 0 &&
                                   (in_c_reg > 1 || c_output_groups > 1));
    wire start_prime_k_line     = initial_line_fill_done ||
                                  (line_rotation_done && line_rotation_has_next);
    wire start_line_rotation    = (state == S_STREAM_LINE_FILLING) && stream_line_fill_done;
    wire start_stream_line_fill = (state == S_PRIME_K_LINE) && first_window_ready &&
                                  (drain_input_active || vertical_advance_remaining > 0 ||
                                   weight_group_ready);

    assign i_ready = initial_fill_active || prime_k_line_active || stream_line_fill_active;
    wire activation_accepted = i_valid && i_ready;
    wire line_buffer_we = activation_accepted;

    wire [7:0] line_buffer_wr_row = initial_fill_active ?
        (pad_reg + initial_fill_count / c_line_size) : row_map[2];
    wire [31:0] line_buffer_wr_addr = initial_fill_active ?
        (initial_fill_count % c_line_size) :
        (prime_k_line_active ? prime_fill_count : (c_prime_fill + stream_fill_count));

    assign o_weight_ready = weight_fill_active;
    wire weight_accepted = i_weight_valid && weight_fill_active;
    assign o_valid = result_pending;
    assign o_acc_valid = result_pending;

    genvar lane_g;
    generate
        for (lane_g = 0; lane_g < G_C_PAR; lane_g = lane_g + 1) begin : gen_outs
            assign o_acc_data[(lane_g+1)*32-1 -: 32] = result_buffer[lane_g];
            assign o_data[(lane_g+1)*8-1 -: 8]       = quantized_result_buffer[lane_g];
        end
    endgenerate

    //-----------------------------------------------------------------------
    // 状态机
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            case (state)
                S_IDLE: if (start) state <= S_INITIAL_LINE_FILL;
                S_INITIAL_LINE_FILL: if (initial_line_fill_done) state <= S_PRIME_K_LINE;
                S_PRIME_K_LINE: begin
                    if (first_window_ready) begin
                        if (drain_input_active || vertical_advance_remaining > 0)
                            state <= S_STREAM_LINE_FILLING;
                        else if (weight_group_ready)
                            state <= S_CALC_AND_SLIDING_WINDOW;
                    end
                end
                S_CALC_AND_SLIDING_WINDOW: if (calculation_done) state <= S_STREAM_LINE_FILLING;
                S_STREAM_LINE_FILLING: if (stream_line_fill_done) state <= S_LINE_ROTATION;
                S_LINE_ROTATION: begin
                    if (line_rotation_done) begin
                        if (line_rotation_has_next)
                            state <= S_PRIME_K_LINE;
                        else
                            state <= S_IDLE;
                    end
                end
            endcase
        end
    end

    //-----------------------------------------------------------------------
    // 初始行填充
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            initial_fill_active   <= 1'b0;
            initial_line_fill_done <= 1'b0;
            initial_fill_count    <= 0;
        end else begin
            initial_line_fill_done <= 1'b0;
            if (start_line_fill) begin
                initial_fill_active <= 1'b1;
                initial_fill_count  <= 0;
            end else if (initial_fill_active) begin
                if (c_initial_fill == 0) begin
                    // k=1 时无需初始行填充，直接完成
                    initial_fill_active    <= 1'b0;
                    initial_line_fill_done <= 1'b1;
                end else if (activation_accepted) begin
                    if (initial_fill_count == c_initial_fill - 1) begin
                        initial_fill_active    <= 1'b0;
                        initial_line_fill_done <= 1'b1;
                    end else begin
                        initial_fill_count <= initial_fill_count + 1;
                    end
                end
            end
        end
    end

    //-----------------------------------------------------------------------
    // prime K 行填充
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            prime_k_line_active <= 1'b0;
            first_window_ready  <= 1'b0;
            prime_fill_count    <= 0;
        end else begin
            if (start_prime_k_line) begin
                prime_fill_count <= 0;
                if (row_valid[2]) begin
                    prime_k_line_active <= 1'b1;
                    first_window_ready  <= 1'b0;
                end else begin
                    prime_k_line_active <= 1'b0;
                    first_window_ready  <= 1'b1;
                end
            end else if (prime_k_line_active) begin
                if (activation_accepted) begin
                    if (prime_fill_count == c_prime_fill - 1) begin
                        prime_k_line_active <= 1'b0;
                        first_window_ready  <= 1'b1;
                    end else begin
                        prime_fill_count <= prime_fill_count + 1;
                    end
                end
            end
        end
    end

    //-----------------------------------------------------------------------
    // 行缓冲写入
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (line_buffer_we)
            line_buffer[line_buffer_wr_row][line_buffer_wr_addr] <= i_data;
    end

    //-----------------------------------------------------------------------
    // 权重加载（slice 按输入通道循环；每窗口每通道重新加载）
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            weight_fill_active  <= 1'b0;
            weight_group_ready  <= 1'b0;
            weight_fill_count   <= 0;
        end else begin
            if (start_weight_fill) begin
                weight_fill_active <= 1'b1;
                weight_group_ready <= 1'b0;
                weight_fill_count  <= 0;
            end else if (weight_fill_active) begin
                if (weight_accepted) begin
                    weight_buffer[weight_fill_count] <= $signed(i_weight_data);
                    if (weight_fill_count == c_weight_fill - 1) begin
                        weight_fill_active <= 1'b0;
                        weight_group_ready <= 1'b1;
                    end else begin
                        weight_fill_count <= weight_fill_count + 1;
                    end
                end
            end
        end
    end

    //-----------------------------------------------------------------------
    // 计算核心（conv: 全输入通道；dw: 对角，输入通道=输出通道）
    //-----------------------------------------------------------------------
    integer lane, v_row, v_col, v_output_channel, v_ci;
    integer v_input_col, v_line_addr, v_required_extra_columns, v_required_stream_bytes;
    reg signed [8:0]  v_act_lane [0:G_C_PAR-1];
    reg signed [16:0] v_product;
    reg signed [31:0] v_raw_acc, v_biased_acc, v_act_acc;
    reg signed [63:0] v_rq, v_rq_shifted;

    always @(posedge clk) begin
        if (!rst_n) begin
            calculation_active              <= 1'b0;
            calculation_done                <= 1'b0;
            calculation_waiting_for_weights <= 1'b0;
            calculation_kernel_count        <= 0;
            calculation_channel_count       <= 0;
            weight_refill_request           <= 1'b0;
            result_pending                  <= 1'b0;
            next_window_pending             <= 1'b0;
            window_column_count             <= 0;
            output_group_count              <= 0;
            for (lane = 0; lane < G_C_PAR; lane = lane + 1) begin
                accumulators[lane]            <= 32'sd0;
                result_buffer[lane]           <= 32'sd0;
                quantized_result_buffer[lane] <= 8'sd0;
            end
        end else begin
            weight_refill_request <= 1'b0;

            if (state != S_CALC_AND_SLIDING_WINDOW) begin
                calculation_active              <= 1'b0;
                calculation_done                <= 1'b0;
                calculation_waiting_for_weights <= 1'b0;
                calculation_kernel_count        <= 0;
                calculation_channel_count       <= 0;
                next_window_pending             <= 1'b0;
                window_column_count             <= 0;
                output_group_count              <= 0;
                result_pending                  <= 1'b0;
            end else if (!calculation_done) begin
                if (result_pending) begin
                    if (i_acc_ready) begin
                        result_pending <= 1'b0;
                        if (window_column_count < c_output_width - 1) begin
                            window_column_count <= window_column_count + 1;
                            next_window_pending <= 1'b1;
                        end else if (output_group_count < c_output_groups - 1) begin
                            output_group_count <= output_group_count + 1;
                            window_column_count <= 0;
                            calculation_active  <= 1'b0;
                            calculation_kernel_count <= 0;
                            calculation_channel_count <= 0;
                            calculation_waiting_for_weights <= 1'b1;
                            weight_refill_request <= 1'b1;
                            for (lane = 0; lane < G_C_PAR; lane = lane + 1)
                                accumulators[lane] <= 32'sd0;
                        end else begin
                            calculation_active <= 1'b0;
                            calculation_done   <= 1'b1;
                        end
                    end
                end else if (next_window_pending) begin
                    v_required_extra_columns = $signed(window_column_count) * $signed({1'b0, stride_reg}) - $signed({1'b0, pad_reg});
                    if (v_required_extra_columns < 0)
                        v_required_extra_columns = 0;
                    else if (v_required_extra_columns > in_w_reg - k_reg)
                        v_required_extra_columns = in_w_reg - k_reg;
                    v_required_stream_bytes = v_required_extra_columns * in_c_reg;

                    if (stream_bytes_available >= v_required_stream_bytes) begin
                        next_window_pending <= 1'b0;
                        calculation_kernel_count <= 0;
                        calculation_channel_count <= 0;
                        for (lane = 0; lane < G_C_PAR; lane = lane + 1)
                            accumulators[lane] <= 32'sd0;
                        if (in_c_reg == 1) begin
                            calculation_active <= 1'b1;
                        end else begin
                            calculation_active <= 1'b0;
                            calculation_waiting_for_weights <= 1'b1;
                            weight_refill_request <= 1'b1;
                        end
                    end
                end else if (calculation_waiting_for_weights) begin
                    if (weight_group_ready && !weight_fill_active && !weight_refill_request) begin
                        calculation_waiting_for_weights <= 1'b0;
                        calculation_active <= 1'b1;
                        calculation_kernel_count <= 0;
                    end
                end else if (!calculation_active) begin
                    calculation_active <= 1'b1;
                    calculation_kernel_count <= 0;
                    calculation_channel_count <= 0;
                    for (lane = 0; lane < G_C_PAR; lane = lane + 1)
                        accumulators[lane] <= 32'sd0;
                end else begin
                    // ---- MAC 主循环（conv/dw）----
                    v_row = calculation_kernel_count / k_reg;
                    v_col = calculation_kernel_count % k_reg;
                    v_input_col = $signed(window_column_count) * $signed({1'b0, stride_reg}) + v_col - $signed({1'b0, pad_reg});

                    // 每 lane 激活：conv 共享输入通道；dw 各自通道（oc = 输出通道）
                    for (lane = 0; lane < G_C_PAR; lane = lane + 1) begin
                        v_output_channel = output_group_count * G_C_PAR + lane;
                        v_ci = (type_reg == 4) ? v_output_channel : calculation_channel_count;
                        v_act_lane[lane] = 9'sd0;
                        if (v_output_channel < out_c_reg && row_valid[v_row] &&
                            v_input_col >= 0 && v_input_col < in_w_reg && v_ci < in_c_reg) begin
                            v_line_addr = v_input_col * in_c_reg + v_ci;
                            v_act_lane[lane] = $signed(line_buffer[row_map[v_row]][v_line_addr]);
                        end
                        v_product = v_act_lane[lane] * weight_buffer[lane * c_kernel_size + calculation_kernel_count];
                        accumulators[lane] <= accumulators[lane] + v_product;
                    end

                    // kernel 完成：kcount == k²-1（conv 切通道/dw 或 conv 末通道则窗口完成）
                    if (calculation_kernel_count == c_kernel_size - 1) begin
                        calculation_kernel_count <= 0;
                        if (type_reg == 4 ||
                            calculation_channel_count == in_c_reg - 1) begin
                            // ---- 窗口完成：requant 所有 lane ----
                            for (lane = 0; lane < G_C_PAR; lane = lane + 1) begin
                                v_output_channel = output_group_count * G_C_PAR + lane;
                                v_raw_acc = accumulators[lane] +
                                    v_act_lane[lane] * weight_buffer[lane * c_kernel_size + calculation_kernel_count];
                                result_buffer[lane] <= v_raw_acc;

                                // ---- activation（raw 域）----
                                v_biased_acc = v_raw_acc + bias_store[v_output_channel];
                                case (act_reg)
                                    2'd0: v_act_acc = v_biased_acc;
                                    2'd1: v_act_acc = v_biased_acc[31] ? 32'sd0 : v_biased_acc;
                                    default: begin
                                        v_act_acc = v_biased_acc[31] ? 32'sd0 : v_biased_acc;
                                        if (v_act_acc > rcl6_store[v_output_channel])
                                            v_act_acc = rcl6_store[v_output_channel];
                                    end
                                endcase

                                // ---- requant ----
                                v_rq = $signed(v_act_acc) * $signed({1'b0, rq_m_store[v_output_channel]});
                                if (rq_r_store[v_output_channel] > 0)
                                    v_rq = v_rq + (64'sd1 << (rq_r_store[v_output_channel] - 1));
                                v_rq_shifted = v_rq >>> rq_r_store[v_output_channel];

                                if (v_rq_shifted > 64'sd127)
                                    quantized_result_buffer[lane] <= 8'sd127;
                                else if (v_rq_shifted < -64'sd128)
                                    quantized_result_buffer[lane] <= -8'sd128;
                                else
                                    quantized_result_buffer[lane] <= v_rq_shifted[7:0];
                            end

                            calculation_active <= 1'b0;
                            result_pending    <= 1'b1;
                        end else begin
                            // 本通道 kernel 完成，切换输入通道（refill 下一 slice）
                            calculation_active <= 1'b0;
                            calculation_channel_count <= calculation_channel_count + 1;
                            calculation_waiting_for_weights <= 1'b1;
                            weight_refill_request <= 1'b1;
                        end
                    end else begin
                        calculation_kernel_count <= calculation_kernel_count + 1;
                    end
                end
            end
        end
    end

    //-----------------------------------------------------------------------
    // 流式行填充
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            stream_line_fill_active <= 1'b0;
            stream_line_fill_done   <= 1'b0;
            stream_fill_count       <= 0;
            stream_bytes_available  <= 0;
        end else begin
            if (start_stream_line_fill) begin
                stream_fill_count <= 0;
                if (c_stream_fill == 0) begin
                    // in_w == k：无流式填充需求，直接完成
                    stream_line_fill_active <= 1'b0;
                    stream_line_fill_done   <= 1'b1;
                    stream_bytes_available  <= 0;
                end else if (row_valid[2]) begin
                    stream_line_fill_active <= 1'b1;
                    stream_line_fill_done   <= 1'b0;
                    stream_bytes_available  <= 0;
                end else begin
                    stream_line_fill_active <= 1'b0;
                    stream_line_fill_done   <= 1'b1;
                    stream_bytes_available  <= c_stream_fill;
                end
            end else if (stream_line_fill_active) begin
                if (activation_accepted) begin
                    stream_bytes_available <= stream_bytes_available + 1;
                    if (stream_fill_count == c_stream_fill - 1) begin
                        stream_line_fill_active <= 1'b0;
                        stream_line_fill_done   <= 1'b1;
                    end else begin
                        stream_fill_count <= stream_fill_count + 1;
                    end
                end
            end
        end
    end

    //-----------------------------------------------------------------------
    // 行旋转
    //-----------------------------------------------------------------------
    integer r;
    integer v_new_input_row;
    reg v_do_rotation, v_drain_rotation;

    always @(posedge clk) begin
        if (!rst_n) begin
            spare_row <= 3;
            output_row_count <= 0;
            logical_top_row  <= -1;
            vertical_advance_remaining <= 0;
            drain_input_active <= 1'b0;
            line_rotation_done <= 1'b0;
            line_rotation_has_next <= 1'b0;
            for (r = 0; r < 3; r = r + 1) begin
                row_map[r] <= r;
                row_valid[r] <= (r >= pad_reg && r < in_h_reg + pad_reg) ? 1'b1 : 1'b0;
            end
        end else if (state == S_IDLE) begin
            spare_row <= 3;
            output_row_count <= 0;
            logical_top_row  <= -1;
            vertical_advance_remaining <= 0;
            drain_input_active <= 1'b0;
            line_rotation_done <= 1'b0;
            line_rotation_has_next <= 1'b0;
            for (r = 0; r < 3; r = r + 1) begin
                row_map[r] <= r;
                row_valid[r] <= (r >= pad_reg && r < in_h_reg + pad_reg) ? 1'b1 : 1'b0;
            end
        end else begin
            line_rotation_done <= 1'b0;
            if (start_line_rotation) begin
                v_new_input_row = logical_top_row + k_reg;
                v_do_rotation   = 1'b0;
                v_drain_rotation = 1'b0;
                if (drain_input_active) begin
                    if (v_new_input_row >= 0 && v_new_input_row < in_h_reg) begin
                        v_do_rotation   = 1'b1;
                        v_drain_rotation = 1'b1;
                    end else begin
                        drain_input_active <= 1'b0;
                        line_rotation_has_next <= 1'b0;
                    end
                end else if (output_row_count < c_output_height - 1) begin
                    v_do_rotation = 1'b1;
                end else if (v_new_input_row >= 0 && v_new_input_row < in_h_reg) begin
                    drain_input_active <= 1'b1;
                    v_do_rotation   = 1'b1;
                    v_drain_rotation = 1'b1;
                end else begin
                    line_rotation_has_next <= 1'b0;
                end

                if (v_do_rotation) begin
                    for (r = 0; r < 2; r = r + 1) begin
                        row_map[r]   <= row_map[r+1];
                        row_valid[r] <= row_valid[r+1];
                    end
                    row_map[2]   <= spare_row;
                    spare_row    <= row_map[0];
                    row_valid[2] <= (v_new_input_row >= 0 && v_new_input_row < in_h_reg) ? 1'b1 : 1'b0;
                    logical_top_row <= logical_top_row + 1;
                    line_rotation_has_next <= 1'b1;

                    if (v_drain_rotation) begin
                        vertical_advance_remaining <= 0;
                    end else if (vertical_advance_remaining == 0) begin
                        if (stride_reg == 1) begin
                            output_row_count <= output_row_count + 1;
                            vertical_advance_remaining <= 0;
                        end else begin
                            vertical_advance_remaining <= stride_reg - 1;
                        end
                    end else if (vertical_advance_remaining == 1) begin
                        vertical_advance_remaining <= 0;
                        output_row_count <= output_row_count + 1;
                    end else begin
                        vertical_advance_remaining <= vertical_advance_remaining - 1;
                    end
                end
                line_rotation_done <= 1'b1;
            end
        end
    end

    //-----------------------------------------------------------------------
    // o_done
    //-----------------------------------------------------------------------
    reg o_done_r;
    assign o_done = o_done_r;
    always @(posedge clk) begin
        if (!rst_n)
            o_done_r <= 1'b0;
        else begin
            o_done_r <= 1'b0;
            if (state == S_LINE_ROTATION && line_rotation_done && !line_rotation_has_next)
                o_done_r <= 1'b1;
        end
    end

endmodule
