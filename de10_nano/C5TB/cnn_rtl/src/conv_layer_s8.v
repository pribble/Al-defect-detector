//=============================================================================
// conv_layer_s8.v — int8 卷积数据通路（cnn_top 复现阶段 1）
//------------------------------------------------------------------------------
// 架构同构自上游开源项目 conv_layer.vhd（VHDL，MIT）：
//   https://github.com/AlbianSalihu/de1-soc-fpga-accelerated-video-pipeline
// 数值语义按 cnn_top_spec.md §7 对齐（阶段 1 改动）：
//   1) 输入/输出 signed int8（上游为 uint8）
//   2) requant 从"relu 后无符号乘+饱和255"改为"activation 后 signed 乘+算术右移+饱和[-128,127]"
//   3) activation 由参数 G_ACT 选择：0 none / 1 relu / 2 relu6（G_RAW_CLAMP6 为 raw 域上限）
//   bias 加在 raw 域（标准 conv 语义），bias_int/mult/shift 由软件按 STAGE1_plan.md §2 预生成
// 详见 STAGE1_plan.md。
//=============================================================================

module conv_layer_s8 #(
    parameter G_C_IN      = 3,    // 输入通道数
    parameter G_C_OUT     = 4,    // 输出通道数
    parameter G_W_IN      = 4,    // 输入宽
    parameter G_H_IN      = 4,    // 输入高
    parameter G_C_PAR     = 4,    // 并行输出通道 lane 数（G_C_OUT 需为 G_C_PAR 整数倍）
    parameter G_KERNEL    = 3,    // 核边长（3×3）
    parameter G_PADDING   = 1,    // 对称 zero padding
    parameter G_STRIDE    = 1,    // 步长
    parameter G_ACT       = 1,    // 0 none / 1 relu / 2 relu6（阶段 1 综合期固定）
    parameter G_RAW_CLAMP6 = 0    // relu6 的 raw 域 clamp 上限（阶段 1 综合期固定）
)(
    input  wire                clk,
    input  wire                rst_n,

    // 激活流（signed int8，行→列→通道顺序）
    input  wire                i_valid,
    output wire                i_ready,
    input  wire [7:0]          i_data,

    // 权重流（signed int8，每 slice = G_C_PAR × KERNEL²）
    input  wire                i_weight_valid,
    output wire                o_weight_ready,
    input  wire [7:0]          i_weight_data,

    // per-channel 量化参数：01=bias_int(int32) / 10=mult(uint32) / 11=shift(uint8)
    input  wire                cfg_we,
    input  wire [1:0]          cfg_sel,
    input  wire [19:0]         cfg_addr,
    input  wire [31:0]         cfg_wdata,

    // 量化输出（signed int8，G_C_PAR 通道打包）
    output wire                o_valid,
    output wire [G_C_PAR*8-1:0] o_data,
    output wire                o_done,

    // 原始 int32 累加输出（调试/对拍）
    input  wire                i_acc_ready,
    output wire                o_acc_valid,
    output wire [G_C_PAR*32-1:0] o_acc_data
);

    //-----------------------------------------------------------------------
    // 派生常量（对应上游 VHDL 的 constant 段）
    //-----------------------------------------------------------------------
    localparam C_LINE_SIZE          = G_W_IN * G_C_IN;
    localparam C_INITIAL_FILL_ROWS  = G_KERNEL - G_PADDING - 1;
    localparam C_INITIAL_FILL_SIZE  = C_INITIAL_FILL_ROWS * C_LINE_SIZE;
    localparam C_PRIME_FILL_SIZE    = G_KERNEL * G_C_IN;
    localparam C_KERNEL_SIZE        = G_KERNEL * G_KERNEL;
    localparam C_WEIGHT_FILL_SIZE   = G_C_PAR * C_KERNEL_SIZE;
    localparam C_PADDED_WIDTH       = G_W_IN + 2 * G_PADDING;
    localparam C_PADDED_HEIGHT      = G_H_IN + 2 * G_PADDING;
    localparam C_OUTPUT_WIDTH       = (C_PADDED_WIDTH  - G_KERNEL) / G_STRIDE + 1;
    localparam C_OUTPUT_HEIGHT      = (C_PADDED_HEIGHT - G_KERNEL) / G_STRIDE + 1;
    localparam C_OUTPUT_GROUPS      = G_C_OUT / G_C_PAR;
    localparam C_STREAM_FILL_SIZE   = (G_W_IN - G_KERNEL) * G_C_IN;

    //-----------------------------------------------------------------------
    // 状态机（对应 VHDL conv_states）
    //-----------------------------------------------------------------------
    localparam [2:0] S_IDLE                   = 3'd0,
                     S_INITIAL_LINE_FILL      = 3'd1,
                     S_PRIME_K_LINE           = 3'd2,
                     S_CALC_AND_SLIDING_WINDOW= 3'd3,
                     S_STREAM_LINE_FILLING    = 3'd4,
                     S_LINE_ROTATION          = 3'd5;

    reg [2:0] state = S_IDLE;

    //-----------------------------------------------------------------------
    // 存储体
    //-----------------------------------------------------------------------
    reg signed [7:0] weight_buffer [0:C_WEIGHT_FILL_SIZE-1];   // 当前 slice 权重
    reg [7:0] line_buffer [0:G_KERNEL][0:C_LINE_SIZE-1];       // 行缓冲（G_KERNEL+1 行）
    reg signed [31:0] accumulators [0:G_C_PAR-1];              // 累加器
    reg signed [31:0] result_buffer [0:G_C_PAR-1];             // 原始累加结果（调试输出）
    reg [7:0] quantized_result_buffer [0:G_C_PAR-1];           // 量化输出
    reg signed [31:0] bias_store [0:G_C_OUT-1];                // bias_int（raw 域）
    reg [31:0] rq_m_store [0:G_C_OUT-1];                       // requant mult
    reg [7:0]  rq_r_store [0:G_C_OUT-1];                       // requant shift
    reg [7:0] row_map [0:G_KERNEL-1];                          // 逻辑行 → 物理行映射
    reg row_valid [0:G_KERNEL-1];                              // 行有效标志
    reg [7:0] spare_row;                                       // 备用物理行

    //-----------------------------------------------------------------------
    // 控制/计数信号
    //-----------------------------------------------------------------------
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
    integer logical_top_row;    // 有符号（可为负：-G_PADDING）

    //-----------------------------------------------------------------------
    // 组合逻辑（对应 VHDL architecture 体前的 signal 赋值）
    //-----------------------------------------------------------------------
    wire start_line_fill          = (state == S_IDLE);
    wire start_weight_fill        = (state == S_IDLE) || weight_refill_request ||
                                    (line_rotation_done && line_rotation_has_next &&
                                     !drain_input_active && vertical_advance_remaining == 0 &&
                                     (G_C_IN > 1 || C_OUTPUT_GROUPS > 1));
    wire start_prime_k_line       = initial_line_fill_done ||
                                    (line_rotation_done && line_rotation_has_next);
    wire start_line_rotation      = (state == S_STREAM_LINE_FILLING) && stream_line_fill_done;
    wire start_stream_line_fill   = (state == S_PRIME_K_LINE) && first_window_ready &&
                                    (drain_input_active || vertical_advance_remaining > 0 ||
                                     weight_group_ready);

    assign i_ready = initial_fill_active || prime_k_line_active || stream_line_fill_active;
    wire activation_accepted = i_valid && i_ready;
    wire line_buffer_we = activation_accepted;

    wire [7:0] line_buffer_wr_row = initial_fill_active ?
        (G_PADDING + initial_fill_count / C_LINE_SIZE) :
        row_map[G_KERNEL-1];
    wire [31:0] line_buffer_wr_addr = initial_fill_active ?
        (initial_fill_count % C_LINE_SIZE) :
        (prime_k_line_active ? prime_fill_count : (C_PRIME_FILL_SIZE + stream_fill_count));

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
    // 状态机（对应 VHDL controller_process）
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            case (state)
                S_IDLE: state <= S_INITIAL_LINE_FILL;
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
                        else begin
                            state <= S_IDLE;
                        end
                    end
                end
            endcase
        end
    end

    //-----------------------------------------------------------------------
    // 初始行填充（对应 VHDL initial_line_fill_process）
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
                if (activation_accepted) begin
                    if (initial_fill_count == C_INITIAL_FILL_SIZE - 1) begin
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
    // prime K 行填充（对应 VHDL prime_k_line_process）
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            prime_k_line_active <= 1'b0;
            first_window_ready  <= 1'b0;
            prime_fill_count    <= 0;
        end else begin
            if (start_prime_k_line) begin
                prime_fill_count <= 0;
                if (row_valid[G_KERNEL-1]) begin
                    prime_k_line_active <= 1'b1;
                    first_window_ready  <= 1'b0;
                end else begin
                    prime_k_line_active <= 1'b0;
                    first_window_ready  <= 1'b1;
                end
            end else if (prime_k_line_active) begin
                if (activation_accepted) begin
                    if (prime_fill_count == C_PRIME_FILL_SIZE - 1) begin
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
    // 行缓冲写入（对应 VHDL line_buffer_write_process，无复位）
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (line_buffer_we)
            line_buffer[line_buffer_wr_row][line_buffer_wr_addr] <= i_data;
    end

    //-----------------------------------------------------------------------
    // 权重加载（对应 VHDL weight_filling_process）
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
                    if (weight_fill_count == C_WEIGHT_FILL_SIZE - 1) begin
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
    // 量化参数加载（对应 VHDL parameter_loading_process）
    //-----------------------------------------------------------------------
    always @(posedge clk) begin
        if (cfg_we) begin
            if (cfg_addr < G_C_OUT) begin
                case (cfg_sel)
                    2'b01: bias_store[cfg_addr]   <= $signed(cfg_wdata);
                    2'b10: rq_m_store[cfg_addr]   <= cfg_wdata;
                    2'b11: rq_r_store[cfg_addr]   <= cfg_wdata[7:0];
                endcase
            end
        end
    end

    //-----------------------------------------------------------------------
    // 计算核心（对应 VHDL calculation_process）
    //-----------------------------------------------------------------------
    integer lane, v_row, v_col, v_output_channel;
    integer v_input_col, v_line_addr, v_required_extra_columns, v_required_stream_bytes;
    reg signed [8:0]  v_activation;
    reg signed [16:0] v_product;
    reg signed [31:0] v_raw_acc, v_biased_acc, v_act_acc;
    reg signed [63:0] v_rq, v_rq_shifted;

    always @(posedge clk) begin
        if (!rst_n) begin
            calculation_active             <= 1'b0;
            calculation_done               <= 1'b0;
            calculation_waiting_for_weights <= 1'b0;
            calculation_kernel_count       <= 0;
            calculation_channel_count      <= 0;
            weight_refill_request          <= 1'b0;
            result_pending                 <= 1'b0;
            next_window_pending            <= 1'b0;
            window_column_count            <= 0;
            output_group_count             <= 0;
            for (lane = 0; lane < G_C_PAR; lane = lane + 1) begin
                accumulators[lane]              <= 32'sd0;
                result_buffer[lane]             <= 32'sd0;
                quantized_result_buffer[lane]   <= 8'sd0;
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
                        if (window_column_count < C_OUTPUT_WIDTH - 1) begin
                            window_column_count <= window_column_count + 1;
                            next_window_pending <= 1'b1;
                        end else if (output_group_count < C_OUTPUT_GROUPS - 1) begin
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
                    v_required_extra_columns = $signed(window_column_count) * G_STRIDE - G_PADDING;
                    if (v_required_extra_columns < 0)
                        v_required_extra_columns = 0;
                    else if (v_required_extra_columns > G_W_IN - G_KERNEL)
                        v_required_extra_columns = G_W_IN - G_KERNEL;
                    v_required_stream_bytes = v_required_extra_columns * G_C_IN;

                    if (stream_bytes_available >= v_required_stream_bytes) begin
                        next_window_pending <= 1'b0;
                        calculation_kernel_count <= 0;
                        calculation_channel_count <= 0;
                        for (lane = 0; lane < G_C_PAR; lane = lane + 1)
                            accumulators[lane] <= 32'sd0;
                        if (G_C_IN == 1) begin
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
                    // ---- MAC 主循环（组合临时计算 + 非阻塞状态更新）----
                    v_row = calculation_kernel_count / G_KERNEL;
                    v_col = calculation_kernel_count % G_KERNEL;
                    v_input_col = $signed(window_column_count) * G_STRIDE + v_col - G_PADDING;

                    v_activation = 9'sd0;
                    if (row_valid[v_row] && v_input_col >= 0 && v_input_col < G_W_IN) begin
                        v_line_addr = v_input_col * G_C_IN + calculation_channel_count;
                        // 阶段1：signed int8（上游为无符号扩展）
                        v_activation = $signed(line_buffer[row_map[v_row]][v_line_addr]);
                    end

                    for (lane = 0; lane < G_C_PAR; lane = lane + 1) begin
                        v_product = v_activation * weight_buffer[lane * C_KERNEL_SIZE + calculation_kernel_count];
                        accumulators[lane] <= accumulators[lane] + v_product;
                        if (calculation_kernel_count == C_KERNEL_SIZE - 1 &&
                            calculation_channel_count == G_C_IN - 1) begin
                            v_raw_acc = accumulators[lane] + v_product;
                            result_buffer[lane] <= v_raw_acc;

                            v_output_channel = output_group_count * G_C_PAR + lane;

                            // ---- activation（阶段1：G_ACT 可配）----
                            v_biased_acc = v_raw_acc + bias_store[v_output_channel];
                            case (G_ACT)
                                2'd0: v_act_acc = v_biased_acc;                       // none
                                2'd1: v_act_acc = v_biased_acc[31] ? 32'sd0 : v_biased_acc; // relu
                                default: begin                                       // relu6
                                    v_act_acc = v_biased_acc[31] ? 32'sd0 : v_biased_acc;
                                    if (v_act_acc > G_RAW_CLAMP6)
                                        v_act_acc = G_RAW_CLAMP6;
                                end
                            endcase

                            // ---- signed requant：act·mult，round-half-up，算术右移，饱和 int8 ----
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
                    end

                    if (calculation_kernel_count == C_KERNEL_SIZE - 1) begin
                        calculation_kernel_count <= 0;
                        if (calculation_channel_count == G_C_IN - 1) begin
                            calculation_active <= 1'b0;
                            result_pending    <= 1'b1;
                        end else begin
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
    // 流式行填充（对应 VHDL stream_line_fill_process）
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
                if (row_valid[G_KERNEL-1]) begin
                    stream_line_fill_active <= 1'b1;
                    stream_line_fill_done   <= 1'b0;
                    stream_bytes_available  <= 0;
                end else begin
                    stream_line_fill_active <= 1'b0;
                    stream_line_fill_done   <= 1'b1;
                    stream_bytes_available  <= C_STREAM_FILL_SIZE;
                end
            end else if (stream_line_fill_active) begin
                if (activation_accepted) begin
                    stream_bytes_available <= stream_bytes_available + 1;
                    if (stream_fill_count == C_STREAM_FILL_SIZE - 1) begin
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
    // 行旋转（对应 VHDL line_rotation_process）
    //-----------------------------------------------------------------------
    integer r;
    integer v_new_input_row;
    reg v_do_rotation, v_drain_rotation;

    always @(posedge clk) begin
        if (!rst_n) begin
            spare_row <= G_KERNEL;
            output_row_count <= 0;
            logical_top_row  <= -G_PADDING;
            vertical_advance_remaining <= 0;
            drain_input_active <= 1'b0;
            line_rotation_done <= 1'b0;
            line_rotation_has_next <= 1'b0;
            for (r = 0; r < G_KERNEL; r = r + 1) begin
                row_map[r] <= r;
                row_valid[r] <= (r >= G_PADDING && r < G_H_IN + G_PADDING) ? 1'b1 : 1'b0;
            end
        end else if (state == S_IDLE) begin
            spare_row <= G_KERNEL;
            output_row_count <= 0;
            logical_top_row  <= -G_PADDING;
            vertical_advance_remaining <= 0;
            drain_input_active <= 1'b0;
            line_rotation_done <= 1'b0;
            line_rotation_has_next <= 1'b0;
            for (r = 0; r < G_KERNEL; r = r + 1) begin
                row_map[r] <= r;
                row_valid[r] <= (r >= G_PADDING && r < G_H_IN + G_PADDING) ? 1'b1 : 1'b0;
            end
        end else begin
            line_rotation_done <= 1'b0;
            if (start_line_rotation) begin
                v_new_input_row = logical_top_row + G_KERNEL;
                v_do_rotation   = 1'b0;
                v_drain_rotation = 1'b0;
                if (drain_input_active) begin
                    if (v_new_input_row >= 0 && v_new_input_row < G_H_IN) begin
                        v_do_rotation   = 1'b1;
                        v_drain_rotation = 1'b1;
                    end else begin
                        drain_input_active <= 1'b0;
                        line_rotation_has_next <= 1'b0;
                    end
                end else if (output_row_count < C_OUTPUT_HEIGHT - 1) begin
                    v_do_rotation = 1'b1;
                end else if (v_new_input_row >= 0 && v_new_input_row < G_H_IN) begin
                    drain_input_active <= 1'b1;
                    v_do_rotation   = 1'b1;
                    v_drain_rotation = 1'b1;
                end else begin
                    line_rotation_has_next <= 1'b0;
                end

                if (v_do_rotation) begin
                    for (r = 0; r < G_KERNEL - 1; r = r + 1) begin
                        row_map[r]   <= row_map[r+1];
                        row_valid[r] <= row_valid[r+1];
                    end
                    row_map[G_KERNEL-1]   <= spare_row;
                    spare_row             <= row_map[0];
                    row_valid[G_KERNEL-1] <= (v_new_input_row >= 0 && v_new_input_row < G_H_IN) ? 1'b1 : 1'b0;
                    logical_top_row       <= logical_top_row + 1;
                    line_rotation_has_next <= 1'b1;

                    if (v_drain_rotation) begin
                        vertical_advance_remaining <= 0;
                    end else if (vertical_advance_remaining == 0) begin
                        if (G_STRIDE == 1) begin
                            output_row_count <= output_row_count + 1;
                            vertical_advance_remaining <= 0;
                        end else begin
                            vertical_advance_remaining <= G_STRIDE - 1;
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
    // o_done（对应 VHDL controller 中 S_LINE_ROTATION 结束脉冲）
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
