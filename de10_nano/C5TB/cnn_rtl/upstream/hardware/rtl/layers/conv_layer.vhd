library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity conv_layer is
    generic (
        G_C_IN : positive;
        G_C_OUT : positive;
        G_W_IN : positive;
        G_H_IN : positive;
        G_C_PAR : positive;
        G_KERNEL : positive;
        G_PADDING : natural;
        G_STRIDE : positive
    );
    port (
        clk : in std_logic;
        rst_n : in std_logic;

        i_valid : in std_logic;
        i_ready : out std_logic;
        i_data : in std_logic_vector(7 downto 0);

        i_weight_valid : in std_logic; 
        o_weight_ready : out std_logic;
        i_weight_data : in std_logic_vector(7 downto 0);

        cfg_we : in std_logic;
        cfg_sel : in std_logic_vector(1 downto 0);
        cfg_addr : in std_logic_vector(19 downto 0);
        cfg_wdata : in std_logic_vector(31 downto 0);

        o_valid : out std_logic;
        o_data : out std_logic_vector(G_C_PAR * 8 - 1 downto 0);
        o_done : out std_logic;

        i_acc_ready : in std_logic;
        o_acc_valid : out std_logic;
        o_acc_data : out std_logic_vector(G_C_PAR * 32 - 1 downto 0)    
    );
end entity conv_layer;

architecture rtl of conv_layer is

    type conv_states is (
        S_IDLE,
        S_INITIAL_LINE_FILL,
        S_PRIME_K_LINE,
        S_CALC_AND_SLIDING_WINDOW,
        S_STREAM_LINE_FILLING,
        S_LINE_ROTATION
    );

    signal state : conv_states := S_IDLE;

    constant C_LINE_SIZE : positive := G_W_IN * G_C_IN;
    constant C_INITIAL_FILL_ROWS : positive := G_KERNEL - G_PADDING - 1;
    constant C_INITIAL_FILL_SIZE : positive := C_INITIAL_FILL_ROWS * C_LINE_SIZE;
    constant C_PRIME_FILL_SIZE : positive := G_KERNEL * G_C_IN;
    constant C_KERNEL_SIZE : positive := G_KERNEL * G_KERNEL;
    constant C_WEIGHT_FILL_SIZE : positive := G_C_PAR * C_KERNEL_SIZE;
    constant C_PADDED_WIDTH : positive := G_W_IN + 2 * G_PADDING;
    constant C_PADDED_HEIGHT : positive := G_H_IN + 2 * G_PADDING;

    constant C_OUTPUT_WIDTH : positive := ((C_PADDED_WIDTH - G_KERNEL) / G_STRIDE) + 1;
    constant C_OUTPUT_HEIGHT : positive := ((C_PADDED_HEIGHT - G_KERNEL) / G_STRIDE) + 1;
    constant C_OUTPUT_GROUPS : positive := G_C_OUT / G_C_PAR;
    constant C_STREAM_FILL_SIZE : positive := (G_W_IN - G_KERNEL) * G_C_IN;
    
    type t_weight_buffer is array(
        0 to C_WEIGHT_FILL_SIZE - 1
    ) of signed(7 downto 0);

    type t_line_buffer is array (
        0 to G_KERNEL,
        0 to C_LINE_SIZE - 1
    ) of std_logic_vector(7 downto 0);

    type t_accumulator_array is array (
        0 to G_C_PAR - 1
    ) of signed(31 downto 0);

    type t_bias_store is array(
        0 to G_C_OUT - 1
    ) of signed(31 downto 0);

    type t_rq_m_store is array(
        0 to G_C_OUT - 1  
    ) of unsigned(31 downto 0);

    type t_rq_r_store is array(
        0 to G_C_OUT - 1
    ) of unsigned(7 downto 0);

    type t_quantized_array is array(
        0 to G_C_PAR - 1
    ) of std_logic_vector(7 downto 0);

    type t_row_map is array (
        0 to G_KERNEL - 1
    ) of natural range 0 to G_KERNEL;

    type t_row_valid is array(
        0 to G_KERNEL - 1
    ) of std_logic;

    signal weight_buffer : t_weight_buffer;
    signal accumulators : t_accumulator_array;
    signal weight_fill_active : std_logic := '0';
    signal weight_group_ready : std_logic := '0';
    signal weight_accepted : std_logic;
    signal weight_fill_count : natural range 0 to C_WEIGHT_FILL_SIZE - 1 := 0; 

    signal line_buffer : t_line_buffer;

    signal start_line_fill : std_logic;
    signal start_weight_fill : std_logic;
    signal start_prime_k_line : std_logic;

    signal activation_accepted : std_logic;

    signal initial_fill_active : std_logic := '0';
    signal initial_line_fill_done : std_logic := '0';

    signal initial_fill_count : natural range 0 to C_INITIAL_FILL_SIZE - 1 := 0;

    signal prime_k_line_active : std_logic := '0';
    signal first_window_ready : std_logic := '0';

    signal prime_fill_count : natural range 0 to C_PRIME_FILL_SIZE - 1 := 0;

    signal line_buffer_we : std_logic;

    signal line_buffer_wr_row : natural range 0 to G_KERNEL := 0;

    signal line_buffer_wr_addr : natural range 0 to C_LINE_SIZE - 1 := 0;

    signal calculation_active : std_logic := '0';
    signal calculation_done : std_logic := '0';

    signal calculation_waiting_for_weights : std_logic := '0';
    signal calculation_kernel_count : natural range 0 to C_KERNEL_SIZE - 1 := 0; 
    signal calculation_channel_count : natural range 0 to G_C_IN - 1 := 0;
    signal weight_refill_request : std_logic := '0';
    signal start_stream_line_fill : std_logic;
    signal stream_line_fill_active : std_logic := '0';
    signal stream_line_fill_done : std_logic := '0';

    signal stream_fill_count : natural range 0 to C_STREAM_FILL_SIZE - 1 := 0;
    signal stream_bytes_available : natural range 0 to C_STREAM_FILL_SIZE := 0;
    signal window_column_count : natural range 0 to C_OUTPUT_WIDTH - 1 := 0; 
    signal next_window_pending : std_logic := '0';
    signal row_map : t_row_map;
    signal spare_row : natural range 0 to G_KERNEL := G_KERNEL;
    signal output_row_count : natural range 0 to C_OUTPUT_HEIGHT - 1 := 0;
    signal start_line_rotation : std_logic;
    signal line_rotation_done : std_logic := '0';
    signal line_rotation_has_next : std_logic := '0';
    signal result_buffer : t_accumulator_array;
    signal result_pending : std_logic := '0';
    signal bias_store : t_bias_store := (others => (others => '0'));
    signal rq_m_store : t_rq_m_store := (others => to_unsigned(1,32));
    signal rq_r_store : t_rq_r_store := (others => (others => '0'));
    signal quantized_result_buffer : t_quantized_array;

    signal row_valid : t_row_valid := (others => '0');
    signal output_group_count : natural range 0 to C_OUTPUT_GROUPS - 1 := 0;
    signal logical_top_row : integer range -integer(G_PADDING) to G_H_IN + integer(G_PADDING) := -integer(G_PADDING);
    signal vertical_advance_remaining : natural range 0 to G_STRIDE - 1 := 0;
    signal drain_input_active : std_logic := '0';

begin
    start_line_fill   <= '1' when state = S_IDLE else '0';
    start_weight_fill <= '1' when state = S_IDLE or weight_refill_request = '1' or (line_rotation_done = '1' and line_rotation_has_next = '1' and drain_input_active = '0' and vertical_advance_remaining = 0 and (G_C_IN > 1 or C_OUTPUT_GROUPS > 1)) else '0';

    start_prime_k_line <= initial_line_fill_done or (line_rotation_done and line_rotation_has_next);

    start_line_rotation <= '1' when state = S_STREAM_LINE_FILLING and stream_line_fill_done = '1' else '0';

    start_stream_line_fill <= '1' when state = S_PRIME_K_LINE and first_window_ready = '1' and (drain_input_active = '1' or vertical_advance_remaining > 0 or weight_group_ready = '1') else '0';

    i_ready <= initial_fill_active or prime_k_line_active or stream_line_fill_active;

    activation_accepted <= i_valid and i_ready;

    line_buffer_we <= activation_accepted;

    line_buffer_wr_row <= G_PADDING + initial_fill_count / C_LINE_SIZE when initial_fill_active = '1' else row_map(G_KERNEL-1);

    line_buffer_wr_addr <= initial_fill_count mod C_LINE_SIZE when initial_fill_active = '1' else prime_fill_count when prime_k_line_active = '1' else C_PRIME_FILL_SIZE + stream_fill_count;

    o_weight_ready <= weight_fill_active;
    weight_accepted <= i_weight_valid and weight_fill_active;
    o_acc_valid <= result_pending;
    o_valid <= result_pending;

    accumulator_output_generate : 
    for lane in 0 to G_C_PAR - 1 generate
        o_acc_data((lane + 1) * 32 - 1 downto lane * 32) <= std_logic_vector(result_buffer(lane));
    end generate;

    quantized_output_generate :
    for lane in 0 to G_C_PAR - 1 generate
        o_data((lane + 1) * 8 - 1 downto lane * 8) <= quantized_result_buffer(lane);
    end generate;

    controller_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                state <= S_IDLE;
                o_done <= '0';

            else
                o_done <= '0';
                case state is

                    when S_IDLE =>
                        state <= S_INITIAL_LINE_FILL;

                    when S_INITIAL_LINE_FILL =>
                        if initial_line_fill_done = '1' then
                            state <= S_PRIME_K_LINE;
                        end if;

                    when S_PRIME_K_LINE =>
                        if first_window_ready = '1' then
                            if drain_input_active = '1' or vertical_advance_remaining > 0 then
                                state <= S_STREAM_LINE_FILLING;
                            elsif weight_group_ready = '1' then
                                state <= S_CALC_AND_SLIDING_WINDOW;
                            end if;
                        end if;

                    when S_CALC_AND_SLIDING_WINDOW =>
                        if calculation_done = '1' then
                            state <= S_STREAM_LINE_FILLING;    
                        end if;

                    when S_STREAM_LINE_FILLING =>
                        if stream_line_fill_done = '1' then
                            state <= S_LINE_ROTATION;
                        end if;

                    when S_LINE_ROTATION =>
                        if line_rotation_done = '1' then
                            if line_rotation_has_next = '1' then 
                                state <= S_PRIME_K_LINE;
                            else
                                state <= S_IDLE;
                                o_done <= '1';
                            end if;
                        end if;
                end case;
            end if;
        end if;
    end process;


    initial_line_fill_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                initial_fill_active <= '0';
                initial_line_fill_done <= '0';
                initial_fill_count <= 0;

            else
                initial_line_fill_done <= '0';
                if start_line_fill = '1' then
                    initial_fill_active <= '1';
                    initial_fill_count <= 0;

                elsif initial_fill_active = '1' then
                    if activation_accepted = '1' then
                        if initial_fill_count = C_INITIAL_FILL_SIZE - 1 then
                            initial_fill_active <= '0';
                            initial_line_fill_done <= '1';

                        else
                            initial_fill_count <= initial_fill_count + 1;
                        end if;

                    end if;
                end if;
            end if;
        end if;
    end process;


    prime_k_line_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                prime_k_line_active <= '0';
                first_window_ready <= '0';
                prime_fill_count <= 0;

            else
                if start_prime_k_line = '1' then
                    prime_fill_count <= 0;
                    if row_valid(G_KERNEL - 1) = '1' then
                        prime_k_line_active <= '1';
                        first_window_ready <= '0';
                    else
                        prime_k_line_active <= '0';
                        first_window_ready <= '1';
                    end if;

                elsif prime_k_line_active = '1' then
                    if activation_accepted = '1' then

                        if prime_fill_count = C_PRIME_FILL_SIZE - 1 then
                            prime_k_line_active <= '0';
                            first_window_ready <= '1';

                        else
                            prime_fill_count <= prime_fill_count + 1;
                        end if;

                    end if;
                end if;
            end if;
        end if;
    end process;


    line_buffer_write_process : process(clk)
    begin
        if rising_edge(clk) then
            if line_buffer_we = '1' then
                line_buffer(line_buffer_wr_row, line_buffer_wr_addr) <= i_data;
            end if;
        end if;
    end process;


    weight_filling_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                weight_fill_active <= '0';
                weight_group_ready <= '0';
                weight_fill_count <= 0;

            else
                if start_weight_fill = '1' then
                    weight_fill_active <= '1';
                    weight_group_ready <= '0';
                    weight_fill_count <= 0;

                elsif weight_fill_active = '1' then
                    if weight_accepted = '1' then
                        weight_buffer(weight_fill_count) <= signed(i_weight_data);

                        if weight_fill_count = C_WEIGHT_FILL_SIZE - 1 then
                            weight_fill_active <= '0';
                            weight_group_ready <= '1'; 
                        else
                            weight_fill_count <= weight_fill_count + 1;
                        end if;
                    end if;
                end if;
            end if;
        end if;
    end process;

    parameter_loading_process : process(clk)
        variable v_addr : natural;
    begin
        if rising_edge(clk) then
            if cfg_we = '1' then
                v_addr := to_integer(unsigned(cfg_addr));
                if v_addr < G_C_OUT then
                    case cfg_sel is
                        when "01" =>
                            bias_store(v_addr) <= signed(cfg_wdata);
                        when "10" =>
                            rq_m_store(v_addr) <= unsigned(cfg_wdata);
                        when "11" =>
                            rq_r_store(v_addr) <= unsigned(cfg_wdata(7 downto 0));
                        when others =>
                            null;
                    end case;
                end if;
            end if;
        end if; 
    end process;

    calculation_process : process(clk)
        variable v_row : natural;
        variable v_col : natural;
        variable v_line_addr : natural;
        variable v_weight_addr : natural;
        
        variable v_activation : signed(8 downto 0);
        variable v_product : signed(16 downto 0);
        variable v_output_channel : natural range 0 to G_C_OUT - 1;
        variable v_raw_acc : signed(31 downto 0);
        variable v_biased_acc : signed(31 downto 0);
        variable v_acc_pos : signed(31 downto 0);
        variable v_requant_product : unsigned(63 downto 0);
        variable v_requant_shifted : unsigned(63 downto 0);
        variable v_requant_shift : natural range 0 to 255;

        variable v_input_col : integer;
        variable v_required_extra_columns : integer;
        variable v_required_stream_bytes : natural;
                        
    begin
        if rising_edge(clk) then
            if rst_n = '0' then 
                calculation_active <= '0';
                calculation_done <= '0';
                calculation_waiting_for_weights <= '0';
                calculation_kernel_count <= 0;
                calculation_channel_count <= 0;
                weight_refill_request <= '0';
                result_pending <= '0';
                next_window_pending <= '0';
                window_column_count <= 0;
                output_group_count <= 0;

                for lane in 0 to G_C_PAR - 1 loop 
                    accumulators(lane) <= (others => '0');
                    result_buffer(lane) <= (others => '0');
                    quantized_result_buffer(lane) <= (others => '0');
                end loop;
            else
                weight_refill_request <= '0';

                if state /= S_CALC_AND_SLIDING_WINDOW Then
                    calculation_active <= '0';
                    calculation_done <= '0';
                    calculation_waiting_for_weights <= '0';
                    calculation_kernel_count <= 0;
                    calculation_channel_count <= 0;
                    next_window_pending <= '0';
                    window_column_count <= 0;
                    output_group_count <= 0;
                    result_pending <= '0';
                elsif  calculation_done = '0' then
                    if result_pending = '1' then
                        if i_acc_ready = '1' then
                            result_pending <= '0';

                            if window_column_count < C_OUTPUT_WIDTH - 1 then
                                window_column_count <= window_column_count + 1;
                                next_window_pending <= '1';
                            elsif output_group_count < C_OUTPUT_GROUPS -1 then
                                output_group_count <= output_group_count + 1;
                                window_column_count <= 0;
                                calculation_active <= '0';
                                calculation_kernel_count <= 0;
                                calculation_channel_count <= 0;
                                calculation_waiting_for_weights <= '1';
                                weight_refill_request <= '1';

                                for lane in 0 to G_C_PAR - 1 loop
                                    accumulators(lane) <= (others => '0');
                                end loop;
                            else
                                calculation_active <= '0';
                                calculation_done <= '1';
                            end if;
                        end if;
                    elsif next_window_pending = '1' then 
                        v_required_extra_columns := integer(window_column_count) * G_STRIDE - G_PADDING;
                        if v_required_extra_columns < 0 then
                            v_required_extra_columns := 0;

                        elsif v_required_extra_columns > G_W_IN - G_KERNEL then
                            v_required_extra_columns := G_W_IN - G_KERNEL;
                        end if;

                        v_required_stream_bytes := natural(v_required_extra_columns) * G_C_IN;

                        if stream_bytes_available >= v_required_stream_bytes then
                            next_window_pending <= '0';
                            calculation_kernel_count <= 0;
                            calculation_channel_count <= 0;

                            for lane in 0 to G_C_PAR - 1 loop
                                accumulators(lane) <= (others => '0');
                            end loop;

                            if G_C_IN = 1 then
                                calculation_active <= '1';
                            else
                                calculation_active <= '0';
                                calculation_waiting_for_weights <= '1';
                                weight_refill_request <= '1';
                            end if;
                        end if;
                    elsif calculation_waiting_for_weights = '1' then
                        if weight_group_ready = '1' and weight_fill_active = '0' and weight_refill_request = '0' then
                            calculation_waiting_for_weights <= '0';
                            calculation_active <= '1';
                            calculation_kernel_count <= 0;
                        end if;
                    elsif calculation_active = '0' then
                        calculation_active <= '1';
                        calculation_kernel_count <= 0;
                        calculation_channel_count <= 0;
                        
                        for lane in 0 to G_C_PAR - 1 loop
                            accumulators(lane) <= (others => '0');
                        end loop;
                    else
                        v_row := calculation_kernel_count / G_KERNEL;
                        v_col := calculation_kernel_count mod G_KERNEL; 

                        v_input_col := integer(window_column_count) * G_STRIDE + integer(v_col) - G_PADDING;
                        v_activation := (others => '0');
                        if row_valid(v_row) = '1' and v_input_col >= 0 and v_input_col < G_W_IN then
                            v_line_addr := natural(v_input_col) * G_C_IN + calculation_channel_count;
                            v_activation := signed('0' & line_buffer(row_map(v_row),v_line_addr));
                        end if;

                        for lane in 0 to G_C_PAR - 1 loop
                            v_weight_addr := lane * C_KERNEL_SIZE + calculation_kernel_count;
                            v_product := v_activation * weight_buffer(v_weight_addr);
                            accumulators(lane)<=accumulators(lane) + resize(v_product, 32);
                            if calculation_kernel_count = C_KERNEL_SIZE - 1 and calculation_channel_count = G_C_IN - 1 then
                                v_raw_acc := accumulators(lane) + resize(v_product, 32);
                                result_buffer(lane) <= v_raw_acc;

                                v_output_channel := output_group_count * G_C_PAR + lane;
                                v_biased_acc := v_raw_acc + bias_store(v_output_channel);

                                if v_biased_acc(31) = '1' then
                                    v_acc_pos := (others => '0');
                                else
                                    v_acc_pos := v_biased_acc;
                                end if;

                                v_requant_product := unsigned(std_logic_vector(v_acc_pos))*rq_m_store(v_output_channel);

                                v_requant_shift := to_integer(rq_r_store(v_output_channel));

                                if v_requant_shift > 0 then
                                    v_requant_product := v_requant_product + shift_left(to_unsigned(1,64), v_requant_shift - 1);
                                end if;

                                v_requant_shifted := shift_right(v_requant_product, v_requant_shift);
                                
                                if v_requant_shifted > to_unsigned(255, 64) then
                                    quantized_result_buffer(lane) <= (others => '1');
                                else
                                    quantized_result_buffer(lane) <= std_logic_vector(v_requant_shifted(7 downto 0));
                                end if;
                            end if;
                        end loop;

                        if calculation_kernel_count = C_KERNEL_SIZE - 1 then
                            calculation_kernel_count <= 0;

                            if calculation_channel_count = G_C_IN -1 then
                                calculation_active <= '0';
                                result_pending <= '1';
                            else
                                calculation_active <= '0';
                                calculation_channel_count <= calculation_channel_count + 1;
                                calculation_waiting_for_weights <= '1';
                                weight_refill_request <= '1';
                            end if;
                        else
                            calculation_kernel_count <= calculation_kernel_count + 1;
                        end if; 
                    end if;
                end if;
            end if;
        end if;
    end process;

    stream_line_fill_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                stream_line_fill_active <= '0';
                stream_line_fill_done <= '0';
                stream_fill_count <= 0;
                stream_bytes_available <= 0;
            else
                if start_stream_line_fill = '1' then
                    stream_fill_count <= 0;
                    if row_valid(G_KERNEL - 1) = '1' then
                        stream_line_fill_active <= '1';
                        stream_line_fill_done <= '0';
                        stream_bytes_available <= 0;
                    else
                        stream_line_fill_active <= '0';
                        stream_line_fill_done <= '1';
                        stream_bytes_available <= C_STREAM_FILL_SIZE;
                    end if;
                elsif stream_line_fill_active = '1' then 
                    if activation_accepted = '1' then 
                        stream_bytes_available <= stream_bytes_available + 1;
                        if stream_fill_count = C_STREAM_FILL_SIZE - 1 then
                            stream_line_fill_active <= '0';
                            stream_line_fill_done <= '1';
                        else
                            stream_fill_count <= stream_fill_count + 1;
                        end if;
                    end if;
                end if;
            end if;
        end if;
    end process;

    line_rotation_process : process(clk)
        variable v_new_input_row : integer;
        variable v_do_rotation : boolean;
        variable v_drain_rotation : boolean;
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                spare_row <= G_KERNEL;
                output_row_count <= 0;

                logical_top_row <= -integer(G_PADDING);

                vertical_advance_remaining <= 0;
                drain_input_active <= '0';
                line_rotation_done <= '0';
                line_rotation_has_next <= '0';

                for row in 0 to G_KERNEL - 1 loop
                    row_map(row) <= row;

                    if integer(row) - integer(G_PADDING) >= 0 and integer(row) - integer(G_PADDING) < G_H_IN then
                        row_valid(row) <= '1';
                    else
                        row_valid(row) <= '0';
                    end if;
                end loop;
            elsif state = S_IDLE then
                spare_row <= G_KERNEL;
                output_row_count <= 0;

                logical_top_row <= -integer(G_PADDING);

                vertical_advance_remaining <= 0;
                drain_input_active <= '0';
                line_rotation_done <= '0';
                line_rotation_has_next <= '0';

                for row in 0 to G_KERNEL - 1 loop
                    row_map(row) <= row;

                    if integer(row) - integer(G_PADDING) >= 0 and integer(row) - integer(G_PADDING) < G_H_IN then
                        row_valid(row) <= '1';
                    else
                        row_valid(row) <= '0';
                    end if;
                end loop;

            else
                line_rotation_done <= '0';
                if start_line_rotation = '1' then
                    v_new_input_row := logical_top_row + G_KERNEL;
                    v_do_rotation := false;
                    v_drain_rotation := false;
                    if drain_input_active = '1' then
                        if v_new_input_row >= 0 and v_new_input_row < G_H_IN then
                            v_do_rotation := true;
                            v_drain_rotation := true;
                        else
                            drain_input_active <= '0';
                            line_rotation_has_next <= '0';
                        end if;

                    elsif output_row_count < C_OUTPUT_HEIGHT - 1 then
                        v_do_rotation := true;

                    elsif v_new_input_row >= 0 and v_new_input_row < G_H_IN then
                        drain_input_active <= '1';
                        v_do_rotation := true;
                        v_drain_rotation := true;
                    else
                        line_rotation_has_next <= '0';
                    end if;

                    if v_do_rotation then
                        for row in 0 to G_KERNEL - 2 loop
                            row_map(row) <= row_map(row + 1);
                            row_valid(row) <= row_valid(row + 1);
                        end loop;

                        row_map(G_KERNEL - 1) <= spare_row;
                        spare_row <= row_map(0);

                        if v_new_input_row >= 0 and v_new_input_row < G_H_IN then
                            row_valid(G_KERNEL - 1) <= '1';
                        else
                            row_valid(G_KERNEL - 1) <= '0';
                        end if;

                        logical_top_row <= logical_top_row + 1;
                        line_rotation_has_next <= '1';

                        if v_drain_rotation then
                            vertical_advance_remaining <= 0;

                        elsif vertical_advance_remaining = 0 then
                            if G_STRIDE = 1 then
                                output_row_count <= output_row_count + 1;
                                vertical_advance_remaining <= 0;
                            else
                                vertical_advance_remaining <= G_STRIDE - 1;
                            end if;

                        elsif vertical_advance_remaining = 1 then
                            vertical_advance_remaining <= 0;
                            output_row_count <= output_row_count + 1;
                        else
                            vertical_advance_remaining <= vertical_advance_remaining - 1;
                        end if;
                    end if;

                    line_rotation_done <= '1';
                end if;
            end if;
        end if;
    end process;

end architecture rtl;