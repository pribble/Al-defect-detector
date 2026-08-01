library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.env.all;

entity tb_conv_layer is
    generic (
        G_PREFIX : string := "features_0";

        G_C_IN    : positive := 1;
        G_C_OUT   : positive := 64;
        G_H_IN    : positive := 64;
        G_W_IN    : positive := 64;
        G_KERNEL  : positive := 5;
        G_PADDING : natural  := 2;
        G_STRIDE  : positive := 1;
        G_C_PAR   : positive := 64;

        G_VECS : string :=
            "verification/vectors/default/";

        G_RAW_FILE : string :=
            "verification/results/default/raw_acc/" &
            "features_0_raw_acc.bin";

        G_STALL_PERIOD : natural := 7;

        G_PROGRESS_STEP : natural := 10;

        G_TIMEOUT_CYCLES : positive := 1000000
    );
end entity tb_conv_layer;

architecture sim of tb_conv_layer is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_KERNEL_SIZE : positive :=
        G_KERNEL * G_KERNEL;

    constant C_OUTPUT_WIDTH : positive :=
        (
            (
                G_W_IN +
                2 * G_PADDING -
                G_KERNEL
            ) /
            G_STRIDE
        ) + 1;

    constant C_OUTPUT_HEIGHT : positive :=
        (
            (
                G_H_IN +
                2 * G_PADDING -
                G_KERNEL
            ) /
            G_STRIDE
        ) + 1;

    constant C_OUTPUT_GROUPS : positive :=
        G_C_OUT / G_C_PAR;

    constant C_INPUT_BYTE_COUNT : positive :=
        G_H_IN *
        G_W_IN *
        G_C_IN;

    constant C_WEIGHT_FILE_BYTE_COUNT : positive :=
        G_C_OUT *
        G_C_IN *
        C_KERNEL_SIZE;

    constant C_WEIGHT_BATCH_SIZE : positive :=
        G_C_PAR *
        C_KERNEL_SIZE;

    function weight_batch_count return positive is
    begin
        if G_C_IN = 1 then

            if C_OUTPUT_GROUPS = 1 then
                return 1;
            else
                return
                    C_OUTPUT_HEIGHT *
                    C_OUTPUT_GROUPS;
            end if;

        else
            return
                C_OUTPUT_HEIGHT *
                C_OUTPUT_GROUPS *
                C_OUTPUT_WIDTH *
                G_C_IN;
        end if;
    end function;

    constant C_WEIGHT_BATCH_COUNT : positive :=
        weight_batch_count;

    constant C_WEIGHT_ACCEPT_COUNT : positive :=
        C_WEIGHT_BATCH_COUNT *
        C_WEIGHT_BATCH_SIZE;

    constant C_OUTPUT_TRANSFER_COUNT : positive :=
        C_OUTPUT_HEIGHT *
        C_OUTPUT_GROUPS *
        C_OUTPUT_WIDTH;

    constant C_OUTPUT_BYTE_COUNT : positive :=
        C_OUTPUT_HEIGHT *
        C_OUTPUT_WIDTH *
        G_C_OUT;

    constant C_RAW_BYTE_COUNT : positive :=
        C_OUTPUT_BYTE_COUNT * 4;

    constant C_INPUT_FILE : string :=
        G_VECS &
        G_PREFIX &
        "_in.bin";

    constant C_WEIGHT_FILE : string :=
        G_VECS &
        G_PREFIX &
        "_weights.bin";

    constant C_BIAS_FILE : string :=
        G_VECS &
        G_PREFIX &
        "_biases.bin";

    constant C_RQ_M_FILE : string :=
        G_VECS &
        G_PREFIX &
        "_requant_m.bin";

    constant C_RQ_R_FILE : string :=
        G_VECS &
        G_PREFIX &
        "_requant_r.bin";

    constant C_OUTPUT_FILE : string :=
        G_VECS &
        G_PREFIX &
        "_out.bin";

    type t_binary_file is file of character;

    type t_byte_array is array (
        natural range <>
    ) of std_logic_vector(7 downto 0);

    type t_int32_array is array (
        natural range <>
    ) of signed(31 downto 0);

    impure function load_byte_file (
        constant p_path  : string;
        constant p_count : positive
    ) return t_byte_array is

        file v_file : t_binary_file;

        variable v_status :
            file_open_status;

        variable v_result :
            t_byte_array(0 to p_count - 1);

        variable v_character :
            character;

        variable v_value :
            natural;

    begin
        file_open(
            v_status,
            v_file,
            p_path,
            read_mode
        );

        assert v_status = open_ok
            report
                "Unable to open vector file: " &
                p_path
            severity failure;

        for index in 0 to p_count - 1 loop

            assert not endfile(v_file)
                report
                    p_path &
                    " ended early at byte " &
                    integer'image(index) &
                    "; expected " &
                    integer'image(p_count) &
                    " bytes"
                severity failure;

            read(v_file, v_character);

            v_value :=
                character'pos(v_character);

            v_result(index) :=
                std_logic_vector(
                    to_unsigned(v_value, 8)
                );

        end loop;

        assert endfile(v_file)
            report
                p_path &
                " contains more than the expected " &
                integer'image(p_count) &
                " bytes"
            severity failure;

        file_close(v_file);

        return v_result;
    end function;

    impure function load_int32_file (
        constant p_path  : string;
        constant p_count : positive
    ) return t_int32_array is

        file v_file : t_binary_file;

        variable v_status :
            file_open_status;

        variable v_result :
            t_int32_array(0 to p_count - 1);

        variable v_character :
            character;

        variable v_word :
            std_logic_vector(31 downto 0);

        variable v_byte :
            natural;

    begin
        file_open(
            v_status,
            v_file,
            p_path,
            read_mode
        );

        assert v_status = open_ok
            report
                "Unable to open raw accumulator file: " &
                p_path
            severity failure;

        for index in 0 to p_count - 1 loop

            for byte_index in 0 to 3 loop

                assert not endfile(v_file)
                    report
                        p_path &
                        " ended early while reading accumulator " &
                        integer'image(index)
                    severity failure;

                read(v_file, v_character);

                v_byte :=
                    character'pos(v_character);

                v_word(
                    byte_index * 8 + 7
                    downto
                    byte_index * 8
                ) :=
                    std_logic_vector(
                        to_unsigned(v_byte, 8)
                    );

            end loop;

            v_result(index) :=
                signed(v_word);

        end loop;

        assert endfile(v_file)
            report
                p_path &
                " contains more than the expected " &
                integer'image(p_count * 4) &
                " bytes"
            severity failure;

        file_close(v_file);

        return v_result;
    end function;

    constant C_INPUT_MEMORY :
        t_byte_array(0 to C_INPUT_BYTE_COUNT - 1) :=
        load_byte_file(
            C_INPUT_FILE,
            C_INPUT_BYTE_COUNT
        );

    constant C_WEIGHT_MEMORY :
        t_byte_array(0 to C_WEIGHT_FILE_BYTE_COUNT - 1) :=
        load_byte_file(
            C_WEIGHT_FILE,
            C_WEIGHT_FILE_BYTE_COUNT
        );

    constant C_BIAS_MEMORY :
        t_byte_array(0 to G_C_OUT * 4 - 1) :=
        load_byte_file(
            C_BIAS_FILE,
            G_C_OUT * 4
        );

    constant C_RQ_M_MEMORY :
        t_byte_array(0 to G_C_OUT * 4 - 1) :=
        load_byte_file(
            C_RQ_M_FILE,
            G_C_OUT * 4
        );

    constant C_RQ_R_MEMORY :
        t_byte_array(0 to G_C_OUT - 1) :=
        load_byte_file(
            C_RQ_R_FILE,
            G_C_OUT
        );

    constant C_EXPECTED_OUTPUT :
        t_byte_array(0 to C_OUTPUT_BYTE_COUNT - 1) :=
        load_byte_file(
            C_OUTPUT_FILE,
            C_OUTPUT_BYTE_COUNT
        );

    constant C_EXPECTED_RAW :
        t_int32_array(0 to C_OUTPUT_BYTE_COUNT - 1) :=
        load_int32_file(
            G_RAW_FILE,
            C_OUTPUT_BYTE_COUNT
        );

    function little_endian_word (
        constant p_memory : t_byte_array;
        constant p_base   : natural
    ) return std_logic_vector is

        variable v_result :
            std_logic_vector(31 downto 0);

    begin
        v_result(7 downto 0) :=
            p_memory(p_base + 0);

        v_result(15 downto 8) :=
            p_memory(p_base + 1);

        v_result(23 downto 16) :=
            p_memory(p_base + 2);

        v_result(31 downto 24) :=
            p_memory(p_base + 3);

        return v_result;
    end function;

    signal clk :
        std_logic := '0';

    signal rst_n :
        std_logic := '0';

    signal streams_enabled :
        std_logic := '0';

    signal i_valid :
        std_logic := '0';

    signal i_ready :
        std_logic;

    signal i_data :
        std_logic_vector(7 downto 0) :=
        (others => '0');

    signal i_weight_valid :
        std_logic := '0';

    signal o_weight_ready :
        std_logic;

    signal i_weight_data :
        std_logic_vector(7 downto 0) :=
        (others => '0');

    signal cfg_we :
        std_logic := '0';

    signal cfg_sel :
        std_logic_vector(1 downto 0) :=
        (others => '0');

    signal cfg_addr :
        std_logic_vector(19 downto 0) :=
        (others => '0');

    signal cfg_wdata :
        std_logic_vector(31 downto 0) :=
        (others => '0');

    signal o_valid :
        std_logic;

    signal o_data :
        std_logic_vector(
            G_C_PAR * 8 - 1 downto 0
        );

    signal i_acc_ready :
        std_logic := '0';

    signal o_acc_valid :
        std_logic;

    signal o_acc_data :
        std_logic_vector(
            G_C_PAR * 32 - 1 downto 0
        );

    signal input_accept_count :
        natural range
            0 to C_INPUT_BYTE_COUNT :=
        0;

    signal weight_accept_count :
        natural range
            0 to C_WEIGHT_ACCEPT_COUNT :=
        0;

    signal output_transfer_count :
        natural range
            0 to C_OUTPUT_TRANSFER_COUNT :=
        0;

    signal output_cycle_count :
        natural := 0;

    signal output_stall_observed :
        std_logic := '0';

begin

    assert G_C_OUT mod G_C_PAR = 0
        report
            "G_C_OUT must be divisible by G_C_PAR"
        severity failure;

    assert C_RAW_BYTE_COUNT =
           C_OUTPUT_BYTE_COUNT * 4
        report
            "Internal raw-byte calculation error"
        severity failure;

    clk <= not clk after C_CLK_PERIOD / 2;

    dut : entity work.conv_layer
        generic map (
            G_C_IN    => G_C_IN,
            G_C_OUT   => G_C_OUT,
            G_W_IN    => G_W_IN,
            G_H_IN    => G_H_IN,
            G_C_PAR   => G_C_PAR,
            G_KERNEL  => G_KERNEL,
            G_PADDING => G_PADDING,
            G_STRIDE  => G_STRIDE
        )
        port map (
            clk   => clk,
            rst_n => rst_n,

            i_valid => i_valid,
            i_ready => i_ready,
            i_data  => i_data,

            i_weight_valid => i_weight_valid,
            o_weight_ready => o_weight_ready,
            i_weight_data  => i_weight_data,

            cfg_we    => cfg_we,
            cfg_sel   => cfg_sel,
            cfg_addr  => cfg_addr,
            cfg_wdata => cfg_wdata,

            o_valid => o_valid,
            o_data  => o_data,

            i_acc_ready => i_acc_ready,
            o_acc_valid => o_acc_valid,
            o_acc_data  => o_acc_data
        );

    --------------------------------------------------------------------
    -- Load bias, multiplier and right-shift parameters while reset is
    -- asserted. The RTL parameter memories are writable during reset.
    --------------------------------------------------------------------

    configuration_process : process

        procedure write_parameter (
            constant p_sel  :
                std_logic_vector(1 downto 0);

            constant p_addr :
                natural;

            constant p_data :
                std_logic_vector(31 downto 0)
        ) is
        begin
            wait until falling_edge(clk);

            cfg_sel <= p_sel;

            cfg_addr <=
                std_logic_vector(
                    to_unsigned(
                        p_addr,
                        cfg_addr'length
                    )
                );

            cfg_wdata <= p_data;
            cfg_we <= '1';

            wait until rising_edge(clk);
            wait for 1 ns;

            cfg_we <= '0';
        end procedure;

        variable v_word :
            std_logic_vector(31 downto 0);

    begin
        rst_n <= '0';
        streams_enabled <= '0';

        cfg_we <= '0';
        cfg_sel <= (others => '0');
        cfg_addr <= (others => '0');
        cfg_wdata <= (others => '0');

        report
            "START " &
            G_PREFIX &
            " Cin=" &
            integer'image(G_C_IN) &
            " Cout=" &
            integer'image(G_C_OUT) &
            " HxW=" &
            integer'image(G_H_IN) &
            "x" &
            integer'image(G_W_IN) &
            " K=" &
            integer'image(G_KERNEL) &
            " P=" &
            integer'image(G_PADDING) &
            " S=" &
            integer'image(G_STRIDE) &
            " Cpar=" &
            integer'image(G_C_PAR) &
            " Out=" &
            integer'image(C_OUTPUT_HEIGHT) &
            "x" &
            integer'image(C_OUTPUT_WIDTH)
            severity note;

        wait for 4 * C_CLK_PERIOD;

        for output_channel in 0 to G_C_OUT - 1 loop

            v_word :=
                little_endian_word(
                    C_BIAS_MEMORY,
                    output_channel * 4
                );

            write_parameter(
                p_sel  => "01",
                p_addr => output_channel,
                p_data => v_word
            );

        end loop;

        for output_channel in 0 to G_C_OUT - 1 loop

            v_word :=
                little_endian_word(
                    C_RQ_M_MEMORY,
                    output_channel * 4
                );

            write_parameter(
                p_sel  => "10",
                p_addr => output_channel,
                p_data => v_word
            );

        end loop;

        for output_channel in 0 to G_C_OUT - 1 loop

            v_word := (others => '0');

            v_word(7 downto 0) :=
                C_RQ_R_MEMORY(output_channel);

            write_parameter(
                p_sel  => "11",
                p_addr => output_channel,
                p_data => v_word
            );

        end loop;

        wait until falling_edge(clk);

        cfg_we <= '0';

        rst_n <= '1';
        streams_enabled <= '1';

        wait;
    end process;

    --------------------------------------------------------------------
    -- Stream the real activation tensor in HWC order.
    --------------------------------------------------------------------

    activation_driver_process : process(all)
    begin
        i_valid <= '0';
        i_data <= (others => '0');

        if
            streams_enabled = '1' and
            input_accept_count <
            C_INPUT_BYTE_COUNT
        then
            i_valid <= '1';

            i_data <=
                C_INPUT_MEMORY(
                    input_accept_count
                );
        end if;
    end process;

    activation_counter_process : process(clk)
    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                input_accept_count <= 0;

            elsif
                i_valid = '1' and
                i_ready = '1'
            then
                input_accept_count <=
                    input_accept_count + 1;
            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Translate every weight request into the corresponding PyTorch
    -- weight tensor slice:
    --
    -- weights[C_OUT][C_IN][KH][KW]
    --
    -- One DUT batch contains:
    --
    -- lane 0 K*K values,
    -- lane 1 K*K values,
    -- ...
    -- lane G_C_PAR-1 K*K values.
    --------------------------------------------------------------------

    weight_driver_process : process(all)

        variable v_batch_index :
            natural;

        variable v_byte_in_batch :
            natural;

        variable v_lane :
            natural;

        variable v_kernel_index :
            natural;

        variable v_kernel_row :
            natural;

        variable v_kernel_column :
            natural;

        variable v_output_group :
            natural;

        variable v_input_channel :
            natural;

        variable v_batch_in_row :
            natural;

        variable v_group_span :
            natural;

        variable v_group_remainder :
            natural;

        variable v_output_channel :
            natural;

        variable v_file_index :
            natural;

    begin
        i_weight_valid <= '0';
        i_weight_data <= (others => '0');

        if
            streams_enabled = '1' and
            weight_accept_count <
            C_WEIGHT_ACCEPT_COUNT
        then
            v_batch_index :=
                weight_accept_count /
                C_WEIGHT_BATCH_SIZE;

            v_byte_in_batch :=
                weight_accept_count mod
                C_WEIGHT_BATCH_SIZE;

            v_lane :=
                v_byte_in_batch /
                C_KERNEL_SIZE;

            v_kernel_index :=
                v_byte_in_batch mod
                C_KERNEL_SIZE;

            v_kernel_row :=
                v_kernel_index /
                G_KERNEL;

            v_kernel_column :=
                v_kernel_index mod
                G_KERNEL;

            if G_C_IN = 1 then

                if C_OUTPUT_GROUPS = 1 then
                    v_output_group := 0;
                else
                    v_output_group :=
                        v_batch_index mod
                        C_OUTPUT_GROUPS;
                end if;

                v_input_channel := 0;

            else
                v_group_span :=
                    C_OUTPUT_WIDTH *
                    G_C_IN;

                v_batch_in_row :=
                    v_batch_index mod
                    (
                        C_OUTPUT_GROUPS *
                        v_group_span
                    );

                v_output_group :=
                    v_batch_in_row /
                    v_group_span;

                v_group_remainder :=
                    v_batch_in_row mod
                    v_group_span;

                v_input_channel :=
                    v_group_remainder mod
                    G_C_IN;
            end if;

            v_output_channel :=
                v_output_group *
                G_C_PAR +
                v_lane;

            v_file_index :=
                (
                    (
                        (
                            v_output_channel *
                            G_C_IN +
                            v_input_channel
                        ) *
                        G_KERNEL +
                        v_kernel_row
                    ) *
                    G_KERNEL
                ) +
                v_kernel_column;

            i_weight_valid <= '1';

            i_weight_data <=
                C_WEIGHT_MEMORY(v_file_index);
        end if;
    end process;

    weight_counter_process : process(clk)
    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                weight_accept_count <= 0;

            elsif
                i_weight_valid = '1' and
                o_weight_ready = '1'
            then
                weight_accept_count <=
                    weight_accept_count + 1;
            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Periodic output backpressure.
    --------------------------------------------------------------------

    output_cycle_process : process(clk)
    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                output_cycle_count <= 0;
            else
                output_cycle_count <=
                    output_cycle_count + 1;
            end if;

        end if;
    end process;

    output_ready_process : process(all)
    begin
        i_acc_ready <= '0';

        if rst_n = '1' then

            if G_STALL_PERIOD = 0 then
                i_acc_ready <= '1';

            elsif
                output_cycle_count mod
                G_STALL_PERIOD /=
                G_STALL_PERIOD - 1
            then
                i_acc_ready <= '1';
            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Compare:
    --
    --   o_data     against exported uint8 output,
    --   o_acc_data against padded/strided raw-int32 golden.
    --
    -- DUT stream order:
    --
    --   output row
    --     output group
    --       output column
    --         lane
    --
    -- Golden-file order:
    --
    --   output row
    --     output column
    --       output channel
    --------------------------------------------------------------------

    output_checker_process : process(clk)

        variable v_output_row :
            natural;

        variable v_output_group :
            natural;

        variable v_output_column :
            natural;

        variable v_row_offset :
            natural;

        variable v_output_channel :
            natural;

        variable v_golden_index :
            natural;

        variable v_actual_quantized :
            natural;

        variable v_expected_quantized :
            natural;

        variable v_actual_raw :
            signed(31 downto 0);

        variable v_expected_raw :
            signed(31 downto 0);

        variable v_completed_bytes :
            natural;

        variable v_percentage :
            natural;

        variable v_next_percentage :
            natural := G_PROGRESS_STEP;

    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                output_transfer_count <= 0;

                v_next_percentage :=
                    G_PROGRESS_STEP;

            elsif
                o_valid = '1' and
                i_acc_ready = '1'
            then
                assert o_acc_valid = '1'
                    report
                        G_PREFIX &
                        ": o_valid asserted without o_acc_valid"
                    severity failure;

                assert
                    output_transfer_count <
                    C_OUTPUT_TRANSFER_COUNT
                    report
                        G_PREFIX &
                        ": unexpected extra output transfer"
                    severity failure;

                if
                    output_transfer_count <
                    C_OUTPUT_TRANSFER_COUNT
                then
                    v_output_row :=
                        output_transfer_count /
                        (
                            C_OUTPUT_GROUPS *
                            C_OUTPUT_WIDTH
                        );

                    v_row_offset :=
                        output_transfer_count mod
                        (
                            C_OUTPUT_GROUPS *
                            C_OUTPUT_WIDTH
                        );

                    v_output_group :=
                        v_row_offset /
                        C_OUTPUT_WIDTH;

                    v_output_column :=
                        v_row_offset mod
                        C_OUTPUT_WIDTH;

                    for lane in 0 to G_C_PAR - 1 loop

                        v_output_channel :=
                            v_output_group *
                            G_C_PAR +
                            lane;

                        v_golden_index :=
                            (
                                (
                                    v_output_row *
                                    C_OUTPUT_WIDTH +
                                    v_output_column
                                ) *
                                G_C_OUT
                            ) +
                            v_output_channel;

                        v_actual_quantized :=
                            to_integer(
                                unsigned(
                                    o_data(
                                        lane * 8 + 7
                                        downto
                                        lane * 8
                                    )
                                )
                            );

                        v_expected_quantized :=
                            to_integer(
                                unsigned(
                                    C_EXPECTED_OUTPUT(
                                        v_golden_index
                                    )
                                )
                            );

                        assert
                            v_actual_quantized =
                            v_expected_quantized
                            report
                                G_PREFIX &
                                " QUANT MISMATCH" &
                                " row=" &
                                integer'image(v_output_row) &
                                " col=" &
                                integer'image(v_output_column) &
                                " channel=" &
                                integer'image(v_output_channel) &
                                " group=" &
                                integer'image(v_output_group) &
                                " lane=" &
                                integer'image(lane) &
                                " expected=" &
                                integer'image(
                                    v_expected_quantized
                                ) &
                                " got=" &
                                integer'image(
                                    v_actual_quantized
                                )
                            severity failure;

                        v_actual_raw :=
                            signed(
                                o_acc_data(
                                    lane * 32 + 31
                                    downto
                                    lane * 32
                                )
                            );

                        v_expected_raw :=
                            C_EXPECTED_RAW(
                                v_golden_index
                            );

                        assert
                            v_actual_raw =
                            v_expected_raw
                            report
                                G_PREFIX &
                                " RAW MISMATCH" &
                                " row=" &
                                integer'image(v_output_row) &
                                " col=" &
                                integer'image(v_output_column) &
                                " channel=" &
                                integer'image(v_output_channel) &
                                " group=" &
                                integer'image(v_output_group) &
                                " lane=" &
                                integer'image(lane) &
                                " expected=" &
                                integer'image(
                                    to_integer(v_expected_raw)
                                ) &
                                " got=" &
                                integer'image(
                                    to_integer(v_actual_raw)
                                )
                            severity failure;

                    end loop;

                    output_transfer_count <=
                        output_transfer_count + 1;

                    if G_PROGRESS_STEP > 0 then

                        v_completed_bytes :=
                            (
                                output_transfer_count +
                                1
                            ) *
                            G_C_PAR;

                        v_percentage :=
                            (
                                v_completed_bytes *
                                100
                            ) /
                            C_OUTPUT_BYTE_COUNT;

                        if
                            v_percentage >=
                            v_next_percentage and
                            v_completed_bytes <
                            C_OUTPUT_BYTE_COUNT
                        then
                            report
                                "PROGRESS " &
                                G_PREFIX &
                                " " &
                                integer'image(
                                    v_percentage
                                ) &
                                "% " &
                                integer'image(
                                    v_completed_bytes
                                ) &
                                "/" &
                                integer'image(
                                    C_OUTPUT_BYTE_COUNT
                                ) &
                                " output bytes"
                                severity note;

                            while
                                v_next_percentage <=
                                v_percentage
                            loop
                                v_next_percentage :=
                                    v_next_percentage +
                                    G_PROGRESS_STEP;
                            end loop;
                        end if;

                    end if;
                end if;
            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Both packed outputs must remain stable while stalled.
    --------------------------------------------------------------------

    output_stability_process : process(clk)

        variable v_holding :
            boolean := false;

        variable v_held_quantized :
            std_logic_vector(
                G_C_PAR * 8 - 1 downto 0
            ) :=
            (others => '0');

        variable v_held_raw :
            std_logic_vector(
                G_C_PAR * 32 - 1 downto 0
            ) :=
            (others => '0');

    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                v_holding := false;

                v_held_quantized :=
                    (others => '0');

                v_held_raw :=
                    (others => '0');

                output_stall_observed <= '0';

            else
                assert o_valid = o_acc_valid
                    report
                        G_PREFIX &
                        ": o_valid and o_acc_valid disagree"
                    severity failure;

                if v_holding then

                    assert o_valid = '1'
                        report
                            G_PREFIX &
                            ": o_valid dropped while stalled"
                        severity failure;

                    assert o_acc_valid = '1'
                        report
                            G_PREFIX &
                            ": o_acc_valid dropped while stalled"
                        severity failure;

                    assert o_data = v_held_quantized
                        report
                            G_PREFIX &
                            ": quantized output changed while stalled"
                        severity failure;

                    assert o_acc_data = v_held_raw
                        report
                            G_PREFIX &
                            ": raw output changed while stalled"
                        severity failure;

                end if;

                if
                    o_valid = '1' and
                    i_acc_ready = '0'
                then
                    v_holding := true;

                    v_held_quantized :=
                        o_data;

                    v_held_raw :=
                        o_acc_data;

                    output_stall_observed <= '1';

                else
                    v_holding := false;
                end if;
            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Finish only after every activation, requested weight byte and
    -- expected output transfer has completed.
    --------------------------------------------------------------------

    completion_process : process
    begin
        wait until rst_n = '1';

        wait until
            input_accept_count =
                C_INPUT_BYTE_COUNT and
            weight_accept_count =
                C_WEIGHT_ACCEPT_COUNT and
            output_transfer_count =
                C_OUTPUT_TRANSFER_COUNT;

        for cycle in 1 to 16 loop
            wait until rising_edge(clk);

            assert not (
                o_valid = '1' and
                i_acc_ready = '1'
            )
                report
                    G_PREFIX &
                    ": unexpected output after completion"
                severity failure;
        end loop;

        if G_STALL_PERIOD > 0 then
            assert output_stall_observed = '1'
                report
                    G_PREFIX &
                    ": configured output backpressure was not exercised"
                severity failure;
        end if;

        report
            "PASS " &
            G_PREFIX &
            " inputs=" &
            integer'image(C_INPUT_BYTE_COUNT) &
            " weight_batches=" &
            integer'image(C_WEIGHT_BATCH_COUNT) &
            " output_bytes=" &
            integer'image(C_OUTPUT_BYTE_COUNT) &
            " raw and quantized vectors match"
            severity note;

        stop;
        wait;
    end process;

    watchdog_process : process
    begin
        for cycle in 1 to G_TIMEOUT_CYCLES loop
            wait until rising_edge(clk);
        end loop;

        assert false
            report
                "TIMEOUT " &
                G_PREFIX &
                " inputs=" &
                integer'image(input_accept_count) &
                "/" &
                integer'image(C_INPUT_BYTE_COUNT) &
                " weights=" &
                integer'image(weight_accept_count) &
                "/" &
                integer'image(C_WEIGHT_ACCEPT_COUNT) &
                " outputs=" &
                integer'image(output_transfer_count) &
                "/" &
                integer'image(C_OUTPUT_TRANSFER_COUNT)
            severity failure;
    end process;

end architecture sim;