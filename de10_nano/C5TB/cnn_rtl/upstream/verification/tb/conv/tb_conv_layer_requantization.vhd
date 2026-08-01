library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.env.all;

entity tb_conv_layer_requantization is
end entity tb_conv_layer_requantization;

architecture sim of tb_conv_layer_requantization is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_C_IN    : positive := 1;
    constant C_C_OUT   : positive := 4;
    constant C_W_IN    : positive := 4;
    constant C_H_IN    : positive := 3;
    constant C_C_PAR   : positive := 2;
    constant C_KERNEL  : positive := 3;
    constant C_PADDING : natural  := 0;
    constant C_STRIDE  : positive := 1;

    constant C_INPUT_COUNT : positive :=
        C_C_IN * C_W_IN * C_H_IN;

    constant C_KERNEL_SIZE : positive :=
        C_KERNEL * C_KERNEL;

    constant C_OUTPUT_GROUPS : positive :=
        C_C_OUT / C_C_PAR;

    constant C_WEIGHT_COUNT : positive :=
        C_OUTPUT_GROUPS *
        C_C_PAR *
        C_KERNEL_SIZE;

    constant C_OUTPUT_COUNT : positive := 4;

    type t_integer_array is array (
        natural range <>
    ) of integer;

    --------------------------------------------------------------------
    -- Input image:
    --
    --   1   2   3   4
    --   5   6   7   8
    --   9  10  11  12
    --
    -- All weights are 1.
    --
    -- Raw output:
    --
    --   column 0 = 54
    --   column 1 = 63
    --
    -- Output ordering:
    --
    --   group 0, column 0
    --   group 0, column 1
    --   group 1, column 0
    --   group 1, column 1
    --------------------------------------------------------------------

    constant C_INPUT :
        t_integer_array(0 to C_INPUT_COUNT - 1) := (
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12
        );

    constant C_EXPECTED_RAW :
        t_integer_array(0 to C_OUTPUT_COUNT - 1) := (
            54,
            63,
            54,
            63
        );

    --------------------------------------------------------------------
    -- Per-output-channel parameters
    --
    -- Channel 0:
    --   bias = -100
    --   multiplier = 1
    --   shift = 0
    --
    --   ReLU(54 - 100) = 0
    --   ReLU(63 - 100) = 0
    --
    -- Channel 1:
    --   bias = 10
    --   multiplier = 1
    --   shift = 0
    --
    --   54 + 10 = 64
    --   63 + 10 = 73
    --
    -- Channel 2:
    --   bias = 0
    --   multiplier = 3
    --   shift = 2
    --
    --   round(54 * 3 / 4)
    --     = (162 + 2) >> 2
    --     = 41
    --
    --   round(63 * 3 / 4)
    --     = (189 + 2) >> 2
    --     = 47
    --
    -- Channel 3:
    --   bias = 300
    --   multiplier = 1
    --   shift = 0
    --
    --   54 + 300 = 354 -> 255
    --   63 + 300 = 363 -> 255
    --------------------------------------------------------------------

    constant C_EXPECTED_QUANT_LANE_0 :
        t_integer_array(0 to C_OUTPUT_COUNT - 1) := (
             0,
             0,
            41,
            47
        );

    constant C_EXPECTED_QUANT_LANE_1 :
        t_integer_array(0 to C_OUTPUT_COUNT - 1) := (
             64,
             73,
            255,
            255
        );

    signal clk : std_logic := '0';
    signal rst_n : std_logic := '0';

    signal i_valid : std_logic := '0';
    signal i_ready : std_logic;

    signal i_data :
        std_logic_vector(7 downto 0) :=
        (others => '0');

    signal i_weight_valid : std_logic := '0';
    signal o_weight_ready : std_logic;

    signal i_weight_data :
        std_logic_vector(7 downto 0) :=
        (others => '0');

    signal cfg_we : std_logic := '0';

    signal cfg_sel :
        std_logic_vector(1 downto 0) :=
        (others => '0');

    signal cfg_addr :
        std_logic_vector(19 downto 0) :=
        (others => '0');

    signal cfg_wdata :
        std_logic_vector(31 downto 0) :=
        (others => '0');

    signal o_valid : std_logic;

    signal o_data :
        std_logic_vector(
            C_C_PAR * 8 - 1 downto 0
        );

    signal i_acc_ready : std_logic := '0';
    signal o_acc_valid : std_logic;

    signal o_acc_data :
        std_logic_vector(
            C_C_PAR * 32 - 1 downto 0
        );

    signal streams_enabled : std_logic := '0';

    signal input_index :
        natural range 0 to C_INPUT_COUNT := 0;

    signal weight_index :
        natural range 0 to C_WEIGHT_COUNT := 0;

    signal output_index :
        natural range 0 to C_OUTPUT_COUNT := 0;

    signal output_stall_observed : std_logic := '0';

begin

    clk <= not clk after C_CLK_PERIOD / 2;

    dut : entity work.conv_layer
        generic map (
            G_C_IN    => C_C_IN,
            G_C_OUT   => C_C_OUT,
            G_W_IN    => C_W_IN,
            G_H_IN    => C_H_IN,
            G_C_PAR   => C_C_PAR,
            G_KERNEL  => C_KERNEL,
            G_PADDING => C_PADDING,
            G_STRIDE  => C_STRIDE
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
            o_done => open,

            i_acc_ready => i_acc_ready,
            o_acc_valid => o_acc_valid,
            o_acc_data  => o_acc_data
        );

    --------------------------------------------------------------------
    -- Reset and parameter loading
    --------------------------------------------------------------------

    configuration_process : process

        procedure write_parameter (
            constant p_sel  : std_logic_vector(1 downto 0);
            constant p_addr : natural;
            constant p_data : std_logic_vector(31 downto 0)
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

    begin
        rst_n <= '0';
        streams_enabled <= '0';

        cfg_we <= '0';
        cfg_sel <= (others => '0');
        cfg_addr <= (others => '0');
        cfg_wdata <= (others => '0');

        wait for 4 * C_CLK_PERIOD;
        wait until falling_edge(clk);

        rst_n <= '1';

        ----------------------------------------------------------------
        -- Channel 0
        ----------------------------------------------------------------

        write_parameter(
            p_sel  => "01",
            p_addr => 0,
            p_data => std_logic_vector(
                to_signed(-100, 32)
            )
        );

        write_parameter(
            p_sel  => "10",
            p_addr => 0,
            p_data => std_logic_vector(
                to_unsigned(1, 32)
            )
        );

        write_parameter(
            p_sel  => "11",
            p_addr => 0,
            p_data => std_logic_vector(
                to_unsigned(0, 32)
            )
        );

        ----------------------------------------------------------------
        -- Channel 1
        ----------------------------------------------------------------

        write_parameter(
            p_sel  => "01",
            p_addr => 1,
            p_data => std_logic_vector(
                to_signed(10, 32)
            )
        );

        write_parameter(
            p_sel  => "10",
            p_addr => 1,
            p_data => std_logic_vector(
                to_unsigned(1, 32)
            )
        );

        write_parameter(
            p_sel  => "11",
            p_addr => 1,
            p_data => std_logic_vector(
                to_unsigned(0, 32)
            )
        );

        ----------------------------------------------------------------
        -- Channel 2
        ----------------------------------------------------------------

        write_parameter(
            p_sel  => "01",
            p_addr => 2,
            p_data => std_logic_vector(
                to_signed(0, 32)
            )
        );

        write_parameter(
            p_sel  => "10",
            p_addr => 2,
            p_data => std_logic_vector(
                to_unsigned(3, 32)
            )
        );

        write_parameter(
            p_sel  => "11",
            p_addr => 2,
            p_data => std_logic_vector(
                to_unsigned(2, 32)
            )
        );

        ----------------------------------------------------------------
        -- Channel 3
        ----------------------------------------------------------------

        write_parameter(
            p_sel  => "01",
            p_addr => 3,
            p_data => std_logic_vector(
                to_signed(300, 32)
            )
        );

        write_parameter(
            p_sel  => "10",
            p_addr => 3,
            p_data => std_logic_vector(
                to_unsigned(1, 32)
            )
        );

        write_parameter(
            p_sel  => "11",
            p_addr => 3,
            p_data => std_logic_vector(
                to_unsigned(0, 32)
            )
        );

        wait until falling_edge(clk);

        cfg_we <= '0';
        streams_enabled <= '1';

        wait;
    end process;

    --------------------------------------------------------------------
    -- Activation source
    --------------------------------------------------------------------

    activation_driver_process : process(all)
    begin
        i_valid <= '0';
        i_data <= (others => '0');

        if
            streams_enabled = '1' and
            input_index < C_INPUT_COUNT
        then
            i_valid <= '1';

            i_data <=
                std_logic_vector(
                    to_unsigned(
                        C_INPUT(input_index),
                        i_data'length
                    )
                );
        end if;
    end process;

    activation_counter_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                input_index <= 0;

            elsif
                i_valid = '1' and
                i_ready = '1'
            then
                input_index <= input_index + 1;
            end if;
        end if;
    end process;

    --------------------------------------------------------------------
    -- Weight source
    --
    -- Both groups, both lanes, every kernel position use weight 1.
    --------------------------------------------------------------------

    weight_driver_process : process(all)
    begin
        i_weight_valid <= '0';
        i_weight_data <= (others => '0');

        if
            streams_enabled = '1' and
            weight_index < C_WEIGHT_COUNT
        then
            i_weight_valid <= '1';

            i_weight_data <=
                std_logic_vector(
                    to_signed(
                        1,
                        i_weight_data'length
                    )
                );
        end if;
    end process;

    weight_counter_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                weight_index <= 0;

            elsif
                i_weight_valid = '1' and
                o_weight_ready = '1'
            then
                weight_index <= weight_index + 1;
            end if;
        end if;
    end process;

    --------------------------------------------------------------------
    -- Force the first output to remain stalled for several cycles.
    --
    -- After that first output is released, the remainder of the frame
    -- runs without backpressure.
    --------------------------------------------------------------------

    output_ready_process : process(clk)
        variable v_first_output_seen : boolean := false;
        variable v_stall_cycles : natural range 0 to 3 := 0;
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                i_acc_ready <= '0';

                v_first_output_seen := false;
                v_stall_cycles := 0;

                output_stall_observed <= '0';

            else
                if not v_first_output_seen then
                    i_acc_ready <= '0';

                    if o_valid = '1' then
                        v_first_output_seen := true;
                        v_stall_cycles := 3;

                        output_stall_observed <= '1';
                    end if;

                elsif v_stall_cycles > 0 then
                    i_acc_ready <= '0';

                    v_stall_cycles :=
                        v_stall_cycles - 1;

                else
                    i_acc_ready <= '1';
                end if;
            end if;
        end if;
    end process;

    --------------------------------------------------------------------
    -- Valid signals must represent the same held result.
    --------------------------------------------------------------------

    valid_consistency_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '1' then
                assert o_valid = o_acc_valid
                    report
                        "FAIL: o_valid and o_acc_valid disagree."
                    severity failure;
            end if;
        end if;
    end process;

    --------------------------------------------------------------------
    -- Check raw and quantized output values.
    --------------------------------------------------------------------

    output_checker_process : process(clk)
        variable v_raw_lane_0 : integer;
        variable v_raw_lane_1 : integer;

        variable v_quant_lane_0 : integer;
        variable v_quant_lane_1 : integer;
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                output_index <= 0;

            elsif
                o_valid = '1' and
                i_acc_ready = '1'
            then
                assert output_index < C_OUTPUT_COUNT
                    report
                        "FAIL: unexpected extra output."
                    severity failure;

                if output_index < C_OUTPUT_COUNT then

                    v_raw_lane_0 :=
                        to_integer(
                            signed(
                                o_acc_data(
                                    31 downto 0
                                )
                            )
                        );

                    v_raw_lane_1 :=
                        to_integer(
                            signed(
                                o_acc_data(
                                    63 downto 32
                                )
                            )
                        );

                    v_quant_lane_0 :=
                        to_integer(
                            unsigned(
                                o_data(
                                    7 downto 0
                                )
                            )
                        );

                    v_quant_lane_1 :=
                        to_integer(
                            unsigned(
                                o_data(
                                    15 downto 8
                                )
                            )
                        );

                    assert
                        v_raw_lane_0 =
                        C_EXPECTED_RAW(output_index)
                        report
                            "FAIL: raw lane 0 mismatch at output " &
                            integer'image(output_index) &
                            ". Expected " &
                            integer'image(
                                C_EXPECTED_RAW(output_index)
                            ) &
                            ", got " &
                            integer'image(v_raw_lane_0)
                        severity failure;

                    assert
                        v_raw_lane_1 =
                        C_EXPECTED_RAW(output_index)
                        report
                            "FAIL: raw lane 1 mismatch at output " &
                            integer'image(output_index) &
                            ". Expected " &
                            integer'image(
                                C_EXPECTED_RAW(output_index)
                            ) &
                            ", got " &
                            integer'image(v_raw_lane_1)
                        severity failure;

                    assert
                        v_quant_lane_0 =
                        C_EXPECTED_QUANT_LANE_0(output_index)
                        report
                            "FAIL: quantized lane 0 mismatch at output " &
                            integer'image(output_index) &
                            ". Expected " &
                            integer'image(
                                C_EXPECTED_QUANT_LANE_0(
                                    output_index
                                )
                            ) &
                            ", got " &
                            integer'image(v_quant_lane_0)
                        severity failure;

                    assert
                        v_quant_lane_1 =
                        C_EXPECTED_QUANT_LANE_1(output_index)
                        report
                            "FAIL: quantized lane 1 mismatch at output " &
                            integer'image(output_index) &
                            ". Expected " &
                            integer'image(
                                C_EXPECTED_QUANT_LANE_1(
                                    output_index
                                )
                            ) &
                            ", got " &
                            integer'image(v_quant_lane_1)
                        severity failure;

                    output_index <= output_index + 1;
                end if;
            end if;
        end if;
    end process;

    --------------------------------------------------------------------
    -- Both raw and quantized outputs must remain stable while stalled.
    --------------------------------------------------------------------

    output_stability_process : process(clk)
        variable v_holding : boolean := false;

        variable v_held_raw :
            std_logic_vector(
                C_C_PAR * 32 - 1 downto 0
            ) :=
            (others => '0');

        variable v_held_quantized :
            std_logic_vector(
                C_C_PAR * 8 - 1 downto 0
            ) :=
            (others => '0');
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                v_holding := false;
                v_held_raw := (others => '0');
                v_held_quantized := (others => '0');

            else
                if v_holding then
                    assert o_valid = '1'
                        report
                            "FAIL: o_valid dropped while stalled."
                        severity failure;

                    assert o_acc_valid = '1'
                        report
                            "FAIL: o_acc_valid dropped while stalled."
                        severity failure;

                    assert o_acc_data = v_held_raw
                        report
                            "FAIL: raw accumulator changed while stalled."
                        severity failure;

                    assert o_data = v_held_quantized
                        report
                            "FAIL: quantized output changed while stalled."
                        severity failure;
                end if;

                if
                    o_valid = '1' and
                    i_acc_ready = '0'
                then
                    v_holding := true;
                    v_held_raw := o_acc_data;
                    v_held_quantized := o_data;
                else
                    v_holding := false;
                end if;
            end if;
        end if;
    end process;

    --------------------------------------------------------------------
    -- Completion
    --------------------------------------------------------------------

    completion_process : process
    begin
        wait until rst_n = '1';

        wait until
            input_index = C_INPUT_COUNT and
            weight_index = C_WEIGHT_COUNT and
            output_index = C_OUTPUT_COUNT;

        wait until rising_edge(clk);
        wait until rising_edge(clk);

        assert output_stall_observed = '1'
            report
                "FAIL: the backpressure condition was not exercised."
            severity failure;

        assert input_index = C_INPUT_COUNT
            report
                "FAIL: the complete input frame was not accepted."
            severity failure;

        assert weight_index = C_WEIGHT_COUNT
            report
                "FAIL: both output-group weight batches were not accepted."
            severity failure;

        for cycle in 1 to 8 loop
            wait until rising_edge(clk);

            assert not (
                o_valid = '1' and
                i_acc_ready = '1'
            )
                report
                    "FAIL: unexpected output after completion."
                severity failure;
        end loop;

        report
            "PASS: bias, ReLU, requantization, rounding, saturation, " &
            "channel selection and output backpressure are correct."
            severity note;

        stop;
        wait;
    end process;

    watchdog_process : process
    begin
        wait for 20 us;

        assert false
            report
                "TIMEOUT: requantization test did not complete. " &
                "Inputs=" &
                integer'image(input_index) &
                "/" &
                integer'image(C_INPUT_COUNT) &
                ", weights=" &
                integer'image(weight_index) &
                "/" &
                integer'image(C_WEIGHT_COUNT) &
                ", outputs=" &
                integer'image(output_index) &
                "/" &
                integer'image(C_OUTPUT_COUNT)
            severity failure;
    end process;

end architecture sim;