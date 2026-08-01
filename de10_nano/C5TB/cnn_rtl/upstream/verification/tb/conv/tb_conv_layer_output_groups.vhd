library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.env.all;

entity tb_conv_layer_output_groups is
end entity;

architecture sim of tb_conv_layer_output_groups is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_C_IN    : positive := 1;
    constant C_C_OUT   : positive := 4;
    constant C_W_IN    : positive := 4;
    constant C_H_IN    : positive := 3;
    constant C_C_PAR   : positive := 2;
    constant C_KERNEL  : positive := 3;
    constant C_PADDING : natural  := 0;
    constant C_STRIDE  : positive := 1;

    constant C_INPUT_COUNT  : positive :=
        C_C_IN * C_W_IN * C_H_IN;

    constant C_WEIGHT_COUNT : positive :=
        C_C_OUT * C_KERNEL * C_KERNEL;

    constant C_OUTPUT_COUNT : positive := 4;

    type t_integer_array is array (
        natural range <>
    ) of integer;

    -- Input image:
    --
    --  1   2   3   4
    --  5   6   7   8
    --  9  10  11  12

    constant C_INPUT : t_integer_array(
        0 to C_INPUT_COUNT - 1
    ) := (
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    );

    -- Output ordering:
    --
    -- group 0, column 0
    -- group 0, column 1
    -- group 1, column 0
    -- group 1, column 1

    constant C_EXPECTED_LANE_0 :
        t_integer_array(0 to C_OUTPUT_COUNT - 1) := (
             54,
             63,
            -54,
            -63
        );

    constant C_EXPECTED_LANE_1 :
        t_integer_array(0 to C_OUTPUT_COUNT - 1) := (
            108,
            126,
            162,
            189
        );

    signal clk : std_logic := '0';
    signal rst_n : std_logic := '0';

    signal i_valid : std_logic := '0';
    signal i_ready : std_logic;
    signal i_data : std_logic_vector(7 downto 0) :=
        (others => '0');

    signal i_weight_valid : std_logic := '0';
    signal o_weight_ready : std_logic;
    signal i_weight_data : std_logic_vector(7 downto 0) :=
        (others => '0');

    signal i_acc_ready : std_logic := '1';
    signal o_acc_valid : std_logic;
    signal o_acc_data :
        std_logic_vector(C_C_PAR * 32 - 1 downto 0);

    signal input_index :
        natural range 0 to C_INPUT_COUNT := 0;

    signal weight_index :
        natural range 0 to C_WEIGHT_COUNT := 0;

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
            clk => clk,
            rst_n => rst_n,

            i_valid => i_valid,
            i_ready => i_ready,
            i_data => i_data,

            i_weight_valid => i_weight_valid,
            o_weight_ready => o_weight_ready,
            i_weight_data => i_weight_data,

            i_acc_ready => i_acc_ready,
            o_acc_valid => o_acc_valid,
            cfg_we    => '0',
            cfg_sel   => (others => '0'),
            cfg_addr  => (others => '0'),
            cfg_wdata => (others => '0'),

            o_valid => open,
            o_data  => open,
            o_done => open,
            o_acc_data => o_acc_data
        );

    reset_process : process
    begin
        rst_n <= '0';

        wait for 4 * C_CLK_PERIOD;
        wait until rising_edge(clk);

        rst_n <= '1';

        wait;
    end process;

    --------------------------------------------------------------------
    -- Activation stream
    --------------------------------------------------------------------

    input_comb_process : process(all)
    begin
        i_valid <= '0';
        i_data <= (others => '0');

        if input_index < C_INPUT_COUNT then
            i_valid <= '1';

            i_data <= std_logic_vector(
                to_unsigned(
                    C_INPUT(input_index),
                    i_data'length
                )
            );
        end if;
    end process;

    input_counter_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                input_index <= 0;

            elsif i_valid = '1' and i_ready = '1' then
                input_index <= input_index + 1;
            end if;
        end if;
    end process;

    --------------------------------------------------------------------
    -- Weight stream
    --
    -- Group 0:
    --   lane 0: all weights =  1
    --   lane 1: all weights =  2
    --
    -- Group 1:
    --   lane 0: all weights = -1
    --   lane 1: all weights =  3
    --------------------------------------------------------------------

    weight_comb_process : process(all)
        variable v_weight : integer;
    begin
        i_weight_valid <= '0';
        i_weight_data <= (others => '0');

        if weight_index < C_WEIGHT_COUNT then
            i_weight_valid <= '1';

            if weight_index < 9 then
                v_weight := 1;

            elsif weight_index < 18 then
                v_weight := 2;

            elsif weight_index < 27 then
                v_weight := -1;

            else
                v_weight := 3;
            end if;

            i_weight_data <= std_logic_vector(
                to_signed(
                    v_weight,
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

            elsif i_weight_valid = '1' and
                  o_weight_ready = '1' then

                weight_index <= weight_index + 1;
            end if;
        end if;
    end process;

    --------------------------------------------------------------------
    -- Output checking
    --------------------------------------------------------------------

    output_check_process : process
        variable v_output_index : natural := 0;
        variable v_lane_0 : integer;
        variable v_lane_1 : integer;
    begin
        wait until rst_n = '1';

        while v_output_index < C_OUTPUT_COUNT loop
            wait until rising_edge(clk);

            if o_acc_valid = '1' and
               i_acc_ready = '1' then

                v_lane_0 := to_integer(
                    signed(o_acc_data(31 downto 0))
                );

                v_lane_1 := to_integer(
                    signed(o_acc_data(63 downto 32))
                );

                assert v_lane_0 =
                       C_EXPECTED_LANE_0(v_output_index)
                    report
                        "Lane 0 mismatch at output " &
                        integer'image(v_output_index) &
                        ". Expected " &
                        integer'image(
                            C_EXPECTED_LANE_0(
                                v_output_index
                            )
                        ) &
                        ", got " &
                        integer'image(v_lane_0)
                    severity failure;

                assert v_lane_1 =
                       C_EXPECTED_LANE_1(v_output_index)
                    report
                        "Lane 1 mismatch at output " &
                        integer'image(v_output_index) &
                        ". Expected " &
                        integer'image(
                            C_EXPECTED_LANE_1(
                                v_output_index
                            )
                        ) &
                        ", got " &
                        integer'image(v_lane_1)
                    severity failure;

                v_output_index := v_output_index + 1;
            end if;
        end loop;

        assert input_index = C_INPUT_COUNT
            report
                "The complete activation frame was not accepted."
            severity failure;

        assert weight_index = C_WEIGHT_COUNT
            report
                "Both output-group weight sets were not accepted."
            severity failure;

        report
            "PASS: output groups swept horizontally in group-major order."
            severity note;

        stop;
        wait;
    end process;

    watchdog_process : process
    begin
        wait for 20 us;

        assert false
            report
                "TIMEOUT: output-group test did not complete."
            severity failure;
    end process;

end architecture;