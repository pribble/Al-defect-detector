library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.env.all;

use work.conv_tb_pkg.all;

entity tb_conv_layer_horizontal_sliding is
end entity tb_conv_layer_horizontal_sliding;


architecture sim of tb_conv_layer_horizontal_sliding is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_C_IN   : positive := 1;
    constant C_W_IN   : positive := 4;
    constant C_C_PAR  : positive := 2;
    constant C_KERNEL : positive := 3;
    constant C_H_IN  : positive := C_KERNEL;

    constant C_LINE_SIZE : positive :=
        C_W_IN * C_C_IN;

    constant C_INITIAL_FILL_SIZE : positive :=
        (C_KERNEL - 1) * C_LINE_SIZE;

    constant C_PRIME_FILL_SIZE : positive :=
        C_KERNEL * C_C_IN;

    constant C_STREAM_FILL_SIZE : positive :=
        (C_W_IN - C_KERNEL) * C_C_IN;

    constant C_TOTAL_ACTIVATIONS : positive :=
        C_INITIAL_FILL_SIZE +
        C_PRIME_FILL_SIZE +
        C_STREAM_FILL_SIZE;

    constant C_KERNEL_SIZE : positive :=
        C_KERNEL * C_KERNEL;

    constant C_WEIGHT_FILL_SIZE : positive :=
        C_C_PAR * C_KERNEL_SIZE;

    signal clk   : std_logic := '0';
    signal rst_n : std_logic := '0';

    signal i_valid : std_logic := '0';
    signal i_ready : std_logic;
    signal i_data  : std_logic_vector(7 downto 0) :=
        (others => '0');

    signal i_weight_valid : std_logic := '0';
    signal o_weight_ready : std_logic;
    signal i_weight_data  : std_logic_vector(7 downto 0) :=
        (others => '0');

    signal o_acc_valid : std_logic;
    signal o_acc_data  :
        std_logic_vector(C_C_PAR * 32 - 1 downto 0);

    signal acc_valid_count : natural := 0;

begin

    clk <= not clk after C_CLK_PERIOD / 2;


    dut : entity work.conv_layer
        generic map (
            G_C_IN    => C_C_IN,
            G_C_OUT   => C_C_PAR,
            G_W_IN    => C_W_IN,
            G_H_IN    => C_H_IN,
            G_C_PAR   => C_C_PAR,
            G_KERNEL  => C_KERNEL,
    
            G_PADDING => 0,
            G_STRIDE  => 1
        )
        port map (
            clk            => clk,
            rst_n          => rst_n,

            i_valid        => i_valid,
            i_ready        => i_ready,
            i_data         => i_data,

            i_weight_valid => i_weight_valid,
            o_weight_ready => o_weight_ready,
            i_weight_data  => i_weight_data,

            o_acc_valid    => o_acc_valid,
            o_acc_data     => o_acc_data,
            cfg_we    => '0',
            cfg_sel   => (others => '0'),
            cfg_addr  => (others => '0'),
            cfg_wdata => (others => '0'),

            o_valid => open,
            o_data  => open,
            o_done => open,
            i_acc_ready => '1'
        );


    reset_process : process
    begin
        rst_n <= '0';

        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until falling_edge(clk);

        rst_n <= '1';

        wait;
    end process;


    activation_driver : process
    begin
        wait until rst_n = '1';

        send_activation_range(
            p_clk         => clk,
            p_valid       => i_valid,
            p_ready       => i_ready,
            p_data        => i_data,
            p_first_value => 0,
            p_count       => C_TOTAL_ACTIVATIONS
        );

        wait;
    end process;


    weight_driver : process
    begin
        wait until rst_n = '1';

        for index in 0 to C_KERNEL_SIZE - 1 loop
            send_weight_byte(
                p_clk          => clk,
                p_weight_valid => i_weight_valid,
                p_weight_ready => o_weight_ready,
                p_weight_data  => i_weight_data,
                p_value        => 1
            );
        end loop;

        for index in 0 to C_KERNEL_SIZE - 1 loop
            send_weight_byte(
                p_clk          => clk,
                p_weight_valid => i_weight_valid,
                p_weight_ready => o_weight_ready,
                p_weight_data  => i_weight_data,
                p_value        => 255
            );
        end loop;

        wait;
    end process;


    result_counter_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                acc_valid_count <= 0;

            elsif o_acc_valid = '1' then
                acc_valid_count <= acc_valid_count + 1;
            end if;
        end if;
    end process;


    result_process : process
    begin
        wait until o_acc_valid = '1';
        wait for 1 ns;

        assert signed(o_acc_data(31 downto 0)) =
               to_signed(45, 32)
            report
                "FAIL: window 0 lane 0 was not 45."
            severity failure;

        assert signed(o_acc_data(63 downto 32)) =
               to_signed(-45, 32)
            report
                "FAIL: window 0 lane 1 was not -45."
            severity failure;

        wait until o_acc_valid = '0';
        wait until o_acc_valid = '1';
        wait for 1 ns;

        assert signed(o_acc_data(31 downto 0)) =
               to_signed(54, 32)
            report
                "FAIL: window 1 lane 0 was not 54."
            severity failure;

        assert signed(o_acc_data(63 downto 32)) =
               to_signed(-54, 32)
            report
                "FAIL: window 1 lane 1 was not -54."
            severity failure;

        wait until o_acc_valid = '0';

        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for 1 ns;

        assert acc_valid_count = 2
            report
                "FAIL: calculation did not produce exactly two results."
            severity failure;

        assert i_ready = '0'
            report
                "FAIL: activation input remained ready after row completion."
            severity failure;

        report
            "PASS: horizontal sliding produced windows 45/-45 and 54/-54."
            severity note;

        stop;
        wait;
    end process;


    timeout_process : process
    begin
        wait for 2 us;

        assert false
            report
                "FAIL: horizontal sliding test timed out."
            severity failure;
    end process;

end architecture sim;