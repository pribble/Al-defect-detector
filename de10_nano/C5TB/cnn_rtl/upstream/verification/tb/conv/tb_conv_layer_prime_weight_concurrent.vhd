library ieee;
use ieee.std_logic_1164.all;

library std;
use std.env.all;

use work.conv_tb_pkg.all;

entity tb_conv_layer_prime_weight_concurrent is
end entity tb_conv_layer_prime_weight_concurrent;


architecture sim of tb_conv_layer_prime_weight_concurrent is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_C_IN   : positive := 1;
    constant C_W_IN   : positive := 4;
    constant C_C_PAR  : positive := 2;
    constant C_KERNEL : positive := 3;
    constant C_H_IN  : positive := C_KERNEL;
    

    constant C_INITIAL_FILL_SIZE : positive :=
        (C_KERNEL - 1) * C_W_IN * C_C_IN;

    constant C_PRIME_FILL_SIZE : positive :=
        C_KERNEL * C_C_IN;

    constant C_TOTAL_ACTIVATIONS : positive :=
        C_INITIAL_FILL_SIZE + C_PRIME_FILL_SIZE;

    constant C_WEIGHT_FILL_SIZE : positive :=
        C_C_PAR * C_KERNEL * C_KERNEL;

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

    signal activation_done : std_logic := '0';
    signal weight_done     : std_logic := '0';

    signal activation_accept_count : natural := 0;
    signal weight_accept_count     : natural := 0;
    signal prime_overlap_count     : natural := 0;

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
            o_acc_valid => open,
            o_acc_data  => open,
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

        activation_done <= '1';

        wait;
    end process;


    weight_driver : process
    begin
        wait until rst_n = '1';

        send_weight_range(
            p_clk          => clk,
            p_weight_valid => i_weight_valid,
            p_weight_ready => o_weight_ready,
            p_weight_data  => i_weight_data,
            p_first_value  => 0,
            p_count        => C_WEIGHT_FILL_SIZE
        );

        weight_done <= '1';

        wait;
    end process;


    monitor_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                activation_accept_count <= 0;
                weight_accept_count     <= 0;
                prime_overlap_count     <= 0;

            else
                if i_valid = '1' and i_ready = '1' then

                    if activation_accept_count >=
                       C_INITIAL_FILL_SIZE and
                       activation_accept_count <
                       C_TOTAL_ACTIVATIONS then

                        if i_weight_valid = '1' and
                           o_weight_ready = '1' then

                            prime_overlap_count <=
                                prime_overlap_count + 1;
                        end if;
                    end if;

                    activation_accept_count <=
                        activation_accept_count + 1;
                end if;

                if i_weight_valid = '1' and
                   o_weight_ready = '1' then

                    weight_accept_count <=
                        weight_accept_count + 1;
                end if;
            end if;
        end if;
    end process;


    result_process : process
    begin
        wait until activation_done = '1' and
                   weight_done = '1';

        wait for 1 ns;

        assert activation_accept_count =
               C_TOTAL_ACTIVATIONS
            report
                "FAIL: incorrect activation transfer count."
            severity failure;

        assert weight_accept_count =
               C_WEIGHT_FILL_SIZE
            report
                "FAIL: incorrect weight transfer count."
            severity failure;

        assert prime_overlap_count =
               C_PRIME_FILL_SIZE
            report
                "FAIL: prime-K-line filling and weight filling did not run concurrently."
            severity failure;

        assert i_ready = '0'
            report
                "FAIL: activation input remained ready after prime filling."
            severity failure;

        assert o_weight_ready = '0'
            report
                "FAIL: weight input remained ready after weight filling."
            severity failure;

        report
            "PASS: all three prime-K activation transfers overlapped with weight transfers."
            severity note;

        stop;
        wait;
    end process;


    timeout_process : process
    begin
        wait for 1 us;

        assert false
            report
                "FAIL: concurrent prime/weight test timed out."
            severity failure;
    end process;

end architecture sim;