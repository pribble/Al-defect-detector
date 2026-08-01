library ieee;
use ieee.std_logic_1164.all;

library std;
use std.env.all;

use work.conv_tb_pkg.all;

entity tb_conv_layer_weight_filling is
end entity tb_conv_layer_weight_filling;


architecture sim of tb_conv_layer_weight_filling is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_C_IN   : positive := 1;
    constant C_W_IN   : positive := 4;
    constant C_C_PAR  : positive := 2;
    constant C_KERNEL : positive := 3;
    constant C_H_IN  : positive := C_KERNEL;

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


    stimulus_process : process
    begin
        i_weight_valid <= '0';
        i_weight_data  <= (others => '0');

        reset_dut(
            p_clk   => clk,
            p_rst_n => rst_n,
            p_valid => i_valid,
            p_data  => i_data
        );

        wait until rising_edge(clk);
        wait for 1 ns;

        assert o_weight_ready = '1'
            report "FAIL: weight filling did not start."
            severity failure;

        for value in 0 to C_WEIGHT_FILL_SIZE - 2 loop

            if value = 5 then
                i_weight_valid <= '0';

                wait until rising_edge(clk);
                wait for 1 ns;

                assert o_weight_ready = '1'
                    report "FAIL: weight filling stopped during an invalid cycle."
                    severity failure;

                wait until rising_edge(clk);
                wait for 1 ns;

                assert o_weight_ready = '1'
                    report "FAIL: invalid cycles changed the weight count."
                    severity failure;
            end if;

            send_weight_byte(
                p_clk          => clk,
                p_weight_valid => i_weight_valid,
                p_weight_ready => o_weight_ready,
                p_weight_data  => i_weight_data,
                p_value        => value
            );

            wait for 1 ns;

            assert o_weight_ready = '1'
                report "FAIL: weight filling stopped before eighteen weights."
                severity failure;

        end loop;

        send_weight_byte(
            p_clk          => clk,
            p_weight_valid => i_weight_valid,
            p_weight_ready => o_weight_ready,
            p_weight_data  => i_weight_data,
            p_value        => C_WEIGHT_FILL_SIZE - 1
        );

        wait for 1 ns;

        assert o_weight_ready = '0'
            report "FAIL: weight filling did not stop after eighteen weights."
            severity failure;

        report
            "PASS: S_WEIGHT_FILLING accepted exactly eighteen weights."
            severity note;

        stop;
        wait;
    end process;


    timeout_process : process
    begin
        wait for 1 us;

        assert false
            report "FAIL: weight filling test timed out."
            severity failure;
    end process;

end architecture sim;