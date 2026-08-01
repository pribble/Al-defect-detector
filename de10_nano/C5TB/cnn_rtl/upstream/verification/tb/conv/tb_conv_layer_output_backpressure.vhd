library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.env.all;

use work.conv_tb_pkg.all;

entity tb_conv_layer_output_backpressure is
end entity tb_conv_layer_output_backpressure;


architecture sim of tb_conv_layer_output_backpressure is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_C_IN   : positive := 1;
    constant C_W_IN   : positive := 4;
    constant C_H_IN   : positive := 4;
    constant C_C_PAR  : positive := 2;
    constant C_KERNEL : positive := 3;

    constant C_TOTAL_ACTIVATIONS : positive :=
        C_H_IN * C_W_IN * C_C_IN;

    constant C_KERNEL_SIZE : positive :=
        C_KERNEL * C_KERNEL;

    constant C_WEIGHT_FILL_SIZE : positive :=
        C_C_PAR * C_KERNEL_SIZE;

    constant C_RESULT_COUNT : positive := 4;
    constant C_STALL_CYCLES : positive := 4;

    type t_expected_array is array (
        natural range <>
    ) of integer;

    constant C_EXPECTED_LANE_0 :
        t_expected_array(0 to C_RESULT_COUNT - 1) :=
        (
            45,
            54,
            81,
            90
        );

    constant C_EXPECTED_LANE_1 :
        t_expected_array(0 to C_RESULT_COUNT - 1) :=
        (
            -45,
            -54,
            -81,
            -90
        );

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

    signal i_acc_ready : std_logic := '0';
    signal o_acc_valid : std_logic;
    signal o_done : std_logic;
    signal o_acc_data  :
        std_logic_vector(C_C_PAR * 32 - 1 downto 0);
    signal done_seen : std_logic := '0';
    signal done_count : natural range 0 to 2 := 0;  
    signal activation_accept_count : natural := 0;
    signal weight_accept_count     : natural := 0;
    signal output_accept_count     : natural := 0;

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

            i_acc_ready    => i_acc_ready,
            o_acc_valid    => o_acc_valid,
            cfg_we    => '0',
            cfg_sel   => (others => '0'),
            cfg_addr  => (others => '0'),
            cfg_wdata => (others => '0'),

            o_valid => open,
            o_data  => open,
            o_done => o_done,
            o_acc_data     => o_acc_data
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

        -- Lane 0: nine weights of +1.
        for index in 0 to C_KERNEL_SIZE - 1 loop
            send_weight_byte(
                p_clk          => clk,
                p_weight_valid => i_weight_valid,
                p_weight_ready => o_weight_ready,
                p_weight_data  => i_weight_data,
                p_value        => 1
            );
        end loop;

        -- Lane 1: nine weights of -1.
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

    done_monitor_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                done_seen  <= '0';
                done_count <= 0;

            elsif o_done = '1' then
                done_seen <= '1';

                if done_count < 2 then
                    done_count <= done_count + 1;
                end if;
            end if;
        end if;
    end process;

    transfer_monitor_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                activation_accept_count <= 0;
                weight_accept_count     <= 0;
                output_accept_count     <= 0;

            else
                if i_valid = '1' and i_ready = '1' then
                    activation_accept_count <=
                        activation_accept_count + 1;
                end if;

                if i_weight_valid = '1' and
                   o_weight_ready = '1' then

                    weight_accept_count <=
                        weight_accept_count + 1;
                end if;

                if o_acc_valid = '1' and
                   i_acc_ready = '1' then

                    output_accept_count <=
                        output_accept_count + 1;
                end if;
            end if;
        end if;
    end process;


    output_consumer_process : process
        variable v_held_data :
            std_logic_vector(C_C_PAR * 32 - 1 downto 0);
    begin
        i_acc_ready <= '0';

        wait until rst_n = '1';

        for result_index in 0 to C_RESULT_COUNT - 1 loop

            -- Wait for the completed convolution window.
            wait until o_acc_valid = '1';
            wait for 1 ns;

            v_held_data := o_acc_data;

            assert signed(o_acc_data(31 downto 0)) =
                   to_signed(
                       C_EXPECTED_LANE_0(result_index),
                       32
                   )
                report
                    "FAIL: incorrect lane 0 result at index " &
                    integer'image(result_index)
                severity failure;

            assert signed(o_acc_data(63 downto 32)) =
                   to_signed(
                       C_EXPECTED_LANE_1(result_index),
                       32
                   )
                report
                    "FAIL: incorrect lane 1 result at index " &
                    integer'image(result_index)
                severity failure;

            -- Stall the output for several complete clock cycles.
            for stall_cycle in 1 to C_STALL_CYCLES loop
                wait until rising_edge(clk);
                wait for 1 ns;

                assert o_acc_valid = '1'
                    report
                        "FAIL: o_acc_valid cleared while the output " &
                        "was stalled."
                    severity failure;

                assert o_acc_data = v_held_data
                    report
                        "FAIL: o_acc_data changed while the output " &
                        "was stalled."
                    severity failure;
            end loop;

            -- Allow exactly one output handshake.
            wait until falling_edge(clk);
            i_acc_ready <= '1';

            wait until rising_edge(clk);
            wait for 1 ns;

            assert o_acc_valid = '0'
                report
                    "FAIL: o_acc_valid did not clear after the " &
                    "output handshake."
                severity failure;

            -- Stall the next result as well.
            wait until falling_edge(clk);
            i_acc_ready <= '0';

        end loop;

        -- Allow time to detect an unwanted extra result.
        for cycle in 1 to 5 loop
            wait until rising_edge(clk);
        end loop;

        wait for 1 ns;

        assert activation_accept_count =
               C_TOTAL_ACTIVATIONS
            report
                "FAIL: the DUT did not accept exactly sixteen " &
                "activations."
            severity failure;

        assert weight_accept_count =
               C_WEIGHT_FILL_SIZE
            report
                "FAIL: the DUT did not accept exactly eighteen " &
                "weights."
            severity failure;

        assert output_accept_count =
               C_RESULT_COUNT
            report
                "FAIL: the DUT did not complete exactly four " &
                "output handshakes."
            severity failure;

        assert o_acc_valid = '0'
            report
                "FAIL: an unexpected output remained pending after " &
                "the final result."
            severity failure;

        assert i_ready = '1'
            report
                "FAIL: activation input did not become ready for the next frame."
            severity failure;
        
        if done_seen = '0' then
            wait until done_seen = '1';
        end if;

        wait for 1 ns;

        assert done_count = 1
            report
                "FAIL: expected exactly one o_done pulse."
            severity failure;

        assert o_done = '0'
            report
                "FAIL: o_done remained asserted for more than one clock."
            severity failure;

        assert i_ready = '1'
            report
                "FAIL: activation input did not become ready for the next frame."
            severity failure;

        assert o_weight_ready = '1'
            report
                "FAIL: weight input did not become ready for the next frame."
            severity failure;
        assert o_weight_ready = '1'
            report
                "FAIL: weight input did not become ready for the next frame."
            severity failure;

        report
            "PASS: all four outputs remained valid and stable under " &
            "backpressure."
            severity note;

        stop;
        wait;
    end process;


    timeout_process : process
    begin
        wait for 5 us;

        assert false
            report
                "FAIL: output backpressure test timed out."
            severity failure;
    end process;

end architecture sim;