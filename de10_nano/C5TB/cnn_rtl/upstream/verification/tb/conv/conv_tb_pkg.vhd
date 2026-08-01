library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package conv_tb_pkg is

    procedure reset_dut (
        signal p_clk   : in  std_logic;
        signal p_rst_n : out std_logic;
        signal p_valid : out std_logic;
        signal p_data  : out std_logic_vector;
        constant p_cycles : in positive := 3
    );

    procedure send_activation_byte (
        signal p_clk   : in  std_logic;
        signal p_valid : out std_logic;
        signal p_ready : in  std_logic;
        signal p_data  : out std_logic_vector;
        constant p_value : in natural
    );

    procedure send_activation_range (
        signal p_clk   : in  std_logic;
        signal p_valid : out std_logic;
        signal p_ready : in  std_logic;
        signal p_data  : out std_logic_vector;
        constant p_first_value : in natural;
        constant p_count       : in positive
    );

    procedure send_weight_byte (
        signal p_clk          : in  std_logic;
        signal p_weight_valid : out std_logic;
        signal p_weight_ready : in  std_logic;
        signal p_weight_data  : out std_logic_vector;
        constant p_value      : in  natural
    );

    procedure send_weight_range (
        signal p_clk          : in  std_logic;
        signal p_weight_valid : out std_logic;
        signal p_weight_ready : in  std_logic;
        signal p_weight_data  : out std_logic_vector;
        constant p_first_value : in natural;
        constant p_count       : in positive
    );

end package conv_tb_pkg;


package body conv_tb_pkg is

    procedure reset_dut (
        signal p_clk   : in  std_logic;
        signal p_rst_n : out std_logic;
        signal p_valid : out std_logic;
        signal p_data  : out std_logic_vector;
        constant p_cycles : in positive := 3
    ) is
    begin
        p_rst_n <= '0';
        p_valid <= '0';
        p_data  <= (p_data'range => '0');

        for cycle in 1 to p_cycles loop
            wait until rising_edge(p_clk);
        end loop;

        wait until falling_edge(p_clk);
        p_rst_n <= '1';
    end procedure;


    procedure send_activation_byte (
        signal p_clk   : in  std_logic;
        signal p_valid : out std_logic;
        signal p_ready : in  std_logic;
        signal p_data  : out std_logic_vector;
        constant p_value : in natural
    ) is
    begin
        p_data <= std_logic_vector(
            to_unsigned(p_value, p_data'length)
        );

        p_valid <= '1';

        loop
            wait until rising_edge(p_clk);
            exit when p_ready = '1';
        end loop;

        p_valid <= '0';
    end procedure;


    procedure send_activation_range (
        signal p_clk   : in  std_logic;
        signal p_valid : out std_logic;
        signal p_ready : in  std_logic;
        signal p_data  : out std_logic_vector;
        constant p_first_value : in natural;
        constant p_count       : in positive
    ) is
    begin
        for offset in 0 to p_count - 1 loop
            send_activation_byte(
                p_clk   => p_clk,
                p_valid => p_valid,
                p_ready => p_ready,
                p_data  => p_data,
                p_value => p_first_value + offset
            );
        end loop;
    end procedure;


    procedure send_weight_byte (
        signal p_clk          : in  std_logic;
        signal p_weight_valid : out std_logic;
        signal p_weight_ready : in  std_logic;
        signal p_weight_data  : out std_logic_vector;
        constant p_value      : in  natural
    ) is
    begin
        p_weight_data <= std_logic_vector(
            to_unsigned(p_value, p_weight_data'length)
        );

        p_weight_valid <= '1';

        loop
            wait until rising_edge(p_clk);
            exit when p_weight_ready = '1';
        end loop;

        p_weight_valid <= '0';
    end procedure;


    procedure send_weight_range (
        signal p_clk          : in  std_logic;
        signal p_weight_valid : out std_logic;
        signal p_weight_ready : in  std_logic;
        signal p_weight_data  : out std_logic_vector;
        constant p_first_value : in natural;
        constant p_count       : in positive
    ) is
    begin
        for offset in 0 to p_count - 1 loop
            send_weight_byte(
                p_clk          => p_clk,
                p_weight_valid => p_weight_valid,
                p_weight_ready => p_weight_ready,
                p_weight_data  => p_weight_data,
                p_value        => p_first_value + offset
            );
        end loop;
    end procedure;

end package body conv_tb_pkg;