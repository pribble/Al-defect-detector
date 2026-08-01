library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.env.all;

entity tb_conv_layer_traversal is
    generic (
        G_NAME : string := "traversal";

        G_C_IN    : positive := 1;
        G_C_OUT   : positive := 4;
        G_W_IN    : positive := 5;
        G_H_IN    : positive := 5;
        G_C_PAR   : positive := 2;
        G_KERNEL  : positive := 3;
        G_PADDING : natural  := 1;
        G_STRIDE  : positive := 2;

        G_INPUT_GAP_PERIOD  : natural := 0;
        G_WEIGHT_GAP_PERIOD : natural := 0;
        G_STALL_PERIOD      : natural := 0;

        G_TIMEOUT_CYCLES : positive := 500000
    );
end entity tb_conv_layer_traversal;

architecture sim of tb_conv_layer_traversal is

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

    constant C_INPUT_COUNT : positive :=
        G_H_IN * G_W_IN * G_C_IN;

    constant C_WEIGHT_BATCH_SIZE : positive :=
        G_C_PAR * C_KERNEL_SIZE;

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

    constant C_WEIGHT_BYTE_COUNT : positive :=
        C_WEIGHT_BATCH_COUNT *
        C_WEIGHT_BATCH_SIZE;

    constant C_OUTPUT_TRANSFER_COUNT : positive :=
        C_OUTPUT_HEIGHT *
        C_OUTPUT_GROUPS *
        C_OUTPUT_WIDTH;

    function activation_value (
        input_row     : natural;
        input_col     : natural;
        input_channel : natural
    ) return integer is
    begin
        return
            (
                input_row *
                G_W_IN +
                input_col +
                1
            ) *
            (
                input_channel +
                1
            );
    end function;

    function weight_value (
        output_group  : natural;
        lane          : natural;
        input_channel : natural;
        kernel_row    : natural;
        kernel_col    : natural
    ) return integer is
        variable v_code : integer;
    begin
        v_code :=
            (
                integer(output_group) * 4 +
                integer(lane) * 5 +
                integer(input_channel) * 3 +
                integer(kernel_row) * 2 +
                integer(kernel_col)
            ) mod 7;

        return v_code - 3;
    end function;

    function expected_accumulator (
        output_row   : natural;
        output_col   : natural;
        output_group : natural;
        lane         : natural
    ) return integer is

        variable v_sum : integer := 0;

        variable v_input_row : integer;
        variable v_input_col : integer;

        variable v_activation : integer;
        variable v_weight : integer;

    begin
        for input_channel in 0 to G_C_IN - 1 loop

            for kernel_row in 0 to G_KERNEL - 1 loop

                for kernel_col in 0 to G_KERNEL - 1 loop

                    v_input_row :=
                        integer(output_row) *
                        G_STRIDE +
                        integer(kernel_row) -
                        G_PADDING;

                    v_input_col :=
                        integer(output_col) *
                        G_STRIDE +
                        integer(kernel_col) -
                        G_PADDING;

                    if
                        v_input_row >= 0 and
                        v_input_row < G_H_IN and
                        v_input_col >= 0 and
                        v_input_col < G_W_IN
                    then
                        v_activation :=
                            activation_value(
                                natural(v_input_row),
                                natural(v_input_col),
                                input_channel
                            );
                    else
                        v_activation := 0;
                    end if;

                    v_weight :=
                        weight_value(
                            output_group,
                            lane,
                            input_channel,
                            kernel_row,
                            kernel_col
                        );

                    v_sum :=
                        v_sum +
                        v_activation *
                        v_weight;

                end loop;

            end loop;

        end loop;

        return v_sum;
    end function;

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

    signal i_acc_ready : std_logic := '1';
    signal o_acc_valid : std_logic;

    signal o_acc_data :
        std_logic_vector(
            G_C_PAR * 32 - 1 downto 0
        );

    signal input_index :
        natural range 0 to C_INPUT_COUNT := 0;

    signal weight_byte_index :
        natural range 0 to C_WEIGHT_BYTE_COUNT := 0;

    signal output_transfer_index :
        natural range 0 to C_OUTPUT_TRANSFER_COUNT := 0;

    signal input_cycle_count : natural := 0;
    signal weight_cycle_count : natural := 0;
    signal output_cycle_count : natural := 0;

begin

    assert G_C_OUT mod G_C_PAR = 0
        report
            "G_C_OUT must be divisible by G_C_PAR"
        severity failure;

    assert G_PADDING <= G_KERNEL - 2
        report
            "This RTL currently requires " &
            "G_PADDING <= G_KERNEL - 2"
        severity failure;

    assert G_W_IN > G_KERNEL
        report
            "This RTL currently requires " &
            "G_W_IN > G_KERNEL"
        severity failure;

    assert
        G_INPUT_GAP_PERIOD = 0 or
        G_INPUT_GAP_PERIOD > 1
        report
            "G_INPUT_GAP_PERIOD must be zero " &
            "or greater than one"
        severity failure;

    assert
        G_WEIGHT_GAP_PERIOD = 0 or
        G_WEIGHT_GAP_PERIOD > 1
        report
            "G_WEIGHT_GAP_PERIOD must be zero " &
            "or greater than one"
        severity failure;

    assert
        G_STALL_PERIOD = 0 or
        G_STALL_PERIOD > 1
        report
            "G_STALL_PERIOD must be zero " &
            "or greater than one"
        severity failure;

    assert
        activation_value(
            G_H_IN - 1,
            G_W_IN - 1,
            G_C_IN - 1
        ) <= 255
        report
            "The deterministic activation pattern " &
            "exceeds uint8"
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

    cycle_counter_process : process(clk)
    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                input_cycle_count <= 0;
                weight_cycle_count <= 0;
                output_cycle_count <= 0;

            else
                input_cycle_count <=
                    input_cycle_count + 1;

                weight_cycle_count <=
                    weight_cycle_count + 1;

                output_cycle_count <=
                    output_cycle_count + 1;

            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Activation source
    --------------------------------------------------------------------

    input_driver_process : process(all)

        variable v_flat_pixel : natural;

        variable v_input_row : natural;
        variable v_input_col : natural;
        variable v_input_channel : natural;

        variable v_value : integer;
        variable v_gap : boolean;

    begin
        i_valid <= '0';
        i_data <= (others => '0');

        v_gap := false;

        if G_INPUT_GAP_PERIOD > 0 then

            if
                input_cycle_count mod
                G_INPUT_GAP_PERIOD =
                G_INPUT_GAP_PERIOD - 1
            then
                v_gap := true;
            end if;

        end if;

        if
            input_index < C_INPUT_COUNT and
            not v_gap
        then
            v_input_channel :=
                input_index mod G_C_IN;

            v_flat_pixel :=
                input_index / G_C_IN;

            v_input_row :=
                v_flat_pixel / G_W_IN;

            v_input_col :=
                v_flat_pixel mod G_W_IN;

            v_value :=
                activation_value(
                    v_input_row,
                    v_input_col,
                    v_input_channel
                );

            i_valid <= '1';

            i_data <=
                std_logic_vector(
                    to_unsigned(
                        v_value,
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

            elsif
                i_valid = '1' and
                i_ready = '1'
            then
                input_index <=
                    input_index + 1;

            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Weight source
    --
    -- The stream follows the current RTL's weight requests:
    --
    -- C_IN = 1:
    --   one batch per output group and output row
    --   unless there is only one group, in which case the same
    --   weights remain loaded for the full frame.
    --
    -- C_IN > 1:
    --   one batch for every input-channel slice of every output
    --   column, group and output row.
    --------------------------------------------------------------------

    weight_driver_process : process(all)

        variable v_batch_index : natural;
        variable v_byte_in_batch : natural;

        variable v_lane : natural;

        variable v_kernel_index : natural;
        variable v_kernel_row : natural;
        variable v_kernel_col : natural;

        variable v_output_group : natural;
        variable v_input_channel : natural;

        variable v_row_batch_index : natural;
        variable v_group_batch_size : natural;

        variable v_value : integer;
        variable v_gap : boolean;

    begin
        i_weight_valid <= '0';
        i_weight_data <= (others => '0');

        v_gap := false;

        if G_WEIGHT_GAP_PERIOD > 0 then

            if
                weight_cycle_count mod
                G_WEIGHT_GAP_PERIOD =
                G_WEIGHT_GAP_PERIOD - 1
            then
                v_gap := true;
            end if;

        end if;

        if
            weight_byte_index < C_WEIGHT_BYTE_COUNT and
            not v_gap
        then
            v_batch_index :=
                weight_byte_index /
                C_WEIGHT_BATCH_SIZE;

            v_byte_in_batch :=
                weight_byte_index mod
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

            v_kernel_col :=
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
                v_group_batch_size :=
                    C_OUTPUT_WIDTH *
                    G_C_IN;

                v_row_batch_index :=
                    v_batch_index mod
                    (
                        C_OUTPUT_GROUPS *
                        v_group_batch_size
                    );

                v_output_group :=
                    v_row_batch_index /
                    v_group_batch_size;

                v_input_channel :=
                    v_row_batch_index mod
                    G_C_IN;

            end if;

            v_value :=
                weight_value(
                    v_output_group,
                    v_lane,
                    v_input_channel,
                    v_kernel_row,
                    v_kernel_col
                );

            i_weight_valid <= '1';

            i_weight_data <=
                std_logic_vector(
                    to_signed(
                        v_value,
                        i_weight_data'length
                    )
                );
        end if;

    end process;

    weight_counter_process : process(clk)
    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                weight_byte_index <= 0;

            elsif
                i_weight_valid = '1' and
                o_weight_ready = '1'
            then
                weight_byte_index <=
                    weight_byte_index + 1;

            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Output backpressure
    --------------------------------------------------------------------

    output_ready_process : process(all)
    begin
        i_acc_ready <= '1';

        if G_STALL_PERIOD > 0 then

            if
                output_cycle_count mod
                G_STALL_PERIOD =
                G_STALL_PERIOD - 1
            then
                i_acc_ready <= '0';
            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Raw accumulator checking
    --
    -- Expected output order:
    --
    -- output row
    --   output group
    --     horizontal output column
    --------------------------------------------------------------------

    output_checker_process : process(clk)

        variable v_output_row : natural;
        variable v_output_group : natural;
        variable v_output_col : natural;

        variable v_row_offset : natural;

        variable v_actual : integer;
        variable v_expected : integer;

    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                output_transfer_index <= 0;

            elsif
                o_acc_valid = '1' and
                i_acc_ready = '1'
            then
                if
                    output_transfer_index <
                    C_OUTPUT_TRANSFER_COUNT
                then
                    v_output_row :=
                        output_transfer_index /
                        (
                            C_OUTPUT_GROUPS *
                            C_OUTPUT_WIDTH
                        );

                    v_row_offset :=
                        output_transfer_index mod
                        (
                            C_OUTPUT_GROUPS *
                            C_OUTPUT_WIDTH
                        );

                    v_output_group :=
                        v_row_offset /
                        C_OUTPUT_WIDTH;

                    v_output_col :=
                        v_row_offset mod
                        C_OUTPUT_WIDTH;

                    for lane in 0 to G_C_PAR - 1 loop

                        v_actual :=
                            to_integer(
                                signed(
                                    o_acc_data(
                                        (lane + 1) * 32 - 1
                                        downto
                                        lane * 32
                                    )
                                )
                            );

                        v_expected :=
                            expected_accumulator(
                                v_output_row,
                                v_output_col,
                                v_output_group,
                                lane
                            );

                        assert v_actual = v_expected
                            report
                                G_NAME &
                                ": mismatch at output row " &
                                integer'image(v_output_row) &
                                ", group " &
                                integer'image(v_output_group) &
                                ", column " &
                                integer'image(v_output_col) &
                                ", lane " &
                                integer'image(lane) &
                                ". Expected " &
                                integer'image(v_expected) &
                                ", got " &
                                integer'image(v_actual)
                            severity failure;

                    end loop;

                    output_transfer_index <=
                        output_transfer_index + 1;

                else
                    assert false
                        report
                            G_NAME &
                            ": unexpected extra output"
                        severity failure;

                end if;

            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Output stability while stalled
    --------------------------------------------------------------------

    output_stability_process : process(clk)

        variable v_holding : boolean := false;

        variable v_held_data :
            std_logic_vector(
                G_C_PAR * 32 - 1 downto 0
            ) :=
            (others => '0');

    begin
        if rising_edge(clk) then

            if rst_n = '0' then
                v_holding := false;
                v_held_data := (others => '0');

            else
                if v_holding then

                    assert o_acc_valid = '1'
                        report
                            G_NAME &
                            ": o_acc_valid dropped while stalled"
                        severity failure;

                    assert o_acc_data = v_held_data
                        report
                            G_NAME &
                            ": o_acc_data changed while stalled"
                        severity failure;

                end if;

                if
                    o_acc_valid = '1' and
                    i_acc_ready = '0'
                then
                    v_holding := true;
                    v_held_data := o_acc_data;
                else
                    v_holding := false;
                end if;

            end if;

        end if;
    end process;

    --------------------------------------------------------------------
    -- Completion checks
    --------------------------------------------------------------------

    completion_process : process
    begin
        wait until rst_n = '1';

        wait until
            input_index = C_INPUT_COUNT and
            weight_byte_index = C_WEIGHT_BYTE_COUNT and
            output_transfer_index =
                C_OUTPUT_TRANSFER_COUNT;

        wait until rising_edge(clk);
        wait until rising_edge(clk);

        for cycle in 1 to 12 loop
            wait until rising_edge(clk);

            assert not (
                o_acc_valid = '1' and
                i_acc_ready = '1'
            )
                report
                    G_NAME &
                    ": output continued after completion"
                severity failure;

            assert o_weight_ready = '0'
                report
                    G_NAME &
                    ": unexpected extra weight request"
                severity failure;

        end loop;

        report
            "PASS: " &
            G_NAME &
            " inputs=" &
            integer'image(C_INPUT_COUNT) &
            " weight_batches=" &
            integer'image(C_WEIGHT_BATCH_COUNT) &
            " outputs=" &
            integer'image(C_OUTPUT_TRANSFER_COUNT)
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
                "TIMEOUT: " &
                G_NAME &
                " input=" &
                integer'image(input_index) &
                "/" &
                integer'image(C_INPUT_COUNT) &
                " weights=" &
                integer'image(weight_byte_index) &
                "/" &
                integer'image(C_WEIGHT_BYTE_COUNT) &
                " outputs=" &
                integer'image(output_transfer_index) &
                "/" &
                integer'image(C_OUTPUT_TRANSFER_COUNT)
            severity failure;
    end process;

end architecture sim;