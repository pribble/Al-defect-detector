# `conv_layer`

Generic streaming quantized convolution layer for the FPGA inference pipeline.

> **Status: functionally complete and bit-exact in simulation.**
>
> The layer has been verified with directed testbenches and against the exported
> real vectors for all five convolution layers of the current model.

---

## Role

`conv_layer` performs one complete quantized convolution stage:

```text
uint8 activations
    ↓
int8 convolution weights
    ↓
signed int32 accumulation
    ↓
signed int32 bias addition
    ↓
ReLU
    ↓
fixed-point requantization
    ↓
uint8 saturation
    ↓
uint8 output activations
```

The same RTL block supports the model's convolution layers through generics for:

```text
input channels
output channels
input width and height
kernel size
padding
stride
output-channel parallelism
```

---

## Supported arithmetic

For each output channel:

```text
raw_acc =
    Σ(input_activation × signed_weight)

biased_acc =
    raw_acc + bias[channel]

relu_acc =
    max(biased_acc, 0)

product =
    uint64(relu_acc) × uint32(requant_multiplier[channel])

if requant_shift[channel] > 0:
    product += 2^(requant_shift[channel] - 1)

shifted =
    product >> requant_shift[channel]

output =
    min(shifted, 255)
```

The final result is an unsigned 8-bit activation.

The rounding behavior is round-half-up for nonnegative values.

There is currently no output zero-point addition.

---

## Generics

```vhdl
G_C_IN    : positive;
G_C_OUT   : positive;
G_W_IN    : positive;
G_H_IN    : positive;
G_C_PAR   : positive;
G_KERNEL  : positive;
G_PADDING : natural;
G_STRIDE  : positive
```

### `G_C_PAR`

`G_C_PAR` is the number of output channels calculated in parallel.

One output group contains:

```text
G_C_PAR output channels
```

The number of output groups is:

```text
G_C_OUT / G_C_PAR
```

The current implementation requires:

```text
G_C_OUT mod G_C_PAR = 0
```

A partially populated final output group is not currently supported.

---

## Interface

### Clock and reset

```vhdl
clk   : in std_logic;
rst_n : in std_logic;
```

The reset is synchronous and active low.

---

### Activation input stream

```vhdl
i_valid : in  std_logic;
i_ready : out std_logic;
i_data  : in  std_logic_vector(7 downto 0);
```

Activations are unsigned 8-bit values.

A transfer occurs when:

```text
i_valid = 1 and i_ready = 1
```

Input activations are streamed in:

```text
input row
    → input column
        → input channel
```

order.

---

### Weight input stream

```vhdl
i_weight_valid : in  std_logic;
o_weight_ready : out std_logic;
i_weight_data  : in  std_logic_vector(7 downto 0);
```

Weights are signed 8-bit values.

A transfer occurs when:

```text
i_weight_valid = 1 and o_weight_ready = 1
```

The active weight buffer contains:

```text
G_C_PAR × G_KERNEL × G_KERNEL
```

weights.

This represents one kernel slice for:

```text
one output-channel group
one input channel
```

when `G_C_IN > 1`.

The external weight source must provide each requested slice in lane-major order:

```text
lane 0 kernel values
lane 1 kernel values
...
lane G_C_PAR - 1 kernel values
```

Within each lane, kernel values are supplied in row-major order.

---

### Parameter configuration interface

```vhdl
cfg_we    : in std_logic;
cfg_sel   : in std_logic_vector(1 downto 0);
cfg_addr  : in std_logic_vector(19 downto 0);
cfg_wdata : in std_logic_vector(31 downto 0);
```

Parameters are indexed by absolute output-channel number.

```text
cfg_sel = "01"  signed int32 bias
cfg_sel = "10"  unsigned uint32 requantization multiplier
cfg_sel = "11"  unsigned uint8 requantization right shift
cfg_sel = "00"  reserved
```

For bias and multiplier values, all 32 bits of `cfg_wdata` are used.

For right-shift values, only:

```vhdl
cfg_wdata(7 downto 0)
```

is used.

Parameters must be loaded before processing a frame.

They remain stored across frame restarts.

---

### Quantized output stream

```vhdl
o_valid : out std_logic;
o_data  : out std_logic_vector(G_C_PAR * 8 - 1 downto 0);
```

Each accepted output transfer contains `G_C_PAR` final uint8 output channels.

Lane `n` is packed into:

```vhdl
o_data((n + 1) * 8 - 1 downto n * 8)
```

The output transfer ordering is:

```text
output row
    → output group
        → output column
            → lane
```

The absolute output channel represented by one lane is:

```text
output_group × G_C_PAR + lane
```

---

### Raw accumulator debug output

```vhdl
i_acc_ready : in  std_logic;
o_acc_valid : out std_logic;
o_acc_data  : out std_logic_vector(G_C_PAR * 32 - 1 downto 0);
```

The raw output contains the signed int32 convolution accumulators before:

```text
bias
ReLU
requantization
```

Lane `n` is packed into:

```vhdl
o_acc_data((n + 1) * 32 - 1 downto n * 32)
```

The quantized and raw outputs describe the same output transfer:

```text
o_valid = o_acc_valid
```

The current implementation uses `i_acc_ready` as the ready input for both output
interfaces.

While:

```text
o_valid = 1
and
i_acc_ready = 0
```

both `o_data` and `o_acc_data` remain stable.

The raw interface is currently retained for debugging and verification.

---

### Frame completion

```vhdl
o_done : out std_logic;
```

`o_done` pulses for one clock when:

```text
the final output has been accepted
and
all required trailing input draining has completed
and
the layer is returning to its initial state
```

After `o_done`, the layer automatically becomes ready to process another frame.

A global reset is not required between frames.

---

## Output dimensions

The padded input dimensions are:

```text
padded_width  = G_W_IN + 2 × G_PADDING
padded_height = G_H_IN + 2 × G_PADDING
```

The output dimensions are:

```text
output_width =
    ((padded_width - G_KERNEL) / G_STRIDE) + 1

output_height =
    ((padded_height - G_KERNEL) / G_STRIDE) + 1
```

Padding is implemented as virtual zero-valued activation locations.

Padded positions are not read from the activation stream.

---

## Activation buffering

The layer uses:

```text
G_KERNEL active physical row buffers
one spare physical row buffer
```

The physical line buffer therefore contains:

```text
G_KERNEL + 1 rows
```

Each row contains:

```text
G_W_IN × G_C_IN bytes
```

A logical-to-physical row map selects the physical buffers used by the current
convolution window.

After advancing vertically, the logical row roles rotate:

```text
before:
    active rows = A, B, C
    spare row   = D

after:
    active rows = B, C, D
    spare row   = A
```

No activation data is copied during rotation.

Only the row mapping and validity flags are changed.

Virtual top and bottom padding rows are represented through row-valid flags and
produce zero-valued activations.

---

## Horizontal window traversal

For output column `x` and kernel column `kx`, the real input column is:

```text
input_column =
    x × G_STRIDE + kx - G_PADDING
```

When this column is outside:

```text
0 to G_W_IN - 1
```

the activation value is zero.

The controller waits until the required real activation columns have arrived before
starting each output window.

---

## Vertical stride handling

Vertical stride is implemented by performing repeated logical row rotations between
output rows.

`vertical_advance_remaining` tracks how many row advances are still required before
the next output row can be calculated.

Rows that must be consumed from the input stream but are not used by another output
window are drained before the frame completes.

---

## Weight handling

Weight loading runs concurrently with activation preparation.

The first activation window can only begin calculation when both are true:

```text
first_window_ready = 1
weight_group_ready = 1
```

The active weight buffer contains one `G_C_PAR × K × K` slice.

For `G_C_IN > 1`, the layer requests a new weight slice for each input-channel
contribution.

Accumulation continues across all input channels before producing one output result.

For multiple output groups, the layer reloads the corresponding output-channel
weights and repeats the horizontal sweep over the same logical activation rows.

---

## Accumulation and requantization

Activations are converted from unsigned 8-bit to a positive signed value.

Weights are signed 8-bit values.

Each multiplication produces a signed product, which is resized and accumulated into
a signed 32-bit accumulator.

For each output group, all `G_C_PAR` lanes are calculated in parallel.

For `G_C_IN > 1`, accumulators are preserved while successive input-channel weight
slices are processed.

After the final kernel value of the final input channel:

```text
raw accumulator is stored
bias is added
ReLU is applied
requantization is performed
raw and quantized outputs become valid
```

The quantized result is calculated using:

```text
biased_acc = raw_acc + bias

relu_acc = max(biased_acc, 0)

product = relu_acc × requant_multiplier

if requant_shift > 0:
    product += 2^(requant_shift - 1)

shifted = product >> requant_shift

output = min(shifted, 255)
```

---

## Controller state machine

The main controller uses the following states:

```vhdl
type conv_states is (
    S_IDLE,
    S_INITIAL_LINE_FILL,
    S_PRIME_K_LINE,
    S_CALC_AND_SLIDING_WINDOW,
    S_STREAM_LINE_FILLING,
    S_LINE_ROTATION
);
```

Several datapath operations are implemented as concurrent worker processes rather
than as additional controller states.

These include:

```text
weight filling
initial line filling
prime-line filling
stream-line filling
MAC calculation
parameter loading
line-buffer writing
line rotation
```

---

### `S_IDLE`

Initializes a new frame.

Entering `S_IDLE` starts:

```text
initial activation filling
and
first weight-group loading
```

The line-buffer mapping and frame traversal counters are restored for the new frame.

---

### `S_INITIAL_LINE_FILL`

Loads the initial real activation rows required before the bottom row of the first
logical kernel window can be primed.

The number of initial rows is:

```text
G_KERNEL - G_PADDING - 1
```

This accounts for virtual top padding rows.

---

### `S_PRIME_K_LINE`

Loads the first `G_KERNEL` columns of the bottom logical row when that row maps to a
real input row.

When the logical row is a virtual padding row, no activation bytes are consumed and
the first window is marked ready immediately.

Calculation begins when:

```text
first_window_ready = 1
and
weight_group_ready = 1
```

unless additional vertical advancement or trailing input draining is required.

---

### `S_CALC_AND_SLIDING_WINDOW`

Calculates all output columns for the current output group.

For each output window, the calculation traverses:

```text
kernel position
    → input channel
```

contributions.

The horizontal output ordering is:

```text
group 0: all output columns
group 1: all output columns
...
final group: all output columns
```

After the final output group is complete, vertical traversal begins.

---

### `S_STREAM_LINE_FILLING`

Completes the remaining part of the next physical activation row.

This process may overlap with calculation because stream filling is controlled by an
independent worker process.

For virtual bottom-padding rows, no activation bytes are consumed.

---

### `S_LINE_ROTATION`

Rotates the logical row mapping without copying physical activation data.

The state determines whether:

```text
another vertical advance is required
another output row is available
trailing real input rows must be drained
or
the frame is complete
```

When no further rows remain:

```text
o_done pulses
the controller returns to S_IDLE
the layer prepares for another frame
```

---

## Current implementation constraints

The current implementation assumes:

```text
G_C_OUT mod G_C_PAR = 0
G_PADDING <= G_KERNEL - 2
G_W_IN > G_KERNEL
padded width >= kernel size
padded height >= kernel size
square kernels
the same stride in both spatial dimensions
uint8 activations
int8 weights
signed int32 accumulators
```

A partially populated final output group is not supported.

Parameter memories must be loaded before a frame begins.

The external weight source must respond to every asserted `o_weight_ready` request
with the correct output-group and input-channel weight slice.

---

## Verification status

The implementation has been verified incrementally with directed VHDL testbenches.

Covered behavior includes:

```text
initial line filling
prime-line filling
weight filling
concurrent activation and weight preparation
first-window calculation
horizontal sliding
line-buffer rotation
multiple output groups
multiple input channels
padding
stride 2 and stride 3
trailing input-row draining
activation source gaps
weight source gaps
output backpressure
raw output stability
quantized output stability
bias addition
ReLU
requantization
rounding
uint8 saturation
per-channel parameter selection
frame completion
automatic restart
```

A traversal-matrix test verifies combinations of:

```text
padding
stride
output groups
multiple input channels
source gaps
backpressure
trailing-row draining
```

The real-vector testbench loads the exported:

```text
<prefix>_in.bin
<prefix>_weights.bin
<prefix>_biases.bin
<prefix>_requant_m.bin
<prefix>_requant_r.bin
<prefix>_out.bin
```

files.

It compares:

```text
o_acc_data
    against generated raw signed-int32 convolution references

o_data
    against the exported final uint8 model outputs
```

All five real convolution layers currently pass bit-exact comparison:

```text
Ran:    5 layers
Passed: 5 layers
Failed: 0 layers
```

---

## Simulation commands

Run the directed convolution regressions:

```bash
bash verification/sim/run_conv.sh
```

Run the padding, stride, grouping, and traversal matrix:

```bash
bash verification/sim/run_conv_traversal.sh
```

Run the real model vectors:

```bash
bash verification/sim/run_layers.sh
```

Regenerate the raw accumulator references:

```bash
FORCE_GOLDEN=1 \
bash verification/sim/run_layers.sh
```

Run one selected model layer:

```bash
bash verification/sim/run_layers.sh features_0
```

---

## State-machine diagram

The state-machine diagram should show the six main controller states:

```text
S_IDLE
S_INITIAL_LINE_FILL
S_PRIME_K_LINE
S_CALC_AND_SLIDING_WINDOW
S_STREAM_LINE_FILLING
S_LINE_ROTATION
```

Weight filling, MAC calculation, parameter loading, and line-buffer writing are
concurrent worker processes rather than separate main FSM states.

The diagram should also show:

```text
padding-aware initial filling
stride-driven repeated row rotation
trailing input draining
o_done
automatic return to S_IDLE
```

---

## Remaining implementation work

The functional RTL milestone is complete.

Remaining FPGA implementation work includes:

```text
Quartus synthesis
DSP inference inspection
embedded-memory inference inspection
resource-utilization analysis
timing analysis and Fmax
production interface cleanup
possible removal of raw debug outputs
support for partially populated final output groups
```