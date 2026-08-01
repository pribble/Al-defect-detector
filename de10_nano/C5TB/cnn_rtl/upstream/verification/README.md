# Verification

This directory contains the verification infrastructure for the FPGA inference
pipeline.

The current completed verification milestone covers the generic quantized
convolution layer:

```text
hardware/rtl/layers/conv_layer.vhd
```

The verification approach is simulation-first and bit-exact:

```text
Python/model export
    ↓
binary activation, weight, bias, and requantization vectors
    ↓
VHDL testbench running under GHDL
    ↓
cycle-accurate ready/valid stimulus
    ↓
raw int32 and final uint8 comparison
    ↓
hard assertion on any mismatch
```

There is no numerical tolerance.

Every checked RTL value must match the corresponding integer reference exactly.

---

## Current verification status

The convolution RTL has been verified with:

- directed state and datapath tests;
- activation and weight handshake tests;
- line-buffer filling tests;
- logical row-rotation tests;
- horizontal window traversal tests;
- multiple input-channel tests;
- multiple output-group tests;
- padding tests;
- stride-2 and stride-3 tests;
- activation-source gaps;
- weight-source gaps;
- output backpressure;
- raw-output stability while stalled;
- quantized-output stability while stalled;
- signed bias addition;
- ReLU;
- fixed-point requantization;
- round-half-up behavior;
- unsigned 8-bit saturation;
- per-channel parameter selection;
- trailing input-row draining;
- frame-completion signaling;
- automatic restart for the next frame;
- real exported vectors for all five convolution layers.

Current real-vector result:

```text
Convolution real-vector results
  Ran:    5 layers
  Passed: 5 layers
  Failed: 0 layers
```

The convolution implementation is therefore functionally complete and bit-exact in
simulation for the current AlexNet64Gray model.

The following verification areas remain future work:

```text
max-pooling RTL
fully connected RTL
weight-memory provider
external SDRAM controller
stream adapters
top-level pipeline integration
complete-network simulation
Quartus gate-level or post-fit validation
on-board comparison
```

---

## Table of contents

- [Verification philosophy](#verification-philosophy)
- [Current directory structure](#current-directory-structure)
- [Verification levels](#verification-levels)
- [How to run](#how-to-run)
- [Real-vector workflow](#real-vector-workflow)
- [Binary data formats](#binary-data-formats)
- [Tensor and stream ordering](#tensor-and-stream-ordering)
- [Backpressure verification](#backpressure-verification)
- [Frame completion and restart](#frame-completion-and-restart)
- [Generated raw accumulator references](#generated-raw-accumulator-references)
- [Adding a directed convolution test](#adding-a-directed-convolution-test)
- [Adding a real-vector convolution case](#adding-a-real-vector-convolution-case)
- [Exit criteria](#exit-criteria)
- [Future verification work](#future-verification-work)

---

# Verification philosophy

## Bit-exact comparison

The RTL is checked against integer references.

For each convolution output channel:

```text
raw_acc =
    Σ(uint8 activation × int8 weight)
```

The raw signed-int32 result is checked before:

```text
bias
ReLU
requantization
saturation
```

The final quantized result is then checked after:

```text
signed int32 bias addition
ReLU
unsigned fixed-point multiplication
round-half-up
right shift
uint8 saturation
```

The implemented arithmetic is:

```text
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

There is currently no output-zero-point addition.

---

## Hard-failure policy

A mismatch is always a failing test:

```vhdl
assert actual = expected
    report "FAIL: ..."
    severity failure;
```

The test suite does not use:

```text
floating-point tolerance
percentage error thresholds
average error
warning-only mismatches
```

A passing test means every checked value matched exactly.

---

## Handshake-aware verification

The testbenches do not assume that data transfers every cycle.

An activation transfer occurs only when:

```text
i_valid = 1
and
i_ready = 1
```

A weight transfer occurs only when:

```text
i_weight_valid = 1
and
o_weight_ready = 1
```

An output transfer occurs only when:

```text
o_valid = 1
and
i_acc_ready = 1
```

The current convolution layer uses `i_acc_ready` as the ready input for both:

```text
o_data
o_acc_data
```

The testbenches count accepted transfers rather than merely counting cycles with
`valid` asserted.

---

# Current directory structure

The currently implemented verification structure is:

```text
verification/
│
├── README.md
│
├── tb/
│   ├── conv/
│   │   ├── directed convolution testbenches
│   │   ├── traversal and control tests
│   │   ├── backpressure tests
│   │   ├── output-group tests
│   │   └── shared convolution testbench utilities
│   │
│   └── layers/
│       └── tb_conv_layer.vhd
│
├── sim/
│   ├── run_conv.sh
│   ├── run_conv_traversal.sh
│   └── run_layers.sh
│
└── results/
    └── default/
        └── raw_acc/
            └── <prefix>_raw_acc.bin
```

The ML export pipeline provides the real model vectors used by
`tb_conv_layer.vhd`.

Not every module listed in the original architecture plan currently has an RTL
testbench. The implemented verification tree should be treated as authoritative.

---

# Verification levels

The convolution verification is divided into three main levels.

## Level 1 — Directed behavior tests

Directed testbenches use small tensors and simple values so every expected result can
be calculated and inspected easily.

These tests isolate specific behavior such as:

```text
initial line filling
prime-row filling
weight loading
calculation
horizontal sliding
line rotation
output grouping
output backpressure
requantization
frame completion
automatic restart
```

Small test dimensions make it possible to inspect:

```text
accepted activation count
accepted weight count
output count
expected accumulator values
ready/valid timing
completion timing
```

---

## Level 2 — Traversal matrix

Traversal tests exercise combinations of features that may interact incorrectly even
when each feature passes individually.

Covered combinations include:

```text
padding with stride
stride with multiple output groups
multiple input channels with source gaps
padding with stride 3
trailing-row drain
combined source and output backpressure
```

Current traversal cases include:

```text
padding_stride2_groups
multichannel_with_source_gaps
padding_stride3_groups
trailing_row_drain
combined_backpressure
```

---

## Level 3 — Real model vectors

The complete generic `conv_layer` is instantiated using the dimensions and parameters
of each real convolution layer in AlexNet64Gray.

The testbench:

1. loads the exported input activations;
2. loads the exported signed weights;
3. loads the exported biases;
4. loads the exported requantization multipliers;
5. loads the exported requantization shifts;
6. streams activations through the RTL;
7. supplies weight slices in the order requested by the RTL;
8. applies periodic output backpressure;
9. compares raw accumulators;
10. compares final quantized outputs;
11. checks exact transfer counts;
12. checks output stability while stalled.

Each whole convolution layer is run once per simulation.

The test does not run one output group in isolation.

---

# How to run

Run commands from the repository root.

## Directed convolution tests

```bash
bash verification/sim/run_conv.sh
```

This suite covers the main directed convolution behavior.

Typical coverage includes:

```text
initial line fill
prime K line
weight filling
concurrent prime and weight loading
calculation
horizontal sliding
line rotation
output backpressure
output groups
requantization
completion and restart
```

---

## Padding, stride, and traversal matrix

```bash
bash verification/sim/run_conv_traversal.sh
```

This suite focuses on:

```text
padding
stride 2
stride 3
multiple input channels
multiple output groups
source gaps
backpressure
trailing-row draining
```

---

## All real convolution layers

```bash
bash verification/sim/run_layers.sh
```

The runner discovers the convolution layers from the model metadata and executes one
whole-layer simulation for each layer.

Expected final summary:

```text
Convolution real-vector results
  Ran:    5 layers
  Passed: 5 layers
  Failed: 0 layers
```

---

## One selected convolution layer

Pass the exported layer prefix:

```bash
bash verification/sim/run_layers.sh features_0
```

Other convolution prefixes may be selected in the same way.

---

## Regenerate raw accumulator references

```bash
FORCE_GOLDEN=1 \
bash verification/sim/run_layers.sh
```

Use this after changing:

```text
model weights
input vectors
layer geometry
padding
stride
tensor ordering
raw-reference generation logic
```

For normal regression runs, reuse the existing generated raw references.

---

## Enable waveform generation

```bash
WAVE=1 \
bash verification/sim/run_layers.sh features_0
```

Waveform generation is useful for:

```text
ready/valid debugging
weight-request timing
output ordering
counter inspection
line-buffer traversal
backpressure behavior
completion timing
```

---

## Change the output-stall period

The real-vector runner can apply periodic output backpressure.

For example:

```bash
STALL_PERIOD=7 \
bash verification/sim/run_layers.sh
```

The exact default value is defined by the runner.

A smaller period introduces more frequent output stalls.

---

## GHDL large-object allocation

Large convolution configurations may create large elaborated arrays in the
testbench or DUT.

The runner configures the process stack and passes a GHDL maximum stack-allocation
setting.

The allocation limit can be overridden with:

```bash
MAX_STACK_ALLOC_KB=65536 \
bash verification/sim/run_layers.sh
```

To remove GHDL's individual stack-object limit:

```bash
MAX_STACK_ALLOC_KB=0 \
bash verification/sim/run_layers.sh
```

This setting affects simulation only.

It does not change synthesized hardware resources.

---

# Real-vector workflow

The real-vector flow uses two references for every output.

## Final uint8 reference

The exported model provides:

```text
<prefix>_out.bin
```

This contains the final quantized convolution output after:

```text
bias
ReLU
requantization
rounding
saturation
```

The VHDL testbench compares this file against:

```text
o_data
```

---

## Raw signed-int32 reference

The runner generates:

```text
verification/results/default/raw_acc/<prefix>_raw_acc.bin
```

This contains:

```text
Σ(input activation × signed weight)
```

before:

```text
bias
ReLU
requantization
saturation
```

The VHDL testbench compares this file against:

```text
o_acc_data
```

Checking both references separates two major categories of errors:

```text
raw convolution traversal or accumulation error
```

and:

```text
bias, activation, or requantization error
```

---

## Model metadata

The runner uses model export metadata such as:

```text
ml/outputs/runN/fpgaqparms.json
```

to determine:

```text
input channels
output channels
input height
input width
kernel size
padding
stride
vector prefixes
```

The runner selects a valid `G_C_PAR` value that divides `G_C_OUT`.

The configured maximum lane count is controlled by the runner.

---

# Binary data formats

All binary files are raw packed arrays without headers.

Multi-byte integer files use little-endian byte order.

## Convolution input

```text
<prefix>_in.bin
```

| Property | Value |
|---|---|
| Type | `uint8` |
| Layout | HWC |
| Order | row → column → input channel |
| Padding bytes included | No |

All convolution activations are interpreted as unsigned values.

---

## Convolution weights

```text
<prefix>_weights.bin
```

| Property | Value |
|---|---|
| Type | signed `int8` |
| Model layout | output channel × input channel × kernel row × kernel column |
| RTL delivery | active slices supplied in handshake-request order |

The VHDL testbench serves weights according to `o_weight_ready`.

For each requested active slice, the transfer order is:

```text
lane
    → kernel row
        → kernel column
```

The slice corresponds to:

```text
G_C_PAR output channels
one input channel
```

The testbench maps the exported model tensor into this delivery order.

---

## Biases

```text
<prefix>_biases.bin
```

| Property | Value |
|---|---|
| Type | signed `int32` |
| Endianness | little-endian |
| Count | `C_OUT` |
| Index | absolute output channel |

Bias values are written through the RTL configuration interface before frame
processing.

---

## Requantization multipliers

```text
<prefix>_requant_m.bin
```

| Property | Value |
|---|---|
| Type | unsigned `uint32` |
| Endianness | little-endian |
| Count | `C_OUT` |
| Index | absolute output channel |

---

## Requantization shifts

```text
<prefix>_requant_r.bin
```

| Property | Value |
|---|---|
| Type | unsigned `uint8` |
| Count | `C_OUT` |
| Index | absolute output channel |

---

## Final convolution output

```text
<prefix>_out.bin
```

| Property | Value |
|---|---|
| Type | `uint8` |
| Layout | HWC |
| Order | output row → output column → output channel |

This is the final exported model output.

---

## Generated raw accumulator output

```text
verification/results/default/raw_acc/<prefix>_raw_acc.bin
```

| Property | Value |
|---|---|
| Type | signed `int32` |
| Endianness | little-endian |
| Layout | HWC |
| Order | output row → output column → output channel |

This file is generated by the verification runner rather than exported as the final
model activation.

---

# Tensor and stream ordering

## Input ordering

The RTL consumes activations in:

```text
input row
    → input column
        → input channel
```

order.

Equivalent layout:

```text
HWC
```

---

## Exported golden output ordering

The exported final output and generated raw accumulator reference use:

```text
output row
    → output column
        → output channel
```

order.

Equivalent layout:

```text
HWC
```

---

## RTL packed output ordering

The RTL produces:

```text
output row
    → output group
        → output column
            → lane
```

where:

```text
absolute_output_channel =
    output_group × G_C_PAR + lane
```

The real-vector testbench converts the RTL group/lane indices to the HWC reference
index before comparison.

This mapping is required because the RTL completes one horizontal sweep for each
output-channel group.

---

# Backpressure verification

The convolution layer contains a holding register for each packed result.

When:

```text
o_valid = 1
and
i_acc_ready = 0
```

the testbench verifies that all of the following remain unchanged:

```text
o_valid
o_acc_valid
o_data
o_acc_data
```

No new result may overwrite the pending result.

Internal traversal counters must not advance until the held result is accepted.

The real-vector test introduces periodic stalls to verify this behavior across full
model layers.

---

# Frame completion and restart

The current convolution layer provides:

```vhdl
o_done : out std_logic;
```

`o_done` pulses for one clock after:

```text
the final output has been accepted
all required trailing input rows have been consumed
the final logical row traversal has completed
```

After completion, the layer returns to its initial state.

The testbenches verify that:

```text
o_done pulses once
o_done does not remain asserted
activation input becomes ready for the next frame
weight input becomes ready for the next frame
no global reset is required
```

The physical line-buffer contents are not explicitly cleared.

The logical row map, validity flags, stride state, drain state, and frame counters are
reinitialized.

Old tests that expected `i_ready = 0` or `o_weight_ready = 0` permanently after one
frame must not be used.

Automatic readiness after completion is now the intended behavior.

---

# Generated raw accumulator references

Raw reference files are generated because the exported final output alone cannot show
whether a mismatch originated in:

```text
window traversal
weight ordering
accumulation
bias
ReLU
rounding
requantization
saturation
```

The raw reference contains only:

```text
Σ(activation × weight)
```

Generation is controlled by:

```bash
FORCE_GOLDEN=1
```

The raw files are stored under:

```text
verification/results/default/raw_acc/
```

Generated references should be regenerated when any source data or convolution
geometry changes.

They should not be regenerated automatically during every regression, because doing
so could hide an unintended reference change.

---

# Adding a directed convolution test

## 1. Select a small configuration

Use dimensions that make every expected transfer easy to inspect.

Example considerations:

```text
small width and height
one or two input channels
one or two output lanes
3×3 kernel
simple weights
small activation values
```

---

## 2. Instantiate the current entity

The testbench must connect the complete implemented interface:

```vhdl
clk
rst_n

i_valid
i_ready
i_data

i_weight_valid
o_weight_ready
i_weight_data

cfg_we
cfg_sel
cfg_addr
cfg_wdata

o_valid
o_data
o_done

i_acc_ready
o_acc_valid
o_acc_data
```

Unused outputs may be mapped to `open`.

Configuration inputs may be tied low only when neutral default parameters are
sufficient for the test.

---

## 3. Use handshake-aware stimulus

Do not increment the activation source index until:

```text
i_valid = 1
and
i_ready = 1
```

Do not increment the weight source index until:

```text
i_weight_valid = 1
and
o_weight_ready = 1
```

Do not count an output until:

```text
o_valid = 1
and
i_acc_ready = 1
```

---

## 4. Check exact values

For raw lanes:

```vhdl
signed(
    o_acc_data((lane + 1) * 32 - 1 downto lane * 32)
)
```

For quantized lanes:

```vhdl
unsigned(
    o_data((lane + 1) * 8 - 1 downto lane * 8)
)
```

Use exact assertions for every expected output.

---

## 5. Check counts

At minimum, verify:

```text
accepted activation count
accepted weight count
accepted output count
completion-pulse count
```

This catches cases where correct values are produced but the stream consumes or emits
the wrong number of elements.

---

## 6. Include a timeout

Every testbench should have a timeout process:

```vhdl
timeout_process : process
begin
    wait for C_TIMEOUT;

    assert false
        report "FAIL: test timed out."
        severity failure;
end process;
```

A blocked handshake or missed pulse must fail rather than hang indefinitely.

---

## 7. Monitor one-cycle pulses safely

A process that begins waiting after an event may miss a one-clock pulse.

For `o_done`, use a sticky monitor when the result-checking process may reach the
completion check late:

```vhdl
signal done_seen  : std_logic := '0';
signal done_count : natural := 0;
```

```vhdl
done_monitor_process : process(clk)
begin
    if rising_edge(clk) then
        if rst_n = '0' then
            done_seen  <= '0';
            done_count <= 0;

        elsif o_done = '1' then
            done_seen  <= '1';
            done_count <= done_count + 1;
        end if;
    end if;
end process;
```

The final test should check that exactly one completion pulse occurred for one frame.

---

## 8. Add the test to the appropriate runner

General directed behavior belongs in:

```text
verification/sim/run_conv.sh
```

Padding, stride, source-gap, and combined traversal behavior belongs in:

```text
verification/sim/run_conv_traversal.sh
```

The runner must stop with a nonzero exit code when compilation, elaboration, or
simulation fails.

---

# Adding a real-vector convolution case

The real-vector runner normally discovers convolution layers from the model metadata.

When adding or changing a model convolution layer:

1. export the layer input;
2. export signed int8 weights;
3. export signed int32 biases;
4. export unsigned requantization multipliers;
5. export unsigned requantization shifts;
6. export final uint8 output;
7. confirm the metadata contains the layer geometry;
8. regenerate the raw accumulator reference;
9. run the selected layer;
10. run all convolution layers.

Required vector set:

```text
<prefix>_in.bin
<prefix>_weights.bin
<prefix>_biases.bin
<prefix>_requant_m.bin
<prefix>_requant_r.bin
<prefix>_out.bin
```

Regenerate the raw reference:

```bash
FORCE_GOLDEN=1 \
bash verification/sim/run_layers.sh <prefix>
```

Then run the normal regression without forcing regeneration:

```bash
bash verification/sim/run_layers.sh <prefix>
```

Finally:

```bash
bash verification/sim/run_layers.sh
```

---

# Exit criteria

## Directed convolution regression

Pass condition:

```text
zero GHDL analysis failures
zero elaboration failures
zero simulation assertion failures
every test reports PASS
```

---

## Traversal matrix

Pass condition:

```text
all configured padding, stride, grouping, channel, gap,
backpressure, and drain combinations pass
```

---

## Real convolution vectors

Pass condition:

```text
all raw signed-int32 outputs match
all final uint8 outputs match
all transfer counts match
all stalled outputs remain stable
all requested layers pass
```

Current status:

```text
Ran:    5
Passed: 5
Failed: 0
```

---

## Functional convolution milestone

The functional convolution milestone is considered complete when:

```text
directed tests pass
traversal tests pass
requantization tests pass
all real convolution layers pass
frame completion works
automatic restart works
```

This milestone has been reached.

Quartus synthesis, resource inference, timing analysis, and hardware deployment are
separate implementation milestones.

---

# Future verification work

## Max-pooling verification

Future tests should cover:

```text
2×2 maximum selection
stride 2
all channels
row-boundary handling
output ordering
backpressure
frame completion
two consecutive frames
real exported vectors
```

---

## Fully connected verification

Future tests should cover:

```text
multi-cycle signed accumulation
weight ordering
bias
ReLU where applicable
requantization
final raw logits
backpressure
frame completion
real exported vectors
```

---

## Weight-provider verification

Future tests should cover:

```text
correct output-group selection
correct input-channel selection
weight-slice ordering
source stalls
prefetching
FIFO boundaries
frame restart
multiple consecutive frames
```

---

## Pipeline verification

The complete pipeline testbench should eventually verify:

```text
input image stream
all convolution layers
all pooling layers
all fully connected layers
argmax
final class result
```

Required integration cases include:

```text
one image
multiple consecutive images
different images without reset
backpressure between every stage
weight-source stalls
parameter loading
no frame-to-frame state leakage
complete bit-exact comparison against Python
```

---

## Quartus and hardware validation

After functional simulation, verification must expand to:

```text
DSP inference
M10K and MLAB inference
ALM and register utilization
post-fit timing
achieved Fmax
clock-domain behavior
external-memory timing
on-board inference results
```

Simulation success proves functional correctness of the modeled RTL.

It does not by itself prove:

```text
resource fit
timing closure
expected DSP packing
expected RAM inference
board-level electrical correctness
```

Those checks belong to the FPGA implementation phase.

---

# Summary

The verification infrastructure has moved beyond the original planned
co-simulation outline.

The current working flow provides:

```text
directed GHDL tests
padding and stride traversal tests
source-gap tests
backpressure tests
raw accumulator checking
final quantized checking
real model vectors
five complete convolution layers passing
frame completion
automatic restart
```

The next verification milestones will follow the remaining RTL implementation:

```text
max pooling
fully connected layers
weight and parameter infrastructure
stream adapters
top-level pipeline
Quartus implementation
on-board validation
```