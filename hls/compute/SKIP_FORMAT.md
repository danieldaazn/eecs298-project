# Pruned-weight format for zero-skipping

**Shared interface:** this is the contract between Kevin's pruning/export and Daniel's
skip hardware. Changing it (especially the block size) needs agreement from both sides.
See CLAUDE.md.

## What the hardware skips

Only the **pointwise (1×1)** layer skips cycles. The pointwise datapath does 8 MACs
per cycle, one **block** of 8 consecutive input channels for one output channel:

```
block (k, cb) = wt[k][8*cb + 0 .. 8*cb + 7]     k  = output channel
                                                cb = block index, 0 .. C_IN/8 - 1
```

If **all 8 weights in a block are zero**, the hardware skips that cycle. A block with
even one nonzero weight costs the full cycle. The depthwise layer doesn't skip: its 9
taps all run in the same cycle, so a zero tap saves nothing, and depthwise is only
~12% of this block's MACs.

## What the export must look like

- **Dense weights, nothing extra.** Export the full `wt[C_OUT][C_IN]` array exactly
  like the unpruned model (same Q1.7 int8 format), with pruned weights set to 0.
  The hardware finds zero blocks itself. No mask, bitmap or sparse index format is
  needed.
- **Prune in aligned blocks of 8 input channels.** For each output channel, zero
  whole groups of input channels `0–7, 8–15, 16–23, …`. The groups must line up with
  multiples of 8 in the layer's input-channel order.
- **Unstructured pruning doesn't help.** Measured: 50% of weights zeroed at random
  skipped **0.4%** of cycles, because 8 random weights are almost never all zero.
- **Bias stays as is.** Pruned blocks don't change how the bias is applied.

## Measured results (32 → 64 channels, 16×16 pixels, 4 blocks per output)

Layer cycles are measured in C/RTL co-simulation of the generated hardware on the K26
part (Vitis HLS 2024.2), not estimated. Same hardware for every row; "bypass" is
`skip_en = 0`.

| Blocks pruned | Weights zero | MAC cycles skipped | Layer cycles, skip on | Bypass | Speedup |
|---|---|---|---|---|---|
| 0% | 1.7% | 0% | 69,798 | 69,798 | 1.00× |
| 30% | 31.7% | 30.5% | 49,830 | 69,798 | 1.40× |
| 50% | 51.7% | 50.0% | 37,030 | 69,798 | 1.88× |
| 70% | 65.5% | 60.5% | 30,118 | 69,798 | 2.32× |
| 90% | 89.1% | 72.7% | 22,182 | 69,798 | 3.15× |
| 100% | 100% | 75.0% (floor) | 20,646 | 69,798 | 3.38× |
| 50% unstructured | 51.2% | 0.4% | 69,542 | 69,798 | 1.00× |

At 200 MHz the dense layer is 0.35 ms; 50% block pruning brings it to 0.19 ms.

Results are bit-identical to the unpruned computation at every level, with skipping
on and with the bypass bit off, in both C-sim and co-sim.

**Why the speedup trails the skip rate:** each pixel has ~15 cycles that skipping
can't remove (loading the pixel's 32 input channels, and the pipeline filling and
draining), on top of the 256 dense MAC cycles. The fewer MAC cycles are left, the
larger that fixed share becomes.

**The 75% ceiling:** every output channel still takes at least 1 cycle, even if all
its blocks are pruned, to write its output (which is then just the requantized
bias). So at high sparsity, an output channel with every block pruned wastes a cycle
on a constant. If whole output channels get pruned, it's better to **remove those
channels from the model** than to export them as all-zero rows.

## Runtime controls

- `skip_en = 1`: skip zero blocks. `skip_en = 0`: run every block (dense), on the same
  hardware, for the pruned-with-skip vs pruned-without-skip comparison.
- `sched_len` (output): cycles per pixel for the current weights. The skip rate is
  `1 - sched_len / (C_OUT * C_IN / 8)`.

## Things that would change this format

- **Block size = the pointwise unroll factor (`PW_UNROLL = 8`).** If that changes,
  the block size changes with it.
- **The fixed-point types** in `include/dsc_types.h` are still placeholders pending
  sign-off.
