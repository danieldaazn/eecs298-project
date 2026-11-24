# Driving the accelerator: notes for the ARM driver

**For Kevin (Week 9 driver), from the compute side.** This covers what the registers
*mean* and how to use them: skipping, `sched_len`, switching weight sets, timing. The
top-level kernel `dsc_top` is Hank's (PR #2), and the register **addresses below are
provisional**:
- the review on #2 asks to move the buffer addresses into the same register bank as
  everything else, which will change them;
- once #2 merges, take every offset from the generated `xdsc_top_hw.h`, never from this
  file.

## What one call does

One call to `dsc_top` runs one complete depthwise-separable block:

1. read the input frame and a parameter blob (weights + biases) from DDR;
2. 3×3 depthwise → ReLU6;
3. 1×1 pointwise with zero-block skipping → ReLU6;
4. write the output frame to DDR.

Weights are reloaded on **every** call, so each call can use a different weight set.
The layer shape is fixed at compile time (currently 32×16×16 → 64, `dsc_types.h`).

## Registers (provisional map, PR #2 @ `ecf8a64`)

Today's IP has **two** AXI-Lite slaves; this is expected to become one.

**`control`:**

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x00 | `AP_CTRL` | R/W | bit 0 `ap_start` (write 1 to start), bit 1 `ap_done` (**clears on read**), bit 2 `ap_idle`, bit 3 `ap_ready` (clears on read), bit 7 `auto_restart`, bit 9 interrupt |
| 0x04 | `GIE` | R/W | bit 0: global interrupt enable |
| 0x08 | `IER` | R/W | bit 0: interrupt on `ap_done`; bit 1: on `ap_ready` |
| 0x0c | `ISR` | R/TOW | bit 0 `ap_done`, bit 1 `ap_ready`; write 1 to clear |
| 0x10 | `stride` | R/W | 2 = stride 2; **anything else runs as stride 1** |
| 0x18 | `skip_en` | R/W | bit 0: 1 = zero-block skipping, 0 = dense bypass |
| 0x20 | `sched_len` | R | pointwise cycles per pixel for the weights just loaded (see below) |
| 0x24 | `sched_len` ctrl | R | bit 0 `sched_len_ap_vld`: **clears on read** |

**`control_r` (64-bit buffer addresses, low word first):**

| Offset | Name |
|---|---|
| 0x10 / 0x14 | `in` (input frame) |
| 0x1c / 0x20 | `params` (parameter blob) |
| 0x28 / 0x2c | `out` (output frame) |

Both "clears on read" bits mean: **read each exactly once per run.** A second read
returns 0 even though the run finished.

## Per-call sequence (polling)

```c
write64(ctrl_r, IN,     in_phys);       // physically contiguous, non-cacheable buffers
write64(ctrl_r, PARAMS, params_phys);
write64(ctrl_r, OUT,    out_phys);
write32(ctrl, STRIDE,   1);             // or 2
write32(ctrl, SKIP_EN,  1);             // 0 = dense bypass, same hardware
write32(ctrl, AP_CTRL,  0x1);           // ap_start
while (!(read32(ctrl, AP_CTRL) & 0x2)) ;  // ap_done (clears on this read)
uint32_t len = read32(ctrl, SCHED_LEN);   // valid once ap_done is seen
```

- **Use non-cacheable DMA buffers.** The PL's HP ports aren't cache-coherent (Hank's note
  on #2). Cached buffers give stale input or output with no error.
- **Buffer formats:** `in` and `out` are HWC; `params` is the packed blob described in
  `hls/top/include/dsc_top.h`. `pack_input` / `pack_params` in
  `hls/top/tb/tb_dsc_top.cpp` are a C reference for the byte layouts. The test-vector
  files (`VECTORS.md`) are in PyTorch layout, so the driver packs once at load time.

## `skip_en` and `sched_len`

- **`skip_en = 0` is a true bypass:** the same bitstream runs every MAC cycle. That gives
  the three benchmark configurations on one bitstream:
  - pruned weights with skip;
  - pruned weights without skip (`skip_en = 0`);
  - unpruned weights.
- **`sched_len` is the hardware's own count of pointwise cycles per output pixel** for the
  weights it just loaded: 256 dense, down to 64 minimum (with 64 output channels). It's
  the cheapest end-to-end check that the right weights arrived:
  - `python3 hls/compute/tools/dsc_vectors.py check <case>` prints the expected value
    for a weight set;
  - with `skip_en = 0` it must read exactly 256.

  A mismatch means the parameter blob is wrong (layout, address or cache).
- **MAC-cycle skip rate** = `1 − sched_len / 256`. Measured speedup will be a bit lower
  because of fixed per-layer overhead (see Latency).

## Switching operating points

Keep one parameter blob per sparsity level (e.g. 30/50/70% pruned) in DDR, and change
only the `params` address between calls. No reconfiguration and no reset are needed.
This is the actuator the energy-adaptive governor in PR #3 would use, along with
`skip_en` and call pacing.

## Latency to expect (measured in co-sim, 16×16 stride 1, 200 MHz)

| Configuration | Cycles | Time |
|---|---|---|
| `dsc_top` as in #2, dense | 71,282 | 0.36 ms |
| `dsc_top` as in #2, 90% blocks pruned | 22,898 | 0.11 ms |
| with the timing fix + overlapped load (proposed for #2), dense | 67,208 | 0.34 ms |
| with the timing fix + overlapped load (proposed for #2), 90% pruned | 18,824 | 0.09 ms |

Per call, latency ≈ (pixels × per-pixel pointwise cost) + a fixed setup of a few hundred
cycles (loading weights, building the skip schedule). Stride 2 is capped near 2× by the
line buffer (see `results/README.md`).

**Clock:** run the PL clock at **200 MHz only with the BRAM timing fix** proposed on #2.
Without it, `dsc_top` closes at ~181 MHz post-route.

## Measuring

- **Latency:** time many back-to-back calls and divide. One call is a fraction of a
  millisecond, close to timer and polling noise.
- **Energy:** see PR #3. Measure over seconds of continuous calls and subtract an idle
  baseline, since the INA260 reads whole-SOM power in ~10 mW steps.
