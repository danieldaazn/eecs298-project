# Memory subsystem, streaming & top-level architecture (Hank)

Covers Hank's Weeks 5–7 (architecture, line buffer / streams / AXI, early
integration, ping-pong), and prepares Weeks 8–10 (co-sim, board topology, latency
breakdown). Code: `hls/memory/` (streaming stages) and `hls/top/` (the fused top-level
kernel; joint Hank/Daniel interface).

## Status

| | State |
|---|---|
| C-sim, native ints (`make csim`) | **Passes**: `hls/memory` 11 tests, `hls/top` 21 cases, Daniel's 29 (on the merged code) |
| C-sim, `ap_int` types | **Passes**, all three suites, using Xilinx's open-source [`ap_int` headers](https://github.com/Xilinx/HLS_arbitrary_Precision_Types) with g++ (no Vitis) |
| Mutation checks | Broken row shift, stride parity, byte lanes, burst address, short read, bias/weight decode, output packing, `skip_en` ignored, missing stride clamp, and (overlapped `pw_stage`) no buffer swap, prefetch into the live half, prefetch past the last pixel, wrong prefetch beat — all caught |
| csynth / co-sim / IP export | **Pass** (Daniel, Vitis HLS 2024.2, `xck26-sfvc784-2LV-c`, at `ecf8a64`): every pipelined loop at its target II, co-sim bit-exact on all 43 calls |
| Post-route timing | `ecf8a64`: 5.54 ns (~181 MHz), **misses 200 MHz**. This branch (`a50c1b2`, BRAM output register + overlapped `pw_stage` + `s_dw` 16): **4.329 ns, ~231 MHz, met**; co-sim bit-exact on all 43 calls (Daniel, 2026-11-26) |
| Resources (post-route) | 3,704 LUT / 5,330 FF / 19 DSP / 30 BRAM18 (~3 % / 2 % / 2 % / 10 % of the K26) |
| Board flow | `board/`, see `board/README.md`. **Bitstream built** by Daniel (Vivado 2024.2, KR260 preset): WNS +0.677 ns at 200 MHz, 5,056 LUT (4.3 %). Overlay, `install.sh` and `dsc_bringup` on the board: **not yet run** |

To repeat the `ap_int` C-sim without Vitis, from `hls/top/`:

```sh
AP=/path/to/HLS_arbitrary_Precision_Types/include
g++ -std=c++14 -O2 -DDSC_USE_AP_TYPES -I$AP -Iinclude -I../memory/include -I../memory/compat \
    -I../compute/include -I../compute/golden src/dsc_top.cpp ../memory/src/mem_stages.cpp \
    ../compute/src/*.cpp ../compute/golden/dsc_golden.cpp tb/tb_dsc_top.cpp -o /tmp/tb && /tmp/tb
```

## Decisions and sign-offs

| Item | Decision | Status |
|---|---|---|
| Fixed-point types (`dsc_types.h`) | Accept Daniel's u8 Q3.5 / s8 Q1.7 / 32-bit acc as-is. In the fused design the accumulator never leaves the core, so only the 8-bit activation sets stream widths | **Hank signed off** 2026-11-19. Kevin still to confirm |
| Window size | Confirmed from Daniel's code: `dw_window_mac` takes one full 3×3 window of **one channel per cycle**. Line buffer sized for that | Done |
| Topology | **Fused** DW→PW in one dataflow kernel; DW output never goes to DRAM | **Daniel signed off** (review of PR #2) |
| DRAM activation layout | **HWC** (channel-interleaved), 16 channels per 128-bit beat | **Needs Kevin**: export and driver must produce/consume HWC |
| Parameter blob layout | One buffer, layout in `hls/top/include/dsc_top.h` | **Needs Kevin** |
| Top-level signature and pragmas | `dsc_top` in `hls/top/include/dsc_top.h`; all registers in **one** AXI-Lite bank `control` (Daniel's requested change, done) | **Daniel signed off**. Kevin still to confirm (register map below) |
| Skip-logic reuse | Daniel's schedule factored into `pw_build_sched` / `pw_sched_step` | **Daniel approved**: same IIs in co-sim, his tests pass on the merged code |
| `pw_wt` storage | `BIND_STORAGE ... impl=bram latency=2` (BRAM output register) | Daniel's timing fix, taken as-is: 5.54 → 4.24 ns post-route, +1 cycle per pixel |
| `pw_stage` structure | Overlapped load: one II=1 pipeline over all (pixel, schedule entry) pairs; the next pixel is prefetched into a double buffer | Daniel's design, taken as-is: removes the ~15-cycle per-pixel gap (3.22× → 3.72× at full sparsity) |
| `hls::LineBuffer` | Not used: it lived in `hls_video.h`, which Vitis HLS removed. Built from partitioned arrays instead (the Vitis-recommended pattern) | — |

## Dataflow topology

```
           gmem_rd (128b)                                                    gmem_wr (128b)
DRAM ──► read_input ──s_in──► window_gen ──s_win──► dw_stage ──s_dw──► pw_stage ──s_out──► write_output ──► DRAM
          (burst)    128b      line buf     72b    9 MAC/cyc   64b    8 MAC/cyc   128b    row ping-pong
                     d=32      2×W×C B      d=4    (Daniel)    d=16   + skip,     d=8     2×(wo·4) beats
                                                                      input dbl buf
```

`dsc_top` first loads the parameter blob and builds the skip schedule (once per call,
about 1k cycles), then runs the dataflow region above.

| Stream | Element | Rate | Why this width |
|---|---|---|---|
| `s_in` | 128 b = 16 ch of one pixel | 1 beat / 16 cycles | = one AXI beat; no re-packing at the AXI side |
| `s_win` | `win_t`, 9 × 8 b | 1 / cycle | exactly what `dw_window_mac` consumes per cycle |
| `s_dw` | 64 b = 8 channels | 1 / 8 cycles | = one pointwise MAC cycle (`PW_UNROLL`), as Daniel's core assumes |
| `s_out` | 128 b = 16 out ch | 4 per pixel | = one AXI beat |

**Window generator** (`window_gen`): walks a (H+1)×(W+1) grid. The extra row and
column are the bottom/right zero padding, fed without reading the stream. For each
grid point it walks the C channels at II=1: two rows come from the line buffer, the
new row from `s_in`, and a 2-column history per channel (`hist[C][3][2]`) forms the
window. Top/left padding is masking, so the buffers are never cleared between frames.
`line` and `hist` are `static` so C-sim keeps their contents between calls like the
RAMs do; the back-to-back-frame test then really runs frames 2–3 on top of frame 1's
leftovers.

## Cycle model and bottleneck

Per call, with the placeholder layer (16×16, 32→64 channels):

- `window_gen`: (H+1)(W+1)·C = **9,248** cycles, independent of stride.
- `dw_stage`: ho·wo·C = 8,192 (stride 1) / 2,048 (stride 2).
- `pw_stage`: ho·wo·n, where n = `sched_len` (256 dense, ≥64 at full sparsity). The
  overlapped load hides the input gather, so there is no per-pixel overhead left.
  (Before it: ho·wo·(n + ~15).)
- Fixed per call: ~1.7k cycles measured (parameter load + schedule build, then
  pipeline fill, since the first window needs a row and a half of input, and drain).

Measured (Daniel's co-sim of this branch, `a50c1b2`, rows `dsc_top_final` in
`hls/compute/results/cosim_results.csv`; stride 2 also shown at the old `s_dw` depth 8):

| sched_len (pruned) | Stride 1 cycles | speedup | Stride 2, `s_dw` 8 | Stride 2, `s_dw` 16 | speedup | stride-2 bound |
|---|---|---|---|---|---|---|
| 256 (dense) | 67,208 | 1.00× | 18,880 | **18,061** | 1.00× | pointwise |
| 166 / 186 (30 %) | 44,171 | 1.52× | 15,380 | **13,581** | 1.33× | pointwise |
| 142 / 141 (50 %) | 38,025 | 1.77× | 13,126 | **11,201** | 1.61× | pointwise / line buffer |
| 67 / 68 (90 %) | 18,824 | 3.57× | 9,843 | 9,843 | 1.83× | line buffer |
| 64 (all zero) | 18,061 | 3.72× | 9,801 | 9,801 | 1.84× | line buffer |

**Stride 1 is exactly `256·sched_len + 1,672` cycles** (±5 at every sparsity level):
pointwise-bound, no per-pixel overhead left, and a fixed ~1.7k per call (parameter
load + schedule build ~1k, pipeline fill/drain).

**Stride-2 FIFO finding (Week 8), confirmed in co-sim.** At stride 2, `window_gen`
emits windows only on every other input row. During each window-less row, pointwise
drains the 2 pixels a depth-8 `s_dw` holds and then idles. **`s_dw` is now 16 deep
(4 pixels).** Measured: −4.3 % (dense), −11.7 % (30 %), −14.7 % (50 %) stride-2 cycles,
no change at stride 1 or at 90 %+. Stride-2 dense is now exactly pointwise-bound
(18,061 = 64·256 + ~1.7k). Cost: 8 more 64-bit FIFO entries (SRL, a few LUTs).
`hls/memory/tools/dataflow_model.py` found the effect but underestimated it at dense
and 30 % (it predicted 0 and −5 %). It matches stride 1 to within 5 cycles, but at
stride 2 it misses some stall in the window path, so **don't quote its stride-2
numbers**; the co-sim table above supersedes them.

**Finding: at stride 2 the line buffer caps the skip speedup at ~1.9×.** The window
generator walks every input pixel at one channel per cycle, but stride 2 has 4× fewer
output pixels, so above ~50 % block sparsity pointwise outruns it. This doesn't matter
for stride-1 layers. If Kevin's model has stride-2 blocks at high sparsity, the fix is
a 2-channels-per-cycle window generator (2 windows/cycle, and 2 `dw_window_mac`
instances in `dw_stage`). Not done; raise it once Kevin's layer list exists.

**Memory is not the bottleneck.** Stride 1 moves 27.5 KB per call (8 KB in, 16 KB out,
2.9 KB params): 0.4 B/cycle dense and 1.4 B/cycle at 100 % sparsity, against
16 B/cycle for one 128-bit port. The design is compute-bound by ~10–40×.

## Ping-pong vs single buffer

- **Input side: no explicit ping-pong.** `read_input` bursts freely into `s_in`
  (depth 32 = 512 cycles of `window_gen` demand, well above DRAM latency), and the
  line buffer is the on-chip buffer. Dataflow already overlaps transfer with compute.
- **Output side: row ping-pong** (`write_output`). A `DATAFLOW` pragma on the row loop
  turns the row buffer into a PIPO, so row r+1 is collected while row r is burst out,
  and the AXI write channel only sees complete back-to-back rows (no mid-burst stalls
  from the slow, sparsity-dependent pointwise rate). Cost: 2 × (wo·4) beats × 16 B =
  2 KB.
- **Honest tradeoff:** a single buffer stalls pointwise only if `s_out` fills while a
  row bursts out. A 64-beat burst takes ~150–200 cycles. `s_out` (2 pixels) holds
  ~540 cycles of pointwise output when dense, so dense layers wouldn't notice. At
  full sparsity it holds only ~160 cycles, which is about the burst time, so that is
  where ping-pong pays off (a few % at most). To A/B in Week 8, delete the `DATAFLOW`
  pragma in `write_output` (that turns it into a single buffer) and compare co-sim
  cycles at 0 % and 90 % sparsity, plus BRAM. Deepening `s_out` is the other cheap
  option.

## FIFO depths

The chain is linear (no reconvergent paths), so it can't deadlock on depth. Per
Daniel's design, latency is **fixed per layer** (the schedule depends only on weights),
so no FIFO has to absorb data-dependent latency. Depths only need to keep pointwise
from starving:

| FIFO | Depth | Reasoning |
|---|---|---|
| `s_in` | 32 | lets the read burst run a full row ahead and hide DRAM latency |
| `s_win` | 4 | producer and consumer are both II=1 |
| `s_dw` | 16 | 4 pixels. `pw_stage` prefetches the next pixel's 4 beats during schedule entries 16–19, and dw produces a pixel in 32 cycles vs ≥64 for pw, so at stride 1 they are always there. 16 rather than 8 covers stride 2's window-less rows (see "Stride-2 FIFO finding") |
| `s_out` | 8 | 2 pixels; covers write_output's row turnaround |

Week 8 result (Daniel's co-sim): no deadlock or overflow at any sparsity, and stride 1
is pointwise-bound as modeled. Latency really is fixed per layer. The one depth change
is `s_dw` 8 → 16, for stride 2 (above).

**`HLS 200-1449` warnings** ("process has both a predecessor and reads an input from its
caller") on `dw_stage`, `pw_stage` and the writer: the weights and `sched` are built
before the dataflow region and passed in. Harmless here. It only stops *overlapping
consecutive calls* (call i+1's load during call i's drain), which we don't do because
`load_params` runs first each call. Cost: the ~1.7k fixed cycles per call, ~2.5 % of a
dense call.

## BRAM budget

The placeholder layer (W=16, C=32, C_out=64) is tiny. Formulas so the real numbers fall
out once Kevin picks image sizes (all compile-time maxima in `dsc_types.h`):

| Buffer | Size | 16×16×32 | e.g. W=112, C=64 |
|---|---|---|---|
| Line buffer `line` | 2 · W · C bytes, 2 banks | 1 KB (2 BRAM18) | 14 KB (~8 BRAM18) |
| Window history `hist` | 6 · C bytes | 192 B (LUTRAM) | 384 B |
| Output ping-pong | 2 · W_out · C_out bytes | 2 KB (LUTRAM or ≤8 BRAM18) | 14 KB |
| PW weights + schedule | C_out · C_in + 256 entries | 2 KB + ~0.6 KB | — |

The K26 has 288 BRAM18 + 64 URAM, so the buffers are not a constraint at MobileNet
sizes. **Measured (post-route): 30 BRAM18 for the whole `dsc_top`** (~10 %). Most of
that is `pw_wt` split into 8 cyclic banks (one RAMB18 each, mostly empty at 2 KB
total) and the m_axi adapters' FIFOs, not the line buffer. The ping-pong buffer is
2 KB, so keeping it costs at most a few BRAM18; not worth trading away.

## AXI

- 128-bit data on both masters (the native width of the Zynq US+ `S_AXI_HP*_FPD`
  ports). `max_*_burst_length=64` (1 KB; the adapter splits at 4 KB boundaries).
- Two bundles: `gmem_rd` (input + params, which are never concurrent: params load
  first) and `gmem_wr` (output). That gives two masters, so reads and writes don't
  contend.
- `depth=` on every `m_axi` port is set for co-sim only.

## Top-level interface (proposed; `hls/top/include/dsc_top.h`)

```cpp
void dsc_top(const axi_word_t *in, const axi_word_t *params, axi_word_t *out,
             int stride, bool skip_en, int &sched_len);
```

All scalars and the three buffer addresses are `s_axilite` registers in bundle
`control`, plus `ap_ctrl_hs` (start/done/idle + interrupt). **Kevin: take register
offsets from the generated `xdsc_top_hw.h` after csynth, don't hand-code them.**

Final map (one bank). Generated by Vitis at `a50c1b2` and committed as
`board/sw/xdsc_top_hw.h` in #4, where `board/sw/dsc_regs.h` checks it at compile time:

| Offset | Register |
|---|---|
| 0x00 / 0x04 / 0x08 / 0x0c | `AP_CTRL` / `GIE` / `IER` / `ISR` (see `hls/compute/DRIVER_GUIDE.md`) |
| 0x10, 0x14 | `in_r` low, high |
| 0x1c, 0x20 | `params` low, high |
| 0x28, 0x2c | `out_r` low, high |
| 0x34 | `stride` |
| 0x3c | `skip_en` |
| 0x44 | `sched_len` (read-only) |
| 0x48 | `sched_len_ap_vld` (bit 0, clears on read) |

Behavior worth knowing for the driver:

- `stride` other than 2 runs as 1. A stray value can't hang the pipeline (tested).
- `sched_len` is valid after `ap_done`. The skip rate is `1 - sched_len/256`.
- `skip_en = 0` is Daniel's dense bypass on the same bitstream (pruned-with-skip vs
  pruned-without-skip, for Week 10).

## Week 9: board integration (`board/`)

Everything is scripted. `board/README.md` has the steps, from the HLS export to a
passing run on the KR260.

- **Block design** (`board/vivado/build_bd.tcl`), all at `pl_clk0` = 200 MHz:
  - `m_axi_gmem_rd` → `S_AXI_HP0_FPD` and `m_axi_gmem_wr` → `S_AXI_HP1_FPD`, each
    through its own SmartConnect, 128-bit.
  - `s_axi_control` ← `M_AXI_HPM0_FPD` via SmartConnect, at **0xA000_0000**, 64 KB.
  - `interrupt` → `pl_ps_irq0[0]` (GIC SPI 89).
  - The script fails without copying a bitstream if WNS < 0.
- **PS configuration:** the KR260 board preset plus six overrides, marked in the
  script for **Kevin's review** (he owns the PS side). Kria Ubuntu loads the PL at
  runtime and leaves the PS as booted, so the device-tree overlay is what actually sets
  the PL clock and the AFI widths.
- **Firmware** (`board/firmware/`): `xmutil loadapp dsc_accel` loads a `.bit.bin`, a
  `.dtbo` and a `shell.json` from `/lib/firmware/xilinx/dsc_accel/`. `install.sh`
  builds them on the board and loads them. The overlay binds the accelerator to
  `generic-uio`, so the driver runs entirely from user space: registers via `mmap`,
  the interrupt via `read()`.
- **Buffers:** the HP ports are not cache-coherent. The bring-up program uses one
  u-dma-buf opened `O_SYNC` (uncached), so no cache maintenance is needed. It copies
  into the buffer with aligned 64-bit accesses, because uncached memory on arm64 is
  Device memory, where `memcpy`'s unaligned accesses can fault. Kevin's driver can
  copy this.
- **Bring-up / benchmark** (`board/sw/dsc_bringup`):
  - Checks every output byte against the golden model, plus the overrun guard and
    `sched_len`, at 0–100 % block sparsity × both strides × skip on/off.
  - Accepts Kevin's VECTORS.md case folders (`--case DIR`).
  - Times each point and writes a CSV for the results table.
  - Register offsets are the prediction above. Copying the generated `xdsc_top_hw.h`
    into `board/sw/` turns them into compile-time checks.

## Week 10: latency breakdown

What the numbers should look like, from co-sim and the model:

- **Compute, stride 1:** `256·sched_len + 1,672` cycles. Dense is 67,208 cycles, 336 µs.
- **Memory traffic:** ~11 KB read (input + params) and ≤16 KB written per call. At
  ~3.2 GB/s per HP port that is under 5 µs on each port, and it streams in parallel
  with compute rather than before or after it.
- So the block is **compute-bound at every sparsity level**. What limits speedup is
  the fixed ~1.7k-cycle prologue and drain, plus (at stride 2) the line buffer's
  9-reads-per-output ceiling. DRAM bandwidth isn't a factor.

`dsc_bringup` prints measured time next to this model:

- skip on, skip off, and the speedup;
- the equivalent cycles at 200 MHz;
- the model's predicted cycles.

The measured time includes a few µs of user-space start/finish handshake. Measured
minus model gives three things:

- the per-call software overhead;
- whether DRAM latency shows up at all (it shouldn't);
- for Kevin's report, the memory-bound vs compute-bound split.

The same numbers settle ping-pong buffering. Double-buffering whole frames only pays off
if transfer time is comparable to compute. Predicted, it is under 5 µs against 90–336 µs,
so it isn't. Once the board confirms that, it is the evidence for the single-buffer
streaming design (see "Ping-pong vs single buffer").

## Verified by Daniel's Vitis runs (at `ecf8a64`)

1. `window_gen` at II=1 (`WG_ROWS_WG_COLS_WG_CH`), `DW_PIX_DW_CH` at II=1.
2. `write_output`'s DATAFLOW-in-a-loop synthesized as a nested dataflow region (PIPO).
3. `pw_wt` / `dw_wt` partitioning carried into the dataflow processes (`PW_RUN` II=1).
4. `load_params` at the predicted IIs; `PW_SK_MASK` at II=2.
5. Co-sim pass (43 calls), IP export pass.

## Confirmed on this branch (Daniel, 2026-11-26, `a50c1b2`)

`run_hls_win.bat cosim export impl`: co-sim bit-exact on all 43 calls, every pipelined
loop at II=1, post-route 4.329 ns, register map as above. The full KR260 bitstream
(PR #4) meets 200 MHz with WNS +0.677 ns.

## Open items for the team

- **Kevin:** confirm the fixed-point types; produce HWC activations and the parameter
  blob (`hls/top/include/dsc_pack.h` is the C implementation of the exact byte layout,
  shared by the testbench and the board bring-up); tell us the test-image sizes and
  whether the model has stride-2 blocks at high sparsity (line-buffer ceiling).
- **Kevin:** review the PS overrides in `board/vivado/build_bd.tcl` (Daniel's build
  applied them without errors; nobody has reviewed them yet).
- **All:** the kernel runs **one fixed layer shape** (compile-time `DW_H/W/C`,
  `PW_C_OUT`). A real MobileNet has many shapes, so we need to pick between runtime
  dimensions with compile-time maxima (my preference: one bitstream, and the line
  buffer and FIFOs are already sized by maxima) and one kernel per shape. This touches
  `dsc_types.h` and Daniel's cores, so it's a joint decision.
