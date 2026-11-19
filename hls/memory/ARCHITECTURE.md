# Memory subsystem, streaming & top-level architecture (Hank)

Covers Hank's Weeks 5–7 (architecture, line buffer / streams / AXI, early
integration, ping-pong), and prepares Weeks 8–10 (co-sim, board topology, latency
breakdown). Code: `hls/memory/` (streaming stages) and `hls/top/` (the fused top-level
kernel; joint Hank/Daniel interface).

## Status

| | State |
|---|---|
| C-sim, native ints (`make csim`) | **Passes**: `hls/memory` 11 tests, `hls/top` 21 tests, Daniel's 21 still pass after the refactor |
| C-sim, `ap_int` types | **Passes**, all three suites, using Xilinx's open-source [`ap_int` headers](https://github.com/Xilinx/HLS_arbitrary_Precision_Types) with g++ (no Vitis) |
| Mutation checks | Broken row shift, stride parity, byte lanes, burst address, short read, bias/weight decode, output packing, `skip_en` ignored, missing stride clamp — all caught |
| csynth / co-sim / timing | **Not run**: no Vitis on the machine this was written on. Scripts ready: `run_hls.tcl` in both dirs |
| Board | Not started (Week 9) |

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
| Topology | **Fused** DW→PW in one dataflow kernel; DW output never goes to DRAM | Hank's decision; Daniel to sign off the top |
| DRAM activation layout | **HWC** (channel-interleaved), 16 channels per 128-bit beat | **Needs Kevin**: export and driver must produce/consume HWC |
| Parameter blob layout | One buffer, layout in `hls/top/include/dsc_top.h` | **Needs Kevin** |
| Top-level signature and pragmas | `dsc_top` in `hls/top/include/dsc_top.h` | **Needs Daniel and Kevin** |
| Skip-logic reuse | Daniel's schedule factored into `pw_build_sched` / `pw_sched_step` (commit "Factor the pointwise skip schedule…"); behavior-preserving, his 21 tests pass | **Needs Daniel's review**, plus a csynth/co-sim rerun to confirm his cycle counts didn't move |
| `hls::LineBuffer` | Not used: it lived in `hls_video.h`, which Vitis HLS removed. Built from partitioned arrays instead (the Vitis-recommended pattern) | — |

## Dataflow topology

```
           gmem_rd (128b)                                                    gmem_wr (128b)
DRAM ──► read_input ──s_in──► window_gen ──s_win──► dw_stage ──s_dw──► pw_stage ──s_out──► write_output ──► DRAM
          (burst)    128b      line buf     72b    9 MAC/cyc   64b    8 MAC/cyc   128b    row ping-pong
                     d=32      2×W×C B      d=4    (Daniel)    d=8    + skip      d=8     2×(wo·4) beats
```

`dsc_top` first loads the parameter blob and builds the skip schedule (once per call,
about 570 cycles), then runs the dataflow region above.

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
The back-to-back-frame test checks that no stale data leaks across frames.

## Cycle model and bottleneck

Per call, with the placeholder layer (16×16, 32→64 channels):

- `window_gen`: (H+1)(W+1)·C = **9,248** cycles, independent of stride.
- `dw_stage`: ho·wo·C = 8,192 (stride 1) / 2,048 (stride 2).
- `pw_stage`: ho·wo·(n + ~15), where n = `sched_len` (256 dense, ≥64 at full sparsity)
  and ~15 is Daniel's measured per-pixel overhead (input gather + pipeline fill).

The model matches Daniel's measured co-sim layer cycles (69,798 / 49,830 / 37,030 at
0 / 30 / 50 %) to a constant ~420 cycles (his once-per-layer prologue).

| Blocks pruned | Stride 1: PW cycles | bound | speedup | Stride 2: PW cycles | bound | speedup |
|---|---|---|---|---|---|---|
| 0% | 69,376 | PW | 1.00× | 17,344 | PW | 1.00× |
| 30% | 49,408 | PW | 1.40× | 12,352 | PW | 1.40× |
| 50% | 36,608 | PW | 1.90× | 9,152 | **line buffer** (9,248) | 1.88× |
| 90% | 21,760 | PW | 3.19× | 5,440 | **line buffer** | 1.88× |

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
| `s_dw` | 8 | 2 pixels; pointwise gathers 4 at once, and dw produces a pixel in 32 cycles vs ≥79 for pw |
| `s_out` | 8 | 2 pixels; covers write_output's row turnaround |

Week 8: check co-sim's dataflow viewer for stalls on `s_dw` (pw starving), which
should only happen at stride 2 with high sparsity (see above).

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
sizes.

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
Behavior worth knowing for the driver:

- `stride` other than 2 runs as 1. A stray value can't hang the pipeline (tested).
- `sched_len` is valid after `ap_done`. The skip rate is `1 - sched_len/256`.
- `skip_en = 0` is Daniel's dense bypass on the same bitstream (pruned-with-skip vs
  pruned-without-skip, for Week 10).

## Week 9 draft: Vivado block-design topology (for Kevin's IPI assembly)

Proposed split: Hank owns the memory/AXI side below, Kevin owns the PS configuration
and the driver.

- `m_axi_gmem_rd` → `S_AXI_HP0_FPD`, `m_axi_gmem_wr` → `S_AXI_HP1_FPD`, each through
  its own SmartConnect, 128-bit.
- `s_axi_control` ← `M_AXI_HPM0_FPD` (or LPD) via SmartConnect. Its base address comes
  from the Vivado address editor and goes in Kevin's driver.
- `interrupt` → `pl_ps_irq0`. One PL clock at 200 MHz to start, which is also the
  clock the energy-adaptive feature would scale.
- **The HP ports are not cache-coherent.** The driver must use non-cacheable,
  physically contiguous buffers (e.g. u-dma-buf/CMA), or flush before start and
  invalidate after done. If that's painful, `S_AXI_HPC0/1_FPD` are coherent, at some
  bandwidth cost, which we don't need anyway.

## Week 10 plan: latency breakdown

Measure total cycles (ap_start→ap_done) at each sparsity point and both strides, then
compare against the model above: prologue (~570) + max(pointwise, window_gen) +
drain. The gap is the "memory-bound vs compute-bound" evidence: given the bandwidth
numbers, it should be small at stride 1, and show the line-buffer ceiling at stride 2.

## Unverified (needs Vitis; do on the first csynth)

1. `window_gen` reaching II=1: relies on the `DEPENDENCE ... dependent=false` pragmas.
   The true reuse distance is ≥C iterations.
2. `write_output`'s **DATAFLOW-in-a-loop** being accepted as a nested dataflow region
   with a PIPO `buf`. Fallback: remove that pragma (single buffer, see above).
3. Array partitioning of `pw_wt` / `dw_wt` propagating from `dsc_top` into the
   dataflow region's processes (needed for 8-wide and 9-wide reads at II=1).
4. `load_params` II (stated as 1–2 per loop; it's a ~200-cycle prologue either way).
5. Timing at 200 MHz, resources, co-sim pass, and whether Daniel's cycle counts are
   unchanged after the refactor.

## Open items for the team

- **Kevin:** confirm the fixed-point types; produce HWC activations and the parameter
  blob (`pack_input` / `pack_params` in `hls/top/tb/tb_dsc_top.cpp` are a C reference
  for the exact byte layout); tell us the test-image sizes and whether the model has
  stride-2 blocks at high sparsity (line-buffer ceiling).
- **Daniel:** review the refactor commit and the stream adapters `dw_stage` /
  `pw_stage` in `hls/top/src/dsc_top.cpp`; sign off the `dsc_top` signature.
- **All:** the kernel runs **one fixed layer shape** (compile-time `DW_H/W/C`,
  `PW_C_OUT`). A real MobileNet has many shapes, so we need to pick between runtime
  dimensions with compile-time maxima (my preference: one bitstream, and the line
  buffer and FIFOs are already sized by maxima) and one kernel per shape. This touches
  `dsc_types.h` and Daniel's cores, so it's a joint decision.
