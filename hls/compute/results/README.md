# Results: compute core and fused block

All numbers here are **measured** on the K26 part (`xck26-sfvc784-2LV-c`) with Vitis HLS /
Vivado 2024.2: cycle counts from C/RTL co-simulation, timing and resources from Vivado
place & route. Raw data: `cosim_results.csv`, `timing_results.csv`. Figures:
`make_plots.py` → `fig_speedup.png`, `fig_latency.png`.

## Zero-block skipping speedup

Same hardware, `skip_en` on vs off (bypass). From `cosim_results.csv`:

| Blocks pruned | MAC cycles skipped | `pointwise_skip_top` (s1) | **`pointwise_skip_pipe_top`** (s1) | `dsc_top` stride 1 | `dsc_top` stride 2 |
|---|---|---|---|---|---|
| 30% | ~28–31% | 1.40× | **1.44×** | 1.48× | 1.22× |
| 50% | ~45–50% | 1.88× | **1.94×** | 1.69× | 1.41× |
| 70% | ~61–64% | 2.32× | **2.72×** | — | — |
| 90% | ~73% | 3.15× | **3.70×** | 3.11× | 1.92× |
| all zero | 75% (cap) | 3.38× | **3.93×** | 3.22× | 1.96× |
| 50% unstructured | 0.4% | 1.00× | **1.00×** | 1.00× | 1.00× |

(Each test case draws its own random pruning, so the same "50%" lands on slightly
different skip rates in different suites. Speedups are always against the same kernel's
own bypass run.)

- **Speedup tracks the ideal `1 / (1 − skipped)` closely at stride 1.**
- **The gap at high sparsity is fixed per-pixel overhead, and the prototype removes it.**
  `pointwise_skip_top` spends ~16.6 cycles per pixel loading the input and restarting
  the pipeline. `pointwise_skip_pipe_top` loads the next pixel during the current one,
  so a call costs exactly `sched_len × pixels + 405` cycles: ~1.6 cycles per pixel of
  one-time setup, spread over the layer. At 90% pruning that's 3.70× instead of 3.15×,
  and with all weights zero 3.93× against a 4× ideal. The same structure would carry
  over to `pw_stage` in `dsc_top`.
- **Stride 2 caps near 2×.** The line buffer walks every input pixel at one channel per
  cycle, so depthwise, not pointwise, becomes the bottleneck.
- **Unstructured pruning gives nothing.** A block of 8 random weights is almost never
  all zero, so pruning must zero whole aligned blocks (`../SKIP_FORMAT.md`).

## Timing (post-route, 200 MHz target)

From `timing_results.csv`:

| Design | HLS estimate | Post-synth | **Post-route** | Fmax | 200 MHz? | LUT / FF / DSP / BRAM18 |
|---|---|---|---|---|---|---|
| `pointwise_skip_top` | 3.64 ns | 3.41 ns | **4.26 ns** | ~235 MHz | met | 557 / 607 / 8 / 0 |
| `pointwise_skip_pipe_top` (prototype) | 3.64 ns | 3.41 ns | **4.19 ns** | ~238 MHz | met | 663 / 576 / 10 / 0 |
| `dsc_top` (PR #2) | 3.65 ns | 5.09 ns | **5.54 ns** | ~181 MHz | **not met** | 3,662 / 5,162 / 17 / 30 |
| `dsc_top` + BRAM output register (fix) | 3.65 ns | 3.43 ns | **4.24 ns** | **~236 MHz** | **met** | 3,639 / 5,212 / 17 / 30 |
| `dsc_top` + fix + overlapped load | 3.65 ns | 3.43 ns | **4.28 ns** | **~234 MHz** | **met** | 3,704 / 5,330 / 19 / 30 |

**HLS estimates were wrong in both directions:**
- **Timing was optimistic.** HLS predicted ~274 MHz for both designs.
- **LUT use was pessimistic.** HLS estimated 12.4k LUTs for `dsc_top`; it actually
  uses 3.7k.
- **Takeaway:** only post-route numbers belong in the report.

### Why `dsc_top` misses 200 MHz

The worst path (slack −0.536 ns, 8 logic levels, 77% logic delay) runs from the
pointwise weight block RAM (`pw_wt_13`, `RAMB18E2`) straight into the DSP
multiply-add in `pw_stage`'s `PW_RUN` loop. In one cycle it does:

1. a block-RAM read **without the output register** (slow clock-to-out);
2. the DSP pre-adder and multiplier;
3. two adder stages into the partial sum.

`pointwise_skip_top` has the same multiply-add but receives its weights through a port,
not from an on-chip BRAM, which is why it meets timing with room to spare.

**Fix (tested): enable the BRAM output register**, one line in `dsc_top.cpp` after the
`pw_wt` partition pragma:

```cpp
#pragma HLS BIND_STORAGE variable=pw_wt type=ram_t2p impl=bram latency=2
```

Tested on a scratch copy of PR #2 (`ecf8a64` + this line):

- **Timing met:** post-route 4.243 ns (~236 MHz), slack +0.757 ns, up from 5.536 ns.
  The new worst path is the DSP multiply-add alone.
- **Resources:** unchanged (−23 LUT, +50 FF).
- **Throughput:** `PW_RUN` stays at II=1 with one more pipeline stage (depth 10 → 11).
- **Cost: +0.35% cycles.** The dense stride-1 layer goes from 71,282 to 71,534 cycles, about
  one cycle per pixel. Every pruning level shifts by the same ~250 cycles (rows
  `dsc_top+bram_lat2` in `cosim_results.csv`).
- **Correctness:** co-sim passes, bit-exact on all cases.

It's Hank's file, so this goes to him as a suggested change for #2.

## Overlapped load in the fused block (experiment on #2)

The prototype's structure ported into `pw_stage` in `dsc_top`, together with the BRAM fix
above. Tested on a scratch copy of #2 (`ecf8a64` + both changes; kept on a local-only
branch, not pushed to Hank's branch).

- **Correctness:** co-sim passes, bit-exact on all 43 calls, 0 dependence warnings.
- **Throughput:** `PW_RUN` at II=1.
- **Timing:** post-route 4.28 ns (~234 MHz), timing met.
- **Resources:** essentially unchanged.

Speedup vs. the same build's bypass (rows `dsc_top+bram_lat2+overlap` in `cosim_results.csv`):

| Blocks pruned | `dsc_top` as in #2 | + fix + overlapped load |
|---|---|---|
| dense (cycles, stride 1) | 71,282 | **67,208** (−5.7%) |
| 30% | 1.48× | **1.52×** |
| 50% | 1.69× | **1.77×** |
| 90% | 3.11× | **3.57×** |
| all weights zero | 3.22× | **3.72×** |
| stride 2, 90% | 1.92× | 1.92× (line-buffer cap) |

**This is the strongest fused-block result so far.** Stride 2 is unchanged, as
expected: there, depthwise and the line buffer are the bottleneck, not pointwise.

## Estimated power and energy per inference

**These are estimates, not measurements.** Vivado's post-route `report_power` is
*vectorless*: it assumes default switching activity, and rates its own confidence
"Medium". It covers the PL only, with no ARM processor or DDR. Board measurement
(INA260, see PR #3) replaces this when available. Data: `power_results.csv`.

| Design | Dynamic | PL device static | Total PL |
|---|---|---|---|
| `dsc_top` as in #2 | 96 mW | 289 mW | 385 mW |
| `dsc_top` + both proposed changes | 100 mW | 289 mW | 389 mW |
| `pointwise_skip_pipe_top` (standalone) | 19 mW | 288 mW | 307 mW |

- **Dynamic power comes from:** block RAM 30 mW, clocks 27 mW, logic 24 mW, signals
  14 mW, DSPs 4 mW.
- **Device static (~289 mW)** is drawn by the PL whether or not the accelerator runs.

**Energy per block** = power × measured co-sim latency, for the fused block with both
changes at 200 MHz, stride 1, skipping on (`fig_energy.png`):

| Blocks pruned | Latency | Dynamic energy | Total PL energy |
|---|---|---|---|
| dense | 336 µs | 34 µJ | 131 µJ |
| 30% | 221 µs | 22 µJ | 86 µJ |
| 50% | 190 µs | 19 µJ | 74 µJ |
| 90% | 94 µs | 9 µJ | 37 µJ |
| all weights zero | 90 µs | 9 µJ | 35 µJ |

**Why energy falls with sparsity:** the run is shorter at roughly the same power.

**Both estimates are simplifications:**
- Vectorless power doesn't vary with sparsity. Real dynamic power while skipping should
  be similar or a bit lower, so the pruned-case energy here is, if anything, an
  overestimate.
- Attributing all the device static to the inference assumes the PL would otherwise be
  off; if it stays powered anyway, only the dynamic column counts.

## Final design (PR #2 @ `a50c1b2`) and the KR260 bitstream

Hank merged all three proposed changes into #2: the single register bank, the BRAM
output register, and the overlapped `pw_stage`. He also deepened `s_dw` from 8 to 16.
Measured on his branch (rows `dsc_top_final`; `kr260_full_system` in
`timing_results.csv`):

- **Correctness:** co-sim passes, bit-exact on all 43 calls.
- **Stride 1:** cycle-for-cycle identical to the experiment above (3.72× with all
  weights zero, 93% of ideal).
- **Timing:** `dsc_top` post-route **4.329 ns (~231 MHz)**. 3,698 LUT / 5,292 FF / 19 DSP /
  30 BRAM18. Estimated power 101 mW dynamic + 289 mW PL static.

**Stride 2 with `s_dw` depth 16** (vs depth 8, same tests):

| Stride 2 | depth 8 | depth 16 | change | speedup now |
|---|---|---|---|---|
| dense | 18,880 | 18,061 | −4.3% | 1.00× |
| 30% blocks pruned | 15,380 | 13,581 | −11.7% | 1.33× |
| 50% | 13,126 | 11,201 | −14.7% | 1.61× |
| 90% / all zero | 9,843 / 9,801 | 9,843 / 9,801 | 0 | 1.83× / 1.84× |

- The 90% speedup *ratio* went down (1.92× → 1.83×) only because the dense baseline got
  faster; the high-sparsity cycle counts are unchanged.
- Hank's cycle model predicted no change for dense and −5% at 30%, so it needs
  recalibrating before it's quoted.

**KR260 bitstream** (`board/vivado/build_bd.tcl`, PR #4; PS + SmartConnects + `dsc_top`,
KR260 board preset):

- **Timing:** all constraints met at 200 MHz, WNS **+0.677 ns** (~231 MHz).
- **Utilization:** 5,056 LUTs (4.3%), 7,069 FFs (3.0%), 15 BRAM tiles (10.4%).
- **Register map** (one bank, from the generated `xdsc_top_hw.h`): `in` 0x10,
  `params` 0x1c, `out` 0x28, `stride` 0x34, `skip_en` 0x3c, `sched_len` 0x44 (valid
  0x48). It matches `board/sw/dsc_regs.h`'s static_asserts.

**Next is on-board measurement:** latency, FPS and energy with the INA260.
