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
