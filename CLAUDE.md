# EECS298 Final Project — Hardware-Accelerated Depthwise-Separable Convolution Engine

An FPGA accelerator for depthwise-separable convolution (MobileNet-style) targeting the
**AMD Kria KR260**, built with **Vitis HLS (C++)** and assembled in **Vivado IP Integrator**.
Two features distinguish it from a plain conv accelerator:

1. **Zero-skipping / pruning hardware** — a Skip-Zero Control Unit that detects pruned
   zero-weights and dynamically skips the corresponding MAC cycles. ReLU6-based vision models
   produce many zeros naturally, so skipping those cycles is free throughput.
2. **Energy-adaptive operation** — read the Xilinx **SysMon** on the KR260 and adjust pipeline
   depth and clock speed in response to power/thermal conditions.

Source of truth for scope: `interim-layout.docx.pdf` (the project proposal) in the repo root.
The plans below are this team's working interpretation of that proposal — detailed enough to
start from, but each student should adjust their own section as design decisions firm up.

## Team

| Student | Name | Email | Core responsibility |
| --- | --- | --- | --- |
| Student 1 | Daniel Nguyen | danien12@uci.edu | Compute Engine & Zero-Skipping Logic |
| Student 2 | Hank Li | hankanl@uci.edu | Memory Subsystem, Streaming & Interconnect |
| Student 3 | Kevin Chhim | kevinlyc@uci.edu | Model Prep, Software Driver & Power Profiling |

## Shared interfaces — change only by agreement

These three things are touched by more than one person's code. Whoever proposes a change
should get explicit sign-off from whoever else depends on it before implementing — a silent
change here invalidates someone else's already-verified work.

- **Fixed-point type definitions** (`ap_fixed<W,I>` widths/scales for weights, activations, and
  accumulators) — needed by Daniel's compute kernels, Hank's buffer/stream widths, and Kevin's
  PyTorch quantization export. Lock this in Week 5.
- **Pruned-weight encoding** — how a "this weight is zero" decision reaches the hardware
  (e.g. a per-weight flag bit vs. a sparse index/run-length format). Written by Kevin (comes
  out of the pruning step), consumed by Daniel's zero-detection logic. The encoding's
  granularity must match the skip granularity Daniel designs around — these two decisions are
  really one decision and should be made together, not in parallel.
- **Top-level HLS function signature & interface pragmas** — the contract between Hank's
  streaming/AXI side and Daniel's compute core (stream types, burst widths, control registers).
  Agree on this before either side writes RTL, since it's expensive to change after both sides
  have built against it.

## Student plans

Each plan below is organized by the project's 6 weeks, but these are starting points, not a
rigid script — if your design decisions push a task into a different week, that's fine; just
keep the people depending on your outputs aware of the slip.

### Daniel — Compute Engine & Zero-Skipping Logic (Pruning Hardware)

**Scope:** C++ HLS for the depthwise (3×3) and pointwise (1×1) compute layers, plus the
zero-detection logic that dynamically skips MAC cycles on pruned zero-weights.

- **Week 5 — Architecture.** Propose the fixed-point types (coordinate with Kevin, who needs
  matching scales for quantization export, and Hank, who sizes buffers around them). Sketch
  the MAC datapath: MACs/cycle, unroll factor, target initiation interval. The key open design
  decision is **skip granularity** — per-element, per-tap, or per-channel-block skipping. Finer
  granularity skips more cycles but makes the skip-control state machine and its interaction
  with Hank's FIFOs more complex; coarser granularity is simpler to pipeline and verify.
  Recommend starting coarse (e.g. per-channel or per-row) and only going finer if profiling
  shows it's worth the complexity. Whatever granularity you pick, it has to match the encoding
  Kevin produces — settle that jointly, not independently. Draft the top-level function
  signature with Hank before either of you writes RTL.
- **Week 6 — Implementation.** Build the depthwise and pointwise kernels *without* skip logic
  first, and get them through C-sim — that's your correctness baseline. Then add zero-detection
  on top, reading Kevin's pruned-weight encoding and conditionally bypassing the
  multiply-accumulate. Apply `PIPELINE`/`UNROLL` pragmas and start checking rough `csynth`
  resource/II estimates now rather than waiting for Week 8 to discover a timing problem.
- **Week 7 — Verification & pruning.** Integrate Kevin's real pruned weights (not synthetic
  test zeros) and confirm skip logic actually fires and produces numerically identical results
  to the unpruned computation. Run C-sim against Kevin's golden vectors in both pruned and
  unpruned configurations. Start logging the measured fraction of MACs actually skipped — this
  number, cross-referenced with Kevin's chosen sparsity level, is one of the project's headline
  results.
- **Week 8 — Co-simulation.** First real integration with Hank's streaming blocks — expect
  interface mismatches at the stream boundary and budget time for them. Pay particular
  attention to timing closure around the skip-control logic specifically, since
  conditional/variable-latency paths are the likeliest source of a scheduling surprise. Export
  the IP block once co-sim passes.
- **Week 9 — Board support.** You're not leading IP Integrator assembly (Kevin is), but be
  available to explain your core's control/status registers. Worth adding a runtime-togglable
  "bypass zero-skipping" bit if it's cheap — it lets Week 10 benchmarking directly compare
  pruned-with-skip vs. pruned-without-skip vs. unpruned on the same bitstream.
- **Week 10 — Benchmarking.** Help explain any gap between the theoretical skip rate and the
  measured speedup (memory-bound stalls and pipeline bubbles are the usual suspects).

**Decide early, with the team, not alone:** skip granularity; whether skipped cycles produce
fixed or variable latency (variable latency pushes complexity onto Hank's FIFOs); how sparsity
pattern (structured vs. unstructured) affects whether skipping saves cycles only, or also chip
area.

### Hank — Memory Subsystem, Streaming & Interconnect (AXI/BRAM)

**Scope:** line buffers (`hls::LineBuffer`), sliding windows, internal FIFOs (`hls::stream`),
and AXI Master (`m_axi`) interfaces for burst DRAM access, plus ping-pong BRAM buffering to
overlap memory transfers with compute.

- **Week 5 — Architecture.** Decide the dataflow topology: how many pipeline stages, what
  streams connect them, and BRAM budget, sized against the test-image dimensions Kevin plans
  to use. Pick an initial `m_axi` burst width against expected DRAM bandwidth — you can tune
  this later. Confirm with Daniel the exact window size (3×3) the depthwise kernel consumes per
  cycle so the line buffer's dimensions are right from the start.
- **Week 6 — Implementation.** Build the line buffer / sliding-window generator and the
  `hls::stream` FIFOs between line-buffer → compute → output. Implement the AXI Master
  read/write path — start non-burst for simplicity, then move to burst once correctness is
  established. Build a standalone pass-through testbench so you can verify streaming
  independent of Daniel's compute kernel being finished.
- **Week 7 — Early integration.** Connect to Daniel's compute kernel as soon as it exists in
  any form, even before his skip logic is done — flushing out interface bugs early is cheaper
  than finding them in Week 8's co-sim. Validate sliding-window output against Kevin's test
  images. Implement and simulate the ping-pong buffering.
- **Week 8 — Co-simulation.** This is the real stress test for your FIFO depths and interface
  pragmas. If Daniel's skip logic produces variable-latency output, this is where you'll find
  out whether your FIFOs can absorb it without overflow/underflow — don't assume fixed latency
  without checking. Review BRAM utilization and adjust ping-pong sizing for the area/performance
  tradeoff.
- **Week 9 — Lead board integration.** This is your subsystem's home turf — take the lead on
  the Vivado block design's memory/AXI topology (interconnect, address map), while Kevin
  configures the Zynq Processing System and ARM-facing control. Agree explicitly on that split
  so you're not duplicating work. Make sure the DMA/memory-map addresses match what Kevin's
  software driver expects to read/write.
- **Week 10 — Benchmarking support.** Produce a latency breakdown (memory-bound vs.
  compute-bound time) — this is the data Kevin needs to explain the overall numbers, and it's
  also the evidence for whether ping-pong buffering is actually paying off.

**Decide early, with the team, not alone:** single deep buffer vs. double (ping-pong) buffer,
based on observed DMA latency vs. compute latency; AXI burst length, and whether it should
adapt to image size; FIFO depth, especially once Daniel's latency behavior (fixed vs.
variable) is known.

### Kevin — Model Prep, Software Driver & Power Profiling

**Scope:** PyTorch baseline model, structured pruning, quantization, and C testbench
verification; Vivado IP Integrator assembly, ARM software drivers, test-image benchmarking,
and energy profiling (Vivado Power Analyzer + Xilinx SysMon).

- **Week 5 — Baseline & encoding.** Stand up the PyTorch baseline model (MobileNet or
  equivalent depthwise-separable architecture) and get it producing correct outputs on a small
  set of test images — these become the reference vectors for everyone's testbenches. Decide
  the pruning strategy (structured channel pruning vs. structured block pruning) in direct
  coordination with Daniel, since it must match his skip-unit granularity — this is a joint
  decision, not Kevin's alone. Decide the quantization scheme (e.g. symmetric int/fixed-point)
  and propagate the agreed fixed-point formats to Daniel and Hank. Define and document the
  pruned-weight encoding Daniel's hardware will consume.
- **Week 6 — Export & driver skeleton.** Export baseline (unpruned) weights/activations in the
  agreed format, and build the golden C/C++ testbench vectors Daniel and Hank need for C-sim.
  Start an ARM software driver skeleton (register-map placeholders) now so Week 9 isn't a cold
  start once real addresses exist.
- **Week 7 — Pruning & test vectors.** Apply structured pruning to the baseline model,
  fine-tune to recover accuracy, and export both pruned weights and the matching
  sparsity-mask/encoding — this directly unblocks Daniel's verification task, so treat it as a
  hard dependency others are waiting on. Extend the golden testbench to cover pruned cases.
  Pick 2–3 target sparsity operating points (e.g. ~30%/50%/70%) to give the final report a
  tradeoff curve rather than a single data point.
- **Week 8 — Co-sim support & IPI skeleton.** Keep supplying accurate test vectors as Daniel
  and Hank's co-simulation surfaces edge cases. Start the Vivado IP Integrator project skeleton
  in parallel so Week 9 isn't starting cold. Begin drafting the ARM driver's MMIO calls against
  the tentative register map.
- **Week 9 — Board integration (lead) & driver.** Own the Vivado block design assembly: Zynq
  Processing System configuration, wiring in Daniel's and Hank's exported IP blocks, and
  generating the KR260 bitstream — coordinate the memory-topology/PS-config split with Hank
  (see Hank's Week 9 note). Finish the ARM driver: image loading, triggering the accelerator,
  reading back results, basic timing instrumentation.
- **Week 10 — Validation & benchmarking.** Deploy to the KR260 on Ubuntu and run the full
  validation suite: latency, FPS, and energy-per-inference via Vivado Power Analyzer and
  Xilinx SysMon, across the chosen sparsity operating points (and across clock/pipeline-depth
  settings if the energy-adaptive feature is implemented in time). Assemble the
  pruned-vs-unpruned (and per-sparsity-level) comparison numbers — this results table is the
  project's primary deliverable.

**Decide early, with the team, not alone:** pruning granularity must match Daniel's skip
granularity, agreed in Week 5; how many sparsity operating points are worth the extra
verification/benchmarking effort; whether the SysMon-driven energy-adaptive clock/pipeline
feature is fully implemented or scoped down to future work — flag this by Week 8 if time is
tight, since it's the one feature in the proposal with no hard owner-week assigned above.

## 6-week timeline

| Week | Milestone |
| --- | --- |
| 5 | **Concept & Architecture** — define specifications, establish fixed-point data types, prepare the pruned PyTorch baseline model. |
| 6 | **RTL & HLS Implementation** — write the C++ HLS for depthwise/pointwise cores and the line-buffer streaming blocks. |
| 7 | **Functional Verification & Pruning** — integrate zero-skipping logic for pruned weights; run Vitis HLS C-simulation. |
| 8 | **Synthesis & Co-Simulation** — run C/RTL co-simulation, verify timing closure, export the custom IP block. |
| 9 | **Board Integration & Deployment** — build the Vivado block design, map to the Zynq Processing System, generate the KR260 bitstream. |
| 10 | **Validation & Benchmarking** — deploy drivers on Ubuntu; measure latency, FPS, and energy-per-inference. |

## Target metrics

Latency, frame rate (FPS), and energy-per-inference on the KR260, compared against the
unpruned baseline, across sparsity operating points. Throughput gain from zero-skipping is the
headline result.

## Working notes for Claude

- The repo currently holds only the proposal; there is no build yet. When adding code, keep
  the three ownership areas in separate directories so the split stays legible
  (e.g. `hls/compute/` for Daniel, `hls/memory/` for Hank, `model/` + `sw/` for Kevin).
- HLS C++ is not ordinary C++: synthesizability comes first. No dynamic allocation, no
  recursion, bounded loops, and arrays sized at compile time. Prefer `hls::stream` over shared
  arrays between stages.
- Never change a fixed-point width, the pruned-weight encoding, or an interface pragma as a
  drive-by fix — these are the shared interfaces above, and changing one without sign-off
  invalidates another student's already-verified results. Flag it instead.
- Keep a golden C model alongside any kernel change so C-sim and co-sim stay meaningful.
- Reference works (cited in the proposal): Zhao et al. 2017, *FPGA-based CNN inference
  accelerator synthesized from multi-threaded C software*, IEEE TCAD 36(11)
  (doi:10.1109/TCAD.2017.2662057); Tragoudaras, Stoikos et al. 2022, *Design space exploration
  of a sparse MobileNetV2 using high-level synthesis and sparse matrix techniques on FPGAs*,
  Sensors 22(12):4318 (doi:10.3390/s22124318).
