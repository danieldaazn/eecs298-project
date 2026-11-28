# Live demo: CIFAR-10 through the FPGA blocks on the KR260

A small CNN whose two depthwise-separable blocks run on our `dsc_top` accelerator,
with a dashboard served by the board. Open `http://<board>:8080/` in any browser on
the same network or Tailscale.

```
32x32 RGB ─ ARM: conv3x3 3→16, conv3x3 16→32 s2 ─▶ FPGA block 1 (stride 1, 16×16×32 → 64)
          ─ ARM: conv1x1 64→32 ─▶ FPGA block 2 (stride 2 → 8×8×64) ─ ARM: avg pool, linear → 10 classes
```

**What the dashboard shows:**
- **Live classification:** test images, the prediction, and each block's latency on the
  FPGA and on one A53 core. A **bit-exact counter** compares every FPGA output byte with
  the same block computed on the CPU.
- **Controls:** pick the pruning level (0/50/75/90 % of pointwise 8-channel blocks
  zeroed) and turn zero-skipping on or off. Off is the dense bypass on the same bitstream.
- **Accuracy vs pruning,** measured on the board over the whole CIFAR-10 test set, in the
  background after start-up. The result is cached, so later starts skip it.
- **Benchmark button:** about 40 s. Measures the speedup from skipping per level and
  stride (against the ideal `256 / cycles-per-pixel`), and energy per inference from the
  INA260 (board power while running minus idle, × time per call), against the same
  blocks on an ARM core.
- **Live board power** and PS/PL temperature (AMS).

Without the bitstream the server runs in **SIM mode**: FPGA results come from the golden
model, latencies from the co-sim cycle model, and a red banner says so. That's how the
whole stack was developed and tested before the bitstream reached the board.

## Files

| Path | What |
|---|---|
| `model/train.py` | Trains the network with the hardware's exact arithmetic (Q1.7 / Q3.5, half-up, ReLU6 at 192), block-prunes pointwise weights at the skip granularity, and checks the export against an independent integer model. Writes `model/export/` |
| `model/check_pipeline.py` | Checks the board's C pipeline (`libdsc`, SIM mode, on a PC) against a float64 numpy reference, and reports its accuracy |
| `model/export/` | Exported weights per level (`s<p>/b1`, `s<p>/b2` in VECTORS.md format; `arm.npz`), `summary.json`, the training log. `test_images.npz` (CIFAR-10 test set) is generated, not committed |
| `board/libdsc.cpp` | C side: FPGA calls (UIO + u-dma-buf), the CPU baseline (golden model), the ARM float layers |
| `board/server.py` | Dashboard + JSON API, stdlib + numpy only (both preinstalled on Kria Ubuntu) |
| `board/static/index.html` | The dashboard |
| `board/dsc-demo.service` | systemd unit: at boot, load `dsc_accel` and start the server |
| `deploy.sh` | Copy to the board, build there, install firmware if present, (re)start the service |

## Deploy

On a PC with the repo (the CIFAR-10 python batches are in `cifar-10-batches-py/` under
`--data`):

```sh
python3 demo/model/train.py --data ~/.keras/datasets/cifar-10-batches-py-target   # ~15 min on a GPU
python3 demo/model/check_pipeline.py                                              # optional, ~2 min
demo/deploy.sh ubuntu@<board-ip>
```

One-time board setup is in `board/README.md`: u-dma-buf is installed and loads at boot
on our board. The bitstream goes in `~/eecs298-project/board/firmware/out/` on the board
(`dsc_accel.bit` and/or `.bit.bin`). Then:

```sh
sudo ./board/firmware/install.sh && sudo systemctl restart dsc-demo
```

`deploy.sh` does the same thing automatically if `out/` is already there.

## Before the presentation

1. Power the board on. Within about a minute the dashboard is up, and the badge says
   **Running on the FPGA** (green). If it says SIMULATED, the bitstream isn't installed:
   check `journalctl -u dsc-demo -b`.
2. Let the accuracy measurement finish (a few minutes per level, once; it's cached).
3. Press **Run benchmark** once, so the charts and table are filled in. That's also cached.
4. Take a screenshot as a fallback slide in case the network fails at the venue.
5. During the demo, try 0 % → 90 %, skipping on → off, and **Auto** for a slide show.

## How the numbers are measured

- **FPGA latency:** `ap_start` → `ap_done` as seen from user space, polling. Includes a
  few µs of register access around the accelerator's own cycles.
- **CPU baseline:** the golden model (`hls/compute/golden`), plain C at `-O3`, one A53
  core, dense. It's the reference implementation, not a NEON-optimized kernel; say so
  when quoting the FPGA-vs-CPU ratio.
- **Energy:** whole-SOM power from the INA260, averaged while one block runs back to
  back, minus the idle average taken just before. The FPGA runs are polled, so the
  number includes one A53 core spinning. That's a fair system-level comparison with the
  CPU baseline, which also keeps one core busy.
- **Accuracy:** the board's own pipeline over the 10,000 test images. Training-time
  accuracy (same arithmetic, on the GPU) is shown next to it in the table.
