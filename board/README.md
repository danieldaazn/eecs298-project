# KR260 bring-up: from HLS export to a passing run on the board

| Path | What | Runs on |
|---|---|---|
| `vivado/build_bd.tcl`, `build_bd_win.bat` | Block design, bitstream, XSA, `.bit.bin` | PC with Vivado 2024.2 |
| `firmware/dsc_accel.dts` | Device-tree overlay: PL clock, AFI widths, accelerator node (UIO) | compiled on the board |
| `firmware/install.sh`, `shell.json`, `dsc_accel.bif` | Installs to `/lib/firmware/xilinx/dsc_accel/` and loads with `xmutil` | board |
| `sw/dsc_bringup.cpp`, `dsc_regs.h`, `Makefile` | Self-test vs golden, VECTORS.md cases, latency/speedup table | board (or `make hostsim` on any Linux PC) |

Design summary and the expected numbers: `hls/memory/ARCHITECTURE.md`, "Week 9" and
"Week 10".

**Status:**
- **Checked:**
  - `build_bd.tcl` runs end to end under `tclsh` with Vivado's commands stubbed, covering
    the board-files, no-board-files, `bd`-only, timing-fail and missing-IP paths.
  - `install.sh` was dry-run with stubbed `dtc`, `bootgen` and `xmutil`.
  - `dsc_bringup` passes end to end against the C-sim kernel (`make -C board/sw
    hostsim`). Deliberately broken outputs, overruns and `sched_len` values all make it
    fail.
- **Not checked:** anything that needs real Vivado, `dtc` or the board. That's every
  step below from step 2 on.

## 1. Export the HLS IP (PC, Vitis HLS)

```bat
hls\top\run_hls_win.bat cosim export impl
```

The log (`hls\top\dsc_top_run.log`) must show co-sim PASS. The IP lands in
`hls/top/proj_dsc_top/solution1/impl/ip`.

Then copy `hls/top/proj_dsc_top/solution1/impl/misc/drivers/dsc_top_v1_0/src/xdsc_top_hw.h`
into `board/sw/` and commit it. From then on the build checks the register offsets in
`dsc_regs.h` against it.

## 2. Build the bitstream (PC, Vivado)

First, once: install the KR260 board files (Vivado → Tools → Vivado Store → Boards →
"Kria KR260 Robotics Starter Kit"). Without them the script still builds, on the bare
part with a warning. That bitstream still works with the steps below, but the XSA
can't be used to build a boot image.

```bat
board\vivado\build_bd_win.bat
```

On Linux: `cd board/vivado && vivado -mode batch -source build_bd.tcl`. Use
`build_bd_win.bat bd` to stop after the block design, which is handy for opening it in
the GUI. Takes roughly 20–40 minutes. Results go to `board/firmware/out/`:

| File | Use |
|---|---|
| `dsc_accel.bit` | bitstream |
| `dsc_accel.bit.bin` | the same bitstream in the format the FPGA manager loads |
| `dsc_accel.xsa` | hardware platform |
| `timing.rpt` | timing report |
| `utilization.rpt` | resource report |

The script stops before writing these if timing fails (WNS < 0).

## 3. Set up the board (first time only)

1. Write the **Kria Ubuntu 22.04** image for KR260 to a ≥32 GB microSD card
   (ubuntu.com/download/amd). Boot it, and log in on the USB-UART console (115200 8N1,
   the second of the four ttyUSB ports) or over Ethernet. The first login is
   `ubuntu`/`ubuntu`, which forces a password change. If it doesn't boot, the SOM's
   QSPI boot firmware is probably older than the image needs; AMD's KR260 "Getting
   Started" page has the update.
2. Install the tools:
   ```sh
   sudo apt update
   sudo apt install build-essential device-tree-compiler bootgen-xlnx git linux-headers-$(uname -r)
   xmutil listapps        # must work; xmutil/dfx-mgr ship with the Kria image
   ```
3. Build and load **u-dma-buf**, which provides the physically contiguous buffer the
   accelerator reads and writes:
   ```sh
   git clone https://github.com/ikwzm/udmabuf && cd udmabuf
   make                                   # builds against the running kernel's headers
   sudo insmod u-dma-buf.ko udmabuf0=1048576
   ls /dev/udmabuf0 /sys/class/u-dma-buf/udmabuf0/phys_addr
   ```
   To load it at every boot instead (done on our board):
   ```sh
   sudo mkdir -p /lib/modules/$(uname -r)/extra && sudo cp u-dma-buf.ko /lib/modules/$(uname -r)/extra/
   sudo depmod -a
   echo u_dma_buf | sudo tee /etc/modules-load.d/u-dma-buf.conf
   echo "options u_dma_buf udmabuf0=4194304" | sudo tee /etc/modprobe.d/u-dma-buf.conf
   sudo modprobe u_dma_buf
   ```
   The module's name is `u_dma_buf`, with underscores. An `options u-dma-buf ...` line
   is silently ignored, and you get the module but no `/dev/udmabuf0`. Don't confuse it
   with the kernel's own `/dev/udmabuf` (no 0), which is a different driver.

## 4. Install and load the accelerator (board)

Copy the repo to the board (`git clone`, or `scp -r` from the PC), plus the
`board/firmware/out/` folder from step 2. Then:

```sh
cd eecs298-project
sudo ./board/firmware/install.sh          # later, to just reload: sudo ./board/firmware/install.sh --reload
```

This should end with `dsc_accel` listed as loaded in `xmutil listapps`, and a
`UIO: uioN -> dsc_top` line.

**Never touch 0xA000_0000 while no bitstream is loaded.** An AXI access to an empty PL
hangs the CPU, and only a power cycle recovers it. Always run `install.sh` (or
`xmutil loadapp dsc_accel`) after each boot, before `dsc_bringup`.

## 5. Run the bring-up test (board)

```sh
make -C board/sw
sudo ./board/sw/dsc_bringup --csv results.csv
```

Expected output:
- one `[PASS]` per point (sparsity × stride);
- a latency table (skip on, skip off, speedup, cycles vs the co-sim model);
- `BRING-UP PASSED` at the end.

To run Kevin's exported cases (VECTORS.md format):

```sh
sudo ./board/sw/dsc_bringup --case path/to/case1 --case path/to/case2
```

Cases with `expected_pw_out.bin` are also checked against it, which checks Kevin's
quantization end to end on the hardware.

**Power (PR #3):** run `sudo ./board/sw/dsc_bringup --reps 20000` in one terminal and
`xmutil platformstats -p` in another. Compare against idle.

## Troubleshooting

| Symptom | Likely cause / check |
|---|---|
| `xmutil loadapp` fails | Look at `dmesg \| tail -30`. The overlay's `firmware-name` must match `dsc_accel.bit.bin` in the same folder. |
| `accelerator never idle` | PL not loaded or not clocked. Check `xmutil listapps`, and `sudo cat /sys/kernel/debug/clk/pl0_ref/clk_rate` (expect ~200000000). |
| `no interrupt within 1 s`, but `--poll` passes | Interrupt path. Check `cat /proc/interrupts \| grep dsc_top`; overlay `interrupts = <0 89 4>`; `pl_ps_irq0` in the block design. |
| Every output wrong, `sched_len` right | Buffer path. AFI widths in the overlay (`config-afi` 4..7 must be 0, i.e. 128-bit), the `udmabuf0` physical address, `O_SYNC`. |
| Bus error or hang on the first register access | PL not loaded, or the HPM0 width is wrong (`config-afi` `<14 0xa00>` must match the PS's 128-bit HPM0). |
| `sched_len` wrong, outputs right | Register map. Copy `xdsc_top_hw.h` into `board/sw` (step 1) and rebuild. |
| No `UIO:` line | `uio_pdrv_genirq` isn't binding `generic-uio`. `dsc_bringup` falls back to `/dev/mem` and polling, so results stay valid. |
