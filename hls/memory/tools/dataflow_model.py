#!/usr/bin/env python3
"""Cycle-level model of dsc_top's dataflow region, for sizing FIFOs.

Models window_gen -> s_win -> dw_stage -> s_dw -> pw_stage (overlapped load) at one
iteration per cycle per stage with blocking FIFOs. Not modeled: read_input/write_output
(never the bottleneck: <10 % of port bandwidth) and the once-per-call parameter load +
schedule build, which is added as a constant FIXED fitted to Daniel's stride-1 co-sim
(where it is constant to within 5 cycles across all sparsity levels).

    python3 dataflow_model.py            # compare with co-sim, sweep s_dw depth

Use it for relative comparisons (FIFO depth A vs B). At stride 2 it under-predicts
co-sim by up to ~1k cycles (unmodeled per-row pipeline latency), so absolute stride-2
numbers come from co-sim, not from here.
"""
H = W = 16      # DW_H, DW_W
C = 32          # DW_C
CBLKS = 4       # PW_CBLKS: s_dw beats per pixel
PREFETCH_AT = 16
LAT_DW = 6      # dw_stage pipeline latency (approximate)
FIXED = 1046    # fitted: parameter load + schedule build + fill/drain not modeled


def run(stride, n, d_win=4, d_dw=8):
    ho = (H - 1) // stride + 1
    wo = (W - 1) // stride + 1
    npix = ho * wo
    emit = []                                   # window_gen iterations, in order
    for y in range(H + 1):
        for x in range(W + 1):
            ok = y >= 1 and x >= 1 and (stride == 1 or ((y - 1) % 2 == 0 and (x - 1) % 2 == 0))
            emit += [ok] * C
    s_win = s_dw = 0
    in_flight = []                              # dw outputs not yet visible in s_dw
    wi = dch = 0
    loaded = i = p = t = 0
    total = npix * n
    cyc = 0
    while t < total:
        cyc += 1
        # pw_stage: prologue loads pixel 0, then one schedule entry per cycle; reads the
        # next pixel's beats at entries PREFETCH_AT.. (stalls if s_dw is empty).
        r_dw = 0
        if loaded < CBLKS:
            if s_dw > 0:
                r_dw = 1
                loaded += 1
        else:
            pb = i - PREFETCH_AT
            need = 0 <= pb < CBLKS and p + 1 < npix
            if not need or s_dw > 0:
                r_dw = 1 if need else 0
                t += 1
                if i == n - 1:
                    i, p = 0, p + 1
                else:
                    i += 1
        # dw_stage: one window per cycle, one s_dw beat per 8 channels.
        r_win = 0
        if s_win > 0:
            last = dch % 8 == 7
            if not last or s_dw - r_dw + len(in_flight) < d_dw:
                r_win = 1
                dch += 1
                if last:
                    in_flight.append(LAT_DW)
        # window_gen: walks the padded grid at II=1, blocks only on a full s_win.
        w_win = 0
        if wi < len(emit):
            if not emit[wi]:
                wi += 1
            elif s_win - r_win < d_win:
                w_win = 1
                wi += 1
        s_win += w_win - r_win
        s_dw += sum(1 for v in in_flight if v <= 1) - r_dw
        in_flight = [v - 1 for v in in_flight if v > 1]
    return cyc + FIXED


# Daniel's co-sim, dsc_top + BRAM latency=2 + overlapped pw_stage, s_dw depth 8
# (hls/compute/results/cosim_results.csv): (stride, sched_len) -> cycles.
COSIM = {(1, 256): 67208, (1, 166): 44171, (1, 142): 38025, (1, 67): 18824, (1, 64): 18061,
         (2, 256): 18880, (2, 186): 15380, (2, 141): 13126, (2, 68): 9843, (2, 64): 9801}

if __name__ == "__main__":
    depths = (8, 16, 32)
    print("stride  n    co-sim(d=8) | model: " + "  ".join(f"s_dw={d:<3d}" for d in depths))
    for (s, n), m in COSIM.items():
        r = [run(s, n, d_dw=d) for d in depths]
        print(f"{s:5d} {n:4d}  {m:11d} |        " + "  ".join(f"{x:8d}" for x in r)
              + f"   (16 vs 8: {100 * (r[1] - r[0]) / r[0]:+.1f} %)")
