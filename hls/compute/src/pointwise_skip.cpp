// pointwise_skip.cpp — 1x1 pointwise convolution with zero-block skipping.
//
// Skip granularity: one "block" = PW_UNROLL consecutive input-channel weights of one
// output channel, i.e. wt[k][cb*PW_UNROLL .. cb*PW_UNROLL + PW_UNROLL-1]. That is
// exactly one cycle of the pointwise datapath, so a block whose weights are all zero
// is one MAC cycle that can be dropped. See ../SKIP_FORMAT.md.
//
// How it works:
//   1. Once per layer (per call), scan the weights and record, for each output
//      channel k, a PW_CBLKS-bit mask of which blocks are nonzero. No separate
//      encoding is needed: zeros are detected from the dense weights themselves.
//   2. Turn the masks into a schedule: one entry per nonzero (k, cb) block. An output
//      channel with no nonzero blocks still gets one "bias only" entry so its output
//      (requant of the bias) is written.
//   3. For every pixel, run the schedule as one II=1 pipeline. Each entry does one
//      PW_UNROLL-wide partial sum (or none, for bias-only), seeding the accumulator
//      with the bias on a channel's first entry and writing the output on its last.
//
// Because the weights are the same for every pixel, every pixel runs the same
// schedule: latency is fixed per layer, not data-dependent. Cycles per pixel are
// sum over k of max(1, nonzero blocks of k), versus PW_C_OUT * PW_CBLKS dense.
//
// skip_en = 0 forces every block "nonzero", which reproduces the dense schedule on
// the same hardware: the runtime bypass bit for pruned-with-skip vs without-skip
// benchmarking.

#include "dsc_kernels.h"

void pw_build_sched(const wgt_t wt[PW_C_OUT][PW_C_IN],
                    bool skip_en,
                    pw_sched_t sched[PW_C_OUT * PW_CBLKS],
                    int &n) {
#pragma HLS INLINE
    // ---- 1. per-output-channel nonzero-block masks (once per layer) -------------
    ap_blk_mask_t mask[PW_C_OUT];
PW_SK_MASK:
    for (int k = 0; k < PW_C_OUT; k++) {
        // II=2: a whole weight row is PW_CBLKS reads per bank, 2 ports per bank.
        // Runs once per layer (~130 cycles), so not worth more ports.
#pragma HLS PIPELINE II=2
        ap_blk_mask_t m = 0;
    PW_SK_MASK_BLK:
        for (int cb = 0; cb < PW_CBLKS; cb++) {
#pragma HLS UNROLL
            bool nz = false;
        PW_SK_MASK_LANE:
            for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
                nz = nz || (wt[k][cb * PW_UNROLL + u] != 0);
            }
            if (nz || !skip_en) m |= (ap_blk_mask_t)(1u << cb);
        }
        mask[k] = m;
    }

    // ---- 2. schedule of (k, cb) entries ------------------------------------------
    int cnt = 0;
PW_SK_SCHED:
    for (int i = 0; i < PW_C_OUT * PW_CBLKS; i++) {
#pragma HLS PIPELINE II=1
        const int k  = i / PW_CBLKS;    // PW_CBLKS is a power of two: shift/mask
        const int cb = i % PW_CBLKS;
        const unsigned m = (unsigned)mask[k];
        const bool bit   = (m >> cb) & 1u;
        const bool empty = (m == 0);
        if (bit || (empty && cb == 0)) {
            pw_sched_t e;
            e.k      = k;
            e.cb     = cb;
            e.mac_en = bit;
            e.first  = empty || (m & ((1u << cb) - 1u)) == 0;   // no lower nonzero block
            e.last   = empty || (m >> (cb + 1)) == 0;           // no higher nonzero block
            sched[cnt] = e;
            cnt++;
        }
    }
    n = cnt;
}

bool pw_sched_step(const pw_sched_t &e,
                   const act_t vec[PW_C_IN],
                   const wgt_t wt[PW_C_OUT][PW_C_IN],
                   const acc_t bias[PW_C_OUT],
                   acc_t &acc,
                   act_t &o) {
#pragma HLS INLINE
    const acc_t base = e.first ? bias[e.k] : acc;
    const acc_t part = e.mac_en ? pw_partial_sum(vec, wt[e.k], e.cb) : (acc_t)0;
    acc = base + part;
    o = requant_relu6(acc);
    return e.last;
}

void pointwise_skip_top(const act_t in[PW_C_IN][PW_H][PW_W],
                        const wgt_t wt[PW_C_OUT][PW_C_IN],
                        const acc_t bias[PW_C_OUT],
                        int h, int w,
                        bool skip_en,
                        act_t out[PW_C_OUT][PW_H][PW_W],
                        int &sched_len) {
    // Standalone C-sim / csynth-estimate wrapper; see dsc_kernels.h.
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=PW_UNROLL dim=2
    // Input partitioned like the weights: PW_UNROLL channels readable per cycle, so
    // gathering a pixel's channel vector takes PW_CBLKS cycles, not PW_C_IN.
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=PW_UNROLL dim=1

    // ---- 1-2. mask + schedule (once per layer) -----------------------------------
    pw_sched_t sched[PW_C_OUT * PW_CBLKS];
    int n;
    pw_build_sched(wt, skip_en, sched, n);
    sched_len = n;

    // ---- 3. run the schedule for every pixel -------------------------------------
PW_SK_ROWS:
    for (int y = 0; y < h; y++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16
    PW_SK_COLS:
        for (int x = 0; x < w; x++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16
            act_t vec[PW_C_IN];
#pragma HLS ARRAY_PARTITION variable=vec cyclic factor=PW_UNROLL
        PW_SK_GATHER:
            for (int cb = 0; cb < PW_CBLKS; cb++) {
#pragma HLS PIPELINE II=1
            PW_SK_GATHER_LANE:
                for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
                    vec[cb * PW_UNROLL + u] = in[cb * PW_UNROLL + u][y][x];
                }
            }

            acc_t acc = 0;
        PW_SK_RUN:
            for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64 max=256
                const pw_sched_t e = sched[i];
                act_t o;
                if (pw_sched_step(e, vec, wt, bias, acc, o))
                    out[e.k][y][x] = o;
            }
        }
    }
}
