// pointwise_skip_pipe.cpp — PROTOTYPE: pointwise zero-block skipping with the next
// pixel's input loaded while the current pixel computes.
//
// Same function and results as pointwise_skip_top (same skip schedule, same bit-exact
// outputs). The difference is the per-pixel overhead:
//
//   pointwise_skip_top:  per pixel, load 32 channels (PW_CBLKS cycles), then run the
//                        schedule (sched_len cycles), then drain the pipeline before
//                        the next pixel: ~sched_len + 15 cycles per pixel.
//   this prototype:      ONE pipeline over every (pixel, schedule entry) pair. During
//                        PW_CBLKS entries partway through a pixel, the next pixel's channel
//                        vector is loaded into the other half of a double buffer.
//                        Pixels follow each other with no load or restart gap:
//                        sched_len cycles per pixel, plus a one-time prologue.
//
// The schedule build (steps 1-2) duplicates pointwise_skip_top's on purpose: that
// function is being refactored into pw_build_sched()/pw_sched_step() in PR #2, and
// this prototype stays out of the way of that diff. Once #2 merges, this should call
// pw_build_sched() instead, and if it's worth it, the same loop structure goes into
// pw_stage in hls/top/.

#include "dsc_kernels.h"

// The next pixel is prefetched during schedule entries [PP_PREFETCH_AT, PP_PREFETCH_AT +
// PW_CBLKS) of the current pixel, not from entry 0. That keeps every buffer write well
// away from that buffer's reads (see the DEPENDENCE note below). A pixel always has at
// least PW_C_OUT entries (one per output channel), so the window always fits.
static const int PP_PREFETCH_AT = 16;   // > PP_RUN pipeline depth (9), with margin
static_assert(PW_C_OUT >= PP_PREFETCH_AT + PW_CBLKS,
              "prefetch window must fit in the shortest possible schedule (PW_C_OUT entries)");

void pointwise_skip_pipe_top(const act_t in[PW_C_IN][PW_H][PW_W],
                             const wgt_t wt[PW_C_OUT][PW_C_IN],
                             const acc_t bias[PW_C_OUT],
                             int h, int w,
                             bool skip_en,
                             act_t out[PW_C_OUT][PW_H][PW_W],
                             int &sched_len) {
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=PW_UNROLL dim=2
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=PW_UNROLL dim=1

    // ---- 1-2. mask + schedule (once per layer; same as pointwise_skip_top) ---------
    ap_blk_mask_t mask[PW_C_OUT];
PP_MASK:
    for (int k = 0; k < PW_C_OUT; k++) {
#pragma HLS PIPELINE II=2
        ap_blk_mask_t m = 0;
        for (int cb = 0; cb < PW_CBLKS; cb++) {
#pragma HLS UNROLL
            bool nz = false;
            for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
                nz = nz || (wt[k][cb * PW_UNROLL + u] != 0);
            }
            if (nz || !skip_en) m |= (ap_blk_mask_t)(1u << cb);
        }
        mask[k] = m;
    }
    pw_sched_t sched[PW_C_OUT * PW_CBLKS];
    int n = 0;
PP_SCHED:
    for (int i = 0; i < PW_C_OUT * PW_CBLKS; i++) {
#pragma HLS PIPELINE II=1
        const int k  = i / PW_CBLKS;
        const int cb = i % PW_CBLKS;
        const unsigned m = (unsigned)mask[k];
        const bool bit   = (m >> cb) & 1u;
        const bool empty = (m == 0);
        if (bit || (empty && cb == 0)) {
            pw_sched_t e;
            e.k      = k;
            e.cb     = cb;
            e.mac_en = bit;
            e.first  = empty || (m & ((1u << cb) - 1u)) == 0;
            e.last   = empty || (m >> (cb + 1)) == 0;
            sched[n] = e;
            n++;
        }
    }
    sched_len = n;

    // h/w come from the driver: never index past the compile-time maximum.
    const int hh = (h < 0) ? 0 : (h > PW_H ? PW_H : h);
    const int ww = (w < 0) ? 0 : (w > PW_W ? PW_W : w);
    const int npix = hh * ww;
    if (npix == 0) return;

    // ---- 3. double-buffered channel vectors ------------------------------------------
    // vbuf[0] / vbuf[1] alternate: the current pixel reads vbuf[cur] while the next
    // pixel is written into vbuf[!cur]. The buffer select is part of the ADDRESS, so
    // each cycle reads only the buffer it uses. (Two separate arrays with a
    // `cur ? v1[c] : v0[c]` select made the hardware read both and discard one, and
    // co-sim's dependence checker flagged those discarded reads.)
    // Cyclic partitioning on dim 2 gives PW_UNROLL lanes per cycle, like `in`/`wt`.
    act_t vbuf[2][PW_C_IN];
#pragma HLS ARRAY_PARTITION variable=vbuf cyclic factor=PW_UNROLL dim=2
    // Why no loop-carried dependence through vbuf is real (PP_RUN depth is 9):
    //  * WAR: buffer X is last read by the final entry of pixel p; the first prefetch
    //    write into X is entry PP_PREFETCH_AT of pixel p+1, >= 17 iterations later.
    //  * RAW: the last prefetch write into X is entry PP_PREFETCH_AT + PW_CBLKS - 1 of
    //    pixel p+1; X is first read by entry 0 of pixel p+2, >= n - 19 >= 45 later.
    // (An earlier version prefetched from entry 0, which put a write one iteration
    // after the previous pixel's last read: co-sim flagged it, and it passed only
    // because of how this build happened to be scheduled.)
#pragma HLS DEPENDENCE variable=vbuf inter false

    // Prologue: pixel 0 into vbuf[0] (PW_CBLKS cycles, once).
PP_PROLOGUE:
    for (int cb = 0; cb < PW_CBLKS; cb++) {
#pragma HLS PIPELINE II=1
        for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
            vbuf[0][cb * PW_UNROLL + u] = in[cb * PW_UNROLL + u][0][0];
        }
    }

    // ---- 4. one pipeline over all (pixel, schedule entry) pairs --------------------
    int i = 0;            // schedule entry within the current pixel
    int p = 0;            // current pixel index (row-major)
    int y = 0, x = 0;     // current pixel
    int ny = 0, nx = 1;   // next pixel
    if (nx == ww) { nx = 0; ny = 1; }
    int cur = 0;          // the current pixel's buffer; the next pixel goes to 1 - cur
    acc_t acc = 0;
    const int total = npix * n;
PP_RUN:
    for (int t = 0; t < total; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64*64 max=256*256
        // Prefetch one block of the next pixel per entry, during entries
        // [PP_PREFETCH_AT, PP_PREFETCH_AT + PW_CBLKS) of this pixel.
        const int pb = i - PP_PREFETCH_AT;
        if (pb >= 0 && pb < PW_CBLKS && p + 1 < npix) {
            for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
                const int c = pb * PW_UNROLL + u;
                vbuf[1 - cur][c] = in[c][ny][nx];
            }
        }

        // One schedule entry of the current pixel (same arithmetic as pw_partial_sum).
        const pw_sched_t e = sched[i];
        acc_t part = 0;
        for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
            const int c = e.cb * PW_UNROLL + u;
            const act_t a = vbuf[cur][c];
            part += (acc_t)(a * wt[e.k][c]);
        }
        const acc_t base = e.first ? bias[e.k] : acc;
        acc = base + (e.mac_en ? part : (acc_t)0);
        if (e.last)
            out[e.k][y][x] = requant_relu6(acc);

        // Advance to the next entry / pixel.
        if (i == n - 1) {
            i = 0;
            p++;
            cur = 1 - cur;
            y = ny; x = nx;
            if (nx + 1 == ww) { nx = 0; ny++; } else { nx++; }
        } else {
            i++;
        }
    }
}
