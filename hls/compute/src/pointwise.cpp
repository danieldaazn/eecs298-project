// pointwise.cpp — baseline 1x1 pointwise convolution (no zero-skipping).
//
// Datapath: PW_UNROLL MACs per cycle across input channels. One output activation
// is built over PW_CBLKS = PW_C_IN / PW_UNROLL consecutive cycles (32/8 = 4 with the
// current placeholders): each cycle adds one PW_UNROLL-wide partial sum into the
// accumulator, and the last cycle requantizes and writes the output.
//
// The (output channel, channel block) loops are one flattened pipeline at II=1, so
// back-to-back outputs don't pay a per-output startup latency. (The first version
// called a non-inlined pw_dot() per output from an unpipelined loop: 16 cycles per
// output instead of 4.)

#include "dsc_kernels.h"

acc_t pw_partial_sum(const act_t vec[PW_C_IN],
                     const wgt_t row[PW_C_IN],
                     int cblk) {
#pragma HLS INLINE
    acc_t part = 0;
PW_LANES:
    for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
        const int c = cblk * PW_UNROLL + u;
        part += (acc_t)(vec[c] * row[c]);
    }
    return part;
}

void pointwise_top(const act_t in[PW_C_IN][PW_H][PW_W],
                   const wgt_t wt[PW_C_OUT][PW_C_IN],
                   const acc_t bias[PW_C_OUT],
                   int h, int w,
                   act_t out[PW_C_OUT][PW_H][PW_W]) {
    // Standalone C-sim / csynth-estimate wrapper; see dsc_kernels.h.
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=PW_UNROLL dim=2
    // Input partitioned like the weights: PW_UNROLL channels readable per cycle, so
    // gathering a pixel's channel vector takes PW_CBLKS cycles, not PW_C_IN.
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=PW_UNROLL dim=1
PW_ROWS:
    // h/w come from the driver: never index past the compile-time maximum.
    for (int y = 0; y < h && y < PW_H; y++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16
    PW_COLS:
        for (int x = 0; x < w && x < PW_W; x++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16
            act_t vec[PW_C_IN];
#pragma HLS ARRAY_PARTITION variable=vec cyclic factor=PW_UNROLL
        PW_GATHER:
            for (int cb = 0; cb < PW_CBLKS; cb++) {
#pragma HLS PIPELINE II=1
            PW_GATHER_LANE:
                for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
                    vec[cb * PW_UNROLL + u] = in[cb * PW_UNROLL + u][y][x];
                }
            }

            acc_t acc = 0;
        PW_OUT_CH:
            for (int k = 0; k < PW_C_OUT; k++) {
            PW_CBLK:
                for (int cb = 0; cb < PW_CBLKS; cb++) {
#pragma HLS PIPELINE II=1
                    const acc_t base = (cb == 0) ? bias[k] : acc;
                    acc = base + pw_partial_sum(vec, wt[k], cb);
                    if (cb == PW_CBLKS - 1)
                        out[k][y][x] = requant_relu6(acc);
                }
            }
        }
    }
}
