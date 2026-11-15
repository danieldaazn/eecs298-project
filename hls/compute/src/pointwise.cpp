// pointwise.cpp — baseline 1x1 pointwise convolution (no zero-skipping).
//
// Datapath: PW_UNROLL MACs per cycle across input channels, so one output
// activation takes PW_C_IN / PW_UNROLL cycles (32/8 = 4 with current placeholders).

#include "dsc_kernels.h"

act_t pw_dot(const act_t vec[PW_C_IN],
             const wgt_t row[PW_C_IN],
             acc_t bias) {
#pragma HLS INLINE off
    acc_t acc = bias;
PW_MAC:
    for (int c = 0; c < PW_C_IN; c++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=PW_UNROLL
        acc += (acc_t)(vec[c] * row[c]);
    }
    return requant_relu6(acc);
}

void pointwise_top(const act_t in[PW_C_IN][PW_H][PW_W],
                   const wgt_t wt[PW_C_OUT][PW_C_IN],
                   const acc_t bias[PW_C_OUT],
                   int h, int w,
                   act_t out[PW_C_OUT][PW_H][PW_W]) {
    // Standalone C-sim / csynth-estimate wrapper; see dsc_kernels.h.
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=PW_UNROLL dim=2
PW_ROWS:
    for (int y = 0; y < h; y++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16
    PW_COLS:
        for (int x = 0; x < w; x++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16
            act_t vec[PW_C_IN];
#pragma HLS ARRAY_PARTITION variable=vec cyclic factor=PW_UNROLL
        PW_GATHER:
            for (int c = 0; c < PW_C_IN; c++) {
#pragma HLS PIPELINE II=1
                vec[c] = in[c][y][x];
            }
        PW_OUT_CH:
            for (int k = 0; k < PW_C_OUT; k++) {
                out[k][y][x] = pw_dot(vec, wt[k], bias[k]);
            }
        }
    }
}
