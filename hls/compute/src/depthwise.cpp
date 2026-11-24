// depthwise.cpp — baseline 3x3 depthwise convolution (no zero-skipping).
//
// Datapath: 9 MACs per cycle (one full 3x3 window), one channel per cycle,
// target II=1 in the core. Zero padding is applied when the window is gathered.

#include "dsc_kernels.h"

act_t dw_window_mac(const act_t win[DW_K][DW_K],
                    const wgt_t k[DW_K][DW_K],
                    acc_t bias) {
#pragma HLS INLINE
    acc_t acc = bias;
DW_TAPS_Y:
    for (int ky = 0; ky < DW_K; ky++) {
#pragma HLS UNROLL
    DW_TAPS_X:
        for (int kx = 0; kx < DW_K; kx++) {
#pragma HLS UNROLL
            acc += (acc_t)(win[ky][kx] * k[ky][kx]);
        }
    }
    return requant_relu6(acc);
}

void depthwise_top(const act_t in[DW_C][DW_H][DW_W],
                   const wgt_t wt[DW_C][DW_K][DW_K],
                   const acc_t bias[DW_C],
                   int stride,
                   act_t out[DW_C][DW_HO_MAX][DW_WO_MAX]) {
    // Standalone C-sim / csynth-estimate wrapper; see dsc_kernels.h.
#pragma HLS ARRAY_PARTITION variable=wt complete dim=2
#pragma HLS ARRAY_PARTITION variable=wt complete dim=3
    // Only strides 1 and 2 are supported: anything other than 2 runs as 1, matching
    // dw_out_dim() and the streaming top (hls/top). Indexing as a shift by
    // stride_log2 also avoids a multiplier on the address path.
    const int stride_log2 = (stride == 2) ? 1 : 0;
    const int ho = dw_out_dim(DW_H, stride);
    const int wo = dw_out_dim(DW_W, stride);

DW_ROWS:
    for (int oy = 0; oy < ho; oy++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16
    DW_COLS:
        for (int ox = 0; ox < wo; ox++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=16
        DW_CHANNELS:
            for (int c = 0; c < DW_C; c++) {
#pragma HLS PIPELINE II=1
                act_t win[DW_K][DW_K];
#pragma HLS ARRAY_PARTITION variable=win complete dim=0
            GATHER_Y:
                for (int ky = 0; ky < DW_K; ky++) {
                GATHER_X:
                    for (int kx = 0; kx < DW_K; kx++) {
                        int iy = (oy << stride_log2) + ky - DW_PAD;
                        int ix = (ox << stride_log2) + kx - DW_PAD;
                        bool inside = (iy >= 0) && (iy < DW_H) && (ix >= 0) && (ix < DW_W);
                        win[ky][kx] = inside ? in[c][iy][ix] : (act_t)0;
                    }
                }
                out[c][oy][ox] = dw_window_mac(win, wt[c], bias[c]);
            }
        }
    }
}
