// dsc_kernels.h — baseline (no zero-skipping) depthwise and pointwise compute.
//
// Two layers of API:
//   * Cores (dw_window_mac, pw_partial_sum): the actual MAC datapath. These are what
//     get reused once the real streaming top-level is agreed with Hank.
//     dw_window_mac turns one 3x3 window into one output activation;
//     pw_partial_sum does one cycle's PW_UNROLL MACs of a pointwise output.
//   * Standalone tops (depthwise_top, pointwise_top): array-in/array-out wrappers
//     for C-sim and early csynth estimates ONLY. They are not the agreed top-level
//     interface (that is a shared interface still to be settled with Hank). Their
//     array ports limit achievable II; the real design feeds the cores from Hank's
//     line buffer / window streams.

#ifndef DSC_KERNELS_H
#define DSC_KERNELS_H

#include "dsc_types.h"

// acc (12 frac bits) -> act (Q3.5): round-half-up shift + ReLU6 clamp.
act_t requant_relu6(acc_t acc);

// One depthwise output: 9 MACs (fully unrolled) + bias + requant.
act_t dw_window_mac(const act_t win[DW_K][DW_K],
                    const wgt_t k[DW_K][DW_K],
                    acc_t bias);

// PW_UNROLL MACs: sum of vec[c]*row[c] for c in channel block `cblk`
// (c = cblk*PW_UNROLL .. cblk*PW_UNROLL + PW_UNROLL-1). A full pointwise output is
// bias + the partial sums of blocks 0..PW_CBLKS-1, then requant_relu6.
acc_t pw_partial_sum(const act_t vec[PW_C_IN],
                     const wgt_t row[PW_C_IN],
                     int cblk);

void depthwise_top(const act_t in[DW_C][DW_H][DW_W],
                   const wgt_t wt[DW_C][DW_K][DW_K],
                   const acc_t bias[DW_C],
                   int stride,
                   act_t out[DW_C][DW_HO_MAX][DW_WO_MAX]);

void pointwise_top(const act_t in[PW_C_IN][PW_H][PW_W],
                   const wgt_t wt[PW_C_OUT][PW_C_IN],
                   const acc_t bias[PW_C_OUT],
                   int h, int w,
                   act_t out[PW_C_OUT][PW_H][PW_W]);

#endif // DSC_KERNELS_H
