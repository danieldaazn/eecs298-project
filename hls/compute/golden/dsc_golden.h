// dsc_golden.h — bit-exact software reference for the compute core.
//
// Deliberately written independently of the HLS kernels: plain int arithmetic,
// no HLS types, no shared helper functions, the most obvious loop order. If a
// kernel and this model disagree, the kernel is wrong (or the spec changed).
//
// Layouts (raw Q-format integers, see dsc_types.h):
//   activations  [C][H][W]      raw Q3.5, 0..255
//   dw weights   [C][3][3]      raw Q1.7, -128..127
//   pw weights   [C_OUT][C_IN]  raw Q1.7
//   bias         [C]            raw, 12 fractional bits

#ifndef DSC_GOLDEN_H
#define DSC_GOLDEN_H

#include "dsc_types.h"

// acc (12 frac bits) -> activation (5 frac bits), round-half-up, ReLU6 clamp.
int golden_requant_relu6(long long acc);

// 3x3 depthwise, padding 1, stride 1 or 2.
void golden_depthwise(const int in[DW_C][DW_H][DW_W],
                      const int wt[DW_C][DW_K][DW_K],
                      const int bias[DW_C],
                      int stride,
                      int out[DW_C][DW_HO_MAX][DW_WO_MAX]);

// 1x1 pointwise over an h x w feature map (h <= PW_H, w <= PW_W).
void golden_pointwise(const int in[PW_C_IN][PW_H][PW_W],
                      const int wt[PW_C_OUT][PW_C_IN],
                      const int bias[PW_C_OUT],
                      int h, int w,
                      int out[PW_C_OUT][PW_H][PW_W]);

#endif // DSC_GOLDEN_H
