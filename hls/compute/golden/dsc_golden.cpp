// dsc_golden.cpp — see dsc_golden.h.

#include "dsc_golden.h"

int golden_requant_relu6(long long acc) {
    // Round half up, then floor-divide by 2^REQUANT_SHIFT. Floor division (not
    // C's truncation toward zero) matches an arithmetic right shift for negatives;
    // negatives clamp to 0 anyway, but keep the semantics exact.
    long long num = acc + (1LL << (REQUANT_SHIFT - 1));
    long long den = 1LL << REQUANT_SHIFT;
    long long q = num / den;
    if (num % den != 0 && num < 0) q -= 1;
    if (q < 0) q = 0;
    if (q > ACT_RELU6_MAX) q = ACT_RELU6_MAX;
    return (int)q;
}

void golden_depthwise(const int in[DW_C][DW_H][DW_W],
                      const int wt[DW_C][DW_K][DW_K],
                      const int bias[DW_C],
                      int stride,
                      int out[DW_C][DW_HO_MAX][DW_WO_MAX]) {
    int ho = dw_out_dim(DW_H, stride);
    int wo = dw_out_dim(DW_W, stride);
    for (int c = 0; c < DW_C; c++) {
        for (int oy = 0; oy < ho; oy++) {
            for (int ox = 0; ox < wo; ox++) {
                long long acc = bias[c];
                for (int ky = 0; ky < DW_K; ky++) {
                    for (int kx = 0; kx < DW_K; kx++) {
                        int iy = oy * stride + ky - DW_PAD;
                        int ix = ox * stride + kx - DW_PAD;
                        if (iy < 0 || iy >= DW_H || ix < 0 || ix >= DW_W) continue; // zero pad
                        acc += (long long)in[c][iy][ix] * wt[c][ky][kx];
                    }
                }
                out[c][oy][ox] = golden_requant_relu6(acc);
            }
        }
    }
}

void golden_pointwise(const int in[PW_C_IN][PW_H][PW_W],
                      const int wt[PW_C_OUT][PW_C_IN],
                      const int bias[PW_C_OUT],
                      int h, int w,
                      int out[PW_C_OUT][PW_H][PW_W]) {
    for (int k = 0; k < PW_C_OUT; k++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                long long acc = bias[k];
                for (int c = 0; c < PW_C_IN; c++)
                    acc += (long long)in[c][y][x] * wt[k][c];
                out[k][y][x] = golden_requant_relu6(acc);
            }
        }
    }
}
