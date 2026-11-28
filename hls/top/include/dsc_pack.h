// dsc_pack.h — host-side packers for dsc_top's DRAM buffers (not synthesized).
//
// The one C implementation of the byte layouts documented in dsc_top.h, shared by the
// C-sim testbench (hls/top/tb) and the board bring-up program (board/sw), so the two
// can't drift apart. Kevin's driver can use it as-is.
//
// Inputs are plain ints in PyTorch layout (same as the golden model and VECTORS.md):
//   in      [DW_C][DW_H][DW_W]     raw Q3.5, 0..255
//   dw_wt   [DW_C][3][3]           raw Q1.7, -128..127
//   dw_b    [DW_C]                 raw int32, 12 fractional bits
//   pw_wt   [PW_C_OUT][PW_C_IN]    raw Q1.7
//   pw_b    [PW_C_OUT]             raw int32
// Buffers are bytes; beat b, byte i of a buffer is dst[16*b + i] (AXI is little-endian).

#ifndef DSC_PACK_H
#define DSC_PACK_H

#include <stdint.h>
#include <string.h>
#include "dsc_top.h"

static const int IN_FRAME_BYTES      = IN_FRAME_BEATS * AXI_BYTES;
static const int PARAM_BYTES         = PARAM_BEATS * AXI_BYTES;
static const int OUT_FRAME_BYTES_MAX = OUT_FRAME_BEATS_MAX * AXI_BYTES;

static inline void dsc_put_i32(uint8_t *p, int32_t v) {
    const uint32_t u = (uint32_t)v;
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(u >> (8 * i));
}

// HWC, channel 0 first: byte (y*DW_W + x)*DW_C + c.
static inline void dsc_pack_input(const int in[DW_C][DW_H][DW_W], uint8_t *dst) {
    for (int y = 0; y < DW_H; y++)
        for (int x = 0; x < DW_W; x++)
            for (int c = 0; c < DW_C; c++)
                dst[(y * DW_W + x) * DW_C + c] = (uint8_t)in[c][y][x];
}

static inline void dsc_pack_params(const int dw_wt[DW_C][DW_K][DW_K], const int dw_b[DW_C],
                                   const int pw_wt[PW_C_OUT][PW_C_IN], const int pw_b[PW_C_OUT],
                                   uint8_t *dst) {
    memset(dst, 0, PARAM_BYTES);
    for (int c = 0; c < DW_C; c++)
        for (int ky = 0; ky < DW_K; ky++)
            for (int kx = 0; kx < DW_K; kx++)
                dst[(P_DW_WT + c) * AXI_BYTES + DW_K * ky + kx] = (uint8_t)dw_wt[c][ky][kx];
    for (int c = 0; c < DW_C; c++) dsc_put_i32(dst + P_DW_B * AXI_BYTES + 4 * c, dw_b[c]);
    for (int k = 0; k < PW_C_OUT; k++)
        for (int c = 0; c < PW_C_IN; c++)
            dst[P_PW_WT * AXI_BYTES + k * PW_C_IN + c] = (uint8_t)pw_wt[k][c];
    for (int k = 0; k < PW_C_OUT; k++) dsc_put_i32(dst + P_PW_B * AXI_BYTES + 4 * k, pw_b[k]);
}

// Output byte (y*wo + x)*PW_C_OUT + k -> out[k][y][x].
static inline int dsc_output_at(const uint8_t *src, int wo, int y, int x, int k) {
    return src[(y * wo + x) * PW_C_OUT + k];
}

// Bytes <-> beats, for the C-sim testbench (ap_uint<128> has no defined memory layout,
// so it can't be memcpy'd).
static inline void dsc_bytes_to_beats(const uint8_t *src, axi_word_t *dst, int beats) {
    for (int b = 0; b < beats; b++) {
        axi_word_t w = 0;
        for (int i = 0; i < AXI_BYTES; i++) w |= (axi_word_t)src[AXI_BYTES * b + i] << (8 * i);
        dst[b] = w;
    }
}
static inline void dsc_beats_to_bytes(const axi_word_t *src, uint8_t *dst, int beats) {
    for (int b = 0; b < beats; b++)
        for (int i = 0; i < AXI_BYTES; i++) dst[AXI_BYTES * b + i] = (uint8_t)(src[b] >> (8 * i));
}

#endif // DSC_PACK_H
