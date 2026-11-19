// dsc_top.h — fused depthwise-separable block: the accelerator's top-level kernel.
//
// !!! SHARED INTERFACE — PROPOSED BY HANK, PENDING SIGN-OFF !!!
// The signature, interface pragmas, DRAM layouts and parameter-blob layout below are
// the contract between the streaming side (Hank), the compute core (Daniel) and the
// ARM driver / weight export (Kevin). See hls/memory/ARCHITECTURE.md. Don't change
// them without agreement from all three.
//
//   in      m_axi gmem_rd  HWC input frame, DW_H x DW_W x DW_C act_t, 16 per beat
//   params  m_axi gmem_rd  parameter blob (layout below), read once per call
//   out     m_axi gmem_wr  HWC output, ho x wo x PW_C_OUT act_t, 16 per beat,
//                          ho = dw_out_dim(DW_H, stride), wo likewise
//   stride  s_axilite      1 or 2 (anything other than 2 is treated as 1)
//   skip_en s_axilite      1 = zero-block skipping, 0 = dense bypass (same hardware)
//   sched_len s_axilite out  pointwise cycles per pixel for the loaded weights
//
// Dataflow: read_input -> window_gen -> dw_stage -> pw_stage -> write_output.

#ifndef DSC_TOP_H
#define DSC_TOP_H

#include "mem_types.h"

// ---- parameter blob: one contiguous buffer of 128-bit beats, in this order -------
// Region            beats                    contents
// DW weights        DW_C                     beat c: bytes 0..8 = wt[c][ky][kx],
//                                            ky-major (byte 3*ky+kx), bytes 9..15 = 0
// DW bias           DW_C/4                   int32 little-endian, 4 per beat,
//                                            bias[4b + j] in bytes 4j..4j+3
// PW weights        PW_C_OUT*PW_C_IN/16      row-major wt[k][c]: beat
//                                            k*(PW_C_IN/16) + h holds c = 16h..16h+15
// PW bias           PW_C_OUT/4               as DW bias
// Weights are raw Q1.7 int8, biases raw int32 with 12 fractional bits (dsc_types.h).
static_assert(DW_K * DW_K <= AXI_BYTES, "a DW kernel must fit in one beat");
static_assert(DW_C % 4 == 0 && PW_C_OUT % 4 == 0, "4 biases per beat");
static_assert(PW_C_IN % AXI_BYTES == 0, "PW weight rows must be whole beats");

static const int PW_ROW_BEATS  = PW_C_IN / AXI_BYTES;
static const int P_DW_WT       = 0;
static const int P_DW_B        = P_DW_WT + DW_C;
static const int P_PW_WT       = P_DW_B + DW_C / 4;
static const int P_PW_B        = P_PW_WT + PW_C_OUT * PW_ROW_BEATS;
static const int PARAM_BEATS   = P_PW_B + PW_C_OUT / 4;

static const int OUT_FRAME_BEATS_MAX = DW_HO_MAX * DW_WO_MAX * OUT_PIX_BEATS;

void dsc_top(const axi_word_t *in,
             const axi_word_t *params,
             axi_word_t *out,
             int stride,
             bool skip_en,
             int &sched_len);

#endif // DSC_TOP_H
