// mem_types.h — word/stream types for the memory subsystem (Hank).
//
// Builds on the shared fixed-point types in hls/compute/include/dsc_types.h
// (act_t = u8 Q3.5, signed off by Hank 2026-11-19). Nothing here changes them.
//
// DRAM layout of every activation tensor is HWC (channel-interleaved): pixel (y, x)
// occupies C consecutive bytes, channel 0 first, pixels in raster order. One AXI
// beat is 128 bits = 16 channels of one pixel, byte i of the beat = channel
// (16*beat_in_pixel + i). See ARCHITECTURE.md.

#ifndef MEM_TYPES_H
#define MEM_TYPES_H

#include <stdint.h>
#include "dsc_types.h"

#if defined(__SYNTHESIS__) || defined(DSC_USE_AP_TYPES)
#include <ap_int.h>
typedef ap_uint<128> axi_word_t;   // one m_axi beat: 16 activations
typedef ap_uint<64>  pw_vec_t;     // 8 activations: one pointwise input cycle
#else
typedef unsigned __int128 axi_word_t;
typedef uint64_t          pw_vec_t;
#endif

static const int AXI_BYTES = 16;                 // bytes (= activations) per beat
static const int PW_VEC_LANES = 8;               // activations per pw_vec_t
static_assert(PW_VEC_LANES == PW_UNROLL, "pw_vec_t must carry one pointwise MAC cycle");
static_assert(DW_C % AXI_BYTES == 0, "a pixel must be a whole number of AXI beats");
static_assert(PW_C_OUT % AXI_BYTES == 0, "a pixel must be a whole number of AXI beats");

// Beats per pixel and per frame.
static const int IN_PIX_BEATS  = DW_C / AXI_BYTES;        // depthwise input pixel
static const int OUT_PIX_BEATS = PW_C_OUT / AXI_BYTES;    // pointwise output pixel
static const int IN_FRAME_BEATS = DW_H * DW_W * IN_PIX_BEATS;

// One 3x3 depthwise window of one channel: what dw_window_mac consumes per cycle.
struct win_t {
    act_t v[DW_K][DW_K];
};

// Byte lane i of a packed word. With i a compile-time constant (unrolled loop) this
// is wiring; with a variable i it is a 16:1 (or 8:1) byte mux.
template <typename W>
static inline act_t word_byte(const W &w, int i) {
#pragma HLS INLINE
    return (act_t)(w >> (8 * i));
}

// Byte lane i set to v; the other lanes are kept.
template <typename W>
static inline W word_set_byte(const W &w, int i, act_t v) {
#pragma HLS INLINE
    const W lane_mask = (W)0xFF << (8 * i);
    return (w & ~lane_mask) | ((W)v << (8 * i));
}

#endif // MEM_TYPES_H
