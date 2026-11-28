// mem_stages.h — streaming stages of the memory subsystem (Hank).
//
//   DRAM --m_axi--> read_input --s_in(128b)--> window_gen --s_win(72b)--> [compute]
//   [compute] --s_out(128b)--> write_output --m_axi--> DRAM
//
// Each stage is one dataflow process. See ARCHITECTURE.md for widths, rates and
// FIFO depths.

#ifndef MEM_STAGES_H
#define MEM_STAGES_H

#include <hls_stream.h>
#include "mem_types.h"

// Burst-read one HWC input frame (DW_H x DW_W x DW_C) into a stream of beats.
void read_input(const axi_word_t *in, hls::stream<axi_word_t> &s_in);

// Line buffer + sliding-window generator. Consumes one input frame from s_in and,
// for each output pixel (raster order) and each channel c = 0..DW_C-1, emits the
// zero-padded 3x3 window of channel c centred on input (oy*stride, ox*stride).
// That is exactly the window dw_window_mac consumes: one channel per cycle.
// stride must be 1 or 2. Emits dw_out_dim(DW_H,s) * dw_out_dim(DW_W,s) * DW_C windows.
void window_gen(hls::stream<axi_word_t> &s_in, hls::stream<win_t> &s_win, int stride);

// ---- write_output: row-level ping-pong burst writer ------------------------------
// collect_row fills a row buffer from the stream; burst_row writes it to DRAM as one
// sequential burst. The DATAFLOW pragma on the row loop turns `buf` into a ping-pong
// (PIPO) buffer: row r+1 is collected while row r is being burst out, and the AXI
// write channel only ever sees complete, back-to-back rows. This keeps a slow or
// bursty producer (the pointwise stage) from stalling a burst halfway through.

template <int PIX_BEATS>
static void collect_row(hls::stream<axi_word_t> &s_out,
                        axi_word_t buf[DW_WO_MAX * PIX_BEATS], int wo) {
COLLECT_ROW:
    for (int i = 0; i < wo * PIX_BEATS; i++) {
#pragma HLS PIPELINE II=1
        buf[i] = s_out.read();
    }
}

template <int PIX_BEATS>
static void burst_row(const axi_word_t buf[DW_WO_MAX * PIX_BEATS],
                      axi_word_t *out, int row, int wo) {
    const int base = row * wo * PIX_BEATS;
BURST_ROW:
    for (int i = 0; i < wo * PIX_BEATS; i++) {
#pragma HLS PIPELINE II=1
        out[base + i] = buf[i];
    }
}

// Write ho rows of wo pixels, PIX_BEATS beats per pixel, HWC, to out[0 ..].
template <int PIX_BEATS>
void write_output(hls::stream<axi_word_t> &s_out, axi_word_t *out, int ho, int wo) {
WRITE_ROWS:
    for (int r = 0; r < ho; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=DW_HO_MAX
#pragma HLS DATAFLOW
        axi_word_t buf[DW_WO_MAX * PIX_BEATS];
        collect_row<PIX_BEATS>(s_out, buf, wo);
        burst_row<PIX_BEATS>(buf, out, r, wo);
    }
}

// ---- standalone pass-through top (memory path only, no compute) -----------------
// read_input -> window_gen -> centre tap -> write_output. The output is the input
// frame subsampled at `stride` (stride 1: an exact copy), HWC, DW_C channels.
// Lets the streaming/AXI path be C-simmed and co-simmed independently of the
// compute kernels.
void stream_passthrough_top(const axi_word_t *in, axi_word_t *out, int stride);

#endif // MEM_STAGES_H
