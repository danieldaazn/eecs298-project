// passthrough_top.cpp — memory-path-only top for C-sim and co-sim. See mem_stages.h.

#include "mem_stages.h"

// Stand-in for the compute core: keep each window's centre tap, re-pack 16 channels
// per beat. Same consumption pattern as the depthwise stage (one window per cycle).
static void centre_tap(hls::stream<win_t> &s_win, hls::stream<axi_word_t> &s_out, int npix) {
    axi_word_t w = 0;
CT_PIX:
    for (int p = 0; p < npix; p++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=DW_HO_MAX*DW_WO_MAX
    CT_CH:
        for (int c = 0; c < DW_C; c++) {
#pragma HLS PIPELINE II=1
            const win_t win = s_win.read();
            const int lane = c % AXI_BYTES;
            w = word_set_byte(w, lane, win.v[1][1]);
            if (lane == AXI_BYTES - 1) s_out.write(w);
        }
    }
}

static void passthrough_dataflow(const axi_word_t *in, axi_word_t *out,
                                 int stride, int ho, int wo, int npix) {
#pragma HLS DATAFLOW
    hls::stream<axi_word_t> s_in("s_in");
    hls::stream<win_t>      s_win("s_win");
    hls::stream<axi_word_t> s_out("s_out");
#pragma HLS STREAM variable=s_in  depth=32
#pragma HLS STREAM variable=s_win depth=4
#pragma HLS STREAM variable=s_out depth=8

    read_input(in, s_in);
    window_gen(s_in, s_win, stride);
    centre_tap(s_win, s_out, npix);
    write_output<IN_PIX_BEATS>(s_out, out, ho, wo);
}

void stream_passthrough_top(const axi_word_t *in, axi_word_t *out, int stride) {
#pragma HLS INTERFACE m_axi port=in  bundle=gmem_rd offset=slave depth=IN_FRAME_BEATS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=out bundle=gmem_wr offset=slave depth=IN_FRAME_BEATS max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=stride bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    // Any stride other than 2 runs as 1, matching dw_out_dim. Without this, a bad
    // register value would make window_gen's window count disagree with ho*wo and
    // hang the pipeline.
    const int s  = (stride == 2) ? 2 : 1;
    const int ho = dw_out_dim(DW_H, s);
    const int wo = dw_out_dim(DW_W, s);
    passthrough_dataflow(in, out, s, ho, wo, ho * wo);
}
