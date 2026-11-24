// dsc_top.cpp — fused depthwise-separable top-level kernel. See dsc_top.h.
//
// Glue only: the MAC datapath is Daniel's (dw_window_mac, pw_build_sched,
// pw_sched_step in hls/compute/), the streaming/AXI stages are Hank's
// (hls/memory/). dw_stage and pw_stage below adapt the compute cores to streams.

#include "dsc_top.h"
#include "mem_stages.h"
#include "dsc_kernels.h"

// ---- parameter unpacking -----------------------------------------------------------

static wgt_t byte_to_wgt(act_t b) {
#pragma HLS INLINE
    return (wgt_t)(int8_t)(unsigned)b;
}

// Little-endian int32 j (0..3) of a beat.
static acc_t word_i32(const axi_word_t &w, int j) {
#pragma HLS INLINE
    unsigned u = 0;
    for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
        u |= (unsigned)word_byte(w, 4 * j + i) << (8 * i);
    }
    return (acc_t)(int32_t)u;
}

static void load_params(const axi_word_t *p,
                        wgt_t dw_wt[DW_C][DW_K][DW_K], acc_t dw_b[DW_C],
                        wgt_t pw_wt[PW_C_OUT][PW_C_IN], acc_t pw_b[PW_C_OUT]) {
LP_DW_WT:
    for (int c = 0; c < DW_C; c++) {
#pragma HLS PIPELINE II=1
        const axi_word_t w = p[P_DW_WT + c];
        for (int ky = 0; ky < DW_K; ky++)
            for (int kx = 0; kx < DW_K; kx++)
                dw_wt[c][ky][kx] = byte_to_wgt(word_byte(w, ky * DW_K + kx));
    }
LP_DW_B:
    for (int b = 0; b < DW_C / 4; b++) {
        // 4 writes per beat into one unpartitioned array (1 write port): II=4.
#pragma HLS PIPELINE II=4
        const axi_word_t w = p[P_DW_B + b];
        for (int j = 0; j < 4; j++) dw_b[4 * b + j] = word_i32(w, j);
    }
LP_PW_WT:
    for (int b = 0; b < PW_C_OUT * PW_ROW_BEATS; b++) {
        // 16 bytes into PW_UNROLL cyclic banks = 2 writes per bank, 1 write port: II=2.
#pragma HLS PIPELINE II=2
        const axi_word_t w = p[P_PW_WT + b];
        const int k  = b / PW_ROW_BEATS;
        const int c0 = (b % PW_ROW_BEATS) * AXI_BYTES;
        for (int i = 0; i < AXI_BYTES; i++) pw_wt[k][c0 + i] = byte_to_wgt(word_byte(w, i));
    }
LP_PW_B:
    for (int b = 0; b < PW_C_OUT / 4; b++) {
#pragma HLS PIPELINE II=4
        const axi_word_t w = p[P_PW_B + b];
        for (int j = 0; j < 4; j++) pw_b[4 * b + j] = word_i32(w, j);
    }
}

// ---- compute stages (stream adapters around Daniel's cores) -------------------------

// One window per cycle -> one depthwise output per cycle, re-packed 8 channels per
// pw_vec_t (the pointwise stage's per-cycle input width).
static void dw_stage(hls::stream<win_t> &s_win, hls::stream<pw_vec_t> &s_dw,
                     const wgt_t wt[DW_C][DW_K][DW_K], const acc_t bias[DW_C], int npix) {
    pw_vec_t v = 0;
DW_PIX:
    for (int p = 0; p < npix; p++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=DW_HO_MAX*DW_WO_MAX
    DW_CH:
        for (int c = 0; c < DW_C; c++) {
#pragma HLS PIPELINE II=1
            const win_t win = s_win.read();
            const act_t o = dw_window_mac(win.v, wt[c], bias[c]);
            const int lane = c % PW_VEC_LANES;
            v = word_set_byte(v, lane, o);
            if (lane == PW_VEC_LANES - 1) s_dw.write(v);
        }
    }
}

// One pipeline over every (pixel, schedule entry) pair. The next pixel's PW_CBLKS input
// beats are read from s_dw during entries [PW_PREFETCH_AT, PW_PREFETCH_AT + PW_CBLKS) of
// the current pixel, into the other half of a double buffer, so pixels follow each other
// with no gather or pipeline-restart gap: a pixel costs exactly n cycles (n = sched_len).
// Outputs leave in channel order k = 0..PW_C_OUT-1 (the schedule is k-ordered, one
// `last` entry per k), packed 16 per beat. Cycles per pixel depend only on the weights
// (fixed per layer), never on activation data.
// (Daniel's structure from hls/compute/src/pointwise_skip_pipe.cpp, co-simmed in this
// stage on PR #2; see its notes on the prefetch window and why the buffer half is
// selected by address, not by a mux.)
static const int PW_PREFETCH_AT = 16;   // > PW_RUN pipeline depth, with margin
static_assert(PW_C_OUT >= PW_PREFETCH_AT + PW_CBLKS,
              "prefetch window must fit in the shortest schedule (PW_C_OUT entries)");

static void pw_stage(hls::stream<pw_vec_t> &s_dw, hls::stream<axi_word_t> &s_out,
                     const wgt_t wt[PW_C_OUT][PW_C_IN], const acc_t bias[PW_C_OUT],
                     const pw_sched_t sched[PW_C_OUT * PW_CBLKS], int n, int npix) {
    act_t vbuf[2][PW_C_IN];
#pragma HLS ARRAY_PARTITION variable=vbuf cyclic factor=PW_UNROLL dim=2
    // WAR gap >= PW_PREFETCH_AT + 1 iterations, RAW gap >= n - PW_PREFETCH_AT - PW_CBLKS + 1
    // >= 45: both beyond the pipeline depth, so no loop-carried dependence is real.
#pragma HLS DEPENDENCE variable=vbuf inter false

    if (npix <= 0) return;
PW_PROLOGUE:
    for (int cb = 0; cb < PW_CBLKS; cb++) {
#pragma HLS PIPELINE II=1
        const pw_vec_t v = s_dw.read();
        for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
            vbuf[0][cb * PW_UNROLL + u] = word_byte(v, u);
        }
    }

    int i = 0, p = 0, cur = 0;
    acc_t acc = 0;
    axi_word_t ow = 0;
    const int total = npix * n;
PW_RUN:
    for (int t = 0; t < total; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=64*64 max=256*256
        const int pb = i - PW_PREFETCH_AT;
        if (pb >= 0 && pb < PW_CBLKS && p + 1 < npix) {
            const pw_vec_t v = s_dw.read();
            for (int u = 0; u < PW_UNROLL; u++) {
#pragma HLS UNROLL
                vbuf[1 - cur][pb * PW_UNROLL + u] = word_byte(v, u);
            }
        }
        const pw_sched_t e = sched[i];
        act_t o;
        if (pw_sched_step(e, vbuf[cur], wt, bias, acc, o)) {
            const int lane = e.k % AXI_BYTES;
            ow = word_set_byte(ow, lane, o);
            if (lane == AXI_BYTES - 1) s_out.write(ow);
        }
        if (i == n - 1) { i = 0; p++; cur = 1 - cur; } else { i++; }
    }
}

// ---- top ------------------------------------------------------------------------------

static void dsc_dataflow(const axi_word_t *in, axi_word_t *out,
                         int stride, int ho, int wo, int npix,
                         const wgt_t dw_wt[DW_C][DW_K][DW_K], const acc_t dw_b[DW_C],
                         const wgt_t pw_wt[PW_C_OUT][PW_C_IN], const acc_t pw_b[PW_C_OUT],
                         const pw_sched_t sched[PW_C_OUT * PW_CBLKS], int n) {
#pragma HLS DATAFLOW
    hls::stream<axi_word_t> s_in("s_in");
    hls::stream<win_t>      s_win("s_win");
    hls::stream<pw_vec_t>   s_dw("s_dw");
    hls::stream<axi_word_t> s_out("s_out");
    // Depths: see ARCHITECTURE.md ("FIFO depths"). pw_stage is the bottleneck, so
    // these only need to keep it from starving, not to absorb rate variation. s_dw is
    // 4 pixels so pointwise keeps working through stride 2's window-less input rows
    // (hls/memory/tools/dataflow_model.py: -5 to -13 % cycles at stride 2, 30-50 %).
#pragma HLS STREAM variable=s_in  depth=32
#pragma HLS STREAM variable=s_win depth=4
#pragma HLS STREAM variable=s_dw  depth=16
#pragma HLS STREAM variable=s_out depth=8

    read_input(in, s_in);
    window_gen(s_in, s_win, stride);
    dw_stage(s_win, s_dw, dw_wt, dw_b, npix);
    pw_stage(s_dw, s_out, pw_wt, pw_b, sched, n, npix);
    write_output<OUT_PIX_BEATS>(s_out, out, ho, wo);
}

void dsc_top(const axi_word_t *in,
             const axi_word_t *params,
             axi_word_t *out,
             int stride,
             bool skip_en,
             int &sched_len) {
#pragma HLS INTERFACE m_axi port=in     bundle=gmem_rd offset=slave depth=IN_FRAME_BEATS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=params bundle=gmem_rd offset=slave depth=PARAM_BEATS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=out    bundle=gmem_wr offset=slave depth=OUT_FRAME_BEATS_MAX max_write_burst_length=64
    // Buffer addresses in the same AXI-Lite bank as everything else (without these,
    // offset=slave puts them in a second slave, control_r). Agreed on PR #2.
#pragma HLS INTERFACE s_axilite port=in     bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=out    bundle=control
#pragma HLS INTERFACE s_axilite port=stride    bundle=control
#pragma HLS INTERFACE s_axilite port=skip_en   bundle=control
#pragma HLS INTERFACE s_axilite port=sched_len bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    wgt_t dw_wt[DW_C][DW_K][DW_K];
#pragma HLS ARRAY_PARTITION variable=dw_wt complete dim=2
#pragma HLS ARRAY_PARTITION variable=dw_wt complete dim=3
    acc_t dw_b[DW_C];
    wgt_t pw_wt[PW_C_OUT][PW_C_IN];
#pragma HLS ARRAY_PARTITION variable=pw_wt cyclic factor=PW_UNROLL dim=2
    // BRAM output register: takes the BRAM read off the multiply-add cycle (Daniel's
    // post-route run: 5.536 ns -> 4.243 ns at the 200 MHz target).
#pragma HLS BIND_STORAGE variable=pw_wt type=ram_t2p impl=bram latency=2
    acc_t pw_b[PW_C_OUT];
    pw_sched_t sched[PW_C_OUT * PW_CBLKS];

    // Once per call, before the pipeline starts: weights in (~380 cycles at the IIs in
    // load_params) and skip schedule built (~380 cycles): ~770 cycles, ~1% of a dense
    // stride-1 call.
    load_params(params, dw_wt, dw_b, pw_wt, pw_b);
    int n;
    pw_build_sched(pw_wt, skip_en, sched, n);
    sched_len = n;

    // Any stride other than 2 runs as 1 (matches dw_out_dim); see passthrough_top.cpp.
    const int s  = (stride == 2) ? 2 : 1;
    const int ho = dw_out_dim(DW_H, s);
    const int wo = dw_out_dim(DW_W, s);
    dsc_dataflow(in, out, s, ho, wo, ho * wo, dw_wt, dw_b, pw_wt, pw_b, sched, n);
}
