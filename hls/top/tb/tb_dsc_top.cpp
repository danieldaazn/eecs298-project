// tb_dsc_top.cpp — C-sim of the fused streaming top vs Daniel's golden model.
//
// For each case: generate a CHW image + weights (plain ints), pack them into the
// documented DRAM formats (HWC frame, parameter blob; dsc_top.h) directly from the
// spec, run dsc_top with skipping on and with the bypass, and require both outputs
// to be bit-exact with golden_depthwise -> golden_pointwise. sched_len is checked
// against an independent count. A guard region after the output buffer catches
// out-of-bounds writes.
//
// The pack_* functions double as a reference for Kevin's driver/export.

#include <stdio.h>
#include <stdint.h>
#include "dsc_top.h"
#include "dsc_golden.h"

static uint32_t rng_state = 0x2545f491u;
static uint32_t rng() {  // xorshift32
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static int rand_range(int lo, int hi) { return lo + (int)(rng() % (uint32_t)(hi - lo + 1)); }

// Golden-side tensors (CHW ints).
static int g_in[DW_C][DW_H][DW_W];
static int g_dw_wt[DW_C][DW_K][DW_K];
static int g_dw_b[DW_C];
static int g_dw_out[DW_C][DW_HO_MAX][DW_WO_MAX];
static int g_pw_wt[PW_C_OUT][PW_C_IN];
static int g_pw_b[PW_C_OUT];
static int g_pw_out[PW_C_OUT][PW_H][PW_W];

// DRAM-side buffers.
static axi_word_t d_in[IN_FRAME_BEATS];
static axi_word_t d_params[PARAM_BEATS];
static const int OUT_GUARD = 16;
static axi_word_t d_out[OUT_FRAME_BEATS_MAX + OUT_GUARD];

static int g_failures = 0;

struct Case {
    int act_lo, act_hi;
    int dw_wgt_lo, dw_wgt_hi, dw_b_lo, dw_b_hi;
    int pw_wgt_lo, pw_wgt_hi, pw_b_lo, pw_b_hi;
    int pw_block_zero_pct;    // structured: whole 8-weight blocks zeroed
    int pw_wgt_zero_pct;      // unstructured: single weights zeroed
};

static void gen(const Case &k) {
    for (int c = 0; c < DW_C; c++) {
        for (int y = 0; y < DW_H; y++)
            for (int x = 0; x < DW_W; x++) g_in[c][y][x] = rand_range(k.act_lo, k.act_hi);
        for (int ky = 0; ky < DW_K; ky++)
            for (int kx = 0; kx < DW_K; kx++) g_dw_wt[c][ky][kx] = rand_range(k.dw_wgt_lo, k.dw_wgt_hi);
        g_dw_b[c] = rand_range(k.dw_b_lo, k.dw_b_hi);
    }
    for (int o = 0; o < PW_C_OUT; o++) {
        for (int c = 0; c < PW_C_IN; c++)
            g_pw_wt[o][c] = ((int)(rng() % 100) < k.pw_wgt_zero_pct) ? 0
                                                                      : rand_range(k.pw_wgt_lo, k.pw_wgt_hi);
        for (int cb = 0; cb < PW_CBLKS; cb++)
            if ((int)(rng() % 100) < k.pw_block_zero_pct)
                for (int u = 0; u < PW_UNROLL; u++) g_pw_wt[o][cb * PW_UNROLL + u] = 0;
        g_pw_b[o] = rand_range(k.pw_b_lo, k.pw_b_hi);
    }
}

static void put_byte(axi_word_t &w, int i, int v) {
    w |= (axi_word_t)(unsigned)(v & 0xFF) << (8 * i);
}
static void put_i32(axi_word_t &w, int j, int v) {
    for (int i = 0; i < 4; i++) put_byte(w, 4 * j + i, (int)(((uint32_t)v >> (8 * i)) & 0xFF));
}

static void pack_input() {  // HWC, 16 channels per beat
    for (int y = 0; y < DW_H; y++)
        for (int x = 0; x < DW_W; x++)
            for (int b = 0; b < IN_PIX_BEATS; b++) {
                axi_word_t w = 0;
                for (int i = 0; i < AXI_BYTES; i++) put_byte(w, i, g_in[b * AXI_BYTES + i][y][x]);
                d_in[(y * DW_W + x) * IN_PIX_BEATS + b] = w;
            }
}

static void pack_params() {  // layout table in dsc_top.h
    for (int i = 0; i < PARAM_BEATS; i++) d_params[i] = 0;
    for (int c = 0; c < DW_C; c++)
        for (int ky = 0; ky < DW_K; ky++)
            for (int kx = 0; kx < DW_K; kx++) put_byte(d_params[P_DW_WT + c], 3 * ky + kx, g_dw_wt[c][ky][kx]);
    for (int c = 0; c < DW_C; c++) put_i32(d_params[P_DW_B + c / 4], c % 4, g_dw_b[c]);
    for (int o = 0; o < PW_C_OUT; o++)
        for (int c = 0; c < PW_C_IN; c++)
            put_byte(d_params[P_PW_WT + o * (PW_C_IN / 16) + c / 16], c % 16, g_pw_wt[o][c]);
    for (int o = 0; o < PW_C_OUT; o++) put_i32(d_params[P_PW_B + o / 4], o % 4, g_pw_b[o]);
}

static int expected_sched_len() {
    int total = 0;
    for (int o = 0; o < PW_C_OUT; o++) {
        int nz = 0;
        for (int cb = 0; cb < PW_CBLKS; cb++) {
            bool any = false;
            for (int u = 0; u < PW_UNROLL; u++) any = any || g_pw_wt[o][cb * PW_UNROLL + u] != 0;
            nz += any;
        }
        total += nz ? nz : 1;
    }
    return total;
}

static void run_case(const char *name, const Case &k, int stride) {
    gen(k);
    pack_input();
    pack_params();
    golden_depthwise(g_in, g_dw_wt, g_dw_b, stride, g_dw_out);
    const int ho = dw_out_dim(DW_H, stride), wo = dw_out_dim(DW_W, stride);
    golden_pointwise(g_dw_out, g_pw_wt, g_pw_b, ho, wo, g_pw_out);

    const int out_beats = ho * wo * OUT_PIX_BEATS;
    const axi_word_t sentinel = ~(axi_word_t)0;
    const int dense_len = PW_C_OUT * PW_CBLKS, exp_len = expected_sched_len();
    int mism = 0, interior = 0, total = 0, overrun = 0, len[2] = {-1, -1};

    for (int pass = 0; pass < 2; pass++) {
        const bool skip_en = (pass == 0);
        for (int i = 0; i < OUT_FRAME_BEATS_MAX + OUT_GUARD; i++) d_out[i] = sentinel;
        dsc_top(d_in, d_params, d_out, stride, skip_en, len[pass]);
        for (int y = 0; y < ho; y++)
            for (int x = 0; x < wo; x++)
                for (int o = 0; o < PW_C_OUT; o++) {
                    const axi_word_t w = d_out[(y * wo + x) * OUT_PIX_BEATS + o / AXI_BYTES];
                    const int got = (int)(uint8_t)(w >> (8 * (o % AXI_BYTES)));
                    const int g = g_pw_out[o][y][x];
                    if (skip_en) {
                        total++;
                        if (g > 0 && g < ACT_RELU6_MAX) interior++;
                    }
                    if (g != got) {
                        if (mism < 5)
                            printf("    mismatch (skip_en=%d) y=%d x=%d k=%d golden=%d got=%d\n",
                                   (int)skip_en, y, x, o, g, got);
                        mism++;
                    }
                }
        for (int i = out_beats; i < OUT_FRAME_BEATS_MAX + OUT_GUARD; i++)
            if (d_out[i] != sentinel) overrun++;
    }

    const bool len_ok = len[0] == exp_len && len[1] == dense_len;
    if (!len_ok)
        printf("    sched_len skip=%d (expected %d), bypass=%d (expected %d)\n",
               len[0], exp_len, len[1], dense_len);
    if (overrun) printf("    %d beats written past the end of the output\n", overrun);
    const bool ok = mism == 0 && len_ok && overrun == 0;
    printf("[%s] %-38s s%d %5d outputs, %5d interior, %d mismatches | cycles/px %3d of %3d\n",
           ok ? "PASS" : "FAIL", name, stride, total, interior, mism, len[0], dense_len);
    if (!ok) g_failures++;
}

int main() {
    //                    act       dw wgt     dw bias          pw wgt   pw bias          blk% wgt%
    const Case typical = {0, 192,  -128, 127, -20000, 20000,   -32, 31, -20000, 20000,   0,   0};
    Case full = {0, 255, -128, 127, -200000, 200000, -128, 127, -200000, 200000, 0, 0};
    Case big_bias = typical;  // exercises all 4 bytes + sign of the int32 bias decode
    big_bias.dw_b_lo = -(1 << 24); big_bias.dw_b_hi = (1 << 24);
    big_bias.pw_b_lo = -(1 << 24); big_bias.pw_b_hi = (1 << 24);
    Case sat_hi = {255, 255, 127, 127, 0, 0, 127, 127, 0, 0, 0, 0};
    Case sat_lo = {255, 255, -128, -128, 0, 0, -128, -128, 0, 0, 0, 0};

    printf("-- dsc_top (fused DW -> PW streaming) vs golden, skip on and bypass --\n");
    for (int s = 1; s <= 2; s++) {
        run_case("typical, dense", typical, s);
        Case c = typical;
        c.pw_block_zero_pct = 30;  run_case("30% PW blocks pruned", c, s);
        c.pw_block_zero_pct = 50;  run_case("50% PW blocks pruned", c, s);
        c.pw_block_zero_pct = 90;  run_case("90% PW blocks pruned", c, s);
        c.pw_block_zero_pct = 100; run_case("all PW weights zero", c, s);
        c.pw_block_zero_pct = 0; c.pw_wgt_zero_pct = 50;
        run_case("50% unstructured PW zeros", c, s);
        run_case("full value ranges", full, s);
        run_case("large +/- biases (2^24)", big_bias, s);
        run_case("saturate high", sat_hi, s);
        run_case("saturate low", sat_lo, s);
    }
    // Unsupported stride values must behave as stride 1 (and not hang).
    printf("\n-- out-of-range stride register --\n");
    {
        gen(typical); pack_input(); pack_params();
        golden_depthwise(g_in, g_dw_wt, g_dw_b, 1, g_dw_out);
        golden_pointwise(g_dw_out, g_pw_wt, g_pw_b, DW_HO_MAX, DW_WO_MAX, g_pw_out);
        int bad = 0;
        const int strides[] = {0, 3, -1};
        for (int t = 0; t < 3; t++) {
            int len;
            dsc_top(d_in, d_params, d_out, strides[t], true, len);
            for (int y = 0; y < DW_HO_MAX; y++)
                for (int x = 0; x < DW_WO_MAX; x++)
                    for (int o = 0; o < PW_C_OUT; o++) {
                        const axi_word_t w = d_out[(y * DW_WO_MAX + x) * OUT_PIX_BEATS + o / AXI_BYTES];
                        if ((int)(uint8_t)(w >> (8 * (o % AXI_BYTES))) != g_pw_out[o][y][x]) bad++;
                    }
        }
        printf("[%s] stride 0, 3, -1 run as stride 1           %d mismatches\n", bad ? "FAIL" : "PASS", bad);
        if (bad) g_failures++;
    }

    if (g_failures) {
        printf("\nTOP C-SIM FAILED: %d test(s) failed\n", g_failures);
        return 1;
    }
    printf("\nTOP C-SIM PASSED: all cases bit-exact against golden model\n");
    return 0;
}
