// tb_mem.cpp — C-sim testbench for the memory subsystem, independent of compute.
//
//   1. window_gen unit test: every tap of every emitted window is compared against a
//      direct zero-padded gather from the CHW image (written independently of the
//      line-buffer logic), at stride 1 and 2. Window count must be exact and the
//      input stream fully consumed.
//   2. Back-to-back frames: the second frame must be exact, i.e. no stale line-buffer
//      or window-history data leaks across frames (the buffers are never cleared).
//   3. stream_passthrough_top end to end (read_input -> window_gen -> centre tap ->
//      write_output ping-pong): output must equal the input subsampled at the stride,
//      and nothing may be written past the expected end of the output buffer.

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mem_stages.h"

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rng() {  // xorshift32
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int g_img[DW_C][DW_H][DW_W];                      // reference image, CHW
static axi_word_t g_frame[IN_FRAME_BEATS];               // same image, HWC beats
static const int OUT_GUARD = 16;                         // beats past the end
static axi_word_t g_out[IN_FRAME_BEATS + OUT_GUARD];
static int g_failures = 0;

enum ImgKind { IMG_RANDOM, IMG_ALL_MAX, IMG_COORD };

static void gen_img(ImgKind kind) {
    for (int c = 0; c < DW_C; c++)
        for (int y = 0; y < DW_H; y++)
            for (int x = 0; x < DW_W; x++) {
                int v;
                if (kind == IMG_RANDOM)      v = (int)(rng() & 0xFF);
                else if (kind == IMG_ALL_MAX) v = 255;   // padding (0) can't hide
                else                          v = (y * 37 + x * 11 + c * 5 + 1) & 0xFF;
                g_img[c][y][x] = v;
            }
}

// HWC packing, written directly from the layout spec in mem_types.h.
static void pack_frame() {
    for (int y = 0; y < DW_H; y++)
        for (int x = 0; x < DW_W; x++)
            for (int b = 0; b < IN_PIX_BEATS; b++) {
                axi_word_t w = 0;
                for (int i = 0; i < AXI_BYTES; i++)
                    w |= (axi_word_t)(unsigned)g_img[b * AXI_BYTES + i][y][x] << (8 * i);
                g_frame[(y * DW_W + x) * IN_PIX_BEATS + b] = w;
            }
}

static int ref_tap(int c, int iy, int ix) {
    if (iy < 0 || iy >= DW_H || ix < 0 || ix >= DW_W) return 0;
    return g_img[c][iy][ix];
}

static int lane_of(const axi_word_t &w, int i) {
    return (int)(unsigned)(uint8_t)(w >> (8 * i));
}

static void test_window_gen(const char *name, ImgKind kind, int stride, int frames) {
    int mism = 0, total = 0;
    for (int f = 0; f < frames; f++) {
        gen_img(kind);
        pack_frame();
        hls::stream<axi_word_t> s_in("tb.s_in");
        hls::stream<win_t> s_win("tb.s_win");
        for (int i = 0; i < IN_FRAME_BEATS; i++) s_in.write(g_frame[i]);
        window_gen(s_in, s_win, stride);

        const int ho = dw_out_dim(DW_H, stride), wo = dw_out_dim(DW_W, stride);
        const int expected = ho * wo * DW_C;
        bool count_ok = (int)s_win.size() == expected && s_in.empty();
        if (!count_ok) {
            printf("    frame %d: %zu windows (expected %d), %zu input beats left\n",
                   f, s_win.size(), expected, s_in.size());
            mism++;
            while (!s_win.empty()) s_win.read();
            while (!s_in.empty()) s_in.read();
            continue;
        }
        // Only the last frame is scored when frames > 1: earlier frames just leave
        // state behind in the buffers.
        const bool score = (f == frames - 1);
        for (int oy = 0; oy < ho; oy++)
            for (int ox = 0; ox < wo; ox++)
                for (int c = 0; c < DW_C; c++) {
                    const win_t w = s_win.read();
                    if (!score) continue;
                    for (int ky = 0; ky < DW_K; ky++)
                        for (int kx = 0; kx < DW_K; kx++) {
                            const int iy = oy * stride + ky - DW_PAD;
                            const int ix = ox * stride + kx - DW_PAD;
                            const int g = ref_tap(c, iy, ix);
                            const int k = (int)w.v[ky][kx];
                            total++;
                            if (g != k) {
                                if (mism < 5)
                                    printf("    mismatch oy=%d ox=%d c=%d ky=%d kx=%d ref=%d got=%d\n",
                                           oy, ox, c, ky, kx, g, k);
                                mism++;
                            }
                        }
                }
    }
    printf("[%s] %-40s %6d taps, %d mismatches\n", mism ? "FAIL" : "PASS", name, total, mism);
    if (mism) g_failures++;
}

static void test_passthrough(const char *name, ImgKind kind, int stride) {
    gen_img(kind);
    pack_frame();
    const axi_word_t sentinel = ~(axi_word_t)0;
    for (int i = 0; i < IN_FRAME_BEATS + OUT_GUARD; i++) g_out[i] = sentinel;

    stream_passthrough_top(g_frame, g_out, stride);

    const int ho = dw_out_dim(DW_H, stride), wo = dw_out_dim(DW_W, stride);
    const int out_beats = ho * wo * IN_PIX_BEATS;
    int mism = 0, total = 0;
    for (int oy = 0; oy < ho; oy++)
        for (int ox = 0; ox < wo; ox++)
            for (int c = 0; c < DW_C; c++) {
                const axi_word_t w = g_out[(oy * wo + ox) * IN_PIX_BEATS + c / AXI_BYTES];
                const int k = lane_of(w, c % AXI_BYTES);
                const int g = g_img[c][oy * stride][ox * stride];
                total++;
                if (g != k) {
                    if (mism < 5)
                        printf("    mismatch oy=%d ox=%d c=%d ref=%d got=%d\n", oy, ox, c, g, k);
                    mism++;
                }
            }
    int overrun = 0;
    for (int i = out_beats; i < IN_FRAME_BEATS + OUT_GUARD; i++)
        if (g_out[i] != sentinel) overrun++;
    if (overrun) printf("    %d beats written past the end of the output\n", overrun);
    const bool ok = (mism == 0) && (overrun == 0);
    printf("[%s] %-40s %6d values, %d mismatches\n", ok ? "PASS" : "FAIL", name, total, mism);
    if (!ok) g_failures++;
}

int main() {
    printf("-- window_gen (line buffer + sliding window) vs direct padded gather --\n");
    test_window_gen("window_gen random, stride 1", IMG_RANDOM, 1, 1);
    test_window_gen("window_gen random, stride 2", IMG_RANDOM, 2, 1);
    test_window_gen("window_gen all-255 (padding), stride 1", IMG_ALL_MAX, 1, 1);
    test_window_gen("window_gen all-255 (padding), stride 2", IMG_ALL_MAX, 2, 1);
    test_window_gen("window_gen coordinate image, stride 1", IMG_COORD, 1, 1);
    test_window_gen("window_gen 3 back-to-back frames, s1", IMG_RANDOM, 1, 3);
    test_window_gen("window_gen 3 back-to-back frames, s2", IMG_RANDOM, 2, 3);

    printf("\n-- stream_passthrough_top (AXI read -> window -> AXI write) --\n");
    test_passthrough("passthrough random, stride 1 (identity)", IMG_RANDOM, 1);
    test_passthrough("passthrough random, stride 2 (subsample)", IMG_RANDOM, 2);
    test_passthrough("passthrough coordinate image, stride 1", IMG_COORD, 1);
    test_passthrough("passthrough all-255, stride 2", IMG_ALL_MAX, 2);

    if (g_failures) {
        printf("\nMEM C-SIM FAILED: %d test(s) failed\n", g_failures);
        return 1;
    }
    printf("\nMEM C-SIM PASSED\n");
    return 0;
}
