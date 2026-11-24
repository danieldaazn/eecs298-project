// tb_dsc.cpp — C-sim testbench: baseline kernels vs golden model, bit-exact.
//
// Uses synthetic data from a fixed-seed generator. Once Kevin's golden vectors
// exist, add a test that loads them from file alongside these.
//
// Each test reports how many outputs landed strictly inside (0, ReLU6 max): a test
// where everything clamps to 0 or 6 would pass while checking almost nothing.

#include <stdio.h>
#include <stdint.h>
#include "dsc_types.h"
#include "dsc_kernels.h"
#include "dsc_golden.h"

// ---- deterministic RNG -------------------------------------------------------
static uint32_t rng_state = 0x12345678u;
static uint32_t rng() {  // xorshift32
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static int rand_range(int lo, int hi) {  // inclusive
    return lo + (int)(rng() % (uint32_t)(hi - lo + 1));
}

// ---- buffers (static: too big for a comfortable stack) ------------------------
static int   g_dw_in[DW_C][DW_H][DW_W];
static int   g_dw_wt[DW_C][DW_K][DW_K];
static int   g_dw_b[DW_C];
static int   g_dw_out[DW_C][DW_HO_MAX][DW_WO_MAX];
static act_t k_dw_in[DW_C][DW_H][DW_W];
static wgt_t k_dw_wt[DW_C][DW_K][DW_K];
static acc_t k_dw_b[DW_C];
static act_t k_dw_out[DW_C][DW_HO_MAX][DW_WO_MAX];

static int   g_pw_in[PW_C_IN][PW_H][PW_W];
static int   g_pw_wt[PW_C_OUT][PW_C_IN];
static int   g_pw_b[PW_C_OUT];
static int   g_pw_out[PW_C_OUT][PW_H][PW_W];
static act_t k_pw_in[PW_C_IN][PW_H][PW_W];
static wgt_t k_pw_wt[PW_C_OUT][PW_C_IN];
static acc_t k_pw_b[PW_C_OUT];
static act_t k_pw_out[PW_C_OUT][PW_H][PW_W];

static int g_failures = 0;

// ---- data generation -----------------------------------------------------------
struct GenParams {
    int act_lo, act_hi;
    int wgt_lo, wgt_hi;
    int bias_lo, bias_hi;
    int zero_wgt_pct;   // probability (0-100) a weight is forced to zero
};

static int gen_wgt(const GenParams& p) {
    if ((int)(rng() % 100) < p.zero_wgt_pct) return 0;
    return rand_range(p.wgt_lo, p.wgt_hi);
}

static void gen_dw(const GenParams& p) {
    for (int c = 0; c < DW_C; c++) {
        for (int y = 0; y < DW_H; y++)
            for (int x = 0; x < DW_W; x++)
                g_dw_in[c][y][x] = rand_range(p.act_lo, p.act_hi);
        for (int ky = 0; ky < DW_K; ky++)
            for (int kx = 0; kx < DW_K; kx++)
                g_dw_wt[c][ky][kx] = gen_wgt(p);
        g_dw_b[c] = rand_range(p.bias_lo, p.bias_hi);
    }
}

static void gen_pw(const GenParams& p) {
    for (int c = 0; c < PW_C_IN; c++)
        for (int y = 0; y < PW_H; y++)
            for (int x = 0; x < PW_W; x++)
                g_pw_in[c][y][x] = rand_range(p.act_lo, p.act_hi);
    for (int k = 0; k < PW_C_OUT; k++) {
        for (int c = 0; c < PW_C_IN; c++)
            g_pw_wt[k][c] = gen_wgt(p);
        g_pw_b[k] = rand_range(p.bias_lo, p.bias_hi);
    }
}

// ---- run + compare -------------------------------------------------------------
// Every kernel output buffer is filled with SENTINEL before each call. No valid
// activation can be 255 (ReLU6 caps at ACT_RELU6_MAX = 192), so:
//   * inside the valid region, a kernel that fails to write an output can't pass by
//     inheriting a previous run's correct value;
//   * outside it, any element that is no longer SENTINEL was written out of bounds.
static const int SENTINEL = 255;

static void fill_dw_out() {
    for (int c = 0; c < DW_C; c++)
        for (int y = 0; y < DW_HO_MAX; y++)
            for (int x = 0; x < DW_WO_MAX; x++)
                k_dw_out[c][y][x] = (act_t)SENTINEL;
}

static void fill_pw_out() {
    for (int k = 0; k < PW_C_OUT; k++)
        for (int y = 0; y < PW_H; y++)
            for (int x = 0; x < PW_W; x++)
                k_pw_out[k][y][x] = (act_t)SENTINEL;
}

// Compares a kernel output buffer against the golden one over the whole array:
// [0,h) x [0,w) must match golden, everything else must still be SENTINEL.
// Adds to *mism (prints the first few) and to *interior / *total for valid outputs.
template <int C, int H, int W>
static void compare_out(const int g[C][H][W], const act_t k[C][H][W], int h, int w,
                        const char* tag, int* mism, int* interior, int* total) {
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                const int kv = (int)k[c][y][x];
                const bool valid = (y < h) && (x < w);
                const int want = valid ? g[c][y][x] : SENTINEL;
                if (valid) {
                    (*total)++;
                    if (want > 0 && want < ACT_RELU6_MAX) (*interior)++;
                }
                if (kv != want) {
                    if (*mism < 5)
                        printf("    mismatch%s c=%d y=%d x=%d %s=%d kernel=%d\n", tag, c, y, x,
                               valid ? "golden" : "untouched", want, kv);
                    (*mism)++;
                }
            }
}

// Depthwise output size computed here, independently of the kernel's dw_out_dim().
static int ref_out_dim(int in_dim, int stride) {
    return (in_dim + 2 * 1 - 3) / stride + 1;   // 3x3, padding 1
}

// kernel_stride is passed to the kernel; the spec is that anything other than 2
// behaves as stride 1, so the golden model runs at that effective stride.
static void run_dw(const char* name, int kernel_stride) {
    const int stride = (kernel_stride == 2) ? 2 : 1;
    for (int c = 0; c < DW_C; c++) {
        for (int y = 0; y < DW_H; y++)
            for (int x = 0; x < DW_W; x++)
                k_dw_in[c][y][x] = (act_t)g_dw_in[c][y][x];
        for (int ky = 0; ky < DW_K; ky++)
            for (int kx = 0; kx < DW_K; kx++)
                k_dw_wt[c][ky][kx] = (wgt_t)g_dw_wt[c][ky][kx];
        k_dw_b[c] = (acc_t)g_dw_b[c];
    }
    golden_depthwise(g_dw_in, g_dw_wt, g_dw_b, stride, g_dw_out);
    fill_dw_out();
    depthwise_top(k_dw_in, k_dw_wt, k_dw_b, kernel_stride, k_dw_out);

    const int ho = ref_out_dim(DW_H, stride), wo = ref_out_dim(DW_W, stride);
    int mism = 0, interior = 0, total = 0;
    compare_out<DW_C, DW_HO_MAX, DW_WO_MAX>(g_dw_out, k_dw_out, ho, wo, "", &mism, &interior, &total);
    printf("[%s] %-34s %5d outputs, %5d interior, %d mismatches\n",
           mism ? "FAIL" : "PASS", name, total, interior, mism);
    if (mism) g_failures++;
}

static void load_pw_kernel_inputs() {
    for (int c = 0; c < PW_C_IN; c++)
        for (int y = 0; y < PW_H; y++)
            for (int x = 0; x < PW_W; x++)
                k_pw_in[c][y][x] = (act_t)g_pw_in[c][y][x];
    for (int k = 0; k < PW_C_OUT; k++) {
        for (int c = 0; c < PW_C_IN; c++)
            k_pw_wt[k][c] = (wgt_t)g_pw_wt[k][c];
        k_pw_b[k] = (acc_t)g_pw_b[k];
    }
}

// kh/kw are passed to the kernel; the spec is that they're clamped to PW_H/PW_W, so
// the golden model runs at the clamped size.
static void run_pw(const char* name, int kh, int kw) {
    const int h = kh < PW_H ? kh : PW_H, w = kw < PW_W ? kw : PW_W;
    load_pw_kernel_inputs();
    golden_pointwise(g_pw_in, g_pw_wt, g_pw_b, h, w, g_pw_out);
    fill_pw_out();
    pointwise_top(k_pw_in, k_pw_wt, k_pw_b, kh, kw, k_pw_out);

    int mism = 0, interior = 0, total = 0;
    compare_out<PW_C_OUT, PW_H, PW_W>(g_pw_out, k_pw_out, h, w, "", &mism, &interior, &total);
    printf("[%s] %-34s %5d outputs, %5d interior, %d mismatches\n",
           mism ? "FAIL" : "PASS", name, total, interior, mism);
    if (mism) g_failures++;
}

// ---- zero-block skipping (pointwise) -------------------------------------------
// Block-structured pruning: each PW_UNROLL-wide weight block of each output channel is
// zeroed with probability block_zero_pct. Also used with block_zero_pct = 0 plus the
// GenParams zero_wgt_pct for unstructured (per-weight) zeros.
static void gen_pw_blocks(const GenParams& p, int block_zero_pct) {
    gen_pw(p);
    for (int k = 0; k < PW_C_OUT; k++)
        for (int cb = 0; cb < PW_CBLKS; cb++)
            if ((int)(rng() % 100) < block_zero_pct)
                for (int u = 0; u < PW_UNROLL; u++)
                    g_pw_wt[k][cb * PW_UNROLL + u] = 0;
}

// Expected schedule length, computed independently of the kernel: per output channel,
// the number of blocks with any nonzero weight, but at least 1 (bias-only entry).
static int expected_sched_len() {
    int total = 0;
    for (int k = 0; k < PW_C_OUT; k++) {
        int nz_blocks = 0;
        for (int cb = 0; cb < PW_CBLKS; cb++) {
            bool nz = false;
            for (int u = 0; u < PW_UNROLL; u++)
                if (g_pw_wt[k][cb * PW_UNROLL + u] != 0) nz = true;
            if (nz) nz_blocks++;
        }
        total += nz_blocks > 0 ? nz_blocks : 1;
    }
    return total;
}

// Runs pointwise_skip_top with skipping on and off; both must match the golden model
// bit-exactly, and the schedule lengths must match the independent expectation.
// The output buffer is refilled with SENTINEL before each pass, so the bypass pass
// can't pass on outputs left over from the skip-on pass.
// Which skip kernel run_pw_skip drives. Called directly (not through a function
// pointer): Vitis co-sim only instruments direct calls to the top function.
enum SkipKernel { SKIP_BASE, SKIP_PIPE };

static void run_pw_skip(const char* name, int kh, int kw, SkipKernel which = SKIP_BASE) {
    const int h = kh < PW_H ? kh : PW_H, w = kw < PW_W ? kw : PW_W;
    load_pw_kernel_inputs();
    int zero_wgts = 0;
    for (int k = 0; k < PW_C_OUT; k++)
        for (int c = 0; c < PW_C_IN; c++)
            if (g_pw_wt[k][c] == 0) zero_wgts++;
    golden_pointwise(g_pw_in, g_pw_wt, g_pw_b, h, w, g_pw_out);

    const int dense_len = PW_C_OUT * PW_CBLKS;
    const int exp_len = expected_sched_len();
    int mism = 0, interior = 0, total = 0;
    int len_skip = -1, len_dense = -1;

    for (int pass = 0; pass < 2; pass++) {
        const bool skip_en = (pass == 0);
        int len = -1;
        fill_pw_out();
        if (which == SKIP_PIPE)
            pointwise_skip_pipe_top(k_pw_in, k_pw_wt, k_pw_b, kh, kw, skip_en, k_pw_out, len);
        else
            pointwise_skip_top(k_pw_in, k_pw_wt, k_pw_b, kh, kw, skip_en, k_pw_out, len);
        (skip_en ? len_skip : len_dense) = len;
        int pass_interior = 0, pass_total = 0;
        compare_out<PW_C_OUT, PW_H, PW_W>(g_pw_out, k_pw_out, h, w,
                                          skip_en ? " (skip on)" : " (bypass)",
                                          &mism, &pass_interior, &pass_total);
        if (skip_en) { interior = pass_interior; total = pass_total; }
    }

    bool len_ok = (len_skip == exp_len) && (len_dense == dense_len);
    if (!len_ok)
        printf("    schedule length: skip=%d (expected %d), bypass=%d (expected %d)\n",
               len_skip, exp_len, len_dense, dense_len);
    bool ok = (mism == 0) && len_ok;
    printf("[%s] %-34s %5d outputs, %5d interior, %d mismatches\n",
           ok ? "PASS" : "FAIL", name, total, interior, mism);
    printf("       weights zero %5.1f%% | cycles/pixel %3d of %3d | MAC cycles skipped %5.1f%%\n",
           100.0 * zero_wgts / (PW_C_OUT * PW_C_IN), len_skip, dense_len,
           100.0 * (dense_len - len_skip) / dense_len);
    if (!ok) g_failures++;
}

// Requant checked directly over every accumulator value near the interesting
// boundaries (rounding ties, 0, ReLU6 max) plus a wide sweep.
static void test_requant() {
    int mism = 0, checked = 0;
    for (long long a = -40000; a <= 40000; a++) {
        int g = golden_requant_relu6(a);
        int k = (int)requant_relu6((acc_t)a);
        checked++;
        if (g != k) {
            if (mism < 5) printf("    requant mismatch acc=%lld golden=%d kernel=%d\n", a, g, k);
            mism++;
        }
    }
    const long long extremes[] = {-(1LL << 24), (1LL << 24), -(1LL << 24) + 63, (1LL << 24) - 65};
    for (int i = 0; i < 4; i++) {
        int g = golden_requant_relu6(extremes[i]);
        int k = (int)requant_relu6((acc_t)extremes[i]);
        checked++;
        if (g != k) mism++;
    }
    // Spot-check the spec itself, not just agreement: 64 is exactly 0.5 LSB -> rounds up to 1.
    if (golden_requant_relu6(63) != 0 || golden_requant_relu6(64) != 1 ||
        golden_requant_relu6(-64) != 0 || golden_requant_relu6(1LL << 20) != ACT_RELU6_MAX) {
        printf("    golden requant spot-check failed\n");
        mism++;
    }
    printf("[%s] %-34s %5d values,  %d mismatches\n",
           mism ? "FAIL" : "PASS", "requant_relu6 sweep", checked, mism);
    if (mism) g_failures++;
}

int main() {
    test_requant();

    //                 act       weight      bias            zero%
    GenParams dw_rand = {0, 192, -128, 127, -20000, 20000, 0};
    GenParams dw_sparse = dw_rand;  dw_sparse.zero_wgt_pct = 50;
    GenParams dw_sat = {255, 255, 127, 127, 0, 0, 0};          // all max -> clamps at 6
    GenParams dw_neg = {255, 255, -128, -128, 0, 0, 0};        // most negative -> clamps at 0
    GenParams dw_zero = {0, 192, 0, 0, -20000, 20000, 0};      // zero weights -> bias only
    GenParams dw_full = {0, 255, -128, 127, -20000, 20000, 0}; // full act range

    gen_dw(dw_rand);   run_dw("depthwise random, stride 1", 1);
    gen_dw(dw_rand);   run_dw("depthwise random, stride 2", 2);
    gen_dw(dw_full);   run_dw("depthwise full act range, stride 1", 1);
    gen_dw(dw_sparse); run_dw("depthwise 50% zero weights", 1);
    gen_dw(dw_sat);    run_dw("depthwise saturate high", 1);
    gen_dw(dw_neg);    run_dw("depthwise saturate low", 2);
    gen_dw(dw_zero);   run_dw("depthwise all-zero weights", 1);
    gen_dw(dw_rand);   run_dw("depthwise stride 3 (runs as 1)", 3);
    gen_dw(dw_rand);   run_dw("depthwise stride 0 (runs as 1)", 0);
    // Bias at the documented limit (|bias| <= BIAS_ABS_MAX) with the largest products.
    GenParams dw_bigb = {0, 255, -128, 127, -BIAS_ABS_MAX, BIAS_ABS_MAX, 0};
    gen_dw(dw_bigb);   run_dw("depthwise bias at limit", 2);

    GenParams pw_rand = {0, 192, -32, 31, -20000, 20000, 0};
    GenParams pw_sparse = pw_rand;  pw_sparse.zero_wgt_pct = 50;
    GenParams pw_wide = {0, 255, -128, 127, -200000, 200000, 0};
    GenParams pw_sat = {255, 255, 127, 127, 0, 0, 0};

    gen_pw(pw_rand);   run_pw("pointwise random 16x16", PW_H, PW_W);
    gen_pw(pw_rand);   run_pw("pointwise random 8x8 (stride-2 dw)", 8, 8);
    gen_pw(pw_sparse); run_pw("pointwise 50% zero weights", PW_H, PW_W);
    gen_pw(pw_wide);   run_pw("pointwise full weight range", PW_H, PW_W);
    gen_pw(pw_sat);    run_pw("pointwise saturate high", PW_H, PW_W);
    gen_pw(pw_rand);   run_pw("pointwise h/w too big (clamped)", PW_H + 5, PW_W + 9);
    gen_pw(pw_rand);   run_pw("pointwise h=w=0 (writes nothing)", 0, 0);
    GenParams pw_bigb = {0, 255, -128, 127, -BIAS_ABS_MAX, BIAS_ABS_MAX, 0};
    gen_pw(pw_bigb);   run_pw("pointwise bias at limit", PW_H, PW_W);

    // Zero-block skipping: block-structured sparsity sweep, plus cases that stress
    // the schedule (all zero, unstructured zeros that rarely fill a whole block).
    printf("\n-- pointwise zero-block skipping (skip on and bypass both checked) --\n");
    gen_pw_blocks(pw_rand, 0);   run_pw_skip("skip: dense (0% blocks pruned)", PW_H, PW_W);
    gen_pw_blocks(pw_rand, 30);  run_pw_skip("skip: 30% blocks pruned", PW_H, PW_W);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("skip: 50% blocks pruned", PW_H, PW_W);
    gen_pw_blocks(pw_rand, 70);  run_pw_skip("skip: 70% blocks pruned", PW_H, PW_W);
    gen_pw_blocks(pw_rand, 90);  run_pw_skip("skip: 90% blocks pruned", PW_H, PW_W);
    gen_pw_blocks(pw_rand, 100); run_pw_skip("skip: all weights zero", PW_H, PW_W);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("skip: 50% blocks pruned, 8x8", 8, 8);
    gen_pw_blocks(pw_sparse, 0); run_pw_skip("skip: 50% unstructured zeros", PW_H, PW_W);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("skip: h/w too big (clamped)", PW_H + 3, PW_W + 7);
    gen_pw_blocks(pw_bigb, 50);  run_pw_skip("skip: bias at limit", PW_H, PW_W);

    // Prototype with next-pixel prefetch: every skip case again, plus shapes that
    // stress its pixel-advance logic (1x1, single row/column, odd sizes).
    printf("\n-- pointwise_skip_pipe_top (prototype: overlapped input load) --\n");
    const SkipKernel pipe = SKIP_PIPE;
    gen_pw_blocks(pw_rand, 0);   run_pw_skip("pipe: dense (0% blocks pruned)", PW_H, PW_W, pipe);
    gen_pw_blocks(pw_rand, 30);  run_pw_skip("pipe: 30% blocks pruned", PW_H, PW_W, pipe);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("pipe: 50% blocks pruned", PW_H, PW_W, pipe);
    gen_pw_blocks(pw_rand, 70);  run_pw_skip("pipe: 70% blocks pruned", PW_H, PW_W, pipe);
    gen_pw_blocks(pw_rand, 90);  run_pw_skip("pipe: 90% blocks pruned", PW_H, PW_W, pipe);
    gen_pw_blocks(pw_rand, 100); run_pw_skip("pipe: all weights zero", PW_H, PW_W, pipe);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("pipe: 50% blocks pruned, 8x8", 8, 8, pipe);
    gen_pw_blocks(pw_sparse, 0); run_pw_skip("pipe: 50% unstructured zeros", PW_H, PW_W, pipe);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("pipe: h/w too big (clamped)", PW_H + 3, PW_W + 7, pipe);
    gen_pw_blocks(pw_bigb, 50);  run_pw_skip("pipe: bias at limit", PW_H, PW_W, pipe);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("pipe: 1x1", 1, 1, pipe);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("pipe: 1x16 (one row)", 1, PW_W, pipe);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("pipe: 16x1 (one column)", PW_H, 1, pipe);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("pipe: 3x5", 3, 5, pipe);
    gen_pw_blocks(pw_rand, 50);  run_pw_skip("pipe: h=w=0 (writes nothing)", 0, 0, pipe);

    if (g_failures) {
        printf("\nC-SIM FAILED: %d test(s) failed\n", g_failures);
        return 1;
    }
    printf("\nC-SIM PASSED: all tests bit-exact against golden model\n");
    return 0;
}
