// libdsc.cpp — the demo's C side, loaded by server.py with ctypes (libdsc.so).
//
//   FPGA:  dsc_top through UIO + u-dma-buf. Same device access as
//          board/sw/dsc_bringup.cpp, reduced to what the demo needs.
//   CPU:   the same block on one ARM core (the golden model, plain C), as the baseline
//          and as the bit-exact check of every FPGA result.
//   ARM:   the network's non-FPGA layers (stem convs, 1x1 projection), float32.
//
// The library only touches the hardware if a UIO device named "dsc_top" exists, which
// only our overlay (board/firmware/dsc_accel.dts) creates. Otherwise it runs in SIM
// mode: FPGA calls are computed by the golden model and their latency comes from the
// co-sim cycle model. That way an unloaded PL can never be accessed (that would hang
// the CPU), and the whole demo can be developed on a PC.
//
// Build with -ffp-contract=off so the float layers round identically on x86 and arm64.

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <string>

#include "dsc_golden.h"
#include "dsc_pack.h"
#include "dsc_regs.h"

namespace {

const int NSLOT = 64;                                  // weight sets kept in DMA memory
const size_t OFF_PARAMS = 0x0000, SLOT_STRIDE = 0x1000;
const size_t OFF_IN = NSLOT * SLOT_STRIDE;             // 0x40000
const size_t OFF_OUT = OFF_IN + 0x2000;
const size_t DMA_NEEDED = OFF_OUT + OUT_FRAME_BYTES_MAX;
static_assert(SLOT_STRIDE >= (size_t)PARAM_BYTES && 0x2000 >= IN_FRAME_BYTES, "layout");

volatile uint32_t *g_regs = nullptr;
int g_uio = -1;
bool g_irq = false;
volatile uint8_t *g_dma = nullptr;
uint64_t g_phys = 0;
bool g_fpga = false;
std::string g_mode = "not opened";

// Host copies of each slot's weights, for the CPU path and SIM mode.
struct Slot {
    bool set = false;
    int dw_wt[DW_C][DW_K][DW_K], dw_b[DW_C], pw_wt[PW_C_OUT][PW_C_IN], pw_b[PW_C_OUT];
    int sched_len_skip;
};
Slot g_slot[NSLOT];

double now_us() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

uint32_t rd(uint32_t off) { return g_regs[off / 4]; }
void wr(uint32_t off, uint32_t v) { g_regs[off / 4] = v; }

bool read_line(const std::string &path, char *buf, size_t n) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;
    const bool ok = fgets(buf, (int)n, f) != nullptr;
    fclose(f);
    return ok;
}

// Uncached mapping = Device memory on arm64: aligned 64-bit accesses only.
void to_dma(size_t off, const uint8_t *src, size_t n) {
    volatile uint64_t *d = (volatile uint64_t *)(g_dma + off);
    for (size_t i = 0; i < n / 8; i++) {
        uint64_t v;
        memcpy(&v, src + 8 * i, 8);
        d[i] = v;
    }
}
void from_dma(size_t off, uint8_t *dst, size_t n) {
    const volatile uint64_t *s = (const volatile uint64_t *)(g_dma + off);
    for (size_t i = 0; i < n / 8; i++) {
        const uint64_t v = s[i];
        memcpy(dst + 8 * i, &v, 8);
    }
}

int out_dim(int stride) { return (DW_H + 2 * DW_PAD - DW_K) / stride + 1; }

// Co-sim fit (hls/compute/results/cosim_results.csv, dsc_top_final): stride 1 is
// 256*n + 1672 cycles; stride 2 is pointwise-bound (64*n + 1672) down to the line
// buffer's ~9.8k floor. Used only in SIM mode, and labelled as such.
double model_us(int stride, int n) {
    const int px = out_dim(stride) * out_dim(stride);
    double cyc = (double)px * n + 1672;
    if (stride == 2 && cyc < 9801) cyc = 9801;
    return cyc / 200.0;
}

// One hardware call on the slot's params and whatever input is in DMA memory.
int hw_call(int slot, int stride, int skip_en, double *us, int *sched_len) {
    const uint64_t in = g_phys + OFF_IN, par = g_phys + OFF_PARAMS + slot * SLOT_STRIDE,
                   out = g_phys + OFF_OUT;
    for (int i = 0; !(rd(DSC_AP_CTRL) & DSC_CTRL_IDLE); i++)
        if (i > 1000000) return -2;
    wr(DSC_IN_LO, (uint32_t)in);      wr(DSC_IN_HI, (uint32_t)(in >> 32));
    wr(DSC_PARAMS_LO, (uint32_t)par); wr(DSC_PARAMS_HI, (uint32_t)(par >> 32));
    wr(DSC_OUT_LO, (uint32_t)out);    wr(DSC_OUT_HI, (uint32_t)(out >> 32));
    wr(DSC_STRIDE, (uint32_t)stride);
    wr(DSC_SKIP_EN, skip_en ? 1u : 0u);
    if (g_irq) {
        wr(DSC_GIE, 1);
        wr(DSC_IER, 1);
        const uint32_t unmask = 1;
        if (write(g_uio, &unmask, 4) != 4) return -3;
    } else {
        wr(DSC_GIE, 0);
    }
    __sync_synchronize();
    const double t0 = now_us();
    wr(DSC_AP_CTRL, DSC_CTRL_START);
    if (g_irq) {
        pollfd p = {g_uio, POLLIN, 0};
        if (poll(&p, 1, 1000) != 1) return -4;
        uint32_t cnt;
        if (read(g_uio, &cnt, 4) != 4) return -3;
        wr(DSC_ISR, 1);
    } else {
        for (long i = 0; !(rd(DSC_AP_CTRL) & DSC_CTRL_DONE); i++)
            if (i > 50000000) return -4;
    }
    *us = now_us() - t0;
    __sync_synchronize();
    *sched_len = (int)rd(DSC_SCHED_LEN);
    return 0;
}

void golden(const Slot &s, const uint8_t *in_chw, int stride, uint8_t *out_chw) {
    static int in[DW_C][DW_H][DW_W], dw[DW_C][DW_HO_MAX][DW_WO_MAX], pw[PW_C_OUT][PW_H][PW_W];
    for (int i = 0; i < DW_C * DW_H * DW_W; i++) (&in[0][0][0])[i] = in_chw[i];
    golden_depthwise(in, s.dw_wt, s.dw_b, stride, dw);
    const int ho = out_dim(stride);
    golden_pointwise(dw, s.pw_wt, s.pw_b, ho, ho, pw);
    for (int k = 0; k < PW_C_OUT; k++)
        for (int y = 0; y < ho; y++)
            for (int x = 0; x < ho; x++) out_chw[(k * ho + y) * ho + x] = (uint8_t)pw[k][y][x];
}

inline uint8_t quant_relu6(float v) {          // same rounding as the hardware requant
    float q = floorf(v * 32.0f + 0.5f);
    if (q < 0) q = 0;
    if (q > 192) q = 192;
    return (uint8_t)q;
}

}  // namespace

extern "C" {

// Returns 1 = real FPGA, 0 = SIM mode, <0 = error (FPGA present but DMA unusable).
int dsc_open(int use_irq) {
    g_fpga = false;
    std::string uio;
    for (int i = 0; i < 16 && uio.empty(); i++) {
        char name[64];
        const std::string base = "/sys/class/uio/uio" + std::to_string(i);
        if (read_line(base + "/name", name, sizeof name) && strncmp(name, "dsc_top", 7) == 0)
            uio = "/dev/uio" + std::to_string(i);
    }
    if (uio.empty()) {
        g_mode = "SIM: no dsc_top UIO device (bitstream not loaded); FPGA results computed by the golden model";
        return 0;
    }
    g_uio = open(uio.c_str(), O_RDWR | O_SYNC);
    if (g_uio < 0) { g_mode = "error: open " + uio + ": " + strerror(errno); return -1; }
    void *r = mmap(nullptr, DSC_REG_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, g_uio, 0);
    if (r == MAP_FAILED) { g_mode = "error: mmap registers"; return -1; }
    g_regs = (volatile uint32_t *)r;

    char buf[64];
    if (!read_line("/sys/class/u-dma-buf/udmabuf0/phys_addr", buf, sizeof buf)) {
        g_mode = "error: u-dma-buf not loaded (/sys/class/u-dma-buf/udmabuf0)";
        return -1;
    }
    g_phys = strtoull(buf, nullptr, 0);
    read_line("/sys/class/u-dma-buf/udmabuf0/size", buf, sizeof buf);
    const size_t size = strtoull(buf, nullptr, 0);
    if (size < DMA_NEEDED) { g_mode = "error: udmabuf0 too small"; return -1; }
    const int fd = open("/dev/udmabuf0", O_RDWR | O_SYNC);
    if (fd < 0) { g_mode = "error: open /dev/udmabuf0"; return -1; }
    void *d = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (d == MAP_FAILED) { g_mode = "error: mmap udmabuf0"; return -1; }
    g_dma = (volatile uint8_t *)d;
    g_irq = use_irq != 0;
    g_fpga = true;
    g_mode = "FPGA: dsc_top via " + uio + (g_irq ? " (interrupt)" : " (polling)") +
             ", u-dma-buf @ phys 0x" + [] { char b[32]; snprintf(b, sizeof b, "%llx", (unsigned long long)g_phys); return std::string(b); }() +
             (DSC_REGS_CONFIRMED ? ", register map from xdsc_top_hw.h" : "");
    return 1;
}

const char *dsc_mode() { return g_mode.c_str(); }
void dsc_set_irq(int on) { g_irq = on && g_fpga; }

// Weight set for one block (PyTorch layouts, as in VECTORS.md). Returns the sched_len
// the hardware should report with skipping on.
int dsc_set_params(int slot, const int8_t *dw_wt, const int32_t *dw_b, const int8_t *pw_wt,
                   const int32_t *pw_b) {
    if (slot < 0 || slot >= NSLOT) return -1;
    Slot &s = g_slot[slot];
    for (int c = 0; c < DW_C; c++) {
        for (int k = 0; k < DW_K * DW_K; k++) s.dw_wt[c][k / 3][k % 3] = dw_wt[c * 9 + k];
        s.dw_b[c] = dw_b[c];
    }
    int len = 0;
    for (int k = 0; k < PW_C_OUT; k++) {
        int nz = 0;
        for (int c = 0; c < PW_C_IN; c++) s.pw_wt[k][c] = pw_wt[k * PW_C_IN + c];
        for (int cb = 0; cb < PW_CBLKS; cb++) {
            bool any = false;
            for (int u = 0; u < PW_UNROLL; u++) any = any || s.pw_wt[k][cb * PW_UNROLL + u] != 0;
            nz += any;
        }
        len += nz ? nz : 1;
        s.pw_b[k] = pw_b[k];
    }
    s.sched_len_skip = len;
    s.set = true;
    if (g_fpga) {
        static uint8_t blob[PARAM_BYTES];
        dsc_pack_params(s.dw_wt, s.dw_b, s.pw_wt, s.pw_b, blob);
        to_dma(OFF_PARAMS + slot * SLOT_STRIDE, blob, PARAM_BYTES);
    }
    return len;
}

// One accelerator call: in_chw u8 [32][16][16] -> out_chw u8 [64][ho][ho].
// *us = accelerator latency (start -> done), or the cycle model's in SIM mode.
int dsc_run(int slot, const uint8_t *in_chw, int stride, int skip_en, uint8_t *out_chw, double *us,
            int *sched_len) {
    if (slot < 0 || slot >= NSLOT || !g_slot[slot].set) return -1;
    stride = stride == 2 ? 2 : 1;
    const int ho = out_dim(stride);
    if (!g_fpga) {
        golden(g_slot[slot], in_chw, stride, out_chw);
        *sched_len = skip_en ? g_slot[slot].sched_len_skip : PW_C_OUT * PW_CBLKS;
        *us = model_us(stride, *sched_len);
        return 0;
    }
    static uint8_t hwc[IN_FRAME_BYTES], out[OUT_FRAME_BYTES_MAX];
    for (int c = 0; c < DW_C; c++)
        for (int p = 0; p < DW_H * DW_W; p++) hwc[p * DW_C + c] = in_chw[c * DW_H * DW_W + p];
    to_dma(OFF_IN, hwc, IN_FRAME_BYTES);
    const int rc = hw_call(slot, stride, skip_en, us, sched_len);
    if (rc) return rc;
    const size_t n = (size_t)ho * ho * PW_C_OUT;
    from_dma(OFF_OUT, out, n);
    for (int p = 0; p < ho * ho; p++)
        for (int k = 0; k < PW_C_OUT; k++) out_chw[k * ho * ho + p] = out[p * PW_C_OUT + k];
    return 0;
}

// n back-to-back calls on the input already in DMA memory (for latency and power
// measurements). *total_us = wall time of the whole loop; *min_us = fastest call.
int dsc_run_repeat(int slot, int stride, int skip_en, int n, double *total_us, double *min_us) {
    if (slot < 0 || slot >= NSLOT || !g_slot[slot].set || !g_fpga) return -1;
    stride = stride == 2 ? 2 : 1;
    double best = 1e18;
    const double t0 = now_us();
    for (int i = 0; i < n; i++) {
        double us;
        int len;
        const int rc = hw_call(slot, stride, skip_en, &us, &len);
        if (rc) return rc;
        if (us < best) best = us;
    }
    *total_us = now_us() - t0;
    *min_us = best;
    return 0;
}

// The same block on the CPU (one core, plain C golden model). Returns its time.
double dsc_cpu(int slot, const uint8_t *in_chw, int stride, uint8_t *out_chw) {
    if (slot < 0 || slot >= NSLOT || !g_slot[slot].set) return -1;
    const double t0 = now_us();
    golden(g_slot[slot], in_chw, stride == 2 ? 2 : 1, out_chw);
    return now_us() - t0;
}

// ---- ARM float layers (BN folded on export; demo/model/train.py) ----------------------
// conv3x3, pad 1, NCHW float in -> float out (no activation). Each output accumulates
// bias, then c, ky, kx in that order; the innermost loop runs along x so it vectorizes
// (the per-output order, and so the rounding, doesn't depend on the vector width).
static void conv3x3(const float *in, int cin, int h, int w, const float *wt, const float *b, int cout,
                    int stride, float *out) {
    const int ho = (h + 2 - 3) / stride + 1, wo = (w + 2 - 3) / stride + 1, hp = h + 2, wp = w + 2;
    static float pad[16 * 34 * 34];                        // largest input: 16 x 32 x 32
    for (int i = 0; i < cin * hp * wp; i++) pad[i] = 0;
    for (int c = 0; c < cin; c++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) pad[(c * hp + y + 1) * wp + x + 1] = in[(c * h + y) * w + x];
    for (int k = 0; k < cout; k++) {
        float *o = out + k * ho * wo;
        for (int i = 0; i < ho * wo; i++) o[i] = b[k];
        for (int c = 0; c < cin; c++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    const float wv = wt[((k * cin + c) * 3 + ky) * 3 + kx];
                    for (int y = 0; y < ho; y++) {
                        const float *src = pad + (c * hp + y * stride + ky) * wp + kx;
                        float *dst = o + y * wo;
                        if (stride == 1)
                            for (int x = 0; x < wo; x++) dst[x] += wv * src[x];
                        else
                            for (int x = 0; x < wo; x++) dst[x] += wv * src[2 * x];
                    }
                }
    }
}

// RGB HWC u8 [32][32][3] -> block-1 input, u8 Q3.5 [32][16][16].
void dsc_arm_stem(const uint8_t *rgb, const float *mean, const float *std, const float *w1, const float *b1,
                  const float *w2, const float *b2, uint8_t *out) {
    static float x[3 * 32 * 32], s1[16 * 32 * 32], s2[32 * 16 * 16];
    for (int c = 0; c < 3; c++)
        for (int p = 0; p < 32 * 32; p++) x[c * 1024 + p] = (rgb[p * 3 + c] / 255.0f - mean[c]) / std[c];
    conv3x3(x, 3, 32, 32, w1, b1, 16, 1, s1);
    for (int i = 0; i < 16 * 32 * 32; i++) s1[i] = s1[i] < 0 ? 0 : (s1[i] > 6 ? 6 : s1[i]);
    conv3x3(s1, 16, 32, 32, w2, b2, 32, 2, s2);
    for (int i = 0; i < 32 * 16 * 16; i++) out[i] = quant_relu6(s2[i]);
}

// Block-1 output u8 [64][16][16] -> block-2 input u8 [32][16][16] (1x1 conv + ReLU6).
void dsc_arm_proj(const uint8_t *in, const float *w, const float *b, uint8_t *out) {
    static float x[64 * 256], acc[256];
    for (int i = 0; i < 64 * 256; i++) x[i] = in[i] / 32.0f;
    for (int k = 0; k < 32; k++) {                     // same order as before: bias, then c
        for (int p = 0; p < 256; p++) acc[p] = b[k];
        for (int c = 0; c < 64; c++) {
            const float wv = w[k * 64 + c];
            for (int p = 0; p < 256; p++) acc[p] += wv * x[c * 256 + p];
        }
        for (int p = 0; p < 256; p++) out[k * 256 + p] = quant_relu6(acc[p]);
    }
}

double dsc_now_us() { return now_us(); }

}  // extern "C"
