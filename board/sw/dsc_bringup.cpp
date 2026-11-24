// dsc_bringup.cpp — first-light test and benchmark for dsc_top on the KR260.
//
// Runs on the board (Kria Ubuntu, after `sudo board/firmware/install.sh`):
//   sudo ./dsc_bringup                       self-test + timing, random weights
//   sudo ./dsc_bringup --case DIR [...]      also run VECTORS.md case folders (Kevin's)
//   sudo ./dsc_bringup --csv results.csv     write the timing table as CSV
//   options: --reps N (timed calls per point, default 100), --poll (don't use the IRQ)
//
// What it checks, per random case (sparsity x stride x skip on/off):
//   - every output byte equals the golden model (hls/compute/golden), the same model
//     C-sim and co-sim were checked against;
//   - nothing is written past the end of the output frame;
//   - sched_len is what the weights imply (skip on) or the dense 256 (skip off).
// Then it times each point and prints latency, the equivalent cycles at 200 MHz, the
// co-sim cycle model's prediction, and the skip-on speedup.
//
// Registers: through UIO (node "dsc_top", from board/firmware/dsc_accel.dts) if bound,
// otherwise /dev/mem at 0xA000_0000. Buffers: one u-dma-buf (/dev/udmabuf0), opened
// O_SYNC so CPU accesses are uncached and no cache maintenance is needed (HP0/HP1 are
// not coherent).

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include "dsc_golden.h"
#include "dsc_pack.h"
#include "dsc_regs.h"

static const uint64_t CTRL_PHYS = 0xA0000000ull;  // build_bd.tcl / dsc_accel.dts
static const double   CLK_MHZ   = 200.0;

// Buffer layout inside the u-dma-buf (all 4 KB aligned).
static const size_t OFF_IN = 0x0000, OFF_PARAMS = 0x2000, OFF_OUT = 0x3000;
static const size_t OUT_GUARD = 256;
static const size_t DMA_NEEDED = OFF_OUT + OUT_FRAME_BYTES_MAX + OUT_GUARD;
static_assert(OFF_PARAMS >= OFF_IN + IN_FRAME_BYTES, "input overlaps params");
static_assert(OFF_OUT >= OFF_PARAMS + PARAM_BYTES, "params overlap output");

static void die(const char *what) {
    fprintf(stderr, "dsc_bringup: %s: %s\n", what, strerror(errno));
    exit(2);
}

static double now_us() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

// ---- device access ---------------------------------------------------------------------
struct Dev {
    volatile uint32_t *regs = nullptr;
    int uio_fd = -1;                 // -1: polling only
    volatile uint8_t *dma = nullptr;
    uint64_t dma_phys = 0;
    size_t dma_size = 0;

    uint32_t rd(uint32_t off) const { return regs[off / 4]; }
    void wr(uint32_t off, uint32_t v) const { regs[off / 4] = v; }
};

static bool read_sysfs(const std::string &path, char *buf, size_t n) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;
    const bool ok = fgets(buf, (int)n, f) != nullptr;
    fclose(f);
    return ok;
}

// The uncached mapping is Device memory on arm64, where memcpy's unaligned accesses can
// fault, so copy with aligned 64-bit accesses. All sizes are multiples of 16.
static void to_dma(const Dev &d, size_t off, const uint8_t *src, size_t n) {
    volatile uint64_t *dst = (volatile uint64_t *)(d.dma + off);
    for (size_t i = 0; i < n / 8; i++) {
        uint64_t v;
        memcpy(&v, src + 8 * i, 8);
        dst[i] = v;
    }
}
static void from_dma(const Dev &d, size_t off, uint8_t *dst, size_t n) {
    const volatile uint64_t *src = (const volatile uint64_t *)(d.dma + off);
    for (size_t i = 0; i < n / 8; i++) {
        const uint64_t v = src[i];
        memcpy(dst + 8 * i, &v, 8);
    }
}
static void fill_dma(const Dev &d, size_t off, uint8_t byte, size_t n) {
    std::vector<uint8_t> tmp(n, byte);
    to_dma(d, off, tmp.data(), n);
}

// One accelerator call on the buffers already in DMA memory. Returns false on timeout.
#ifdef DSC_HOST_SIM
// `make hostsim`: no board. The "device" is the C-sim dsc_top, the DMA buffer is plain
// memory at a fake physical address, and a call runs the kernel on it directly. Checks
// everything in this file except the register/UIO/u-dma-buf plumbing.
static const uint64_t SIM_PHYS = 0x70000000ull;
static uint8_t sim_mem[DMA_NEEDED] __attribute__((aligned(16)));
static void open_regs(Dev &, bool) { printf("registers: HOST SIM (C-sim dsc_top, no hardware)\n"); }
static void open_dma(Dev &d) {
    d.dma = sim_mem;
    d.dma_phys = SIM_PHYS;
    d.dma_size = sizeof sim_mem;
}
static bool run_once(const Dev &d, int stride, bool skip_en, int &sched_len, double &us) {
    static axi_word_t in[IN_FRAME_BEATS], par[PARAM_BEATS], out[OUT_FRAME_BEATS_MAX];
    const uint8_t *m = (const uint8_t *)d.dma;
    dsc_bytes_to_beats(m + OFF_IN, in, IN_FRAME_BEATS);
    dsc_bytes_to_beats(m + OFF_PARAMS, par, PARAM_BEATS);
    dsc_bytes_to_beats(m + OFF_OUT, out, OUT_FRAME_BEATS_MAX);  // keep untouched bytes
    const double t0 = now_us();
    dsc_top(in, par, out, stride, skip_en, sched_len);
    us = now_us() - t0;
    dsc_beats_to_bytes(out, (uint8_t *)d.dma + OFF_OUT, OUT_FRAME_BEATS_MAX);
    return true;
}
#else
static void open_regs(Dev &d, bool use_irq) {
    for (int i = 0; i < 16; i++) {
        char name[64];
        if (!read_sysfs("/sys/class/uio/uio" + std::to_string(i) + "/name", name, sizeof name)) continue;
        if (strncmp(name, "dsc_top", 7) != 0) continue;
        const std::string dev = "/dev/uio" + std::to_string(i);
        const int fd = open(dev.c_str(), O_RDWR | O_SYNC);
        if (fd < 0) die(dev.c_str());
        void *p = mmap(nullptr, DSC_REG_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) die("mmap uio");
        d.regs = (volatile uint32_t *)p;
        d.uio_fd = use_irq ? fd : -1;
        printf("registers: %s%s\n", dev.c_str(), use_irq ? " (interrupt mode)" : " (polling)");
        return;
    }
    const int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) die("/dev/mem (no UIO device named dsc_top either)");
    void *p = mmap(nullptr, DSC_REG_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CTRL_PHYS);
    if (p == MAP_FAILED) die("mmap /dev/mem");
    d.regs = (volatile uint32_t *)p;
    printf("registers: /dev/mem @ 0x%llx (polling; UIO not bound)\n", (unsigned long long)CTRL_PHYS);
}

static void open_dma(Dev &d) {
    char buf[64];
    if (!read_sysfs("/sys/class/u-dma-buf/udmabuf0/phys_addr", buf, sizeof buf))
        die("/sys/class/u-dma-buf/udmabuf0/phys_addr (u-dma-buf loaded? see board/README.md)");
    d.dma_phys = strtoull(buf, nullptr, 0);
    if (!read_sysfs("/sys/class/u-dma-buf/udmabuf0/size", buf, sizeof buf)) die("udmabuf0/size");
    d.dma_size = strtoull(buf, nullptr, 0);
    if (d.dma_size < DMA_NEEDED) {
        fprintf(stderr, "udmabuf0 is %zu bytes, need %zu\n", d.dma_size, DMA_NEEDED);
        exit(2);
    }
    const int fd = open("/dev/udmabuf0", O_RDWR | O_SYNC);
    if (fd < 0) die("/dev/udmabuf0");
    void *p = mmap(nullptr, d.dma_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) die("mmap udmabuf0");
    d.dma = (volatile uint8_t *)p;
    printf("buffers:   udmabuf0 @ phys 0x%llx, %zu bytes\n", (unsigned long long)d.dma_phys, d.dma_size);
}

static bool run_once(const Dev &d, int stride, bool skip_en, int &sched_len, double &us) {
    const uint64_t in = d.dma_phys + OFF_IN, par = d.dma_phys + OFF_PARAMS, out = d.dma_phys + OFF_OUT;
    for (int i = 0; !(d.rd(DSC_AP_CTRL) & DSC_CTRL_IDLE); i++)
        if (i > 1000000) { fprintf(stderr, "accelerator never idle (bitstream loaded?)\n"); return false; }
    d.wr(DSC_IN_LO, (uint32_t)in);       d.wr(DSC_IN_HI, (uint32_t)(in >> 32));
    d.wr(DSC_PARAMS_LO, (uint32_t)par);  d.wr(DSC_PARAMS_HI, (uint32_t)(par >> 32));
    d.wr(DSC_OUT_LO, (uint32_t)out);     d.wr(DSC_OUT_HI, (uint32_t)(out >> 32));
    d.wr(DSC_STRIDE, (uint32_t)stride);
    d.wr(DSC_SKIP_EN, skip_en ? 1u : 0u);
    if (d.uio_fd >= 0) {
        d.wr(DSC_GIE, 1);
        d.wr(DSC_IER, 1);                                // ap_done
        const uint32_t unmask = 1;
        if (write(d.uio_fd, &unmask, 4) != 4) die("uio unmask");
    }
    __sync_synchronize();                                // buffers written before start

    const double t0 = now_us();
    d.wr(DSC_AP_CTRL, DSC_CTRL_START);
    if (d.uio_fd >= 0) {
        pollfd pfd = {d.uio_fd, POLLIN, 0};
        if (poll(&pfd, 1, 1000) != 1) { fprintf(stderr, "no interrupt within 1 s\n"); return false; }
        uint32_t count;
        if (read(d.uio_fd, &count, 4) != 4) die("uio read");
        d.wr(DSC_ISR, 1);                                // clear ap_done
    } else {
        for (long i = 0; !(d.rd(DSC_AP_CTRL) & DSC_CTRL_DONE); i++)
            if (i > 50000000) { fprintf(stderr, "ap_done never set\n"); return false; }
    }
    us = now_us() - t0;
    __sync_synchronize();
    sched_len = (int)d.rd(DSC_SCHED_LEN);
    return true;
}
#endif  // DSC_HOST_SIM

// ---- test data -------------------------------------------------------------------------
static int g_in[DW_C][DW_H][DW_W], g_dw_wt[DW_C][DW_K][DW_K], g_dw_b[DW_C];
static int g_dw_out[DW_C][DW_HO_MAX][DW_WO_MAX];
static int g_pw_wt[PW_C_OUT][PW_C_IN], g_pw_b[PW_C_OUT], g_pw_out[PW_C_OUT][PW_H][PW_W];
static uint8_t h_in[IN_FRAME_BYTES], h_params[PARAM_BYTES], h_out[OUT_FRAME_BYTES_MAX + OUT_GUARD];

static uint32_t rng_state = 0x2545F491u;
static uint32_t rng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static int rand_range(int lo, int hi) { return lo + (int)(rng() % (uint32_t)(hi - lo + 1)); }

// Same ranges as tb_dsc_top's "typical" case; block_pct = % of 8-wide PW blocks zeroed.
static void gen_random(int block_pct) {
    for (int c = 0; c < DW_C; c++) {
        for (int y = 0; y < DW_H; y++)
            for (int x = 0; x < DW_W; x++) g_in[c][y][x] = rand_range(0, 192);
        for (int ky = 0; ky < DW_K; ky++)
            for (int kx = 0; kx < DW_K; kx++) g_dw_wt[c][ky][kx] = rand_range(-128, 127);
        g_dw_b[c] = rand_range(-20000, 20000);
    }
    for (int k = 0; k < PW_C_OUT; k++) {
        for (int c = 0; c < PW_C_IN; c++) g_pw_wt[k][c] = rand_range(-32, 31);
        for (int cb = 0; cb < PW_CBLKS; cb++)
            if ((int)(rng() % 100) < block_pct)
                for (int u = 0; u < PW_UNROLL; u++) g_pw_wt[k][cb * PW_UNROLL + u] = 0;
        g_pw_b[k] = rand_range(-20000, 20000);
    }
}

static int out_dim(int n, int stride) { return (n + 2 * DW_PAD - DW_K) / stride + 1; }

static int expected_sched_len() {
    int total = 0;
    for (int k = 0; k < PW_C_OUT; k++) {
        int nz = 0;
        for (int cb = 0; cb < PW_CBLKS; cb++) {
            bool any = false;
            for (int u = 0; u < PW_UNROLL; u++) any = any || g_pw_wt[k][cb * PW_UNROLL + u] != 0;
            nz += any;
        }
        total += nz ? nz : 1;
    }
    return total;
}

// Fraction of all PW blocks that are zero (what skipping can remove).
static double block_sparsity() {
    int z = 0;
    for (int k = 0; k < PW_C_OUT; k++)
        for (int cb = 0; cb < PW_CBLKS; cb++) {
            bool any = false;
            for (int u = 0; u < PW_UNROLL; u++) any = any || g_pw_wt[k][cb * PW_UNROLL + u] != 0;
            z += !any;
        }
    return (double)z / (PW_C_OUT * PW_CBLKS);
}

// ---- one test point --------------------------------------------------------------------
struct Point {
    std::string name;
    int stride;
    double sparsity;
    int sched_len;
    double us_skip, us_dense;
};
static std::vector<Point> g_points;
static int g_failures = 0;

static double median_us(const Dev &d, int stride, bool skip_en, int reps) {
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        int len;
        double us;
        if (!run_once(d, stride, skip_en, len, us)) return -1;
        t.push_back(us);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Uses the g_* tensors. expected_pw (optional, C_OUT x ho x wo bytes) is checked too.
static void run_point(const Dev &d, const std::string &name, int stride, int reps,
                      const uint8_t *expected_pw = nullptr) {
    dsc_pack_input(g_in, h_in);
    dsc_pack_params(g_dw_wt, g_dw_b, g_pw_wt, g_pw_b, h_params);
    to_dma(d, OFF_IN, h_in, IN_FRAME_BYTES);
    to_dma(d, OFF_PARAMS, h_params, PARAM_BYTES);

    golden_depthwise(g_in, g_dw_wt, g_dw_b, stride, g_dw_out);
    const int ho = out_dim(DW_H, stride), wo = out_dim(DW_W, stride);
    golden_pointwise(g_dw_out, g_pw_wt, g_pw_b, ho, wo, g_pw_out);
    const size_t out_bytes = (size_t)ho * wo * PW_C_OUT;
    const int want_len[2] = {PW_C_OUT * PW_CBLKS, expected_sched_len()};

    int mism = 0, exp_mism = 0, overrun = 0, len[2] = {-1, -1};
    bool ran = true;
    for (int skip = 0; skip < 2 && ran; skip++) {
        fill_dma(d, OFF_OUT, 0xA5, OUT_FRAME_BYTES_MAX + OUT_GUARD);
        double us;
        ran = run_once(d, stride, skip, len[skip], us);
        if (!ran) break;
        from_dma(d, OFF_OUT, h_out, OUT_FRAME_BYTES_MAX + OUT_GUARD);
        for (int y = 0; y < ho; y++)
            for (int x = 0; x < wo; x++)
                for (int k = 0; k < PW_C_OUT; k++) {
                    const int got = dsc_output_at(h_out, wo, y, x, k);
                    if (got != g_pw_out[k][y][x]) {
                        if (mism < 3)
                            printf("    skip=%d mismatch k=%d y=%d x=%d hw=%d golden=%d\n", skip, k, y, x,
                                   got, g_pw_out[k][y][x]);
                        mism++;
                    }
                    if (expected_pw && got != expected_pw[(k * ho + y) * wo + x]) exp_mism++;
                }
        for (size_t i = out_bytes; i < OUT_FRAME_BYTES_MAX + OUT_GUARD; i++) overrun += h_out[i] != 0xA5;
    }
    const bool len_ok = len[0] == want_len[0] && len[1] == want_len[1];
    const bool ok = ran && mism == 0 && exp_mism == 0 && overrun == 0 && len_ok;
    if (!len_ok) printf("    sched_len dense=%d (want %d), skip=%d (want %d)\n", len[0], want_len[0], len[1], want_len[1]);
    if (overrun) printf("    %d bytes written past the output frame\n", overrun);
    if (exp_mism) printf("    %d outputs differ from expected_pw_out.bin (skip on + off)\n", exp_mism);
    printf("[%s] %-28s s%d  %5zu outputs, %d mismatches, sched_len %3d/%d\n", ok ? "PASS" : "FAIL",
           name.c_str(), stride, out_bytes, mism, len[1], want_len[0]);
    if (!ok) { g_failures++; return; }

    if (reps > 0) {
        Point p{name, stride, block_sparsity(), len[1], median_us(d, stride, true, reps),
                median_us(d, stride, false, reps)};
        g_points.push_back(p);
    }
}

// ---- VECTORS.md case folders -----------------------------------------------------------
static bool read_file(const std::string &path, uint8_t *dst, size_t n, bool required) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        if (required) fprintf(stderr, "    missing %s\n", path.c_str());
        return false;
    }
    const size_t got = fread(dst, 1, n, f);
    const bool extra = fgetc(f) != EOF;
    fclose(f);
    if (got != n || extra) {
        fprintf(stderr, "    %s: expected exactly %zu bytes\n", path.c_str(), n);
        return false;
    }
    return true;
}
static int32_t le32(const uint8_t *p) {
    return (int32_t)((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24);
}

static void run_case_dir(const Dev &d, const std::string &dir, int reps) {
    int stride = 0, c = 0, h = 0, w = 0, c_out = 0, ver = 0;
    FILE *f = fopen((dir + "/meta.txt").c_str(), "r");
    if (!f) { printf("[FAIL] %s: no meta.txt\n", dir.c_str()); g_failures++; return; }
    char line[256];
    while (fgets(line, sizeof line, f)) {  // "key value" lines, '#' comments (VECTORS.md)
        if (char *hash = strchr(line, '#')) *hash = 0;
        char key[64];
        int val;
        if (sscanf(line, "%63s %d", key, &val) != 2) continue;
        if (!strcmp(key, "format_version")) ver = val;
        else if (!strcmp(key, "stride")) stride = val;
        else if (!strcmp(key, "c")) c = val;
        else if (!strcmp(key, "h")) h = val;
        else if (!strcmp(key, "w")) w = val;
        else if (!strcmp(key, "c_out")) c_out = val;
    }
    fclose(f);
    if (ver != 1 || (stride != 1 && stride != 2) || c != DW_C || h != DW_H || w != DW_W || c_out != PW_C_OUT) {
        printf("[FAIL] %s: meta.txt doesn't match this build (v%d s%d %dx%dx%d->%d)\n", dir.c_str(), ver,
               stride, c, h, w, c_out);
        g_failures++;
        return;
    }
    static uint8_t in8[DW_C * DW_H * DW_W], dwwt8[DW_C * 9], pwwt8[PW_C_OUT * PW_C_IN];
    static uint8_t dwb[DW_C * 4], pwb[PW_C_OUT * 4], e_pw[PW_C_OUT * DW_HO_MAX * DW_WO_MAX];
    const int ho = out_dim(DW_H, stride), wo = out_dim(DW_W, stride);
    if (!read_file(dir + "/dw_in.bin", in8, sizeof in8, true) ||
        !read_file(dir + "/dw_wt.bin", dwwt8, sizeof dwwt8, true) ||
        !read_file(dir + "/dw_b.bin", dwb, sizeof dwb, true) ||
        !read_file(dir + "/pw_wt.bin", pwwt8, sizeof pwwt8, true) ||
        !read_file(dir + "/pw_b.bin", pwb, sizeof pwb, true)) {
        printf("[FAIL] %s: unreadable case\n", dir.c_str());
        g_failures++;
        return;
    }
    const bool have_e = read_file(dir + "/expected_pw_out.bin", e_pw, (size_t)PW_C_OUT * ho * wo, false);
    for (int ch = 0; ch < DW_C; ch++) {
        for (int y = 0; y < DW_H; y++)
            for (int x = 0; x < DW_W; x++) g_in[ch][y][x] = in8[(ch * DW_H + y) * DW_W + x];
        for (int k = 0; k < 9; k++) g_dw_wt[ch][k / 3][k % 3] = (int8_t)dwwt8[ch * 9 + k];
        g_dw_b[ch] = le32(dwb + 4 * ch);
    }
    for (int k = 0; k < PW_C_OUT; k++) {
        for (int ch = 0; ch < PW_C_IN; ch++) g_pw_wt[k][ch] = (int8_t)pwwt8[k * PW_C_IN + ch];
        g_pw_b[k] = le32(pwb + 4 * k);
    }
    std::string name = dir.substr(dir.find_last_of('/') + 1);
    if (name.empty()) name = dir;
    run_point(d, name + (have_e ? "" : " (no expected)"), stride, reps, have_e ? e_pw : nullptr);
}

// ---- main ------------------------------------------------------------------------------
int main(int argc, char **argv) {
    int reps = 100;
    bool use_irq = true;
    const char *csv = nullptr;
    std::vector<std::string> cases;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--case") && i + 1 < argc) cases.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--poll")) use_irq = false;
        else {
            fprintf(stderr, "usage: %s [--reps N] [--case DIR]... [--csv FILE] [--poll]\n", argv[0]);
            return 2;
        }
    }

    Dev d;
    open_regs(d, use_irq);
    open_dma(d);
    printf("register map: %s\n\n", DSC_REGS_CONFIRMED ? "checked against xdsc_top_hw.h"
                                                       : "PREDICTED (copy xdsc_top_hw.h into board/sw to check)");

    const int pcts[] = {0, 30, 50, 70, 90, 100};
    for (int stride = 1; stride <= 2; stride++)
        for (int pct : pcts) {
            gen_random(pct);
            run_point(d, "random, " + std::to_string(pct) + "% blocks pruned", stride, reps);
        }
    for (const std::string &dir : cases) run_case_dir(d, dir, reps);

    if (!g_points.empty()) {
        // Stride-1 model: cycles = 256 * sched_len + 1,672, fitted to Daniel's co-sim (±5).
        printf("\nlatency, median of %d calls (us; cycles = us x %.0f MHz)\n", reps, CLK_MHZ);
        printf("%-30s %-3s %6s %5s %9s %9s %9s %9s %8s\n", "point", "s", "sparse", "len", "skip_us",
               "dense_us", "skip_cyc", "model_cyc", "speedup");
        for (const Point &p : g_points) {
            char model[16] = "-";
            if (p.stride == 1) snprintf(model, sizeof model, "%d", 256 * p.sched_len + 1672);
            printf("%-30s s%-2d %5.0f%% %5d %9.1f %9.1f %9.0f %9s %7.2fx\n", p.name.c_str(), p.stride,
                   100 * p.sparsity, p.sched_len, p.us_skip, p.us_dense, p.us_skip * CLK_MHZ, model,
                   p.us_dense / p.us_skip);
        }
#ifdef DSC_HOST_SIM
        printf("(HOST SIM: these times are the C model on this CPU, not the hardware.)\n");
#endif
        printf("(latency includes the start/finish handshake from user space, a few us;\n"
               " DRAM traffic is ~27 KB per call, ~9 us at one HP port's 3.2 GB/s.)\n");
    }
    if (csv && !g_points.empty()) {
        FILE *f = fopen(csv, "w");
        if (!f) die(csv);
        fprintf(f, "point,stride,block_sparsity,sched_len,skip_us,dense_us,speedup\n");
        for (const Point &p : g_points)
            fprintf(f, "\"%s\",%d,%.4f,%d,%.2f,%.2f,%.4f\n", p.name.c_str(), p.stride, p.sparsity, p.sched_len,
                    p.us_skip, p.us_dense, p.us_dense / p.us_skip);
        fclose(f);
        printf("wrote %s\n", csv);
    }

    printf("\n%s: %d failing point(s)\n", g_failures ? "BRING-UP FAILED" : "BRING-UP PASSED", g_failures);
    return g_failures ? 1 : 0;
}
