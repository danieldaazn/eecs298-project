// dsc_types.h — numeric types and layer dimensions for the compute core.
//
// !!! SHARED INTERFACE — PLACEHOLDER VALUES !!!
// The widths and scales below are Daniel's proposal and have NOT yet been signed
// off by Hank (buffer/stream widths) or Kevin (quantization export). Do not treat
// them as final; change them only by team agreement (see CLAUDE.md).
//
// Representation: every value is stored as its raw two's-complement integer with an
// implied binary point (Q format). This is bit-for-bit what ap_fixed / ap_ufixed hold
// internally, but expressing it as ap_int / ap_uint means the same arithmetic can be
// reproduced exactly with native C++ integers. That makes the golden model and a
// plain-g++ C-sim bit-exact against the Vitis build.
//
//   activation  act_t  : unsigned 8-bit, Q3.5  (== ap_ufixed<8,3>), range [0, 7.97]
//                        ReLU6 output 6.0 is raw 192.
//   weight      wgt_t  : signed 8-bit,   Q1.7  (== ap_fixed<8,1>),  range [-1, 0.992]
//   product            : Q4.12  (5 + 7 fractional bits)
//   accumulator acc_t  : signed 32-bit,  12 fractional bits. Bias is stored in this
//                        scale so it adds directly. 32 bits is generous: worst case
//                        |product| = 255*128 = 32640, times PW_C_IN = 32 terms is
//                        ~2^20, and |bias| <= BIAS_ABS_MAX = 2^24 (below), so |acc|
//                        < 2^25 and ~26 bits would suffice. Hank/Kevin may want this
//                        narrowed for stream width.
//   bias               : acc_t scale, |bias| <= BIAS_ABS_MAX (2^24, i.e. 4096.0).
//                        PART OF THE EXPORT CONTRACT: the requantization assumes it
//                        (acc + rounding offset must not overflow). Real biases are
//                        tiny by comparison (ReLU6's 6.0 is 24,576 in this scale).
//
// Requantization (acc -> act): shift right by (ACC_FRAC - ACT_FRAC) = 7 with
// round-half-up, then clamp to [0, ACT_RELU6_MAX]. This is MobileNetV1-style:
// ReLU6 after both depthwise and pointwise.

#ifndef DSC_TYPES_H
#define DSC_TYPES_H

#include <stdint.h>

#if defined(__SYNTHESIS__) || defined(DSC_USE_AP_TYPES)
#include <ap_int.h>
typedef ap_uint<8>  act_t;
typedef ap_int<8>   wgt_t;
typedef ap_int<32>  acc_t;
#else
// Plain C++ build (no Xilinx headers): bit-identical within the ranges above.
typedef uint8_t act_t;
typedef int8_t  wgt_t;
typedef int32_t acc_t;
#endif

// ---- Q-format constants -----------------------------------------------------
static const int ACT_FRAC      = 5;
static const int WGT_FRAC      = 7;
static const int ACC_FRAC      = ACT_FRAC + WGT_FRAC;   // 12
static const int REQUANT_SHIFT = ACC_FRAC - ACT_FRAC;   // 7
static const int ACT_RELU6_MAX = 6 << ACT_FRAC;         // 192
static const int BIAS_ABS_MAX  = 1 << 24;                // export contract, see above

// ---- Layer dimensions (placeholder test layer, compile-time maxima) -----------
// Sized for a small MobileNet-style block. Kevin's test images will set the real
// values; Hank's line buffers are sized against DW_W.
static const int DW_C     = 32;   // depthwise channels
static const int DW_H     = 16;   // input height
static const int DW_W     = 16;   // input width
static const int DW_K     = 3;    // kernel size (fixed 3x3)
static const int DW_PAD   = 1;    // "same" padding for stride 1
static const int DW_HO_MAX = DW_H;  // output height at stride 1
static const int DW_WO_MAX = DW_W;

static const int PW_C_IN  = DW_C;  // pointwise input channels
static const int PW_C_OUT = 64;    // pointwise output channels
static const int PW_H     = DW_HO_MAX;
static const int PW_W     = DW_WO_MAX;

// Pointwise MACs per cycle (unroll factor across input channels), and the number of
// cycles per pointwise output that implies. PW_C_IN must be a multiple of PW_UNROLL.
static const int PW_UNROLL = 8;
static const int PW_CBLKS  = PW_C_IN / PW_UNROLL;

// Output size for a 3x3 conv with padding 1; stride must be 1 or 2.
// Written as a shift, not "/ stride", so HLS doesn't build a runtime divider.
static inline int dw_out_dim(int in_dim, int stride) {
    const int span = in_dim + 2 * DW_PAD - DW_K;
    return ((stride == 2) ? (span >> 1) : span) + 1;
}

#endif // DSC_TYPES_H
