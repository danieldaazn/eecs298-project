// requant.cpp — accumulator -> activation conversion shared by both kernels.

#include "dsc_kernels.h"

act_t requant_relu6(acc_t acc) {
#pragma HLS INLINE
    // Round half up, then arithmetic shift (floor). acc + 64 cannot overflow:
    // |acc| < 2^25 by construction (see dsc_types.h).
    acc_t r = (acc + (acc_t)(1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT;
    if (r < 0) r = 0;
    if (r > ACT_RELU6_MAX) r = ACT_RELU6_MAX;
    return (act_t)r;
}
