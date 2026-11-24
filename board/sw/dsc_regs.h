// dsc_regs.h — dsc_top AXI-Lite register offsets (one "control" bank).
//
// PREDICTED from Vitis HLS's usual layout for this signature (three 64-bit pointers,
// two scalar inputs, one scalar output with ap_vld); see hls/memory/ARCHITECTURE.md.
// Confirm against the generated header after `run_hls.tcl ... export`:
//   hls/top/proj_dsc_top/solution1/impl/misc/drivers/dsc_top_v1_0/src/xdsc_top_hw.h
// Copy that file next to this one (board/sw/) and commit it: the static_asserts below
// then fail the build if any offset here is wrong.

#ifndef DSC_REGS_H
#define DSC_REGS_H

#include <stdint.h>

constexpr uint32_t DSC_AP_CTRL        = 0x00;  // bit0 ap_start, bit1 ap_done (COR), bit2 ap_idle
constexpr uint32_t DSC_GIE            = 0x04;  // global interrupt enable
constexpr uint32_t DSC_IER            = 0x08;  // bit0 ap_done, bit1 ap_ready
constexpr uint32_t DSC_ISR            = 0x0c;  // same bits, toggle-on-write to clear
constexpr uint32_t DSC_IN_LO          = 0x10;  // in (64-bit physical address)
constexpr uint32_t DSC_IN_HI          = 0x14;
constexpr uint32_t DSC_PARAMS_LO      = 0x1c;  // params
constexpr uint32_t DSC_PARAMS_HI      = 0x20;
constexpr uint32_t DSC_OUT_LO         = 0x28;  // out
constexpr uint32_t DSC_OUT_HI         = 0x2c;
constexpr uint32_t DSC_STRIDE         = 0x34;
constexpr uint32_t DSC_SKIP_EN        = 0x3c;
constexpr uint32_t DSC_SCHED_LEN      = 0x44;
constexpr uint32_t DSC_SCHED_LEN_CTRL = 0x48;  // bit0 ap_vld (clear on read)
constexpr uint32_t DSC_REG_SPAN       = 0x10000;

constexpr uint32_t DSC_CTRL_START = 1u << 0;
constexpr uint32_t DSC_CTRL_DONE  = 1u << 1;
constexpr uint32_t DSC_CTRL_IDLE  = 1u << 2;

#if __has_include("xdsc_top_hw.h")
#include "xdsc_top_hw.h"
#define DSC_REGS_CONFIRMED 1
static_assert(DSC_AP_CTRL == XDSC_TOP_CONTROL_ADDR_AP_CTRL, "AP_CTRL offset");
static_assert(DSC_GIE == XDSC_TOP_CONTROL_ADDR_GIE, "GIE offset");
static_assert(DSC_IER == XDSC_TOP_CONTROL_ADDR_IER, "IER offset");
static_assert(DSC_ISR == XDSC_TOP_CONTROL_ADDR_ISR, "ISR offset");
static_assert(DSC_IN_LO == XDSC_TOP_CONTROL_ADDR_IN_R_DATA, "in offset");
static_assert(DSC_PARAMS_LO == XDSC_TOP_CONTROL_ADDR_PARAMS_DATA, "params offset");
static_assert(DSC_OUT_LO == XDSC_TOP_CONTROL_ADDR_OUT_R_DATA, "out offset");
static_assert(DSC_STRIDE == XDSC_TOP_CONTROL_ADDR_STRIDE_DATA, "stride offset");
static_assert(DSC_SKIP_EN == XDSC_TOP_CONTROL_ADDR_SKIP_EN_DATA, "skip_en offset");
static_assert(DSC_SCHED_LEN == XDSC_TOP_CONTROL_ADDR_SCHED_LEN_DATA, "sched_len offset");
static_assert(DSC_SCHED_LEN_CTRL == XDSC_TOP_CONTROL_ADDR_SCHED_LEN_CTRL, "sched_len ctrl offset");
#else
#define DSC_REGS_CONFIRMED 0
#endif

#endif // DSC_REGS_H
