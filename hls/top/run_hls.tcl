# Vitis HLS script for the fused top-level kernel dsc_top: C-sim + csynth (+ co-sim,
# + IP export).
#   vitis_hls -f run_hls.tcl                     # C-sim + csynth
#   vitis_hls -f run_hls.tcl -tclargs cosim      # + C/RTL co-simulation
#   vitis_hls -f run_hls.tcl -tclargs cosim export   # + export IP for Vivado IPI
#   vitis_hls -f run_hls.tcl -tclargs export impl    # + out-of-context place & route
#   vitis_hls -f run_hls.tcl -tclargs csim_only
# On Windows: run_hls_win.bat [args]. Runs from anywhere (cd's to this directory).
# Same argument conventions as hls/compute/run_hls.tcl. Uses the real hls_stream.h,
# so ../memory/compat is deliberately NOT on the include path.
# "export" writes the IP that board/vivado/build_bd.tcl picks up:
#   proj_dsc_top/solution1/impl/ip (and xilinx_com_hls_dsc_top_1_0.zip).
# "impl" runs Vivado synthesis + P&R on the IP, out of context, for real timing:
#   proj_dsc_top/solution1/impl/report/verilog/dsc_top_export.rpt.
# Run by Daniel on Vitis HLS 2024.2 / K26 at ecf8a64: C-sim, csynth, co-sim, export PASS.

set top dsc_top
set part xck26-sfvc784-2LV-c   ;# KR260 / K26 SOM
set csim_only 0
set cosim 0
set export 0
set impl 0
foreach a $argv {
    if { [string match "xc*" $a] } { set part $a }
    if { $a eq "csim_only" } { set csim_only 1 }
    if { $a eq "cosim" } { set cosim 1 }
    if { $a eq "export" } { set export 1 }
    if { $a eq "impl" } { set impl 1 }
}
puts "run_hls.tcl: top = $top, part = $part, csim_only = $csim_only, cosim = $cosim, export = $export, impl = $impl"

set here [file normalize [file dirname [info script]]]
cd $here
source $here/../memory/win_cosim_tools.tcl

open_project -reset proj_$top
set_top $top

set compute $here/../compute
set memory  $here/../memory
set cflags "-std=c++14 -I$here/include -I$memory/include -I$compute/include -I$compute/golden -DDSC_USE_AP_TYPES"
add_files $here/src/dsc_top.cpp           -cflags $cflags
add_files $memory/src/mem_stages.cpp      -cflags $cflags
add_files $compute/src/requant.cpp        -cflags $cflags
add_files $compute/src/depthwise.cpp      -cflags $cflags
add_files $compute/src/pointwise.cpp      -cflags $cflags
add_files $compute/src/pointwise_skip.cpp -cflags $cflags
add_files -tb $compute/golden/dsc_golden.cpp -cflags $cflags
add_files -tb $here/tb/tb_dsc_top.cpp        -cflags $cflags

open_solution -reset solution1 -flow_target vivado
set_part $part
create_clock -period 5 -name default   ;# 200 MHz

csim_design
if { !$csim_only } {
    csynth_design
    if { $cosim } { win_cosim_zip_fix proj_$top; cosim_design -rtl verilog }
    if { $impl } {
        export_design -flow impl -format ip_catalog -rtl verilog
    } elseif { $export } {
        export_design -format ip_catalog -rtl verilog
    }
}
exit
