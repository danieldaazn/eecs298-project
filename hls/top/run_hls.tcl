# Vitis HLS script for the fused top-level kernel dsc_top: C-sim + csynth (+ co-sim,
# + IP export).
#   vitis_hls -f run_hls.tcl                     # C-sim + csynth
#   vitis_hls -f run_hls.tcl -tclargs cosim      # + C/RTL co-simulation
#   vitis_hls -f run_hls.tcl -tclargs cosim export   # + export IP for Vivado IPI
#   vitis_hls -f run_hls.tcl -tclargs csim_only
# Same argument conventions as hls/compute/run_hls.tcl. Uses the real hls_stream.h,
# so ../memory/compat is deliberately NOT on the include path.
# NOT YET RUN: written without access to Vitis; expect to fix pragma warnings on
# the first csynth (see ../memory/ARCHITECTURE.md, "Unverified").

set top dsc_top
set part xck26-sfvc784-2LV-c   ;# KR260 / K26 SOM
set csim_only 0
set cosim 0
set export 0
foreach a $argv {
    if { [string match "xc*" $a] } { set part $a }
    if { $a eq "csim_only" } { set csim_only 1 }
    if { $a eq "cosim" } { set cosim 1 }
    if { $a eq "export" } { set export 1 }
}
puts "run_hls.tcl: top = $top, part = $part, csim_only = $csim_only, cosim = $cosim, export = $export"

open_project -reset proj_$top
set_top $top

set here [file normalize [file dirname [info script]]]
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
    if { $cosim } { cosim_design -rtl verilog }
    if { $export } { export_design -format ip_catalog -rtl verilog }
}
exit
