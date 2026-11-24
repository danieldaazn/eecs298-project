# Vitis HLS script for the memory-path pass-through top: C-sim + csynth (+ co-sim).
#   vitis_hls -f run_hls.tcl                     # C-sim + csynth
#   vitis_hls -f run_hls.tcl -tclargs cosim      # + C/RTL co-simulation
#   vitis_hls -f run_hls.tcl -tclargs csim_only
#   vitis_hls -f run_hls.tcl -tclargs xczu5ev-sfvc784-2-e   # other part
# Same argument conventions as hls/compute/run_hls.tcl. Uses the real hls_stream.h,
# so compat/ is deliberately NOT on the include path. Runs from anywhere.
# Run by Daniel on Vitis HLS 2024.2 / K26 at ecf8a64: C-sim, csynth, co-sim PASS.

set top stream_passthrough_top
set part xck26-sfvc784-2LV-c   ;# KR260 / K26 SOM
set csim_only 0
set cosim 0
foreach a $argv {
    if { [string match "xc*" $a] } { set part $a }
    if { $a eq "csim_only" } { set csim_only 1 }
    if { $a eq "cosim" } { set cosim 1 }
}
puts "run_hls.tcl: top = $top, part = $part, csim_only = $csim_only, cosim = $cosim"

set here [file normalize [file dirname [info script]]]
cd $here
source $here/win_cosim_tools.tcl

open_project -reset proj_$top
set_top $top

set cflags "-std=c++14 -I$here/include -I$here/../compute/include -DDSC_USE_AP_TYPES"
add_files $here/src/mem_stages.cpp      -cflags $cflags
add_files $here/src/passthrough_top.cpp -cflags $cflags
add_files -tb $here/tb/tb_mem.cpp       -cflags $cflags

open_solution -reset solution1 -flow_target vivado
set_part $part
create_clock -period 5 -name default   ;# 200 MHz, same target as the compute core

csim_design
if { !$csim_only } {
    csynth_design
    if { $cosim } { win_cosim_zip_fix proj_$top; cosim_design -rtl verilog }
}
exit
