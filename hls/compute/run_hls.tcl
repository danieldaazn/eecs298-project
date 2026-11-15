# Vitis HLS script for the baseline compute kernels.
#   vitis_hls -f run_hls.tcl                 # C-sim + csynth for depthwise_top
#   vitis_hls -f run_hls.tcl -tclargs pointwise_top
#
# Untested so far: written on a machine without Vitis. Run it on the lab/Linux box.
# Part is the KR260's K26 SOM.

set top depthwise_top
if { $argc > 0 } { set top [lindex $argv 0] }

open_project -reset proj_$top
set_top $top

set cflags "-std=c++14 -Iinclude -Igolden -DDSC_USE_AP_TYPES"
add_files src/requant.cpp   -cflags $cflags
add_files src/depthwise.cpp -cflags $cflags
add_files src/pointwise.cpp -cflags $cflags
add_files -tb golden/dsc_golden.cpp -cflags $cflags
add_files -tb tb/tb_dsc.cpp         -cflags $cflags

open_solution -reset solution1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period 5 -name default   ;# 200 MHz starting target

csim_design
csynth_design
exit
