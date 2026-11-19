# Vitis HLS script for the baseline compute kernels.
#   vitis_hls -f run_hls.tcl                 # C-sim + csynth for depthwise_top
#   vitis_hls -f run_hls.tcl -tclargs pointwise_top
#
# Untested so far: written on a machine without Vitis. Run it on the lab/Linux box.
# Part is the KR260's K26 SOM.

# Pick the top from the arguments. Some Vitis versions (2024.2 on Windows) pass the
# whole command line in argv ("-f run_hls.tcl -tclargs pointwise_top"), so look for
# a known top name rather than trusting argv's first element.
# Any argument starting with "xc" overrides the device, e.g. a proxy part with the
# same UltraScale+ fabric when the K26 device files aren't installed. (A bare name,
# not "part=...": Windows cmd splits arguments on "=".) Estimates from a proxy are
# indicative only; final numbers must come from the K26 part.
# "csim_only" skips synthesis. C-sim results don't depend on the part, so this can
# run with any installed device when the K26 files are missing.
set top depthwise_top
set part xck26-sfvc784-2LV-c   ;# KR260 / K26 SOM
set csim_only 0
foreach a $argv {
    if { $a eq "depthwise_top" || $a eq "pointwise_top" } { set top $a }
    if { [string match "xc*" $a] } { set part $a }
    if { $a eq "csim_only" } { set csim_only 1 }
}
puts "run_hls.tcl: top = $top, part = $part, csim_only = $csim_only"

open_project -reset proj_$top
set_top $top

# Absolute include paths: C-sim compiles from inside the project directory, where
# relative -I paths would no longer resolve.
set here [file normalize [file dirname [info script]]]
set cflags "-std=c++14 -I$here/include -I$here/golden -DDSC_USE_AP_TYPES"
add_files src/requant.cpp   -cflags $cflags
add_files src/depthwise.cpp -cflags $cflags
add_files src/pointwise.cpp -cflags $cflags
add_files -tb golden/dsc_golden.cpp -cflags $cflags
add_files -tb tb/tb_dsc.cpp         -cflags $cflags

open_solution -reset solution1 -flow_target vivado
set_part $part
create_clock -period 5 -name default   ;# 200 MHz starting target

csim_design
if { !$csim_only } { csynth_design }
exit
