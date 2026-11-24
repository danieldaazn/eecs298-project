# Vitis HLS script for the compute kernels: C-sim + csynth (+ optional co-sim).
#   vitis_hls -f run_hls.tcl                                  # depthwise_top
#   vitis_hls -f run_hls.tcl -tclargs pointwise_top
#   vitis_hls -f run_hls.tcl -tclargs pointwise_skip_top cosim
# On Windows use run_hls_win.bat. Tested with Vitis HLS 2024.2 on the K26 part.
#
# "cosim" adds C/RTL co-simulation after synthesis: the same testbench drives the
# generated RTL for the selected top (other tops still run as C). Slow.

# Pick the top from the arguments. Some Vitis versions (2024.2 on Windows) pass the
# whole command line in argv ("-f run_hls.tcl -tclargs pointwise_top"), so look for
# a known top name rather than trusting argv's first element.
# Any argument starting with "xc" overrides the device, e.g. a proxy part with the
# same UltraScale+ fabric when the K26 device files aren't installed. (A bare name,
# not "part=...": Windows cmd splits arguments on "=".) Estimates from a proxy are
# indicative only; final numbers must come from the K26 part.
# "csim_only" skips synthesis. C-sim results don't depend on the part, so this can
# run with any installed device when the K26 files are missing.
# "impl" runs Vivado synthesis + place & route on the exported IP (out of context) for
# real post-route timing and resources: solution1/impl/report/verilog/<top>_export.rpt.
set top depthwise_top
set part xck26-sfvc784-2LV-c   ;# KR260 / K26 SOM
set csim_only 0
set cosim 0
set impl 0
foreach a $argv {
    if { $a in {depthwise_top pointwise_top pointwise_skip_top pointwise_skip_pipe_top} } { set top $a }
    if { [string match "xc*" $a] } { set part $a }
    if { $a eq "csim_only" } { set csim_only 1 }
    if { $a eq "cosim" } { set cosim 1 }
    if { $a eq "impl" } { set impl 1 }
}
puts "run_hls.tcl: top = $top, part = $part, csim_only = $csim_only, cosim = $cosim, impl = $impl"

# Run from this script's directory, so it works from anywhere (e.g.
# vitis_hls -f hls/compute/run_hls.tcl from the repo root); the project folder
# proj_<top> is created next to this script.
set here [file normalize [file dirname [info script]]]
cd $here

open_project -reset proj_$top
set_top $top

# Absolute include paths: C-sim compiles from inside the project directory, where
# relative -I paths would no longer resolve.
set cflags "-std=c++14 -I$here/include -I$here/golden -DDSC_USE_AP_TYPES"
add_files src/requant.cpp   -cflags $cflags
add_files src/depthwise.cpp -cflags $cflags
add_files src/pointwise.cpp -cflags $cflags
add_files src/pointwise_skip.cpp -cflags $cflags
add_files src/pointwise_skip_pipe.cpp -cflags $cflags
add_files -tb golden/dsc_golden.cpp -cflags $cflags
add_files -tb tb/tb_dsc.cpp         -cflags $cflags

open_solution -reset solution1 -flow_target vivado
set_part $part
create_clock -period 5 -name default   ;# 200 MHz starting target

csim_design
if { !$csim_only } {
    csynth_design
    if { $cosim } {
        # Windows workaround for "WARNING: [HLS 200-2011] Cannot find the 'zip'
        # utility" (without it, co-sim skips saving its process/loop stall records):
        #   Co-sim's dataflow monitor checks for zip with `exec which zip`, then runs
        #   `exec zip ...`. The which.exe and zip.exe it finds first are Vitis's msys64
        #   (old Cygwin) builds, which on newer Windows print "Couldn't compute
        #   FAST_CWD pointer" to stderr; Tcl exec treats any stderr as failure.
        #   Co-sim's scripts run inside this same Vitis process, so a folder put first
        #   on this process's PATH wins. It must hold ONLY clean which.exe (Vivado's
        #   GnuWin32 build) and zip.exe (Vitis's standalone build): putting all of
        #   gnuwin\bin first also swaps make/sh and breaks the co-sim C++ build.
        #   The folder lives in the project dir (not sim/, which co-sim rebuilds).
        #   Nothing in the Xilinx install is modified.
        # NOTE: this does NOT change the co-sim report's Status column. On Vitis
        # 2024.2 under Windows 11 ARM, that column reads "Fail" even for a trivial
        # one-line adder whose co-sim passes. Judge co-sim by the log's
        # "C/RTL co-simulation finished: PASS" line and the post-check output.
        if { $::tcl_platform(platform) eq "windows" &&
             [info exists ::env(XILINX_VIVADO)] && [info exists ::env(XILINX_VITIS)] } {
            set which_exe [file join $::env(XILINX_VIVADO) gnuwin bin which.exe]
            set zip_exe   [file join $::env(XILINX_VITIS) tps win64 zip bin zip.exe]
            if { [file exists $which_exe] && [file exists $zip_exe] } {
                set tooldir [file normalize [file join proj_$top cosim_tools]]
                file mkdir $tooldir
                file copy -force $which_exe $zip_exe $tooldir
                set ::env(PATH) "[file nativename $tooldir];$::env(PATH)"
                puts "run_hls.tcl: co-sim zip workaround: clean which/zip from $tooldir"
            }
        }
        cosim_design -rtl verilog
    }
    if { $impl } { export_design -flow impl -rtl verilog -format ip_catalog }
}
exit
