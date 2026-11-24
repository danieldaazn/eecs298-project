# win_cosim_tools.tcl — Daniel's Windows co-sim workaround (hls/compute/run_hls.tcl),
# as a proc so Hank's scripts can share it. Call just before cosim_design.
#
# Fixes "WARNING: [HLS 200-2011] Cannot find the 'zip' utility" on Windows: co-sim's
# dataflow monitor finds Vitis's old msys64 which.exe/zip.exe, which print a Cygwin
# warning to stderr that Tcl exec treats as failure. A project-local folder holding
# only a clean which.exe (Vivado's GnuWin32) and zip.exe (Vitis's standalone build)
# goes first on this process's PATH. Nothing in the Xilinx install is modified.
# Does NOT change the co-sim report's Status column, which reads "Fail" on Windows 11
# ARM even when co-sim passes: judge by "C/RTL co-simulation finished: PASS" in the log.
proc win_cosim_zip_fix {projdir} {
    if { $::tcl_platform(platform) ne "windows" } { return }
    if { ![info exists ::env(XILINX_VIVADO)] || ![info exists ::env(XILINX_VITIS)] } { return }
    set which_exe [file join $::env(XILINX_VIVADO) gnuwin bin which.exe]
    set zip_exe   [file join $::env(XILINX_VITIS) tps win64 zip bin zip.exe]
    if { ![file exists $which_exe] || ![file exists $zip_exe] } { return }
    set tooldir [file normalize [file join $projdir cosim_tools]]
    file mkdir $tooldir
    file copy -force $which_exe $zip_exe $tooldir
    set ::env(PATH) "[file nativename $tooldir];$::env(PATH)"
    puts "co-sim zip workaround: clean which/zip from $tooldir"
}
