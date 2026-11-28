# build_bd.tcl — KR260 block design, bitstream and XSA for the dsc_top accelerator.
#
#   vivado -mode batch -source build_bd.tcl                 (Linux)
#   build_bd_win.bat                                        (Windows)
#   vivado -mode batch -source build_bd.tcl -tclargs bd     (block design only, no impl)
#
# Prerequisite: the HLS IP exported with OOC implementation, i.e.
#   hls/top/run_hls_win.bat cosim export impl     (or run_hls.tcl ... export impl)
# which leaves the IP catalog in hls/top/proj_dsc_top/solution1/impl/ip.
#
# Output, in board/firmware/out/:
#   dsc_accel.bit       bitstream
#   dsc_accel.bit.bin   the same, in the byte order the ZynqMP FPGA manager loads
#   dsc_accel.xsa       hardware platform (PS config + bitstream), for Vitis/PetaLinux
#   timing.rpt, utilization.rpt
#
# Block design (all on pl_clk0 = 200 MHz):
#
#   PS M_AXI_HPM0_FPD --SmartConnect--> dsc_top s_axi_control   @ 0xA000_0000, 64 KB
#   dsc_top m_axi_gmem_rd --SmartConnect--> PS S_AXI_HP0_FPD    (128-bit, DDR)
#   dsc_top m_axi_gmem_wr --SmartConnect--> PS S_AXI_HP1_FPD    (128-bit, DDR)
#   dsc_top interrupt --> PS pl_ps_irq0[0]                      (GIC SPI 89)
#
# Reads and writes go to different HP ports so the two bursts never queue behind each
# other in the same AFI FIFO.
#
# !!! PS CONFIGURATION — FOR KEVIN'S REVIEW !!!
# Kevin owns the Zynq PS configuration (CLAUDE.md, Week 9). This script applies the
# KR260 board preset and then changes only the six settings in ps_overrides below.
# Kria Ubuntu loads the PL at runtime and leaves the PS as booted, so for the
# xmutil flow these settings document the design rather than configure the chip: the
# clock rate and AFI widths are set by the device-tree overlay in board/firmware.
# They do matter if the XSA is used to build a boot image.
#
# Written without a local Vivado: the flow is checked with stubbed commands only.
# The first real run is on Daniel's Windows VM; see board/README.md.

set here [file dirname [file normalize [info script]]]
set root [file normalize [file join $here .. ..]]
cd $here

set bd_only [expr {[lsearch -exact $argv bd] >= 0}]
set part     xck26-sfvc784-2LV-c
set proj     dsc_kr260
set proj_dir [file join $here proj_kr260]
set ip_dir   [file join $root hls top proj_dsc_top solution1 impl ip]
set out_dir  [file join $root board firmware out]
set app      dsc_accel
set ctrl_base 0xA0000000

if {![file isdirectory $ip_dir]} {
    error "HLS IP not found at $ip_dir. Run hls/top/run_hls_win.bat cosim export impl first."
}
file mkdir $out_dir

# ---- project ---------------------------------------------------------------------------
create_project -force $proj $proj_dir -part $part

# The KR260 board files come from the Vivado board store (Tools > Vivado Store > Boards,
# "Kria KR260 Robotics Starter Kit"). Without them the design still builds on the bare
# part, but the PS has no DDR/MIO preset, so the XSA isn't bootable on its own (the
# bitstream is still fine for xmutil, which leaves the PS as booted).
set board_part [lindex [lsort [get_board_parts -quiet -latest_file_version {*kr260_som*}]] end]
if {$board_part ne ""} {
    set_property board_part $board_part [current_project]
    if {[catch {set_property board_connections {som240_1_connector xilinx.com:kr260_carrier:som240_1_connector:1.0} [current_project]} msg]} {
        puts "WARNING: build_bd.tcl: carrier connection not set ($msg); the PS preset is unaffected."
    }
    puts "build_bd.tcl: board part $board_part"
} else {
    puts "WARNING: build_bd.tcl: KR260 board files not installed; building on part $part only."
}

set_property ip_repo_paths [list $ip_dir] [current_project]
update_ip_catalog -rebuild

# Pick the newest installed version of an IP by name, so the script isn't tied to one
# Vivado release's version numbers.
proc ipdef {name} {
    set defs [lsort [get_ipdefs -filter "NAME == $name"]]
    if {[llength $defs] == 0} { error "IP $name not in the catalog" }
    return [lindex $defs end]
}

# ---- block design ----------------------------------------------------------------------
create_bd_design dsc_bd

set ps [create_bd_cell -type ip -vlnv [ipdef zynq_ultra_ps_e] zynq_ultra_ps_e_0]
if {$board_part ne ""} {
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
}
# For Kevin's review (see header). HPM0 stays at the preset's 128 bits: the SmartConnect
# converts to the 32-bit AXI-Lite slave, and the AFI width register stays at the value
# the KR260's boot firmware already sets (config-afi <14 0xa00> in the overlay).
set ps_overrides [list \
    CONFIG.PSU__USE__M_AXI_GP0             1 \
    CONFIG.PSU__USE__M_AXI_GP1             0 \
    CONFIG.PSU__USE__M_AXI_GP2             0 \
    CONFIG.PSU__USE__S_AXI_GP2             1 \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH        128 \
    CONFIG.PSU__USE__S_AXI_GP3             1 \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH        128 \
    CONFIG.PSU__USE__IRQ0                  1 \
    CONFIG.PSU__FPGA_PL0_ENABLE            1 \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ 200 \
]
set_property -dict $ps_overrides $ps

set acc [create_bd_cell -type ip -vlnv [ipdef dsc_top] dsc_top_0]
set rst [create_bd_cell -type ip -vlnv [ipdef proc_sys_reset] rst_pl0]
set sc_ctrl [create_bd_cell -type ip -vlnv [ipdef smartconnect] sc_ctrl]
set sc_rd   [create_bd_cell -type ip -vlnv [ipdef smartconnect] sc_rd]
set sc_wr   [create_bd_cell -type ip -vlnv [ipdef smartconnect] sc_wr]
foreach sc [list $sc_ctrl $sc_rd $sc_wr] {
    set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1] $sc
}

# Clock and reset: one domain.
set clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
connect_bd_net $clk \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp1_fpd_aclk] \
    [get_bd_pins dsc_top_0/ap_clk] \
    [get_bd_pins rst_pl0/slowest_sync_clk] \
    [get_bd_pins sc_ctrl/aclk] [get_bd_pins sc_rd/aclk] [get_bd_pins sc_wr/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_pl0/ext_reset_in]
connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] \
    [get_bd_pins dsc_top_0/ap_rst_n] \
    [get_bd_pins sc_ctrl/aresetn] [get_bd_pins sc_rd/aresetn] [get_bd_pins sc_wr/aresetn]

# AXI.
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M00_AXI] [get_bd_intf_pins dsc_top_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins dsc_top_0/m_axi_gmem_rd] [get_bd_intf_pins sc_rd/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_rd/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins dsc_top_0/m_axi_gmem_wr] [get_bd_intf_pins sc_wr/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_wr/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP1_FPD]

# Interrupt: ap_done/ap_ready via the IP's ISR, enabled through GIE/IER.
connect_bd_net [get_bd_pins dsc_top_0/interrupt] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

# Addresses: the control registers at a fixed base (the overlay and board/sw rely on
# it); both masters see all of DDR, where u-dma-buf allocates the buffers.
assign_bd_address -offset $ctrl_base -range 64K \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    [get_bd_addr_segs dsc_top_0/s_axi_control/Reg] -force
assign_bd_address

validate_bd_design
save_bd_design

set bd_file [get_files dsc_bd.bd]
make_wrapper -files $bd_file -top -import
set_property top dsc_bd_wrapper [current_fileset]
generate_target all $bd_file

if {$bd_only} {
    puts "build_bd.tcl: block design written ($proj_dir); stopping before implementation."
    return
}

# ---- implementation --------------------------------------------------------------------
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

set status [get_property STATUS [get_runs impl_1]]
if {![string match "*write_bitstream Complete*" $status]} {
    error "implementation did not finish: $status (see $proj_dir/$proj.runs/impl_1)"
}

open_run impl_1
report_timing_summary -file [file join $out_dir timing.rpt]
report_utilization    -file [file join $out_dir utilization.rpt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "build_bd.tcl: WNS = $wns ns at 200 MHz"
# Stop before anything lands where install.sh would pick it up.
if {$wns < 0} {
    error "timing NOT met: WNS = $wns ns; nothing copied to $out_dir. See timing.rpt and $proj_dir."
}

set bit_src [file join $proj_dir $proj.runs impl_1 dsc_bd_wrapper.bit]
set bit     [file join $out_dir $app.bit]
file copy -force $bit_src $bit
write_hw_platform -fixed -include_bit -force -file [file join $out_dir $app.xsa]

# .bit -> .bit.bin for the FPGA manager (same result as `bootgen -process_bitstream bin`;
# install.sh regenerates it with bootgen on the board if bootgen is installed there).
write_cfgmem -force -format bin -interface SMAPx32 -disablebitswap \
    -loadbit "up 0x0 $bit" [file join $out_dir $app.bit.bin]
# write_cfgmem names its output .bin and adds a .prm; normalize.
foreach f [glob -nocomplain [file join $out_dir $app.bit.prm]] { file delete $f }

puts "build_bd.tcl: done. Firmware in $out_dir; next: board/README.md, step 3."
