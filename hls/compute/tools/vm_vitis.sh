#!/bin/bash
# vm_vitis.sh — run this folder's Vitis HLS flow inside a Parallels Windows VM, from macOS.
#
#   tools/vm_vitis.sh <top> [csim_only|cosim|impl|<part>] ...
#   e.g. tools/vm_vitis.sh pointwise_skip_top cosim
#        tools/vm_vitis.sh pointwise_skip_top impl
#
# Copies include/ golden/ src/ tb/ and the run scripts to the VM, runs run_hls_win.bat
# there, then prints the key log lines, the csynth summary, co-sim per-call cycles and
# the post-route implementation report (whichever exist).
#
# Specific to one setup (Daniel's Mac): adjust the settings below for yours. Needs
# Parallels' prlctl, the VM running, and the Mac home folder shared into the VM.

VM="${VM:-Windows 11}"
SRC='\\Mac\Home\Documents\eecs298-project\hls\compute'   # this folder, as the VM sees it
DST='C:\work\compute'                                     # working copy inside the VM

TOP="$1"; shift
[ -z "$TOP" ] && { echo "usage: $0 <top> [args...]"; exit 2; }
ARGS="$*"

ps() { prlctl exec "$VM" powershell -NoProfile -Command "$1" 2>&1; }

# ---- sync (one PowerShell call, retried: prlctl exec fails intermittently) ----------
SYNC="New-Item -ItemType Directory -Force $DST | Out-Null; \
foreach (\$d in 'include','golden','src','tb') { Copy-Item -Recurse -Force $SRC\\\$d $DST\\ }; \
Copy-Item -Force $SRC\\run_hls.tcl,$SRC\\run_hls_win.bat $DST\\; \
if (Test-Path $DST\\proj_$TOP) { Remove-Item -Recurse -Force $DST\\proj_$TOP }; 'SYNC-OK'"
ok=0
for try in 1 2 3; do
  if ps "$SYNC" | grep -q SYNC-OK; then ok=1; break; fi
  sleep 3
done
[ $ok = 1 ] || { echo "vm_vitis: ERROR: could not sync files into the VM"; exit 1; }

# ---- run ----------------------------------------------------------------------------------
prlctl exec "$VM" cmd /c "$DST\\run_hls_win.bat $TOP $ARGS" 2>&1 | tail -3
LOG="$DST\\${TOP}_run.log"
echo "== log highlights ($TOP $ARGS)"
ps "Get-Content $LOG | Select-String -Pattern 'run_hls.tcl:|FAIL|ERROR|CRITICAL|II Violation|Pipelining result|Estimated Fmax|COSIM 212-1000|COSIM 212-4\]|Finished Command|exit code' | ForEach-Object { \$_.Line.Substring(0,[Math]::Min(220,\$_.Line.Length)) }"

echo "== csynth summary"
ps "\$r = Get-ChildItem $DST\\proj_$TOP -Recurse -Filter '${TOP}_csynth.rpt' -ErrorAction SilentlyContinue | Select -First 1; if (\$r) { \$c = Get-Content \$r.FullName; \$i = (\$c | Select-String -Pattern '== Performance Estimates' | Select -First 1).LineNumber; \$c[(\$i)..(\$i+12)]; \$j = (\$c | Select-String -Pattern '== Utilization Estimates' | Select -First 1).LineNumber; \$c[(\$j+3)..(\$j+16)] } else { '(none)' }" | grep -v '^\s*$' | grep -v -- '----'

echo "== co-sim cycles per call"
ps "\$r = '$DST\\proj_$TOP\\solution1\\sim\\report\\verilog\\result.transaction.rpt'; if (Test-Path \$r) { Get-Content \$r } else { '(none)' }"

echo "== post-route implementation report"
ps "\$r = Get-ChildItem $DST\\proj_$TOP\\solution1\\impl -Recurse -Filter '*_export.rpt' -ErrorAction SilentlyContinue | Select -First 1; if (\$r) { Get-Content \$r.FullName } else { '(none)' }"
