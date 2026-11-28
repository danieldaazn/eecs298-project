@echo off
rem Run run_hls.tcl (dsc_top) under Vitis HLS 2024.2 on Windows.
rem   run_hls_win.bat                       -> C-sim + csynth
rem   run_hls_win.bat cosim export          -> + co-sim + IP export (for board\vivado)
rem   run_hls_win.bat export impl           -> + IP export with out-of-context P&R
rem Output goes to dsc_top_run.log next to this script. Same PATH fix as
rem ..\compute\run_hls_win.bat (tee.exe lives under Vitis\, not Vitis_HLS\).

setlocal
if "%XILINX_ROOT%"=="" set XILINX_ROOT=C:\Xilinx
cd /d "%~dp0"
set PATH=%XILINX_ROOT%\Vitis\2024.2\tps\win64\msys64\usr\bin;%PATH%
call "%XILINX_ROOT%\Vitis_HLS\2024.2\settings64.bat"
call vitis_hls -f run_hls.tcl -tclargs %1 %2 %3 %4 > dsc_top_run.log 2>&1
echo vitis_hls exit code %ERRORLEVEL% >> dsc_top_run.log
endlocal
