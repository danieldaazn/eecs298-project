@echo off
rem Run run_hls.tcl under Vitis HLS 2024.2 on Windows.
rem   run_hls_win.bat                  -> depthwise_top
rem   run_hls_win.bat pointwise_top
rem Output goes to <top>_run.log next to this script.
rem
rem Why the PATH line: vitis_hls.bat pipes its log through tee.exe, but on a
rem standard 2024.2 install tee.exe lives under Vitis\, not Vitis_HLS\, so the
rem launcher can't find it. Override XILINX_ROOT if installed somewhere else.

setlocal
if "%XILINX_ROOT%"=="" set XILINX_ROOT=C:\Xilinx
rem Optional extra arguments, passed through to run_hls.tcl:
rem   a device name (anything starting with xc), and/or csim_only
rem   e.g. run_hls_win.bat depthwise_top xc7a35tcpg236-1 csim_only
set TOP=%1
if "%TOP%"=="" set TOP=depthwise_top
set PARTARG=%2 %3

cd /d "%~dp0"
set PATH=%XILINX_ROOT%\Vitis\2024.2\tps\win64\msys64\usr\bin;%PATH%
call "%XILINX_ROOT%\Vitis_HLS\2024.2\settings64.bat"
call vitis_hls -f run_hls.tcl -tclargs %TOP% %PARTARG% > %TOP%_run.log 2>&1
echo vitis_hls exit code %ERRORLEVEL% >> %TOP%_run.log
endlocal
