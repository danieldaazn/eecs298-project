@echo off
rem Build the KR260 block design, bitstream and XSA with Vivado 2024.2 on Windows.
rem   build_bd_win.bat         -> full build (needs the HLS IP: ..\..\hls\top\run_hls_win.bat cosim export impl)
rem   build_bd_win.bat bd      -> block design only, stops before implementation
rem Output goes to ..\firmware\out\ and the log to build_bd.log next to this script.

setlocal
if "%XILINX_ROOT%"=="" set XILINX_ROOT=C:\Xilinx
cd /d "%~dp0"
call "%XILINX_ROOT%\Vivado\2024.2\settings64.bat"
call vivado -mode batch -nojournal -log build_bd.log -source build_bd.tcl -tclargs %1
echo vivado exit code %ERRORLEVEL% >> build_bd.log
endlocal
