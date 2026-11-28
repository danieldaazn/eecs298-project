#!/bin/sh
# install.sh — run ON THE KR260 (Kria Ubuntu 22.04) to install and load the accelerator.
#
#   sudo ./install.sh            install firmware from ./out and load it
#   sudo ./install.sh --reload   just unload whatever app is loaded and load dsc_accel
#
# Needs: out/dsc_accel.bit (and/or out/dsc_accel.bit.bin) from board/vivado/build_bd.tcl,
# plus device-tree-compiler (apt install device-tree-compiler). If bootgen is installed
# (apt install bootgen-xlnx), the .bit.bin is regenerated from the .bit with it, which
# is AMD's documented route; otherwise the .bit.bin Vivado wrote is used as-is.
set -eu

APP=dsc_accel
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/out"
FW=${DSC_FW_DIR:-/lib/firmware/xilinx/$APP}   # override only for testing

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo $0)"; exit 1; }

if [ "${1:-}" != "--reload" ]; then
    command -v dtc >/dev/null || { echo "dtc missing: apt install device-tree-compiler"; exit 1; }
    BUILD=$(mktemp -d)
    trap 'rm -rf "$BUILD"' EXIT

    if command -v bootgen >/dev/null && [ -f "$OUT/$APP.bit" ]; then
        cp "$OUT/$APP.bit" "$BUILD/"
        cp "$HERE/$APP.bif" "$BUILD/"
        (cd "$BUILD" && bootgen -arch zynqmp -image $APP.bif -process_bitstream bin -w on >/dev/null)
        echo "bit.bin: generated with bootgen"
    elif [ -f "$OUT/$APP.bit.bin" ]; then
        cp "$OUT/$APP.bit.bin" "$BUILD/"
        echo "bit.bin: using Vivado's write_cfgmem output (bootgen not installed)"
    else
        echo "no $OUT/$APP.bit or $OUT/$APP.bit.bin; copy board/firmware/out from the Vivado build"
        exit 1
    fi

    dtc -q -@ -I dts -O dtb -o "$BUILD/$APP.dtbo" "$HERE/$APP.dts"

    mkdir -p "$FW"
    cp "$BUILD/$APP.bit.bin" "$BUILD/$APP.dtbo" "$HERE/shell.json" "$FW/"
    echo "installed to $FW:"
    ls -l "$FW"
fi

# generic-uio binding for the accelerator node (harmless if already loaded). Without it
# board/sw falls back to /dev/mem and polling.
if ! grep -q "of_id=generic-uio" /proc/cmdline; then
    modprobe -r uio_pdrv_genirq 2>/dev/null || true
    modprobe uio_pdrv_genirq of_id=generic-uio 2>/dev/null || echo "note: uio_pdrv_genirq not available"
fi

# Only one app can be loaded; unloadapp fails harmlessly if none is. At boot, dfx-mgr may
# still be loading its default app, so retry until our accelerator's UIO node exists.
has_uio() {
    for d in /sys/class/uio/uio*; do
        [ -e "$d/name" ] && [ "$(cat "$d/name")" = "dsc_top" ] && return 0
    done
    return 1
}
tries=0
until has_uio; do
    tries=$((tries + 1))
    if [ $tries -gt 5 ]; then
        echo "dsc_accel did not load after 5 attempts (see dmesg, xmutil listapps)"
        exit 1
    fi
    xmutil unloadapp >/dev/null 2>&1 || true
    xmutil loadapp $APP || true
    has_uio || sleep 2
done
xmutil listapps

for d in /sys/class/uio/uio*; do
    [ -e "$d/name" ] || continue
    if [ "$(cat "$d/name")" = "dsc_top" ]; then echo "UIO: $(basename "$d") -> dsc_top"; fi
done
[ -e /dev/udmabuf0 ] || echo "note: /dev/udmabuf0 missing; see board/README.md (u-dma-buf)"
