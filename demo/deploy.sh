#!/bin/sh
# deploy.sh — copy the repo to the KR260, build there, (re)start the demo service.
#
#   demo/deploy.sh ubuntu@<board-ip>            from the repo root or anywhere
#
# Asks for the board password (ssh, then sudo). Copies the working tree (not the
# Vivado/Vitis project folders), builds libdsc.so and dsc_bringup natively on the board,
# installs board/firmware/out/ into /lib/firmware/xilinx/dsc_accel if it is present
# on the board, and enables dsc-demo.service, which starts at every boot.
# The model export (demo/model/export/, incl. test_images.npz) must exist locally:
# python3 demo/model/train.py --data <cifar dir>.
set -eu
HOST=${1:?usage: deploy.sh user@board}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=eecs298-project

[ -f "$ROOT/demo/model/export/summary.json" ] || { echo "no model export; run demo/model/train.py first"; exit 1; }

rsync -az --delete \
    --exclude '.git/' --exclude 'proj_*/' --exclude 'build/' --exclude '.Xil/' \
    --exclude '*.so' --exclude 'board/sw/dsc_bringup' --exclude 'board/sw/dsc_bringup_hostsim' \
    --exclude 'board/firmware/out/' \
    "$ROOT/" "$HOST:$DEST/"

ssh -t "$HOST" "set -e; cd $DEST
make -s -C demo/board && make -s -C board/sw
sed 's|REPO|'\$HOME/$DEST'|g' demo/board/dsc-demo.service | sudo tee /etc/systemd/system/dsc-demo.service >/dev/null
if [ -f board/firmware/out/dsc_accel.bit ] || [ -f board/firmware/out/dsc_accel.bit.bin ]; then
    sudo ./board/firmware/install.sh
else
    echo 'note: no board/firmware/out/ on the board yet: the demo starts in SIM mode'
fi
sudo systemctl daemon-reload
sudo systemctl enable dsc-demo.service >/dev/null
sudo systemctl restart dsc-demo.service
sleep 3; systemctl --no-pager --lines=5 status dsc-demo.service | tail -6"
echo "dashboard: http://${HOST#*@}:8080/"
