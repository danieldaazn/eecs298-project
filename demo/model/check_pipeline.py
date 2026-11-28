#!/usr/bin/env python3
"""Check the board's C pipeline (demo/board/libdsc.so, SIM mode on a PC) against an
independent numpy float64 reference of the same network, and report its accuracy.

    python3 check_pipeline.py [--export export] [--n 10000]

Per level: byte mismatches of the FPGA blocks' inputs (C float32 vs numpy float64:
only values sitting exactly on a rounding boundary may differ), prediction agreement,
and the C pipeline's test accuracy, which is what the board should measure (the FPGA
blocks are bit-exact with the golden model, and the C layers round identically on
x86 and arm64 with -ffp-contract=off).
"""
import argparse
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'board'))
sys.path.insert(0, HERE)
import server                       # noqa: E402  (libdsc bindings + Level loader)
from train import int_block         # noqa: E402  (independent integer dsc_top)


def conv_ref(x, w, b, stride):
    """float64 conv (pad = k//2), NCHW single image."""
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)))
    ho = (x.shape[1] + 2 * (k // 2) - k) // stride + 1
    out = np.zeros((w.shape[0], ho, ho)) + b[:, None, None]
    for ky in range(k):
        for kx in range(k):
            win = xp[:, ky:ky + stride * (ho - 1) + 1:stride, kx:kx + stride * (ho - 1) + 1:stride]
            out += np.einsum('kc,cyx->kyx', w[:, :, ky, kx], win)
    return out


def q(x):
    return np.clip(np.floor(x * 32 + 0.5), 0, 192).astype(np.uint8)


def ref_pipeline(L, P, img):
    x = (img.astype(np.float64).transpose(2, 0, 1) / 255 - L.mean[:, None, None]) / L.std[:, None, None]
    s1 = np.clip(conv_ref(x, L.w1.astype(np.float64), L.b1, 1), 0, 6)
    a1 = q(conv_ref(s1, L.w2.astype(np.float64), L.b2, 2))
    o1 = int_block(a1, P['b1'], 1)[1]
    a2 = q(conv_ref(o1 / 32.0, L.wp.astype(np.float64).reshape(32, 64, 1, 1), L.bp, 1))
    o2 = int_block(a2, P['b2'], 2)[1]
    logits = L.fc_w @ (o2.reshape(64, -1).mean(axis=1) / 32.0) + L.fc_b
    return a1, a2, int(logits.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default=os.path.join(HERE, 'export'))
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--n-ref', type=int, default=500, help='images also run through the float64 reference')
    args = ap.parse_args()

    if server.lib.dsc_open(0) != 0:
        raise SystemExit('expected SIM mode (run this on a PC, not the board)')
    with open(os.path.join(args.export, 'summary.json')) as f:
        summary = json.load(f)
    t = np.load(os.path.join(args.export, 'test_images.npz'))
    X, Y = np.ascontiguousarray(t['x']), t['y']
    n = min(args.n, len(Y))
    for i, info in enumerate(summary['levels']):
        d = os.path.join(args.export, f's{info["level"]}')
        L = server.Level(d, info, i)
        P = {}
        for blk in ('b1', 'b2'):
            bd = os.path.join(d, blk)
            P[blk] = dict(dw_wt=np.fromfile(f'{bd}/dw_wt.bin', np.int8).reshape(32, 3, 3),
                          dw_b=np.fromfile(f'{bd}/dw_b.bin', '<i4'),
                          pw_wt=np.fromfile(f'{bd}/pw_wt.bin', np.int8).reshape(64, 32),
                          pw_b=np.fromfile(f'{bd}/pw_b.bin', '<i4'))
        correct = agree = mis_a1 = mis_a2 = 0
        for k in range(n):
            img = X[k]
            a1 = np.empty((32, 16, 16), np.uint8)
            server.lib.dsc_arm_stem(img, L.mean, L.std, L.w1, L.b1, L.w2, L.b2, a1)
            o1 = np.empty((64, 16, 16), np.uint8)
            us, ln = ctypes.c_double(), ctypes.c_int()
            server.lib.dsc_run(L.slot['b1'], a1, 1, 1, o1, ctypes.byref(us), ctypes.byref(ln))
            a2 = np.empty((32, 16, 16), np.uint8)
            server.lib.dsc_arm_proj(o1, L.wp, L.bp, a2)
            o2 = np.empty((64, 8, 8), np.uint8)
            server.lib.dsc_run(L.slot['b2'], a2, 2, 1, o2, ctypes.byref(us), ctypes.byref(ln))
            pred = int((L.fc_w @ (o2.reshape(64, -1).mean(axis=1) / 32.0) + L.fc_b).argmax())
            correct += pred == Y[k]
            if k < args.n_ref:
                r1, r2, rp = ref_pipeline(L, P, img)
                mis_a1 += int((r1 != a1).sum())
                mis_a2 += int((r2 != a2).sum())
                agree += rp == pred
        m = min(n, args.n_ref)
        print(f'level {info["level"]:3d}%: C pipeline acc {correct / n:.4f} (n={n}), training {info["gpu_test_acc"]:.4f} | '
              f'vs float64 ref on {m}: block-1 input bytes differ {mis_a1}/{m * 8192}, '
              f'block-2 input {mis_a2}/{m * 8192}, predictions agree {agree}/{m}', flush=True)


if __name__ == '__main__':
    main()
