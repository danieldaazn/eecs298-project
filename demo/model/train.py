#!/usr/bin/env python3
"""Train, block-prune and export the demo CNN (CIFAR-10) whose two DSC blocks run on the FPGA.

    python3 train.py --data DIR [--epochs 40] [--ft-epochs 15] [--out export]

Network (32x32 RGB in):
    ARM   stem1  conv3x3  3->16, stride 1, BN, ReLU6                      32x32x16
    ARM   stem2  conv3x3 16->32, stride 2, BN, ReLU6, quantize to Q3.5     16x16x32
    FPGA  b1     dsc_top stride 1: DW3x3 + ReLU6, PW 32->64 + ReLU6        16x16x64
    ARM   proj   conv1x1 64->32, BN, ReLU6, quantize to Q3.5              16x16x32
    FPGA  b2     dsc_top stride 2                                          8x8x64
    ARM   head   global average pool, linear 64->10

The FPGA blocks are trained with the hardware's exact arithmetic (hls/compute/include/
dsc_types.h): Q1.7 weights, Q3.5 activations, int32 biases with 12 fractional bits,
round-half-up requantization with ReLU6 clamped at 192. In that number format every
partial sum is a multiple of 2^-12 below 2^8, so a float32/float64 forward pass is exact
and the fake-quantized model *is* the integer hardware. check_export() confirms it with
an independent integer implementation before anything is written.

Pruning is the hardware's skip granularity: whole 8-input-channel blocks of a pointwise
output channel (PW_UNROLL = 8), chosen by L2 norm per layer, then fine-tuned with the
mask fixed. Depthwise weights are not pruned (skipping is pointwise only).

Output, per sparsity level p: OUT/s<p>/{b1,b2}/ in VECTORS.md format (one real
activation tile as dw_in.bin, plus expected outputs), OUT/s<p>/arm.npz (ARM layers,
BN folded), and OUT/summary.json. OUT/test_images.npz holds the CIFAR-10 test set.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ACT_ONE, WGT_ONE, ACC_ONE = 32, 128, 4096     # 2^ACT_FRAC, 2^WGT_FRAC, 2^ACC_FRAC
ACT_MAX = 192                                 # ReLU6 in Q3.5
C_IN, C_OUT, UNROLL = 32, 64, 8
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def ste(x, q):
    """Forward q(x), backward identity."""
    return x + (q - x).detach()


def q_act(x):     # -> real value of a Q3.5 activation after ReLU6, half-up
    # The clamp (ReLU6) keeps its real gradient; only the rounding is straight-through.
    y = torch.clamp(x, 0, ACT_MAX / ACT_ONE)
    return ste(y, torch.floor(y * ACT_ONE + 0.5) / ACT_ONE)


def q_wgt(w):
    return ste(w, torch.clamp(torch.round(w * WGT_ONE), -128, 127) / WGT_ONE)


def q_bias(b):
    return ste(b, torch.round(b * ACC_ONE) / ACC_ONE)


class DSCBlock(nn.Module):
    """One dsc_top call: DW 3x3 (pad 1) + requant/ReLU6, PW 1x1 + requant/ReLU6."""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.dw_w = nn.Parameter(torch.randn(C_IN, 1, 3, 3) * 0.25)
        self.dw_b = nn.Parameter(torch.zeros(C_IN))
        self.pw_w = nn.Parameter(torch.randn(C_OUT, C_IN, 1, 1) * (1.5 / C_IN ** 0.5))
        self.pw_b = nn.Parameter(torch.zeros(C_OUT))
        self.register_buffer('mask', torch.ones(C_OUT, C_IN // UNROLL))

    def pw_mask(self):
        return self.mask.repeat_interleave(UNROLL, dim=1).view(C_OUT, C_IN, 1, 1)

    def forward(self, x):
        y = F.conv2d(x, q_wgt(self.dw_w), q_bias(self.dw_b), self.stride, 1, groups=C_IN)
        y = q_act(y)
        z = F.conv2d(y, q_wgt(self.pw_w) * self.pw_mask(), q_bias(self.pw_b))
        return q_act(z)

    def prune(self, frac):
        """Zero the `frac` of (output channel, 8-channel block) pairs with the smallest norm."""
        with torch.no_grad():
            w = (q_wgt(self.pw_w) * self.pw_mask()).view(C_OUT, C_IN // UNROLL, UNROLL)
            norms = w.norm(dim=2).flatten()
            k = int(round(frac * norms.numel()))
            mask = torch.ones_like(norms)
            if k > 0:
                mask[torch.argsort(norms)[:k]] = 0
            self.mask.copy_(mask.view(C_OUT, -1))

    def export(self):
        """Integer tensors exactly as the hardware consumes them."""
        with torch.no_grad():
            dw = torch.clamp(torch.round(self.dw_w * WGT_ONE), -128, 127).view(C_IN, 3, 3)
            pw = torch.clamp(torch.round(self.pw_w * WGT_ONE), -128, 127) * self.pw_mask()
            return dict(dw_wt=dw.cpu().numpy().astype(np.int8),
                        dw_b=torch.round(self.dw_b * ACC_ONE).cpu().numpy().astype(np.int32),
                        pw_wt=pw.view(C_OUT, C_IN).cpu().numpy().astype(np.int8),
                        pw_b=torch.round(self.pw_b * ACC_ONE).cpu().numpy().astype(np.int32))


def conv_bn(cin, cout, k, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = conv_bn(3, 16, 3, 1)
        self.stem2 = conv_bn(16, C_IN, 3, 2)
        self.b1 = DSCBlock(1)
        self.proj = conv_bn(C_OUT, C_IN, 1, 1)
        self.b2 = DSCBlock(2)
        self.fc = nn.Linear(C_OUT, 10)

    def forward(self, x):          # x: normalized float images
        x = F.relu6(self.stem1(x))
        x = q_act(self.stem2(x))
        x = self.b1(x)
        x = q_act(self.proj(x))
        x = self.b2(x)
        return self.fc(x.mean(dim=(2, 3)))

    def sched_len(self):
        out = {}
        for name, b in (('b1', self.b1), ('b2', self.b2)):
            nz = b.export()['pw_wt'].reshape(C_OUT, C_IN // UNROLL, UNROLL).any(axis=2).sum(axis=1)
            out[name] = int(np.maximum(nz, 1).sum())
        return out


# ---- data (whole dataset on the GPU; augmentation on the GPU) --------------------------
def load(data_dir, dev):
    def arr(train):
        ds = torchvision.datasets.CIFAR10(data_dir, train=train, download=False)
        return torch.tensor(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets)
    (xtr, ytr), (xte, yte) = arr(True), arr(False)
    return xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)


def normalize(x_u8):
    return (x_u8.float() / 255 - MEAN.to(x_u8.device)) / STD.to(x_u8.device)


def augment(x):                    # random crop (pad 4) + horizontal flip, uint8 NCHW
    n = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (n,), device=x.device)
    j = torch.randint(0, 9, (n,), device=x.device)
    rows = (i.view(n, 1) + torch.arange(32, device=x.device)).view(n, 1, 32, 1).expand(n, 3, 32, 40)
    xp = torch.gather(xp, 2, rows)
    cols = (j.view(n, 1) + torch.arange(32, device=x.device)).view(n, 1, 1, 32).expand(n, 3, 32, 32)
    xp = torch.gather(xp, 3, cols)
    flip = torch.rand(n, device=x.device) < 0.5
    xp[flip] = xp[flip].flip(3)
    return xp


def evaluate(model, x, y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for k in range(0, x.shape[0], 1000):
            correct += (model(normalize(x[k:k + 1000])).argmax(1) == y[k:k + 1000]).sum().item()
    return correct / x.shape[0]


def train(model, data, epochs, lr, log):
    xtr, ytr, xte, yte = data
    # No weight decay on the FPGA blocks: it shrinks their outputs below the Q3.5 step.
    blk = [p for n, p in model.named_parameters() if n.startswith(('b1.', 'b2.'))]
    rest = [p for n, p in model.named_parameters() if not n.startswith(('b1.', 'b2.'))]
    opt = torch.optim.AdamW([{'params': rest, 'weight_decay': 5e-4}, {'params': blk, 'weight_decay': 0}], lr=lr)
    steps = epochs * (xtr.shape[0] // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr.shape[0], device=xtr.device)
        for k in range(0, xtr.shape[0] - 127, 128):
            idx = perm[k:k + 128]
            loss = F.cross_entropy(model(normalize(augment(xtr[idx]))), ytr[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ep == epochs - 1 or ep % 5 == 4:
            log(f'  epoch {ep + 1:3d}/{epochs}  loss {loss.item():.3f}  test acc {evaluate(model, xte, yte):.4f}')


# ---- export: ARM layers (BN folded) + integer reference check --------------------------
def fold(cb):
    conv, bn = cb[0], cb[1]
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * s.view(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * s
    return w.detach().cpu().numpy().astype(np.float32), b.detach().cpu().numpy().astype(np.float32)


def int_block(a, p, stride):
    """Independent integer dsc_top (same as hls/compute/golden): a uint8 [32,16,16]."""
    a = a.astype(np.int64)
    ap = np.pad(a, ((0, 0), (1, 1), (1, 1)))
    ho = (16 + 2 - 3) // stride + 1
    acc = np.zeros((C_IN, ho, ho), np.int64) + p['dw_b'].astype(np.int64)[:, None, None]
    for ky in range(3):
        for kx in range(3):
            win = ap[:, ky:ky + stride * (ho - 1) + 1:stride, kx:kx + stride * (ho - 1) + 1:stride]
            acc += win * p['dw_wt'][:, ky, kx].astype(np.int64)[:, None, None]
    d = np.clip((acc + 64) >> 7, 0, ACT_MAX)
    acc2 = np.einsum('kc,cyx->kyx', p['pw_wt'].astype(np.int64), d) + p['pw_b'].astype(np.int64)[:, None, None]
    return d.astype(np.uint8), np.clip((acc2 + 64) >> 7, 0, ACT_MAX).astype(np.uint8)


def block_inputs(model, x_u8):
    """The two FPGA blocks' uint8 inputs and outputs as the trained model computes them."""
    model.eval()
    with torch.no_grad():
        x = F.relu6(model.stem1(normalize(x_u8).to(model.fc.weight.dtype)))
        a1 = q_act(model.stem2(x))
        o1 = model.b1(a1)
        a2 = q_act(model.proj(o1))
        o2 = model.b2(a2)
    u8 = lambda t: torch.round(t * ACT_ONE).to(torch.uint8).cpu().numpy()
    return u8(a1), u8(o1), u8(a2), u8(o2)


def check_export(model, x_u8):
    """The fake-quant model and the integer reference must agree on every output byte."""
    a1, o1, a2, o2 = block_inputs(model, x_u8)
    p1, p2 = model.b1.export(), model.b2.export()
    bad = 0
    for n in range(x_u8.shape[0]):
        bad += int((int_block(a1[n], p1, 1)[1] != o1[n]).sum())
        bad += int((int_block(a2[n], p2, 2)[1] != o2[n]).sum())
    return bad


def write_case(d, p, a, stride):
    os.makedirs(d, exist_ok=True)
    with open(f'{d}/meta.txt', 'w') as f:
        f.write(f'format_version 1\nstride {stride}\nc {C_IN}\nh 16\nw 16\nc_out {C_OUT}\n')
    a.astype(np.uint8).tofile(f'{d}/dw_in.bin')
    p['dw_wt'].tofile(f'{d}/dw_wt.bin')
    p['dw_b'].astype('<i4').tofile(f'{d}/dw_b.bin')
    p['pw_wt'].tofile(f'{d}/pw_wt.bin')
    p['pw_b'].astype('<i4').tofile(f'{d}/pw_b.bin')
    dw, pw = int_block(a, p, stride)
    dw.tofile(f'{d}/expected_dw_out.bin')
    pw.tofile(f'{d}/expected_pw_out.bin')


def export(model, out, level, xte, acc):
    d = f'{out}/s{level}'
    os.makedirs(d, exist_ok=True)
    w1, b1 = fold(model.stem1)
    w2, b2 = fold(model.stem2)
    wp, bp = fold(model.proj)
    np.savez(f'{d}/arm.npz', stem1_w=w1, stem1_b=b1, stem2_w=w2, stem2_b=b2, proj_w=wp, proj_b=bp,
             fc_w=model.fc.weight.detach().cpu().numpy(), fc_b=model.fc.bias.detach().cpu().numpy(),
             mean=MEAN.flatten().numpy(), std=STD.flatten().numpy())
    a1, _, a2, _ = block_inputs(model, xte[:1])
    write_case(f'{d}/b1', model.b1.export(), a1[0], 1)
    write_case(f'{d}/b2', model.b2.export(), a2[0], 2)
    sl = model.sched_len()
    return dict(level=level, gpu_test_acc=acc, sched_len=sl,
                pw_block_sparsity={k: float(1 - getattr(model, k).mask.mean().item()) for k in ('b1', 'b2')})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder containing cifar-10-batches-py/')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'export'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--ft-epochs', type=int, default=15)
    ap.add_argument('--levels', default='0,50,75,90')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    logf = open(f'{args.out}/train_log.txt', 'w')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    data = load(args.data, dev)
    xte, yte = data[2], data[3]
    np.savez_compressed(f'{args.out}/test_images.npz', x=xte.permute(0, 2, 3, 1).cpu().numpy(),
                        y=yte.cpu().numpy(), classes=np.array(CLASSES))

    model = Net().to(dev)
    log(f'dense training, {args.epochs} epochs')
    train(model, data, args.epochs, 3e-3, log)
    dense = {k: v.clone() for k, v in model.state_dict().items()}

    summary = []
    for level in [int(v) for v in args.levels.split(',')]:
        model.load_state_dict(dense)
        if level:
            model.b1.prune(level / 100)
            model.b2.prune(level / 100)
            log(f'{level}% PW blocks pruned, fine-tuning {args.ft_epochs} epochs')
            train(model, data, args.ft_epochs, 1e-3, log)
        acc = evaluate(model, xte, yte)
        bad = check_export(model.cpu().double(), xte[:200].cpu())
        model.float().to(dev)
        log(f'level {level}%: test acc {acc:.4f}, sched_len {model.sched_len()}, '
            f'integer-reference mismatches on 200 images: {bad}')
        if bad:
            raise SystemExit('export check failed: fake-quant model != integer reference')
        summary.append(export(model.cpu(), args.out, level, xte.cpu(), acc))
        model.to(dev)
    with open(f'{args.out}/summary.json', 'w') as f:
        json.dump(dict(classes=CLASSES, levels=summary), f, indent=1)
    log('done')


if __name__ == '__main__':
    main()
