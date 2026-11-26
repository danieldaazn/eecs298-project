# Test vectors from the model

**Status: PROPOSAL, not yet agreed with Kevin.** This is the file format for getting
real model data (weights, activations, reference outputs) into the hardware
testbenches. Nothing else depends on it yet, so it's cheap to change. Say so before
building an exporter against it.

## What a case is

One **case** = one depthwise-separable block (3×3 depthwise → 1×1 pointwise, ReLU6
after each), as one folder of raw binary files:

| File | Type | Shape (row-major) | Contents |
|---|---|---|---|
| `meta.txt` | text | | `key value` lines, see below |
| `dw_in.bin` | uint8 | `[C][H][W]` | block input, activations Q3.5 |
| `dw_wt.bin` | int8 | `[C][3][3]` | depthwise weights, Q1.7 |
| `dw_b.bin` | int32 LE | `[C]` | depthwise bias, 12 fractional bits |
| `pw_wt.bin` | int8 | `[C_out][C]` | pointwise weights, Q1.7 |
| `pw_b.bin` | int32 LE | `[C_out]` | pointwise bias, 12 fractional bits |
| `expected_dw_out.bin` | uint8 | `[C][Ho][Wo]` | *optional:* reference depthwise output |
| `expected_pw_out.bin` | uint8 | `[C_out][Ho][Wo]` | *optional:* reference block output |

```
# meta.txt
format_version 1
stride 1        # 1 or 2
c 32            # depthwise channels = pointwise input channels
h 16
w 16
c_out 64
```

`Ho = (H + 2 − 3) / stride + 1` (3×3, padding 1), same for `Wo`.

**These are PyTorch's own layouts:** activations `[C,H,W]`, depthwise weight
`[C,1,3,3]`, pointwise weight `[C_out,C,1,1]`, flattened. The hardware's DRAM layout
(HWC activations, packed 128-bit parameter beats) is a separate concern, handled by one
packer (`pack_input` / `pack_params` in `hls/top/tb/tb_dsc_top.cpp`, later the ARM
driver), so the exporter never needs to know it.

## Number formats (from `include/dsc_types.h`, placeholders pending sign-off)

| Quantity | Format | float → raw | Range |
|---|---|---|---|
| activation | unsigned Q3.5 | `round(a × 32)` | 0 … 255 (ReLU6's 6.0 = 192) |
| weight | signed Q1.7 | `round(w × 128)` | −128 … 127 (−1.0 … 0.992) |
| bias | int32, 12 fractional bits | `round(b × 4096)` | \|raw\| ≤ 2^24 (`BIAS_ABS_MAX`) |

**Rounding is half UP:** `floor(x + 0.5)`. This matters. `torch.round` and `np.round`
round half to *even*, which disagrees on exact .5 cases. The hardware's requantization
(accumulator → activation) also rounds half up, so a reference computed with
`torch.round` will be off by ±1 in a few outputs. The tools detect that and say so.

## Exporting from PyTorch

```python
import os
import numpy as np

def q(x, frac_bits, lo, hi):
    """Quantize like the hardware: round half up, then clamp."""
    return np.clip(np.floor(np.asarray(x, np.float64) * (1 << frac_bits) + 0.5), lo, hi)

def export_case(d, act, dw_w, dw_b, pw_w, pw_b, stride):
    """act: [C,H,W] float block input (already ReLU6'd). dw_w: [C,1,3,3], pw_w:
    [C_out,C,1,1] float weights with batch-norm folded in. Biases: float, per channel."""
    os.makedirs(d, exist_ok=True)
    C, H, W = act.shape
    q(act, 5, 0, 255).astype(np.uint8).tofile(f"{d}/dw_in.bin")
    q(dw_w.reshape(C, 3, 3), 7, -128, 127).astype(np.int8).tofile(f"{d}/dw_wt.bin")
    q(dw_b, 12, -2**24, 2**24).astype("<i4").tofile(f"{d}/dw_b.bin")
    q(pw_w.reshape(pw_w.shape[0], C), 7, -128, 127).astype(np.int8).tofile(f"{d}/pw_wt.bin")
    q(pw_b, 12, -2**24, 2**24).astype("<i4").tofile(f"{d}/pw_b.bin")
    with open(f"{d}/meta.txt", "w") as f:
        f.write(f"format_version 1\nstride {stride}\nc {C}\nh {H}\nw {W}\nc_out {pw_w.shape[0]}\n")
```

(`.detach().cpu().numpy()` the tensors first.)

**Expected outputs:** the most useful `expected_*.bin` come from *your* integer
simulation of the quantized model (half-up rounding, ReLU6 clamp at 192 after each
layer). If those match the hardware bit-exactly, quantization agrees end to end. If
you don't have one yet, `dsc_vectors.py expect <case>` fills them in from this spec.
That still checks the kernels, but not your quantization. Float-model outputs can't be
compared bit-exactly; judge those by accuracy, not here.

## Running

From `hls/compute/`:

```bash
python3 tools/dsc_vectors.py check path/to/case      # validate + sparsity report
make vectors VEC="path/to/case1 path/to/case2"     # check, then run the kernels
make example-vectors                               # try it on a synthetic sweep
```

**`check`** fails a case if:
- files are missing or the wrong size;
- dimensions don't match the hardware build;
- a bias exceeds the limit;
- the stride isn't 1 or 2;
- `expected_*.bin` disagree with the integer reference (with a rounding-mode hint when
  every difference is ±1).

It **warns** about:
- zeros the hardware can't skip (not whole aligned blocks of 8);
- output channels with every block pruned (see `SKIP_FORMAT.md`);
- input activations above 6.0;
- weights that look clipped (a spike at −128/127, typical when batch-norm-folded weights
  exceed Q1.7's ±1.0 range).

It also reports the skip rate the hardware will get for those weights.

**`tb_vectors`** runs the actual kernels:
- golden model vs `expected_*.bin`;
- depthwise → pointwise;
- pointwise with skipping on and with bypass, each bit-exact against golden;
- `sched_len` checked against an independent count.

It reports the measured skip rate. Exit code 0 only if every case passes.

## Current limitation: one layer shape

The kernels are compiled for **32 × 16 × 16 → 64** (`DW_C`, `DW_H`, `DW_W`,
`PW_C_OUT` in `dsc_types.h`). A case of any other shape is rejected with a clear
message. Real MobileNet layers are other shapes, so for now pick (or crop) a block
that matches, or we agree new compile-time sizes. Whether to support runtime
dimensions is the open question in Hank's #2.
