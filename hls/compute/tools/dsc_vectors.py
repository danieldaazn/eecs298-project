#!/usr/bin/env python3
"""dsc_vectors.py — test vectors for one depthwise-separable block (DW 3x3 -> PW 1x1).

File format: see ../VECTORS.md. In short, a case is a folder of raw little-endian
binary files in PyTorch's native layouts, plus a small meta.txt.

Commands
  check   <case_dir>...   Validate a case against the hardware contracts (dims, value
                          ranges, bias limit, stride), report its sparsity and the
                          skip rate the hardware will get, and, if expected_*.bin
                          files are present, compare them with the bit-exact integer
                          reference. Exit code 1 if any case has an error.
  expect  <case_dir>...   Write expected_dw_out.bin / expected_pw_out.bin from the
                          bit-exact integer reference (overwrites).
  example <case_dir>      Write a synthetic case (random data, optional block pruning)
                          including expected outputs. For testing the flow without
                          real model data.

The hardware dimensions and Q formats are read from include/dsc_types.h, so this
tool and the hardware can't silently disagree about them.
"""

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES_H = os.path.join(HERE, "..", "include", "dsc_types.h")

FORMAT_VERSION = 1
FILES = {
    # name           dtype       shape (in terms of meta dims)
    "dw_in.bin":      ("uint8",  lambda m: (m["c"], m["h"], m["w"])),
    "dw_wt.bin":      ("int8",   lambda m: (m["c"], 3, 3)),
    "dw_b.bin":       ("<i4",    lambda m: (m["c"],)),
    "pw_wt.bin":      ("int8",   lambda m: (m["c_out"], m["c"])),
    "pw_b.bin":       ("<i4",    lambda m: (m["c_out"],)),
}
EXPECTED = {
    "expected_dw_out.bin": ("uint8", lambda m: (m["c"], m["ho"], m["wo"])),
    "expected_pw_out.bin": ("uint8", lambda m: (m["c_out"], m["ho"], m["wo"])),
}


# ---- hardware constants, parsed from dsc_types.h -------------------------------------
def hw_constants():
    text = open(TYPES_H).read()
    vals = {}
    for name, expr in re.findall(r"static const int\s+(\w+)\s*=\s*([^;]+);", text):
        expr = expr.strip()
        try:
            vals[name] = int(eval(expr, {"__builtins__": {}}, dict(vals)))
        except Exception:
            pass  # e.g. expressions we don't need
    need = ["ACT_FRAC", "REQUANT_SHIFT", "ACT_RELU6_MAX", "BIAS_ABS_MAX",
            "DW_C", "DW_H", "DW_W", "PW_C_IN", "PW_C_OUT", "PW_UNROLL"]
    missing = [n for n in need if n not in vals]
    if missing:
        sys.exit(f"could not parse {missing} from {TYPES_H}")
    return vals


HW = hw_constants()


def out_dim(n, stride):
    return (n + 2 * 1 - 3) // stride + 1


# ---- bit-exact integer reference (same arithmetic as golden/dsc_golden.cpp) ----------
def requant_relu6(acc):
    sh = HW["REQUANT_SHIFT"]
    q = (acc + (1 << (sh - 1))) >> sh          # numpy >> on int64 is arithmetic (floor)
    return np.clip(q, 0, HW["ACT_RELU6_MAX"]).astype(np.uint8)


def ref_depthwise(x, wt, b, stride):
    c, h, w = x.shape
    ho, wo = out_dim(h, stride), out_dim(w, stride)
    p = np.zeros((c, h + 2, w + 2), dtype=np.int64)
    p[:, 1:-1, 1:-1] = x
    acc = np.broadcast_to(b.astype(np.int64)[:, None, None], (c, ho, wo)).copy()
    for ky in range(3):
        for kx in range(3):
            win = p[:, ky:ky + stride * (ho - 1) + 1:stride, kx:kx + stride * (wo - 1) + 1:stride]
            acc += win * wt[:, ky, kx].astype(np.int64)[:, None, None]
    return requant_relu6(acc)


def ref_pointwise(x, wt, b):
    acc = np.einsum("kc,chw->khw", wt.astype(np.int64), x.astype(np.int64))
    acc += b.astype(np.int64)[:, None, None]
    return requant_relu6(acc)


def reference(case):
    dw = ref_depthwise(case["dw_in.bin"], case["dw_wt.bin"], case["dw_b.bin"], case["meta"]["stride"])
    pw = ref_pointwise(dw, case["pw_wt.bin"], case["pw_b.bin"])
    return dw, pw


# ---- case I/O --------------------------------------------------------------------------
class CaseError(Exception):
    pass


def read_meta(d):
    path = os.path.join(d, "meta.txt")
    if not os.path.isfile(path):
        raise CaseError("missing meta.txt")
    meta = {}
    for ln, line in enumerate(open(path), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise CaseError(f"meta.txt line {ln}: expected 'key value', got {line!r}")
        k, v = parts
        try:
            meta[k] = int(v)
        except ValueError:
            raise CaseError(f"meta.txt line {ln}: value for {k!r} is not an integer")
    for k in ("format_version", "stride", "c", "h", "w", "c_out"):
        if k not in meta:
            raise CaseError(f"meta.txt: missing key {k!r}")
    if meta["format_version"] != FORMAT_VERSION:
        raise CaseError(f"meta.txt: format_version {meta['format_version']}, this tool reads {FORMAT_VERSION}")
    if meta["stride"] not in (1, 2):
        raise CaseError(f"meta.txt: stride must be 1 or 2, got {meta['stride']}")
    meta["ho"], meta["wo"] = out_dim(meta["h"], meta["stride"]), out_dim(meta["w"], meta["stride"])
    return meta


def load_array(d, name, dtype, shape, required=True):
    path = os.path.join(d, name)
    if not os.path.isfile(path):
        if required:
            raise CaseError(f"missing {name}")
        return None
    want = int(np.prod(shape)) * np.dtype(dtype).itemsize
    got = os.path.getsize(path)
    if got != want:
        raise CaseError(f"{name}: {got} bytes, expected {want} for shape {shape} {np.dtype(dtype).name}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def load_case(d):
    meta = read_meta(d)
    case = {"meta": meta}
    for name, (dtype, shape) in FILES.items():
        case[name] = load_array(d, name, dtype, shape(meta))
    for name, (dtype, shape) in EXPECTED.items():
        case[name] = load_array(d, name, dtype, shape(meta), required=False)
    return case


def write_case(d, meta, arrays):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(f"format_version {FORMAT_VERSION}\n")
        for k in ("stride", "c", "h", "w", "c_out"):
            f.write(f"{k} {meta[k]}\n")
    for name, arr in arrays.items():
        dtype = (FILES.get(name) or EXPECTED.get(name))[0]
        np.ascontiguousarray(arr, dtype=dtype).tofile(os.path.join(d, name))


# ---- check --------------------------------------------------------------------------------
def check_case(d):
    """Returns (errors, warnings, report_lines)."""
    errors, warnings, rep = [], [], []
    try:
        m = read_meta(d)
    except CaseError as e:
        return [str(e)], [], []
    # Dimensions first: a mismatch here would otherwise surface as confusing file-size
    # errors, since file sizes are checked against the dims in meta.txt.
    hw_dims = (HW["DW_C"], HW["DW_H"], HW["DW_W"], HW["PW_C_OUT"])
    if (m["c"], m["h"], m["w"], m["c_out"]) != hw_dims:
        return [f"dims {m['c']}x{m['h']}x{m['w']} -> {m['c_out']} (meta.txt) don't match the hardware build "
                f"{hw_dims[0]}x{hw_dims[1]}x{hw_dims[2]} -> {hw_dims[3]} (include/dsc_types.h). The kernels "
                "are compiled for one layer shape; see VECTORS.md."], [], []
    try:
        case = load_case(d)
    except CaseError as e:
        return [str(e)], [], []
    if HW["PW_C_IN"] != HW["DW_C"]:
        errors.append("dsc_types.h: PW_C_IN != DW_C, the block can't be fused")

    for name in ("dw_b.bin", "pw_b.bin"):
        b = case[name].astype(np.int64)
        worst = int(np.abs(b).max())
        if worst > HW["BIAS_ABS_MAX"]:
            errors.append(f"{name}: |bias| up to {worst} exceeds BIAS_ABS_MAX = {HW['BIAS_ABS_MAX']}")

    # Weights piled up at the int8 limits mean the float weights didn't fit Q1.7
    # ([-1.0, 0.992]) and were clipped during quantization: an accuracy problem that
    # no bit-exact check can see.
    # Clipping shows up as a spike AT the limit relative to the next value in, not
    # merely as values at the limit (evenly spread weights also land there sometimes).
    for name in ("dw_wt.bin", "pw_wt.bin"):
        wq = case[name].astype(np.int64)
        for lim, inner in ((127, 126), (-128, -127)):
            at, nxt = int((wq == lim).sum()), int((wq == inner).sum())
            # min count keeps chance hits in small arrays (DW is only C*9 weights) quiet
            if at >= max(8, 0.01 * wq.size) and at > 3 * max(nxt, 1):
                warnings.append(f"{name}: {at} weights at {lim} but only {nxt} at {inner}: the float weights "
                                "probably exceed Q1.7's [-1.0, 0.992] range and were clipped during "
                                "quantization (common after folding batch-norm). Raise this before "
                                "signing off the types.")

    x = case["dw_in.bin"]
    over = int((x > HW["ACT_RELU6_MAX"]).sum())
    if over:
        warnings.append(f"dw_in.bin: {over} values above {HW['ACT_RELU6_MAX']} (6.0). Fine for the hardware, "
                        "but a ReLU6 output never exceeds 6.0; check the activation scale.")

    # pointwise block structure
    U = HW["PW_UNROLL"]
    pw = case["pw_wt.bin"]
    cblks = m["c"] // U
    if m["c"] % U:
        errors.append(f"c = {m['c']} is not a multiple of the block size {U}")
    else:
        blocks = pw.reshape(m["c_out"], cblks, U)
        zero_blk = np.all(blocks == 0, axis=2)                      # [c_out][cblks]
        nz_per_k = (~zero_blk).sum(axis=1)
        sched_len = int(np.maximum(nz_per_k, 1).sum())
        dense = m["c_out"] * cblks
        zero_w = int((pw == 0).sum())
        zero_w_in_dead_blocks = int(zero_blk.sum()) * U
        stray = zero_w - zero_w_in_dead_blocks                       # zeros the hardware can't use
        dead_k = int((nz_per_k == 0).sum())
        rep.append(f"pointwise weights zero:   {100.0 * zero_w / pw.size:5.1f}%")
        rep.append(f"pointwise blocks pruned:  {100.0 * zero_blk.mean():5.1f}%  ({int(zero_blk.sum())} of {zero_blk.size} blocks of {U})")
        rep.append(f"sched_len (cycles/pixel): {sched_len} of {dense}  ->  MAC cycles skipped {100.0 * (dense - sched_len) / dense:5.1f}%")
        # Only worth a warning if the model is actually sparse (a dense layer has a few
        # incidental zero weights) and most of that sparsity is unusable.
        if zero_w > 0.05 * pw.size and stray > 0.25 * zero_w:
            warnings.append(f"{100.0 * stray / zero_w:.0f}% of the zero weights are in blocks that still have a "
                            f"nonzero weight, so the hardware can't skip them. Prune whole aligned blocks of "
                            f"{U} input channels (SKIP_FORMAT.md).")
        if dead_k:
            warnings.append(f"{dead_k} output channel(s) have every block pruned: each still costs 1 cycle per "
                            "pixel for a constant output. Consider removing those channels from the model.")
    dwz = int((case["dw_wt.bin"] == 0).sum())
    rep.append(f"depthwise weights zero:   {100.0 * dwz / case['dw_wt.bin'].size:5.1f}%  (depthwise doesn't skip)")

    # expected outputs vs the bit-exact reference
    if not errors:
        ref_dw, ref_pw = reference(case)
        for name, ref in (("expected_dw_out.bin", ref_dw), ("expected_pw_out.bin", ref_pw)):
            exp = case[name]
            if exp is None:
                rep.append(f"{name}: not present (run 'expect' to generate, or export from the model)")
                continue
            diff = exp.astype(np.int32) - ref.astype(np.int32)
            bad = int(np.count_nonzero(diff))
            if bad == 0:
                rep.append(f"{name}: matches the integer reference exactly")
            else:
                mx = int(np.abs(diff).max())
                msg = (f"{name}: {bad} of {diff.size} values differ from the integer reference "
                       f"(max |diff| {mx})")
                if mx == 1:
                    msg += (". All differences are +/-1: almost certainly a rounding-mode mismatch. "
                            "The hardware rounds half UP (floor(x + 0.5)); torch.round rounds half to even.")
                errors.append(msg)
    return errors, warnings, rep


def cmd_check(args):
    failed = 0
    for d in args.cases:
        errors, warnings, rep = check_case(d)
        status = "FAIL" if errors else "OK"
        print(f"== {d}: {status}")
        for line in rep:
            print(f"   {line}")
        for w in warnings:
            print(f"   WARNING: {w}")
        for e in errors:
            print(f"   ERROR: {e}")
        failed += bool(errors)
    return 1 if failed else 0


def cmd_expect(args):
    rc = 0
    for d in args.cases:
        try:
            case = load_case(d)
        except CaseError as e:
            print(f"{d}: ERROR: {e}")
            rc = 1
            continue
        dw, pw = reference(case)
        write_case(d, case["meta"], {"expected_dw_out.bin": dw, "expected_pw_out.bin": pw})
        print(f"{d}: wrote expected_dw_out.bin {dw.shape}, expected_pw_out.bin {pw.shape}")
    return rc


def cmd_example(args):
    rng = np.random.default_rng(args.seed)
    m = {"stride": args.stride, "c": HW["DW_C"], "h": HW["DW_H"], "w": HW["DW_W"], "c_out": HW["PW_C_OUT"]}
    U = HW["PW_UNROLL"]
    x = rng.integers(0, HW["ACT_RELU6_MAX"] + 1, size=(m["c"], m["h"], m["w"]), dtype=np.int64)
    dw_wt = rng.integers(-128, 128, size=(m["c"], 3, 3))
    dw_b = rng.integers(-20000, 20001, size=m["c"])
    pw_wt = rng.integers(-32, 32, size=(m["c_out"], m["c"]))
    pw_b = rng.integers(-20000, 20001, size=m["c_out"])
    if args.block_prune:
        blocks = pw_wt.reshape(m["c_out"], m["c"] // U, U)
        blocks[rng.random(blocks.shape[:2]) < args.block_prune / 100.0] = 0
    if args.unstructured:
        pw_wt[rng.random(pw_wt.shape) < args.unstructured / 100.0] = 0
    case = {"meta": dict(m, ho=out_dim(m["h"], m["stride"]), wo=out_dim(m["w"], m["stride"])),
            "dw_in.bin": x, "dw_wt.bin": dw_wt, "dw_b.bin": dw_b, "pw_wt.bin": pw_wt, "pw_b.bin": pw_b}
    dw, pw = reference(case)
    arrays = {k: v for k, v in case.items() if k != "meta"}
    arrays.update({"expected_dw_out.bin": dw, "expected_pw_out.bin": pw})
    write_case(args.case, m, arrays)
    print(f"{args.case}: example case written (stride {args.stride}, {args.block_prune}% blocks pruned, "
          f"{args.unstructured}% unstructured zeros, seed {args.seed})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check");   p.add_argument("cases", nargs="+"); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("expect");  p.add_argument("cases", nargs="+"); p.set_defaults(fn=cmd_expect)
    p = sub.add_parser("example"); p.add_argument("case")
    p.add_argument("--stride", type=int, choices=(1, 2), default=1)
    p.add_argument("--block-prune", type=float, default=0.0, help="%% of PW blocks to zero")
    p.add_argument("--unstructured", type=float, default=0.0, help="%% of PW weights to zero at random")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_example)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
