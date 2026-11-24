#!/usr/bin/env python3
"""make_plots.py — figures for the report from measured co-sim data (cosim_results.csv).

    python3 make_plots.py        # writes fig_speedup.png, fig_efficiency.png, fig_latency.png

Every number plotted is a measured C/RTL co-simulation cycle count; nothing is estimated.
Speedup = cycles with skipping off (bypass) / cycles with skipping on, same hardware.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CLOCK_MHZ = 200.0

# Reference palette (light mode), categorical slots in fixed order. Every series also
# gets its own marker and a direct label, so identity never rests on color alone.
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [
    # (kernel, stride, label, color, marker)
    ("pointwise_skip_top",      1, "Pointwise only, stride 1",            "#2a78d6", "o"),
    ("dsc_top",                 1, "Fused DW→PW block, stride 1",          "#eb6834", "s"),
    ("dsc_top",                 2, "Fused DW→PW block, stride 2",          "#1baf7a", "^"),
    ("pointwise_skip_pipe_top", 1, "Pointwise, overlapped load (proto.)",  "#6250d6", "D"),
    ("dsc_top+bram_lat2+overlap", 1, "Fused block + overlapped load (exp.)", "#e87ba4", "P"),
]


def load():
    rows = []
    with open(os.path.join(HERE, "cosim_results.csv")) as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            r["stride"] = int(r["stride"])
            for k in ("block_prune_pct", "sched_len", "cycles_skip_on", "cycles_bypass"):
                r[k] = int(r[k])
            r["skipped_pct"] = 100.0 * (256 - r["sched_len"]) / 256
            r["speedup"] = r["cycles_bypass"] / r["cycles_skip_on"]
            rows.append(r)
    return rows


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def series_points(rows, kernel, stride):
    """(skipped %, speedup) for one series: dense + block-pruned cases (not unstructured)."""
    return sorted((r["skipped_pct"], r["speedup"]) for r in rows
                  if r["kernel"] == kernel and r["stride"] == stride
                  and (r["block_prune_pct"] > 0 or r["case"] == "dense"))


def ideal(x_pct):
    return 1.0 / (1.0 - x_pct / 100.0)


def fig_speedup(rows):
    """Two panels: standalone pointwise kernels | fused DW->PW block. Measured points
    (markers only: cases are sparse, and straight segments between distant points would
    cut above the convex ideal curve and misread as better-than-ideal)."""
    panels = [
        ("Standalone pointwise kernel (stride 1)", ["pointwise_skip_top", "pointwise_skip_pipe_top"]),
        ("Fused DW→PW block (dsc_top)", ["dsc_top", "dsc_top+bram_lat2+overlap"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    xs = [i * 0.5 for i in range(0, 151)]
    for ax, (title, kernels) in zip(axes, panels):
        style(ax)
        ax.plot(xs, [ideal(x) for x in xs], color=TEXT2, linewidth=1.2, linestyle=(0, (4, 3)),
                zorder=1, label="Ideal: 1 / (1 − skipped)")
        for kernel, stride, label, color, marker in SERIES:
            if kernel not in kernels:
                continue
            pts = series_points(rows, kernel, stride)
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, linestyle="none", marker=marker, markersize=7, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
            ax.annotate(f"{y[-1]:.2f}×", (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=TEXT)
        ax.set_xlim(-2, 82)
        ax.set_ylim(0.8, 4.2)
        ax.set_xlabel("Pointwise MAC cycles skipped (%)", color=TEXT, fontsize=10)
        ax.set_title(title, color=TEXT, fontsize=10.5, loc="left")
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TEXT)
    axes[0].set_ylabel("Speedup vs. skipping off (×)", color=TEXT, fontsize=10)
    fig.suptitle("Zero-block skipping: measured speedup (C/RTL co-sim, K26)", color=TEXT,
                 fontsize=11.5, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Same hardware with skip_en on vs. off; each point is one test case. "
             "Stride 2 (fused block) is capped near 2× by the line buffer: see fig_efficiency.png.",
             fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path = os.path.join(HERE, "fig_speedup.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_efficiency(rows):
    """Measured speedup as a percentage of the ideal for the same skip rate."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for kernel, stride, label, color, marker in SERIES:
        pts = [(x, 100.0 * y / ideal(x)) for x, y in series_points(rows, kernel, stride) if x > 0]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
        ax.annotate(f"{y[-1]:.0f}%", (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT)
    ax.axhline(100, color=TEXT2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(21, 101.5, "ideal", color=TEXT2, fontsize=8.5)
    ax.set_xlim(20, 84)
    ax.set_ylim(40, 108)
    ax.set_xlabel("Pointwise MAC cycles skipped (%)", color=TEXT, fontsize=10)
    ax.set_ylabel("Speedup achieved, % of ideal", color=TEXT, fontsize=10)
    ax.set_title("How much of the skipped work turns into speedup", color=TEXT, fontsize=11, loc="left")
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=TEXT)
    fig.text(0.01, 0.005, "Ideal = 1 / (1 − skipped). Overlapped load removes the per-pixel "
             "load/restart overhead; stride 2 is limited by the line buffer.", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(HERE, "fig_efficiency.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_latency(rows):
    """Fused-block latency at 200 MHz, skip on vs. off, by pruning level (stride 1)."""
    data = sorted((r for r in rows if r["kernel"] == "dsc_top" and r["stride"] == 1
                   and not r["case"].startswith("50% unstructured")),
                  key=lambda r: r["block_prune_pct"])
    if not data:
        return None
    labels = ["dense" if r["block_prune_pct"] == 0 else
              ("all zero" if r["block_prune_pct"] == 100 else f"{r['block_prune_pct']}%") for r in data]
    off = [r["cycles_bypass"] / CLOCK_MHZ / 1000.0 for r in data]     # ms
    on = [r["cycles_skip_on"] / CLOCK_MHZ / 1000.0 for r in data]

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(False, axis="x")
    n = len(data)
    width = 0.36
    gap = 0.02   # surface gap between the paired bars
    xo = [i - width / 2 - gap / 2 for i in range(n)]
    xn = [i + width / 2 + gap / 2 for i in range(n)]
    ax.bar(xo, off, width, color="#2a78d6", label="Skipping off (bypass)", zorder=2)
    ax.bar(xn, on, width, color="#eb6834", label="Skipping on", zorder=2)
    for x, v in zip(xn, on):
        ax.text(x, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.text(xo[0], off[0] + 0.006, f"{off[0]:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Pointwise weight blocks pruned", color=TEXT, fontsize=10)
    ax.set_ylabel(f"Latency per block at {CLOCK_MHZ:.0f} MHz (ms)", color=TEXT, fontsize=10)
    ax.set_title("Fused depthwise-separable block (16×16, 32→64), stride 1",
                 color=TEXT, fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=TEXT)
    ax.set_ylim(0, max(off) * 1.32)
    fig.tight_layout()
    path = os.path.join(HERE, "fig_latency.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_energy(rows):
    """Estimated PL energy per block = Vivado vectorless power x measured co-sim latency.
    For the fused block with both proposed changes (the version that meets 200 MHz)."""
    pw = {}
    with open(os.path.join(HERE, "power_results.csv")) as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            pw[(r["design"], r["variant"])] = (float(r["dynamic_w"]), float(r["device_static_w"]))
    dyn_w, stat_w = pw[("dsc_top", "bram_lat2+overlap")]
    data = sorted((r for r in rows if r["kernel"] == "dsc_top+bram_lat2+overlap" and r["stride"] == 1
                   and not r["case"].startswith("50% unstructured")), key=lambda r: r["block_prune_pct"])
    labels = ["dense" if r["block_prune_pct"] == 0 else
              ("all zero" if r["block_prune_pct"] == 100 else f"{r['block_prune_pct']}%") for r in data]
    t_us = [r["cycles_skip_on"] / CLOCK_MHZ for r in data]           # cycles / MHz = microseconds
    e_dyn = [dyn_w * t for t in t_us]                                 # W x us = uJ
    e_stat = [stat_w * t for t in t_us]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(False, axis="x")
    x = range(len(data))
    ax.bar(x, e_dyn, 0.56, color="#2a78d6", label=f"Accelerator dynamic ({dyn_w * 1000:.0f} mW)", zorder=2)
    ax.bar(x, e_stat, 0.56, bottom=e_dyn, color="#c3c2b7", label=f"PL device static ({stat_w * 1000:.0f} mW, always on)",
           zorder=2, edgecolor=SURFACE, linewidth=1.5)
    for i, (d, st) in enumerate(zip(e_dyn, e_stat)):
        ax.text(i, d + st + 2, f"{d + st:.0f} µJ", ha="center", va="bottom", fontsize=8.5, color=TEXT)
        ax.text(i, d / 2, f"{d:.0f}", ha="center", va="center", fontsize=8, color=SURFACE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Pointwise weight blocks pruned (skipping on)", color=TEXT, fontsize=10)
    ax.set_ylabel("Estimated PL energy per block (µJ)", color=TEXT, fontsize=10)
    ax.set_ylim(0, (e_dyn[0] + e_stat[0]) * 1.22)
    ax.set_title("Estimated energy per depthwise-separable block (16×16, 32→64, 200 MHz)",
                 color=TEXT, fontsize=10.5, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=TEXT)
    fig.text(0.01, 0.005, "ESTIMATE: Vivado vectorless power (default toggle rates) × measured co-sim latency.\n"
             "PL only; excludes the ARM PS and DDR. Board measurement pending.", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(HERE, "fig_energy.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    rows = load()
    for p in (fig_speedup(rows), fig_efficiency(rows), fig_latency(rows), fig_energy(rows)):
        if p:
            print("wrote", p)
