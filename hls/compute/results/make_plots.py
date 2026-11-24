#!/usr/bin/env python3
"""make_plots.py — figures for the report from measured co-sim data (cosim_results.csv).

    python3 make_plots.py        # writes fig_speedup.png, fig_latency.png next to this file

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


def fig_speedup(rows):
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)

    # Ideal: speedup = 1 / (1 - skipped), if skipping removed cycles with no overhead.
    xs = [i * 0.5 for i in range(0, 151)]
    ax.plot(xs, [1.0 / (1.0 - x / 100.0) for x in xs], color=TEXT2, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=1)
    ax.text(73.5, 3.9, "ideal: 1 / (1 − skipped)", color=TEXT2, fontsize=8.5, ha="right")

    for kernel, stride, label, color, marker in SERIES:
        pts = sorted((r["skipped_pct"], r["speedup"]) for r in rows
                     if r["kernel"] == kernel and r["stride"] == stride and r["block_prune_pct"] > 0
                     or (r["kernel"] == kernel and r["stride"] == stride and r["case"] == "dense"))
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.annotate(f"{y[-1]:.2f}×", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT)

    # Unstructured pruning: lots of zero weights, nothing skippable.
    un = [r for r in rows if r["kernel"] == "pointwise_skip_top" and r["case"].startswith("50% unstructured")]
    if un:
        ax.annotate("50% of weights zeroed at random\n(not block-aligned): ~1.0×",
                    (un[0]["skipped_pct"], un[0]["speedup"]), xytext=(9, 0.86),
                    textcoords="data", fontsize=8, color=TEXT2,
                    arrowprops=dict(arrowstyle="-", color=TEXT2, linewidth=0.8))

    ax.set_xlim(-2, 80)
    ax.set_ylim(0.8, 4.2)
    ax.set_xlabel("Pointwise MAC cycles skipped (%)", color=TEXT, fontsize=10)
    ax.set_ylabel("Layer speedup vs. skipping off (×)", color=TEXT, fontsize=10)
    ax.set_title("Zero-block skipping: measured speedup (C/RTL co-sim, K26)",
                 color=TEXT, fontsize=11, loc="left")
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=TEXT)
    fig.text(0.01, 0.005, "Same hardware with skip_en on vs. off. Stride 2 is capped by the line buffer "
             "(1 channel/cycle over every input pixel).", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(HERE, "fig_speedup.png")
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


if __name__ == "__main__":
    rows = load()
    for p in (fig_speedup(rows), fig_latency(rows)):
        if p:
            print("wrote", p)
