#!/usr/bin/env python3
"""Build the SSIM-floor figure (Fig. 5 in the manuscript) from floor_summary.json.

The row order is derived from the data, descending by mean SSIM. The v1 figure used a
hand-written order list, which put Rotated 1 deg above Missing 50% ink and Shifted 1 px
above the blank canvas, neither of which matches the values being plotted.

    python3 make_fig_floor.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "results", "fig6-ssim-floor.png")

LABEL = {
    "perfect_copy": "Perfect copy",
    "drop_25pct": "Missing 25% ink",
    "drop_50pct": "Missing 50% ink",
    "drop_75pct": "Missing 75% ink",
    "rotate_1deg": "Rotated 1$\\degree$",
    "rotate_2deg": "Rotated 2$\\degree$",
    "rotate_5deg": "Rotated 5$\\degree$",
    "blank_canvas": "BLANK CANVAS",
    "shift_1px": "Shifted 1 px",
    "shift_2px": "Shifted 2 px",
    "shift_5px": "Shifted 5 px",
    "shift_10px": "Shifted 10 px",
    "wrong_drawing": "Wrong reference",
    "random_scribble": "Random scribble",
}

GREY, RED, INK = "#7f8c8d", "#c0392b", "#34495e"


def main():
    summ = json.load(open(os.path.join(RESULTS, "floor_summary.json")))
    order = sorted(summ, key=lambda k: -summ[k]["ssim"]["mean"])

    means = np.array([summ[c]["ssim"]["mean"] * 100 for c in order])
    sds = np.array([summ[c]["ssim"]["sd"] * 100 for c in order])
    blank = summ["blank_canvas"]["ssim"]["mean"] * 100

    fig, ax = plt.subplots(figsize=(3.45, 2.42))
    y = np.arange(len(order))
    cols = [RED if c == "blank_canvas" else GREY for c in order]

    ax.barh(y, means, xerr=sds, color=cols, height=0.72,
            error_kw=dict(lw=0.8, ecolor=INK))
    ax.axvline(blank, color=RED, ls="--", lw=1.1, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[c] for c in order], fontsize=8.5)
    for tick, c in zip(ax.get_yticklabels(), order):
        if c == "blank_canvas":
            tick.set_color(RED)
            tick.set_fontweight("bold")

    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("SSIM (%)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.xaxis.grid(True, color="#ecf0f1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=420)
    print("wrote", os.path.normpath(OUT))
    print("row order (descending mean SSIM):")
    for c, m in zip(order, means):
        print("  %-16s %6.2f%%" % (c, m))


if __name__ == "__main__":
    main()
