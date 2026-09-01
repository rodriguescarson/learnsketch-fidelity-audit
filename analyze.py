#!/usr/bin/env python3
"""
Analysis of ssim_floor.py output.

Two questions:
  1. Mechanism. Is the blank-canvas SSIM explained by how sparse the reference
     edge map is? (Predicted by the audit: empty-in-both windows return ~1.)
  2. Discrimination. Can each metric separate a basically-correct drawing from
     no drawing at all? Reported as AUC, which needs no threshold choice.

Writes results/analysis.json and results/figure_floor.png.

    python3 analyze.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# A fidelity metric must at minimum rank these two groups apart.
ACCEPTABLE = ["perfect_copy", "drop_25pct", "shift_1px", "shift_2px",
              "rotate_1deg", "rotate_2deg"]
UNACCEPTABLE = ["blank_canvas", "wrong_drawing", "random_scribble"]

HIGHER_IS_BETTER = {"ssim": True, "ssim_win11": True, "iou": True,
                    "dice": True, "chamfer": False,
                    # learned metrics, present only after learned_metrics.py has run
                    "lpips": False, "dists": False}


def auc(pos, neg):
    """P(a random acceptable drawing outranks a random unacceptable one)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / d) if d else float("nan")


def main():
    rows = json.load(open(os.path.join(RESULTS, "floor_raw.json")))

    # Merge the learned-metric run if it exists. Same corpus, same seed, same
    # iteration order, so rows join on (reference, condition).
    learned_path = os.path.join(RESULTS, "learned_raw.json")
    metrics = [m for m in HIGHER_IS_BETTER if m not in ("lpips", "dists")]
    if os.path.exists(learned_path):
        lut = {(r["reference"], r["condition"]): r
               for r in json.load(open(learned_path))}
        missing = [k for k in ((r["reference"], r["condition"]) for r in rows)
                   if k not in lut]
        if missing:
            print(f"(learned_raw.json covers {len(rows)-len(missing)}/{len(rows)} "
                  f"rows; re-run learned_metrics.py without --quick)\n")
        else:
            for r in rows:
                j = lut[(r["reference"], r["condition"])]
                r["lpips"], r["dists"] = j["lpips"], j["dists"]
            metrics += ["lpips", "dists"]
    else:
        print("(no learned_raw.json; run learned_metrics.py for LPIPS and DISTS)\n")

    out = {}

    # ---- 1. mechanism: blank-canvas score vs how sparse the reference is ----
    blank = [r for r in rows if r["condition"] == "blank_canvas"]
    dens = [r["ink_density"] * 100 for r in blank]
    bs = [r["ssim"] * 100 for r in blank]
    out["mechanism"] = {
        "n": len(blank),
        "pearson_r_inkdensity_vs_blank_ssim": pearson(dens, bs),
        "ink_density_pct": {"min": min(dens), "max": max(dens), "mean": float(np.mean(dens))},
        "blank_ssim_pct": {"min": min(bs), "max": max(bs), "mean": float(np.mean(bs))},
    }
    print("1. MECHANISM")
    print(f"   Blank-canvas SSIM vs reference ink density: r = "
          f"{out['mechanism']['pearson_r_inkdensity_vs_blank_ssim']:+.3f}  (n={len(blank)})")
    print(f"   Sparser reference -> higher score for drawing nothing.")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 100)]:
        sel = [b for d, b in zip(dens, bs) if lo <= d < hi]
        if sel:
            print(f"     ink {lo:>2}-{hi:<3}%  n={len(sel):>2}  "
                  f"blank canvas scores {np.mean(sel):6.2f}%  (max {max(sel):.2f}%)")

    # ---- 2. discrimination -------------------------------------------------
    print("\n2. DISCRIMINATION  (AUC: can the metric rank a roughly-correct")
    print("   drawing above no drawing / the wrong drawing / a scribble?)")
    out["discrimination"] = {}
    for m in metrics:
        hib = HIGHER_IS_BETTER[m]
        s = 1.0 if hib else -1.0
        pos = [s * r[m] for r in rows if r["condition"] in ACCEPTABLE]
        neg = [s * r[m] for r in rows if r["condition"] in UNACCEPTABLE]
        a = auc(pos, neg)
        out["discrimination"][m] = {"auc": a, "n_pos": len(pos), "n_neg": len(neg)}
        verdict = "near chance" if a < 0.65 else ("usable" if a < 0.9 else "strong")
        print(f"   {m:11s} AUC = {a:.3f}   {verdict}")

    # ---- 3. the ordering failure that matters ------------------------------
    print("\n3. ORDERING FAILURES  (share of references where a BLANK CANVAS")
    print("   outscores the stated condition)")
    out["blank_beats"] = {}
    byref = {}
    for r in rows:
        byref.setdefault(r["reference"], {})[r["condition"]] = r
    for cond in ["shift_1px", "shift_2px", "shift_5px", "shift_10px",
                 "rotate_2deg", "rotate_5deg", "drop_50pct", "drop_75pct",
                 "wrong_drawing", "random_scribble"]:
        rates = {}
        for m in metrics:
            better = HIGHER_IS_BETTER[m]
            hits = [((v["blank_canvas"][m] > v[cond][m]) if better
                     else (v["blank_canvas"][m] < v[cond][m]))
                    for v in byref.values() if cond in v]
            rates[m] = float(np.mean(hits))
        out["blank_beats"][cond] = rates
        extra = "".join(f"  {m.upper()} {rates[m]*100:5.1f}%"
                        for m in ("lpips", "dists") if m in rates)
        print(f"   vs {cond:16s} SSIM {rates['ssim']*100:5.1f}%"
              f"     IoU {rates['iou']*100:5.1f}%{extra}")

    with open(os.path.join(RESULTS, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ---- figure ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summ = json.load(open(os.path.join(RESULTS, "floor_summary.json")))
        order = ["perfect_copy", "drop_25pct", "rotate_1deg", "drop_50pct", "rotate_2deg",
                 "drop_75pct", "shift_1px", "blank_canvas", "shift_2px", "rotate_5deg",
                 "shift_5px", "shift_10px", "wrong_drawing", "random_scribble"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

        y = np.arange(len(order))
        cols = ["#c0392b" if c == "blank_canvas" else "#7f8c8d" for c in order]
        ax[0].barh(y, [summ[c]["ssim"]["mean"] * 100 for c in order],
                   xerr=[summ[c]["ssim"]["sd"] * 100 for c in order],
                   color=cols, height=.7, error_kw=dict(lw=.7, ecolor="#34495e"))
        ax[0].axvline(summ["blank_canvas"]["ssim"]["mean"] * 100, color="#c0392b", ls="--", lw=1)
        ax[0].set_yticks(y); ax[0].set_yticklabels(order, fontsize=8)
        ax[0].invert_yaxis(); ax[0].set_xlabel("SSIM (%)"); ax[0].set_xlim(0, 105)
        ax[0].set_title("SSIM: blank canvas (red) outranks\ncorrect drawings that are slightly off",
                        fontsize=9)

        ax[1].barh(y, [summ[c]["iou"]["mean"] for c in order],
                   xerr=[summ[c]["iou"]["sd"] for c in order],
                   color=cols, height=.7, error_kw=dict(lw=.7, ecolor="#34495e"))
        ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlabel("IoU (dilated, r=2px)")
        ax[1].invert_yaxis()          # must match ax[0], or the rows read reversed
        ax[1].set_xlim(0, 1.05)
        ax[1].set_title("IoU on dilated edge maps:\nblank canvas pinned at 0", fontsize=9)
        for a_ in ax:
            a_.grid(axis="x", alpha=.25); a_.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "figure_floor.png"), dpi=200)
        print(f"\nwrote {RESULTS}/figure_floor.png")
    except Exception as e:                                    # noqa: BLE001
        print(f"\n(figure skipped: {e})")

    print(f"wrote {RESULTS}/analysis.json")


if __name__ == "__main__":
    main()
