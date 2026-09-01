#!/usr/bin/env python3
"""
Robustness checks behind the AUC claim, for review findings F4, F5 and playbook item 6.

The headline comparison is SSIM AUC 0.766 against IoU 0.984. Three things could make that
gap an artifact rather than a result:

  1. The acceptable/unacceptable split could be doing the work. The split is a priori (a
     condition is labelled by how it was generated, never by its score), but it excludes five
     middle conditions. Re-run with the middle folded into the acceptable side.
  2. The perfect copy could be doing the work. It is the only positive that is not a degraded
     drawing, and it scores 100% SSIM / 1.000 IoU by construction. Drop it.
  3. The IoU dilation radius could have been selected on the evaluation set. It is fixed at
     2 px in ssim_floor.py and never swept. Sweep it.

Writes results/robustness.json.

    python3 robustness.py
"""

import json
import os

import numpy as np

from ssim_floor import DILATE_R, build_corpus, conditions, m_iou, SEED

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# The a priori split used in analyze.py. A condition is assigned by how it was generated.
ACCEPTABLE = ["perfect_copy", "drop_25pct", "shift_1px", "shift_2px",
              "rotate_1deg", "rotate_2deg"]
UNACCEPTABLE = ["blank_canvas", "wrong_drawing", "random_scribble"]
# Neither clearly good nor clearly bad, so excluded from the headline AUC.
MIDDLE = ["drop_50pct", "drop_75pct", "shift_5px", "shift_10px", "rotate_5deg"]

RADII = [1, 2, 3, 4, 5]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def auc_over(rows, metric, positives, negatives):
    pos = [r[metric] for r in rows if r["condition"] in positives]
    neg = [r[metric] for r in rows if r["condition"] in negatives]
    return auc(pos, neg), len(pos), len(neg)


def main():
    rows = json.load(open(os.path.join(RESULTS, "floor_raw.json")))
    out = {"label_provenance": (
        "Conditions are labelled a priori by how they were generated, not by any human "
        "judgement of the rendered image and not by any metric score. No labeller saw the "
        "scores because no labeller was involved.")}

    # ---- 1 + 2. does the split or the perfect copy carry the AUC? -----------
    variants = {
        "headline": (ACCEPTABLE, UNACCEPTABLE),
        "no_perfect_copy": ([c for c in ACCEPTABLE if c != "perfect_copy"], UNACCEPTABLE),
        "middle_as_acceptable": (ACCEPTABLE + MIDDLE, UNACCEPTABLE),
        "middle_as_acceptable_no_perfect": (
            [c for c in ACCEPTABLE + MIDDLE if c != "perfect_copy"], UNACCEPTABLE),
    }
    out["auc_variants"] = {}
    print("AUC under different definitions of 'acceptable'")
    print("%-34s %8s %8s   %s" % ("variant", "SSIM", "IoU", "n pos/neg"))
    for name, (pos, neg) in variants.items():
        a_s, np_, nn = auc_over(rows, "ssim", pos, neg)
        a_i, _, _ = auc_over(rows, "iou", pos, neg)
        out["auc_variants"][name] = {"ssim": a_s, "iou": a_i, "n_pos": np_, "n_neg": nn}
        print("%-34s %8.3f %8.3f   %d/%d" % (name, a_s, a_i, np_, nn))

    # ---- 3. was the dilation radius selected on the evaluation set? ---------
    print("\nIoU dilation radius sweep (r=%d is what the paper uses)" % DILATE_R)
    corpus = build_corpus(False)
    rng = np.random.default_rng(SEED)
    per_radius = {r: [] for r in RADII}
    for i, (name, e) in enumerate(corpus):
        other = corpus[(i + 1) % len(corpus)][1]
        conds = conditions(e, other, rng)
        for cond, s in conds.items():
            for r in RADII:
                per_radius[r].append({"condition": cond, "iou": m_iou(e, s, r=r)})

    out["dilation_sweep"] = {}
    print("%6s %10s %10s %14s" % ("radius", "AUC", "blank IoU", "AUC no-perfect"))
    for r in RADII:
        a, _, _ = auc_over(per_radius[r], "iou", ACCEPTABLE, UNACCEPTABLE)
        a_np, _, _ = auc_over(per_radius[r],
                              "iou", [c for c in ACCEPTABLE if c != "perfect_copy"],
                              UNACCEPTABLE)
        blank = float(np.mean([x["iou"] for x in per_radius[r]
                               if x["condition"] == "blank_canvas"]))
        out["dilation_sweep"][str(r)] = {"auc": a, "auc_no_perfect": a_np, "blank_iou": blank}
        print("%6d %10.3f %10.3f %14.3f" % (r, a, blank, a_np))

    ssim_headline = out["auc_variants"]["headline"]["ssim"]
    beats = [r for r in RADII if out["dilation_sweep"][str(r)]["auc"] > ssim_headline]
    out["iou_beats_ssim_at_radii"] = beats
    print("\nIoU exceeds SSIM's %.3f at every radius tested: %s"
          % (ssim_headline, beats == RADII))

    with open(os.path.join(RESULTS, "robustness.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote %s/robustness.json" % RESULTS)


if __name__ == "__main__":
    main()
