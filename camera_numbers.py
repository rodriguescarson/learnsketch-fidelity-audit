#!/usr/bin/env python3
"""Counts, intervals and checks reported in the paper, recounted from the stored results.

Reads only results/{floor_raw,learned_raw}.json, which the released harness
regenerates byte for byte, and writes results/camera_numbers.json. No metric is
recomputed from images here; every value is a recount of stored per-pair scores.

    python3 camera_numbers.py

What it adds, and why:
  1. counts and Wilson 95% intervals behind the three headline inversion rates;
  2. a reference-level (cluster) bootstrap interval for each AUC and for IoU minus SSIM;
  3. how often IoU and Dice TIE a blank canvas at zero (the paper says they never rank it
     above; ties are the honest remainder);
  4. the AUC with the blank canvas taken out of the unacceptable group, so IoU's zero for an
     empty canvas (true by construction) is not what carries its AUC;
  5. a duplicate check: scikit-image's `cat` and `chelsea` are the same photograph, so the
     "wrong reference" for nat:cat is an exact copy of it, and two synthetic drawings
     coincide. Every headline number is recomputed without the duplicates.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEED = 20261021
B = 10000

ACC = ["perfect_copy", "drop_25pct", "shift_1px", "shift_2px", "rotate_1deg", "rotate_2deg"]
UNACC = ["blank_canvas", "wrong_drawing", "random_scribble"]
# Conditions that depend only on the reference (no random draw), used to detect duplicates.
DETERMINISTIC = ["blank_canvas", "shift_1px", "shift_2px", "shift_5px", "shift_10px",
                 "rotate_1deg", "rotate_2deg", "rotate_5deg"]
HIB = {"ssim": True, "ssim_win11": True, "iou": True, "dice": True,
       "chamfer": False, "lpips": False, "dists": False}


def load():
    rows = json.load(open(os.path.join(RES, "floor_raw.json")))
    lut = {(r["reference"], r["condition"]): r
           for r in json.load(open(os.path.join(RES, "learned_raw.json")))}
    for r in rows:
        j = lut[(r["reference"], r["condition"])]
        r["lpips"], r["dists"] = j["lpips"], j["dists"]
    by = {}
    for r in rows:
        by.setdefault(r["reference"], {})[r["condition"]] = r
    return by


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    w = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(w / (len(pos) * len(neg)))


def auc_refs(by, refs, m, acc=ACC, unacc=UNACC):
    s = 1.0 if HIB[m] else -1.0
    pos = [s * by[k][c][m] for k in refs for c in acc]
    neg = [s * by[k][c][m] for k in refs for c in unacc]
    return auc(pos, neg)


def beats(by, refs, m, cond):
    """Blank canvas strictly better than `cond` (count), and ties."""
    h = HIB[m]
    win = sum((by[k]["blank_canvas"][m] > by[k][cond][m]) if h
              else (by[k]["blank_canvas"][m] < by[k][cond][m]) for k in refs)
    tie = sum(by[k]["blank_canvas"][m] == by[k][cond][m] for k in refs)
    return win, tie


def headline(by, refs):
    b = np.array([by[k]["blank_canvas"]["ssim"] for k in refs]) * 100
    d = np.array([by[k]["blank_canvas"]["ink_density"] for k in refs]) * 100
    out = {"n_refs": len(refs),
           "blank_ssim_mean_pct": float(b.mean()), "blank_ssim_max_pct": float(b.max()),
           "pearson_r_ink_vs_blank": float(np.corrcoef(d, b)[0, 1]),
           "auc": {m: auc_refs(by, refs, m) for m in HIB}}
    for m in HIB:
        out.setdefault("blank_beats", {})[m] = {
            c: beats(by, refs, m, c) for c in ("shift_2px", "wrong_drawing", "random_scribble")}
    return out


def main():
    by = load()
    refs = list(by)
    out = {"source": "results/floor_raw.json + learned_raw.json"}

    # 1. headline rates with counts and Wilson intervals
    out["inversions_ssim"] = {}
    for c in ("shift_2px", "wrong_drawing", "random_scribble"):
        k, t = beats(by, refs, "ssim", c)
        out["inversions_ssim"][c] = {"k": k, "n": len(refs), "rate": k / len(refs),
                                     "wilson95": wilson(k, len(refs)), "ties": t}

    # 2. reference-level bootstrap for the AUCs and for IoU - SSIM
    rng = np.random.default_rng(SEED)
    boot = {m: [] for m in HIB}
    diff = []
    for _ in range(B):
        samp = [refs[i] for i in rng.integers(0, len(refs), len(refs))]
        a = {m: auc_refs(by, samp, m) for m in HIB}
        for m in HIB:
            boot[m].append(a[m])
        diff.append(a["iou"] - a["ssim"])
    out["auc_point"] = {m: auc_refs(by, refs, m) for m in HIB}
    out["auc_boot95"] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                         for m, v in boot.items()}
    out["auc_iou_minus_ssim"] = {"point": out["auc_point"]["iou"] - out["auc_point"]["ssim"],
                                 "boot95": [float(np.percentile(diff, 2.5)),
                                            float(np.percentile(diff, 97.5))],
                                 "share_of_resamples_le_0": float(np.mean(np.array(diff) <= 0)),
                                 "resamples": B, "seed": SEED}
    # learned metrics and Chamfer against IoU, on the same resamples
    out["auc_minus_iou"] = {}
    for m in ("lpips", "dists", "chamfer"):
        dm = np.array(boot[m]) - np.array(boot["iou"])
        out["auc_minus_iou"][m] = {"point": out["auc_point"][m] - out["auc_point"]["iou"],
                                   "boot95": [float(np.percentile(dm, 2.5)),
                                              float(np.percentile(dm, 97.5))],
                                   "share_of_resamples_le_0": float(np.mean(dm <= 0))}

    # 3. ties for the overlap metrics
    out["ties_with_blank"] = {m: {c: beats(by, refs, m, c)[1]
                                  for c in ("shift_2px", "shift_5px", "shift_10px",
                                            "wrong_drawing", "random_scribble")}
                              for m in ("iou", "dice")}

    # 4. AUC without the blank canvas among the unacceptable conditions
    out["auc_without_blank"] = {m: auc_refs(by, refs, m, ACC, ["wrong_drawing", "random_scribble"])
                                for m in HIB}

    # 5. duplicates
    dup = []
    for i, a in enumerate(refs):
        for b_ in refs[i + 1:]:
            if all(by[a][c]["ssim"] == by[b_][c]["ssim"] for c in DETERMINISTIC) and \
                    by[a]["perfect_copy"]["ink_density"] == by[b_]["perfect_copy"]["ink_density"]:
                dup.append([a, b_])
    self_wrong = [k for k in refs if by[k]["wrong_drawing"]["ssim"] == 1.0
                  and by[k]["wrong_drawing"]["iou"] == 1.0]
    out["duplicates"] = {"identical_reference_pairs": dup,
                         "wrong_drawing_identical_to_reference": self_wrong}
    drop = {p[0] for p in dup}           # keep one of each pair (drops nat:cat, syn:1_16)
    if self_wrong:
        drop |= set(self_wrong)
    keep = [k for k in refs if k not in drop]
    out["published_65"] = headline(by, refs)
    out["deduplicated"] = headline(by, keep)
    out["deduplicated"]["dropped"] = sorted(drop)

    # 6. the two halves of the corpus separately (natural photographs vs synthetic drawings)
    out["by_half"] = {}
    for half, pre in (("natural", "nat:"), ("synthetic", "syn:")):
        sub = [k for k in refs if k.startswith(pre)]
        out["by_half"][half] = {
            "n": len(sub),
            "ssim_blank_beats": {c: beats(by, sub, "ssim", c)[0]
                                 for c in ("shift_2px", "wrong_drawing", "random_scribble")},
            "auc": {m: auc_refs(by, sub, m) for m in ("ssim", "iou", "lpips", "dists")}}

    with open(os.path.join(RES, "camera_numbers.json"), "w") as f:
        json.dump(out, f, indent=1)

    p = out
    print("SSIM inversion counts (blank above):")
    for c, v in p["inversions_ssim"].items():
        print("  %-16s %d/%d = %.1f%%  Wilson95 [%.1f, %.1f]" % (
            c, v["k"], v["n"], 100 * v["rate"], 100 * v["wilson95"][0], 100 * v["wilson95"][1]))
    print("AUC point and reference-bootstrap 95%% (B=%d):" % B)
    for m in HIB:
        print("  %-10s %.3f [%.3f, %.3f]   without blank in unacceptable: %.3f" % (
            m, p["auc_point"][m], *p["auc_boot95"][m], p["auc_without_blank"][m]))
    d = p["auc_iou_minus_ssim"]
    print("IoU - SSIM: %.3f [%.3f, %.3f]" % (d["point"], *d["boot95"]))
    for m, v in p["auc_minus_iou"].items():
        print("%s - IoU: %.4f [%.4f, %.4f]  share<=0 %.3f" % (
            m, v["point"], *v["boot95"], v["share_of_resamples_le_0"]))
    print("ties with blank:", p["ties_with_blank"])
    for half, v in p["by_half"].items():
        print("%s (n=%d): SSIM blank above shift2/wrong/scribble %s; AUC %s" % (
            half, v["n"], v["ssim_blank_beats"], {m: round(a, 3) for m, a in v["auc"].items()}))
    print("duplicates:", p["duplicates"], "dropped:", p["deduplicated"]["dropped"])
    for tag in ("published_65", "deduplicated"):
        h = p[tag]
        print("%s: n=%d blank mean %.2f max %.2f r %.3f" % (
            tag, h["n_refs"], h["blank_ssim_mean_pct"], h["blank_ssim_max_pct"],
            h["pearson_r_ink_vs_blank"]))
        print("   AUC", {m: round(v, 4) for m, v in h["auc"].items()})
        print("   blank beats (win, tie):", {m: h["blank_beats"][m] for m in ("ssim", "iou", "lpips", "dists")})


if __name__ == "__main__":
    main()
