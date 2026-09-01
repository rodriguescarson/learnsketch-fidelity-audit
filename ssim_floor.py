#!/usr/bin/env python3
"""
Does the SSIM score in a sketch-fidelity system have a usable floor?

The manuscript reports SSIM
between a user's thresholded drawing and the Canny edge map of a reference, and
reads "consistently above 93%" as evidence of accurate drawing. Both images are
near-binary and mostly empty background, so the concern is that most local
windows match trivially and the reported range describes the top of a scale
whose bottom was never measured.

The experiment scores a battery of deliberately bad "sketches" against the same
references. If a blank canvas also scores high, the metric cannot separate a
good drawing from no drawing, and the paper's headline number means little.

Pipeline mirrors the manuscript's Section III:
    reference photo -> grayscale -> Canny -> binary edge map (white on black)
    "sketch"        -> same binary convention
    score           -> SSIM over local windows

Baselines computed alongside SSIM, per finding 8 (no baseline comparison):
    IoU and Dice on dilated edge maps, and symmetric Chamfer distance.

    python3 ssim_floor.py            # full run, writes results/
    python3 ssim_floor.py --quick    # fewer references, for a smoke test

Deterministic: fixed seed, no network access at run time.
"""

import argparse
import json
import os
import warnings

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

SEED = 20260801
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Canny thresholds. The manuscript does not report its own (audit finding 11),
# so we use the common OpenCV defaults and report them.
CANNY_LO, CANNY_HI = 100, 200
CANVAS = 256          # all references resized to CANVAS x CANVAS
DILATE_R = 2          # tolerance radius for the overlap baselines


# ---------------------------------------------------------------- corpus ----

def _to_gray_square(img, n=CANVAS):
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        rng = a.max() - a.min()
        a = np.zeros_like(a) if rng == 0 else (a - a.min()) / rng * 255.0
        a = a.astype(np.uint8)
    return cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)


def natural_references(quick=False):
    """Canny edge maps of bundled photographs, plus crops of the large ones."""
    from skimage import data
    names = ["astronaut", "camera", "cat", "chelsea", "clock", "coffee", "coins",
             "horse", "moon", "page", "rocket", "text", "brick", "grass", "gravel",
             "cell", "colorwheel", "immunohistochemistry"]
    big = ["retina", "hubble_deep_field"]
    if quick:
        names, big = names[:6], []

    out = []
    for n in names:
        try:
            out.append((f"nat:{n}", _to_gray_square(getattr(data, n)())))
        except Exception:
            continue
    # extra patches from the high-resolution images, to widen the ink-density range
    for n in big:
        try:
            a = np.asarray(getattr(data, n)())
            if a.ndim == 3:
                a = cv2.cvtColor(a[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            h, w = a.shape[:2]
            for i, (y, x) in enumerate([(0, 0), (h // 2, 0), (0, w // 2), (h // 2, w // 2)]):
                out.append((f"nat:{n}{i}", _to_gray_square(a[y:y + h // 2, x:x + w // 2])))
        except Exception:
            continue
    return out


def synthetic_line_drawings(count, rng):
    """
    Line drawings in the style of the manuscript's own test set, which is a
    flower, a square and a circle. Drawn directly as binary edge maps.
    """
    out = []
    for i in range(count):
        c = np.zeros((CANVAS, CANVAS), np.uint8)
        kind = i % 5
        cx = cy = CANVAS // 2
        if kind == 0:                                     # polygon
            k = int(rng.integers(3, 9))
            r = int(rng.integers(60, 110))
            th = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.random() * 2 * np.pi
            pts = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1).astype(np.int32)
            cv2.polylines(c, [pts], True, 255, 1)
        elif kind == 1:                                   # concentric circles
            for j in range(int(rng.integers(1, 5))):
                cv2.circle(c, (cx, cy), int(30 + j * rng.integers(15, 30)), 255, 1)
        elif kind == 2:                                   # flower / petal rosette
            k = int(rng.integers(5, 10))
            R = int(rng.integers(55, 90))
            for j in range(k):
                a = 2 * np.pi * j / k
                cv2.ellipse(c, (int(cx + R * 0.55 * np.cos(a)), int(cy + R * 0.55 * np.sin(a))),
                            (int(R * 0.45), int(R * 0.22)), np.degrees(a), 0, 360, 255, 1)
            cv2.circle(c, (cx, cy), int(R * 0.25), 255, 1)
        elif kind == 3:                                   # nested rectangles
            for j in range(int(rng.integers(1, 4))):
                s = int(rng.integers(40, 100)) - j * 18
                if s > 8:
                    cv2.rectangle(c, (cx - s, cy - s), (cx + s, cy + s), 255, 1)
        else:                                             # star / spiral
            t = np.linspace(0, rng.integers(3, 8) * np.pi, 600)
            r = np.linspace(8, rng.integers(70, 115), 600)
            pts = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1).astype(np.int32)
            cv2.polylines(c, [pts], False, 255, 1)
        out.append((f"syn:{kind}_{i:02d}", c))
    return out


def build_corpus(quick=False):
    """Returns [(name, edge_map_uint8_binary)]."""
    rng = np.random.default_rng(SEED)
    refs = []
    for name, gray in natural_references(quick):
        e = cv2.Canny(gray, CANNY_LO, CANNY_HI)
        if e.mean() > 0.5:                    # skip images with essentially no edges
            refs.append((name, (e > 0).astype(np.uint8) * 255))
    refs += synthetic_line_drawings(8 if quick else 40, rng)
    return refs


# ------------------------------------------------------------- conditions ---

def shift(e, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(e, M, e.shape[::-1], flags=cv2.INTER_NEAREST, borderValue=0)


def rotate(e, deg):
    M = cv2.getRotationMatrix2D((e.shape[1] / 2, e.shape[0] / 2), deg, 1.0)
    return cv2.warpAffine(e, M, e.shape[::-1], flags=cv2.INTER_NEAREST, borderValue=0)


def drop(e, frac, rng):
    ys, xs = np.nonzero(e)
    if len(ys) == 0:
        return e.copy()
    keep = rng.random(len(ys)) >= frac
    out = np.zeros_like(e)
    out[ys[keep], xs[keep]] = 255
    return out


def scribble(e, rng):
    """Random strokes with the same ink budget as the reference: wrong drawing."""
    out = np.zeros_like(e)
    budget = int((e > 0).sum())
    if budget == 0:
        return out
    n = e.shape[0]
    while (out > 0).sum() < budget:
        p1 = rng.integers(0, n, 2)
        p2 = np.clip(p1 + rng.integers(-n // 3, n // 3, 2), 0, n - 1)
        cv2.line(out, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 255, 1)
    return out


def conditions(e, other, rng):
    """The sketch variants scored against reference `e`."""
    return {
        "perfect_copy":   e.copy(),
        "blank_canvas":   np.zeros_like(e),
        "wrong_drawing":  other.copy(),
        "random_scribble": scribble(e, rng),
        "shift_1px":      shift(e, 1, 1),
        "shift_2px":      shift(e, 2, 2),
        "shift_5px":      shift(e, 5, 5),
        "shift_10px":     shift(e, 10, 10),
        "rotate_1deg":    rotate(e, 1),
        "rotate_2deg":    rotate(e, 2),
        "rotate_5deg":    rotate(e, 5),
        "drop_25pct":     drop(e, 0.25, rng),
        "drop_50pct":     drop(e, 0.50, rng),
        "drop_75pct":     drop(e, 0.75, rng),
    }


# ---------------------------------------------------------------- metrics ---

def m_ssim(a, b, win=7):
    return float(ssim(a, b, data_range=255, win_size=win))


def _dilate(x, r=DILATE_R):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(x, k) > 0


def m_iou(a, b, r=DILATE_R):
    A, B = _dilate(a, r), _dilate(b, r)
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else 1.0


def m_dice(a, b, r=DILATE_R):
    A, B = _dilate(a, r), _dilate(b, r)
    s = A.sum() + B.sum()
    return float(2 * (A & B).sum() / s) if s else 1.0


def m_chamfer(a, b):
    """Symmetric mean chamfer distance in pixels. Lower is better."""
    A, B = (a > 0), (b > 0)
    if not A.any() or not B.any():
        return float(np.hypot(*a.shape))          # worst case: image diagonal
    dtA = cv2.distanceTransform((~A).astype(np.uint8), cv2.DIST_L2, 3)
    dtB = cv2.distanceTransform((~B).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (dtA[B].mean() + dtB[A].mean()))


METRICS = {"ssim": m_ssim, "iou": m_iou, "dice": m_dice, "chamfer": m_chamfer}


# -------------------------------------------------------------------- run ---

def run(quick=False):
    rng = np.random.default_rng(SEED)
    corpus = build_corpus(quick)
    print(f"corpus: {len(corpus)} references "
          f"({sum(1 for n, _ in corpus if n.startswith('nat'))} natural, "
          f"{sum(1 for n, _ in corpus if n.startswith('syn'))} synthetic)")

    rows = []
    for i, (name, e) in enumerate(corpus):
        other = corpus[(i + 1) % len(corpus)][1]          # a different reference
        density = float((e > 0).mean())
        for cond, s in conditions(e, other, rng).items():
            r = {"reference": name, "condition": cond, "ink_density": density}
            for mn, fn in METRICS.items():
                r[mn] = fn(e, s)
            r["ssim_win11"] = m_ssim(e, s, win=11)
            rows.append(r)
    return rows


def summarise(rows):
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    out = {}
    for c in conds:
        sub = [r for r in rows if r["condition"] == c]
        out[c] = {m: {"mean": float(np.mean([r[m] for r in sub])),
                      "sd": float(np.std([r[m] for r in sub], ddof=1)),
                      "min": float(np.min([r[m] for r in sub])),
                      "max": float(np.max([r[m] for r in sub]))}
                  for m in list(METRICS) + ["ssim_win11"]}
        out[c]["n"] = len(sub)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    rows = run(a.quick)
    summ = summarise(rows)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "floor_raw.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(RESULTS, "floor_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)

    n = summ["perfect_copy"]["n"]
    print(f"\nn = {n} references per condition. Canny ({CANNY_LO},{CANNY_HI}), "
          f"{CANVAS}x{CANVAS}, SSIM win=7, IoU/Dice dilation r={DILATE_R}px.\n")
    hdr = f"{'condition':17s} {'SSIM %':>16s} {'IoU':>14s} {'Dice':>14s} {'Chamfer px':>12s}"
    print(hdr); print("-" * len(hdr))
    for c, v in summ.items():
        print(f"{c:17s} "
              f"{v['ssim']['mean']*100:7.2f} ± {v['ssim']['sd']*100:5.2f} "
              f"{v['iou']['mean']:7.3f} ± {v['iou']['sd']:4.3f} "
              f"{v['dice']['mean']:7.3f} ± {v['dice']['sd']:4.3f} "
              f"{v['chamfer']['mean']:8.2f}")

    blank = summ["blank_canvas"]["ssim"]
    perf = summ["perfect_copy"]["ssim"]
    print(f"\nSSIM dynamic range, perfect copy vs blank canvas: "
          f"{perf['mean']*100:.2f}% -> {blank['mean']*100:.2f}% "
          f"(spread {(perf['mean']-blank['mean'])*100:.2f} points)")
    print(f"Blank canvas SSIM range across references: "
          f"{blank['min']*100:.2f}% to {blank['max']*100:.2f}%")
    print(f"\nwrote {RESULTS}/floor_summary.json")


if __name__ == "__main__":
    main()
