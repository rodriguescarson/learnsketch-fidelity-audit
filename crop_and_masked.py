#!/usr/bin/env python3
"""
Crop and masking checks. CPU only, offline, about 1 minute.

Two checks on the scoring step, both on the same 910 pairs as extra_metrics.py:

1. The deployed pipeline crops the drawing to its bounding box and rescales it to the reference
   size before SSIM (paper Sec. III-E). The original crop code is not recoverable, so this is a
   re-implementation of the described step, NOT the app's code. Variants:
     crop_draw    drawing cropped to its ink bounding box, resized to the reference size
     crop_both    drawing AND reference cropped to their own ink bounding boxes and resized
   A blank canvas has no bounding box; what the app does is unknown, so two conventions:
     _keep   the blank canvas is left whole (no crop possible)
     _zero   the blank canvas is scored 0 (an empty drawing is rejected outright)
   Resizing uses nearest neighbour (stays binary); interpolation was not varied in the app either.
2. A sparsity-aware SSIM: the mean of the SSIM map over windows where at least one image has ink
   (a window of 7x7 touching any nonzero pixel), so windows empty in both images cannot report
   agreement. This is a mask guard, not CatSIM.

3. Chamfer with an exact distance transform next to the original 3x3-mask one (X13 check).

    python3 crop_and_masked.py     # writes results/crop_and_masked.json
"""
import json
import os
import time
import warnings

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

import extra_metrics as em
import ssim_floor as sf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
B = 10000


def _crop(x):
    ys, xs = np.nonzero(x)
    if len(ys) == 0:
        return None
    c = x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(c, x.shape[::-1], interpolation=cv2.INTER_NEAREST)


def _make_crop(both, blank):
    def f(ref, drw):
        d = _crop(drw)
        if d is None:
            if blank == "zero":
                return 0.0
            d = drw
            r = ref
        else:
            r = _crop(ref) if both else ref
            if r is None:
                r = ref
        return em.m_ssim_own(r, d)
    return f


def m_ssim_active(a, b, win=7, L=255.0):
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    n = win * win
    k = n / (n - 1)
    ux, uy = uniform_filter(a, win), uniform_filter(b, win)
    vx = k * (uniform_filter(a * a, win) - ux * ux)
    vy = k * (uniform_filter(b * b, win) - uy * uy)
    vxy = k * (uniform_filter(a * b, win) - ux * uy)
    s = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    ink = uniform_filter(((a > 0) | (b > 0)).astype(np.float64), win) > 1e-9
    pad = (win - 1) // 2
    sl = (slice(pad, -pad), slice(pad, -pad))
    m = ink[sl]
    return float(s[sl][m].mean()) if m.any() else 1.0


def m_chamfer_exact(a, b):
    """As ssim_floor.m_chamfer but with the exact Euclidean distance transform (the original uses
    cv2's 3x3 mask, about 4.5% low on a 3 px gap). Empty canvas gets the image diagonal."""
    A, B_ = (a > 0), (b > 0)
    if not A.any() or not B_.any():
        return float(np.hypot(*a.shape))
    dtA = cv2.distanceTransform((~A).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dtB = cv2.distanceTransform((~B_).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(0.5 * (dtA[B_].mean() + dtB[A].mean()))


def main():
    t0 = time.time()
    corpus = sf.build_corpus()
    metrics = {"ssim": lambda x, y: sf.m_ssim(x, y),
               "ssim_active": m_ssim_active,
               "crop_draw_keep": _make_crop(False, "keep"), "crop_draw_zero": _make_crop(False, "zero"),
               "crop_both_keep": _make_crop(True, "keep"), "crop_both_zero": _make_crop(True, "zero"),
               "iou": lambda x, y: sf.m_iou(x, y),
               "chamfer_3x3": sf.m_chamfer, "chamfer_exact": m_chamfer_exact}
    for m in metrics:
        em.HIB[m] = not m.startswith("chamfer")
    by = em.score_rows(corpus, metrics)
    refs = list(by)
    out = {"n_refs": len(refs), "note": "re-implementation of the described crop step, not the app's code",
           "bootstrap_resamples": B, "metrics": {}}
    for m in metrics:
        s, _ = em.summarise_metric(by, refs, m, B)
        s["blank_canvas_mean"] = em.blank_mean(by, refs, m)
        s["condition_means"] = {c: float(np.mean([by[k][c][m] for k in refs])) for c in by[refs[0]] if c != "ink"}
        out["metrics"][m] = s
    out["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "results", "crop_and_masked.json"), "w") as f:
        json.dump(out, f, indent=1)
    for m, v in out["metrics"].items():
        k = [v["inversions"][c]["k"] for c in em.INV_CONDS]
        print(f"{m:15s} inv(shift2,wrong,scribble) {k}  pooled {v['auc_pooled']:.3f}  within {v['auc_within_ref']:.3f}"
              f"  blank {v['blank_canvas_mean']:.3f}")


if __name__ == "__main__":
    main()
