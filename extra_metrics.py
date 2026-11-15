#!/usr/bin/env python3
"""
Extra comparators and sensitivity sweeps.

CPU only, offline, about 2 minutes. Reuses the corpus and degradation battery of
ssim_floor.py unchanged and scores the same 910 pairs with metrics the main run
did not include:

  ssim_dil     SSIM computed on the edge maps dilated by the same 2 px as IoU
  ssim_blur    SSIM computed on Gaussian-blurred maps (sigma = 2 px)
  msssim       multi-scale SSIM, 3 scales (256, 128, 64 px), 7x7 uniform window
  mse          mean squared error on the 0/255 maps (lower is better)

plus the two already in the paper (ssim, iou) so every row of the output is on
the same footing. It then repeats the headline test under a sweep of the Canny
thresholds and of the working image size.

CatSIM (Thompson & Maitra 2020) is NOT run: the `catsim` package is not
installed here, no package may be installed, and a re-implementation from memory
would not be CatSIM. It stays listed as untested.

Primary outcome (named, per the revision): the WITHIN-REFERENCE inversion rate,
the share of references on which a blank canvas strictly outranks the stated
degraded-but-correct or wrong drawing, with a Wilson 95% interval. Secondary:
pooled AUC (every acceptable row against every unacceptable row, across
references, ties 0.5) with a reference-level cluster bootstrap, and the
within-reference AUC (acceptable vs unacceptable rows of the same reference,
averaged over references).

    python3 extra_metrics.py        # writes results/extra_metrics.json
"""
import json
import math
import os
import time
import warnings

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as sk_ssim

import ssim_floor as sf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SEED = 20261030
B = 10000
B_SWEEP = 2000

ACC = ["perfect_copy", "drop_25pct", "shift_1px", "shift_2px", "rotate_1deg", "rotate_2deg"]
UNACC = ["blank_canvas", "wrong_drawing", "random_scribble"]
INV_CONDS = ["shift_2px", "wrong_drawing", "random_scribble"]
HIB = {"ssim": True, "ssim_dil": True, "ssim_blur": True, "msssim": True,
       "mse": False, "iou": True}
BLUR_SIGMA = 2.0


# ---------------------------------------------------------------- metrics ---

def _ssim_terms(a, b, win=7, L=255.0):
    """Mean SSIM and mean contrast-structure term, same convention as scikit-image
    (uniform window, sample covariance, border cropped)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    n = win * win
    cov_norm = n / (n - 1)
    ux, uy = uniform_filter(a, win), uniform_filter(b, win)
    uxx, uyy, uxy = uniform_filter(a * a, win), uniform_filter(b * b, win), uniform_filter(a * b, win)
    vx, vy, vxy = cov_norm * (uxx - ux * ux), cov_norm * (uyy - uy * uy), cov_norm * (uxy - ux * uy)
    cs = (2 * vxy + C2) / (vx + vy + C2)
    lum = (2 * ux * uy + C1) / (ux * ux + uy * uy + C1)
    pad = (win - 1) // 2
    sl = (slice(pad, -pad), slice(pad, -pad))
    return float((lum * cs)[sl].mean()), float(cs[sl].mean())


def m_ssim_own(a, b, win=7):
    return _ssim_terms(a, b, win)[0]


def _down2(x):
    h, w = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
    x = x[:h, :w].astype(np.float64)
    return 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])


MS_W = np.array([0.0448, 0.2856, 0.3001])
MS_W = MS_W / MS_W.sum()


def m_msssim(a, b, win=7):
    """Multi-scale SSIM (Wang, Simoncelli, Bovik 2003) with 3 scales; the usual 5
    scales need >160 px at an 11-px window and do not fit 256 px with 7 px."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    val = 1.0
    for j, w in enumerate(MS_W):
        s, cs = _ssim_terms(a, b, win)
        term = s if j == len(MS_W) - 1 else cs
        val *= max(term, 0.0) ** w
        if j < len(MS_W) - 1:
            a, b = _down2(a), _down2(b)
    return float(val)


def m_ssim_dil(a, b, r=sf.DILATE_R):
    return m_ssim_own(sf._dilate(a, r).astype(np.uint8) * 255, sf._dilate(b, r).astype(np.uint8) * 255)


def m_ssim_blur(a, b, sigma=BLUR_SIGMA):
    ka = cv2.GaussianBlur(a.astype(np.float64), (0, 0), sigma)
    kb = cv2.GaussianBlur(b.astype(np.float64), (0, 0), sigma)
    return m_ssim_own(ka, kb)


def m_mse(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float((d * d).mean())


# ----------------------------------------------------------------- stats ----

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def score_rows(corpus, metrics, seed=sf.SEED):
    """Same loop and same rng order as ssim_floor.run(), with a chosen metric set."""
    rng = np.random.default_rng(seed)
    by = {}
    for i, (name, e) in enumerate(corpus):
        other = corpus[(i + 1) % len(corpus)][1]
        cd = sf.conditions(e, other, rng)
        by[name] = {c: {m: fn(e, s) for m, fn in metrics.items()} | {"ink": float((e > 0).mean())}
                    for c, s in cd.items()}
    return by


def _signed(by, refs, m, conds):
    s = 1.0 if HIB[m] else -1.0
    return np.array([[s * by[k][c][m] for c in conds] for k in refs])


def wins_matrix(by, refs, m):
    """W[i, j] = wins of reference i's acceptable rows over reference j's unacceptable rows."""
    P, N = _signed(by, refs, m, ACC), _signed(by, refs, m, UNACC)
    gt = (P[:, None, :, None] > N[None, :, None, :]).sum((2, 3))
    eq = (P[:, None, :, None] == N[None, :, None, :]).sum((2, 3))
    return gt + 0.5 * eq


def boot_idx(n, boot_n):
    """Reference-level resamples. Seeded by (n, boot_n) only, so every metric scored on the same
    references sees the SAME resamples and paired differences between metrics are valid."""
    return np.random.default_rng(SEED + n * 7 + boot_n).integers(0, n, (boot_n, n))


def summarise_metric(by, refs, m, boot_n, rng=None):
    n = len(refs)
    W = wins_matrix(by, refs, m)
    per = len(ACC) * len(UNACC)
    auc = float(W.sum() / (per * n * n))
    within = float(np.trace(W) / (per * n))
    boots, wboots = [], []
    for idx in boot_idx(n, boot_n):
        boots.append(W[np.ix_(idx, idx)].sum() / (per * n * n))
        wboots.append(np.trace(W[np.ix_(idx, idx)]) / (per * n))
    inv = {}
    for c in INV_CONDS:
        x = _signed(by, refs, m, ["blank_canvas", c])
        k = int((x[:, 0] > x[:, 1]).sum())
        t = int((x[:, 0] == x[:, 1]).sum())
        inv[c] = {"k": k, "n": n, "rate": k / n, "wilson95": wilson(k, n), "ties": t}
    return {"auc_pooled": auc, "auc_pooled_boot95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "auc_within_ref": within,
            "auc_within_ref_boot95": [float(np.percentile(wboots, 2.5)), float(np.percentile(wboots, 97.5))],
            "inversions": inv}, np.array(boots)


def blank_mean(by, refs, m):
    return float(np.mean([by[k]["blank_canvas"][m] for k in refs]))


# ------------------------------------------------------------------ main ----

def main():
    t0 = time.time()
    out = {"seed": SEED, "bootstrap_resamples": B, "blur_sigma_px": BLUR_SIGMA,
           "dilation_px": sf.DILATE_R, "msssim_scales": 3, "catsim": "not run: package not installed, no installs allowed"}

    # sanity: own SSIM equals scikit-image's on a few pairs
    corpus = sf.build_corpus()
    a, b = corpus[0][1], corpus[1][1]
    out["own_ssim_vs_skimage_maxabs"] = max(
        abs(m_ssim_own(x, y) - sk_ssim(x, y, data_range=255, win_size=7))
        for x, y in [(a, b), (a, a), (a, np.zeros_like(a)), (corpus[30][1], corpus[31][1])])

    metrics = {"ssim": lambda x, y: sf.m_ssim(x, y), "ssim_dil": m_ssim_dil, "ssim_blur": m_ssim_blur,
               "msssim": m_msssim, "mse": m_mse, "iou": lambda x, y: sf.m_iou(x, y)}
    by = score_rows(corpus, metrics)
    refs = list(by)
    rng = np.random.default_rng(SEED)

    main_out, boots = {}, {}
    for m in metrics:
        main_out[m], boots[m] = summarise_metric(by, refs, m, B, rng)
        main_out[m]["blank_canvas_mean"] = blank_mean(by, refs, m)
    out["main_65"] = main_out
    out["main_65_vs_iou_auc_pooled"] = {}
    for m in metrics:
        if m == "iou":
            continue
        d = boots["iou"] - boots[m]
        out["main_65_vs_iou_auc_pooled"][m] = {
            "iou_minus_metric": main_out["iou"]["auc_pooled"] - main_out[m]["auc_pooled"],
            "boot95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    # per-half (unit = reference)
    out["by_half"] = {}
    for half, pre in (("photograph_edge_maps", "nat:"), ("procedural_drawings", "syn:")):
        sub = [k for k in refs if k.startswith(pre)]
        out["by_half"][half] = {"n": len(sub)} | {
            m: summarise_metric(by, sub, m, 1000, rng)[0] for m in ("ssim", "ssim_dil", "ssim_blur", "msssim", "mse", "iou")}

    # ---- sweeps: Canny thresholds and working size, headline metrics only ----
    sweep_metrics = {"ssim": metrics["ssim"], "ssim_dil": m_ssim_dil, "ssim_blur": m_ssim_blur, "iou": metrics["iou"]}
    out["canny_sweep"] = {}
    lo0, hi0 = sf.CANNY_LO, sf.CANNY_HI
    for lo, hi in [(30, 90), (50, 150), (100, 200), (150, 300), (200, 400)]:
        sf.CANNY_LO, sf.CANNY_HI = lo, hi
        c2 = sf.build_corpus()
        by2 = score_rows(c2, sweep_metrics)
        r2 = list(by2)
        res = {"n_refs": len(r2),
               "mean_ink_pct": float(np.mean([by2[k]["perfect_copy"]["ink"] for k in r2]) * 100)}
        for m in sweep_metrics:
            s, _ = summarise_metric(by2, r2, m, B_SWEEP, rng)
            res[m] = {"auc_pooled": s["auc_pooled"], "auc_boot95": s["auc_pooled_boot95"],
                      "inversions_k": {c: s["inversions"][c]["k"] for c in INV_CONDS},
                      "blank_mean": blank_mean(by2, r2, m)}
        out["canny_sweep"][f"{lo}_{hi}"] = res
    sf.CANNY_LO, sf.CANNY_HI = lo0, hi0

    out["size_sweep"] = {}
    for n in (128, 256, 512):
        sf.CANVAS = n
        sf._to_gray_square.__defaults__ = (n,)
        c2 = sf.build_corpus()
        for tag, r in (("dilation_2px", 2), ("dilation_scaled", max(1, round(2 * n / 256)))):
            mets = {"ssim": metrics["ssim"], "iou": (lambda x, y, r=r: sf.m_iou(x, y, r)),
                    "ssim_dil": (lambda x, y, r=r: m_ssim_dil(x, y, r))}
            by2 = score_rows(c2, mets)
            r2 = list(by2)
            res = {"n_refs": len(r2), "dilation_px": r}
            for m in mets:
                s, _ = summarise_metric(by2, r2, m, B_SWEEP, rng)
                res[m] = {"auc_pooled": s["auc_pooled"], "auc_boot95": s["auc_pooled_boot95"],
                          "inversions_k": {c: s["inversions"][c]["k"] for c in INV_CONDS},
                          "blank_mean": blank_mean(by2, r2, m)}
            out["size_sweep"][f"{n}_{tag}"] = res
    sf.CANVAS = 256
    sf._to_gray_square.__defaults__ = (256,)

    out["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "extra_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"own vs skimage SSIM max abs diff: {out['own_ssim_vs_skimage_maxabs']:.2e}")
    print("\nmain, n=65 (inversions = blank outranks shift2 / wrong / scribble; pooled AUC [boot95]; within-ref AUC)")
    for m, v in main_out.items():
        k = [v["inversions"][c]["k"] for c in INV_CONDS]
        print(f"  {m:10s} inv {k}  pooled {v['auc_pooled']:.3f} {v['auc_pooled_boot95'][0]:.3f}-{v['auc_pooled_boot95'][1]:.3f}"
              f"  within {v['auc_within_ref']:.3f}  blank mean {v['blank_canvas_mean']:.3f}")
    print("\nCanny sweep")
    for k, v in out["canny_sweep"].items():
        print(" ", k, f"ink {v['mean_ink_pct']:.1f}%", {m: (round(v[m]['auc_pooled'], 3), v[m]['inversions_k']) for m in sweep_metrics})
    print("\nSize sweep")
    for k, v in out["size_sweep"].items():
        print(" ", k, {m: (round(v[m]["auc_pooled"], 3), v[m]["inversions_k"]) for m in v if isinstance(v[m], dict)})
    print(f"\nruntime {out['runtime_s']} s")


if __name__ == "__main__":
    main()
