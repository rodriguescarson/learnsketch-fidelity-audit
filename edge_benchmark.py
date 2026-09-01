#!/usr/bin/env python3
"""
Re-measure the four edge operators in ONE harness.

This settles the manuscript's edge-operator comparison. Its Table I
(Canny 0.01 s, Laplacian 0.25 s, Sobel 2.32 s, Prewitt 2.49 s) and its Fig. 4
(all four operators between 2 and 16 ms) disagree by 20x to 1000x and rank the
operators differently. Canny cannot be 232x faster than Sobel: it computes a
Sobel-equivalent gradient internally and then adds non-maximum suppression and
hysteresis on top. The likely cause is an optimised library call for Canny
timed against hand-written loops for the others.

Here every operator goes through OpenCV on identical input at identical
resolution, with warmup and repeated trials, and the post-processing each one
needs to produce a usable line drawing is included in its timing.

It also re-renders the operator outputs with normalisation, which settles
finding 3: in the manuscript's Table I the Laplacian panel is a flat grey field
and the Sobel and Prewitt panels are near-black, because raw signed gradient
magnitude was displayed without scaling.

    python3 edge_benchmark.py

Writes results/edge_benchmark.json, results/figure_edge.png, and the
normalised operator panels alongside the numbers.
"""

import json
import os
import platform
import subprocess
import time

import cv2
import numpy as np

from ssim_floor import CANNY_HI, CANNY_LO, CANVAS, _to_gray_square

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(HERE, "..", "latex", "figures")

REPS = 50          # timed repetitions per image per operator
WARMUP = 5         # discarded, so library init is not charged to the first operator


# ------------------------------------------------------------- operators ----
# Each returns a uint8 binary line drawing, so the comparison is like for like.

def op_canny(g):
    return cv2.Canny(g, CANNY_LO, CANNY_HI)


def _binarise(mag):
    """Normalise a gradient magnitude to 0-255 and threshold with Otsu."""
    m = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, b = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def op_sobel(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return _binarise(cv2.magnitude(gx, gy))


PREWITT_X = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
PREWITT_Y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)


def op_prewitt(g):
    gx = cv2.filter2D(g, cv2.CV_32F, PREWITT_X)
    gy = cv2.filter2D(g, cv2.CV_32F, PREWITT_Y)
    return _binarise(cv2.magnitude(gx, gy))


def op_laplacian(g):
    return _binarise(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)))


OPS = {"Canny": op_canny, "Sobel": op_sobel,
       "Prewitt": op_prewitt, "Laplacian": op_laplacian}


# ------------------------------------------------------------------ host ----

def host():
    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() or cpu
    except Exception:                                     # noqa: BLE001
        pass
    return {"cpu": cpu, "platform": platform.platform(),
            "python": platform.python_version(), "opencv": cv2.__version__,
            "threads": cv2.getNumThreads()}


# ------------------------------------------------------------------- run ----

def main():
    from skimage import data
    names = ["astronaut", "camera", "cat", "chelsea", "clock", "coffee", "coins",
             "horse", "moon", "page", "rocket", "text", "brick", "grass", "gravel",
             "cell", "colorwheel", "immunohistochemistry"]
    imgs = []
    for n in names:
        try:
            imgs.append(_to_gray_square(getattr(data, n)()))
        except Exception:                                 # noqa: BLE001
            continue

    h = host()
    print(f"host: {h['cpu']}")
    print(f"      OpenCV {h['opencv']}, {h['threads']} threads, Python {h['python']}")
    print(f"      {len(imgs)} images at {CANVAS}x{CANVAS}, {REPS} reps each "
          f"(+{WARMUP} warmup)\n")

    # warmup, charged to nobody
    for f in OPS.values():
        for _ in range(WARMUP):
            f(imgs[0])

    per_op = {}
    for name, f in OPS.items():
        times = []
        for g in imgs:
            for _ in range(REPS):
                t0 = time.perf_counter()
                f(g)
                times.append((time.perf_counter() - t0) * 1000.0)   # ms
        a = np.array(times)
        per_op[name] = {"mean_ms": float(a.mean()), "sd_ms": float(a.std(ddof=1)),
                        "median_ms": float(np.median(a)),
                        "p95_ms": float(np.percentile(a, 95)),
                        "min_ms": float(a.min()), "max_ms": float(a.max()),
                        "n": int(a.size)}

    print(f"{'operator':11s} {'mean (ms)':>16s} {'median':>8s} {'p95':>8s} {'x Canny':>8s}")
    print("-" * 56)
    base = per_op["Canny"]["median_ms"]
    for name, v in sorted(per_op.items(), key=lambda kv: kv[1]["median_ms"]):
        print(f"{name:11s} {v['mean_ms']:8.3f} ± {v['sd_ms']:5.3f} "
              f"{v['median_ms']:8.3f} {v['p95_ms']:8.3f} {v['median_ms']/base:8.2f}")

    out = {"host": h, "reps": REPS, "warmup": WARMUP, "n_images": len(imgs),
           "resolution": CANVAS, "canny_thresholds": [CANNY_LO, CANNY_HI],
           "operators": per_op}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "edge_benchmark.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ---- re-render the operator panels on the manuscript's reference image ---
    # Timing above uses a square 256x256 so every operator sees identical work.
    # The PANELS must instead keep the reference photograph's native aspect
    # ratio (492x248), or the subject renders vertically stretched and the rows
    # of Table I no longer match the reference row.
    ref = cv2.imread(os.path.join(FIGDIR, "tab1-reference.png"), cv2.IMREAD_COLOR)
    if ref is not None:
        h0, w0 = ref.shape[:2]
        w = 720
        h = int(round(w * h0 / w0))
        g = cv2.cvtColor(cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        for name, f in OPS.items():
            cv2.imwrite(os.path.join(FIGDIR, f"tab1n-{name.lower()}.png"), f(g))
        print(f"\nre-rendered 4 operator panels at {w}x{h} "
              f"(source aspect {w0/h0:.3f}) into latex/figures/")

    # ---- figure ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = sorted(per_op, key=lambda k: per_op[k]["median_ms"])
        fig, ax = plt.subplots(figsize=(5.0, 2.5))
        ax.bar(order, [per_op[k]["median_ms"] for k in order],
               yerr=[per_op[k]["sd_ms"] for k in order],
               color=["#c0392b" if k == "Canny" else "#7f8c8d" for k in order],
               error_kw=dict(lw=.8, ecolor="#34495e"), width=.62)
        ax.set_ylabel("median time (ms)", fontsize=9)
        ax.set_title(f"Edge operators, one harness, {CANVAS}x{CANVAS}, "
                     f"n={len(imgs)}x{REPS}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "figure_edge.png"), dpi=400)
        fig.savefig(os.path.join(FIGDIR, "fig4-edge-benchmark.png"), dpi=400)
        print("wrote figure_edge.png and latex/figures/fig4-edge-benchmark.png")
    except Exception as e:                                # noqa: BLE001
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
