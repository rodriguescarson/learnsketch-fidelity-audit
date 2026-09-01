#!/usr/bin/env python3
"""
Do learned perceptual metrics survive the same test SSIM fails?

Section V of the manuscript used to concede that LPIPS and DISTS were named but
never measured, so the IoU recommendation rested on a comparison against SSIM,
Dice and Chamfer only. This script closes that gap. It rebuilds exactly the
corpus and degradation ladder of ssim_floor.py (same seed, same iteration
order, so the sketches are identical bit for bit) and scores every pair with:

    LPIPS (AlexNet backbone, Zhang et al. 2018)
    DISTS (Ding et al. 2022)

Both are distances: lower means more similar. Both were trained and validated
on natural images, which is the property under test here, since the reference
and the drawing are sparse binary edge maps and are nothing like natural
images.

Writes results/learned_raw.json in the same row schema as floor_raw.json, so
analyze.py merges it automatically and reports LPIPS and DISTS AUC alongside
SSIM, IoU, Dice and Chamfer, computed by the same function over the same
a priori acceptable/unacceptable split.

    python3 learned_metrics.py            # full run, 65 references
    python3 learned_metrics.py --quick    # smoke test

Deterministic: fixed seed, CPU inference, models in eval mode. Needs torch,
torchvision, lpips and piq; pretrained weights are fetched on first run and
cached, so the first run needs the network and later runs do not.
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from ssim_floor import SEED, RESULTS, build_corpus, conditions  # noqa: E402

DEVICE = torch.device("cpu")   # CPU, not MPS: reproducibility beats speed here


def _to_batch(images):
    """(N,256,256) uint8 edge maps -> (N,3,256,256) float in [0,1]."""
    a = np.stack(images).astype(np.float32) / 255.0
    t = torch.from_numpy(a).unsqueeze(1).repeat(1, 3, 1, 1)
    return t.to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import lpips
    import piq

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)

    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(DEVICE).eval()
    # reduction="none": piq defaults to a batch mean, which silently
    # collapses all 14 conditions into one number.
    dists_net = piq.DISTS(reduction="none").to(DEVICE).eval()

    rng = np.random.default_rng(SEED)          # same stream as ssim_floor.run()
    corpus = build_corpus(args.quick)
    print(f"corpus: {len(corpus)} references, 14 conditions each")

    rows = []
    for i, (name, e) in enumerate(corpus):
        other = corpus[(i + 1) % len(corpus)][1]
        density = float((e > 0).mean())
        conds = conditions(e, other, rng)       # identical to ssim_floor.py
        names = list(conds)
        sketches = _to_batch([conds[c] for c in names])
        ref = _to_batch([e] * len(names))

        # LPIPS wants [-1,1]; piq's DISTS wants [0,1].
        d_lpips = lpips_net(ref * 2 - 1, sketches * 2 - 1).flatten().cpu().numpy()
        d_dists = dists_net(ref, sketches).flatten().cpu().numpy()

        assert len(d_lpips) == len(names) and len(d_dists) == len(names), (
            f"metric returned {len(d_lpips)}/{len(d_dists)} values for "
            f"{len(names)} conditions; check the reduction setting")
        for c, dl, dd in zip(names, d_lpips, d_dists):
            rows.append({"reference": name, "condition": c,
                         "ink_density": density,
                         "lpips": float(dl), "dists": float(dd)})
        print(f"  [{i+1:>2}/{len(corpus)}] {name}", end="\r")

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "learned_raw.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)

    # ---- cost, so the paper's "costs no more to compute" is measured ------
    import time
    from ssim_floor import m_ssim, m_iou
    ref_img, sk_img = corpus[0][1], conditions(corpus[0][1], corpus[1][1],
                                               np.random.default_rng(SEED))["shift_2px"]
    r1, s1 = _to_batch([ref_img]), _to_batch([sk_img])
    timing = {}
    for label, fn in [("ssim", lambda: m_ssim(ref_img, sk_img)),
                      ("iou", lambda: m_iou(ref_img, sk_img)),
                      ("lpips", lambda: float(lpips_net(r1 * 2 - 1, s1 * 2 - 1))),
                      ("dists", lambda: float(dists_net(r1, s1)))]:
        for _ in range(5):
            fn()                                   # warm up
        t0 = time.perf_counter()
        n = 50
        for _ in range(n):
            fn()
        timing[label] = (time.perf_counter() - t0) / n * 1000.0
    with open(os.path.join(RESULTS, "learned_timing.json"), "w") as f:
        json.dump({"ms_per_pair_cpu_256x256": timing, "repetitions": 50}, f, indent=1)

    conds = list(dict.fromkeys(r["condition"] for r in rows))
    print("\n\nMean distance per condition (lower = judged more similar):\n")
    hdr = f"{'condition':17s} {'LPIPS':>16s} {'DISTS':>16s}"
    print(hdr); print("-" * len(hdr))
    for c in conds:
        sub = [r for r in rows if r["condition"] == c]
        l = [r["lpips"] for r in sub]; d = [r["dists"] for r in sub]
        print(f"{c:17s} {np.mean(l):8.4f} ± {np.std(l, ddof=1):5.4f} "
              f"{np.mean(d):8.4f} ± {np.std(d, ddof=1):5.4f}")
    print(f"\nCost per pair, one 256x256 comparison, CPU, mean of 50:")
    base = timing["ssim"]
    for k, v in timing.items():
        print(f"   {k:6s} {v:9.3f} ms   {v/base:8.1f}x SSIM")

    print(f"\nwrote {path}  ({len(rows)} rows)")
    print("run  python3 analyze.py  for the AUC comparison against SSIM and IoU")


if __name__ == "__main__":
    main()
