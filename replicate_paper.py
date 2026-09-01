#!/usr/bin/env python3
"""
Replicate the manuscript's pilot table on its own images, then score the
controls it never ran.

The Table II rows are 4-panel strips (INPUT A | INPUT B | DIFFERENCE | RESULT)
recovered from inside the .doc. INPUT A is the Canny reference, INPUT B is the
drawing that was scored. The manuscript reports 97.95%, 98.72% and 93.11%.

This script asks three questions:
  1. Does the reported score reproduce from the published panels?
  2. What does a BLANK CANVAS score against the same references?
  3. Do overlap-based metrics (IoU, Dice, Chamfer) behave better?

    python3 replicate_paper.py
"""

import json
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ssim_floor import (m_chamfer, m_dice, m_iou, m_ssim, rotate, shift)

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "data", "paper_pairs")
RESULTS = os.path.join(HERE, "results")

# What the manuscript prints in Table II, in row order.
REPORTED = {"flower": 97.95, "square": 98.72, "circle": 93.11}


def load_binary(path):
    """Panel -> binary edge map. Any non-black pixel counts as ink."""
    a = cv2.imread(path, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return ((g > 40).astype(np.uint8)) * 255


def main():
    rows = []
    print(f"{'case':9s} {'reported':>9s} {'SSIM(A,B)':>10s} {'SSIM(A,blank)':>14s} "
          f"{'ink%':>6s} {'IoU(A,B)':>9s} {'IoU(A,blank)':>13s} {'Chamfer':>8s}")
    print("-" * 88)

    for case in ("flower", "square", "circle"):
        A = load_binary(os.path.join(PAIRS, f"{case}_inputA.png"))
        B = load_binary(os.path.join(PAIRS, f"{case}_inputB.png"))
        B = cv2.resize(B, A.shape[::-1], interpolation=cv2.INTER_NEAREST)
        blank = np.zeros_like(A)

        r = {
            "case": case,
            "reported_pct": REPORTED[case],
            "ssim_AB": m_ssim(A, B) * 100,
            "ssim_A_blank": m_ssim(A, blank) * 100,
            "ssim_A_scribbleshift": m_ssim(A, shift(A, 5, 5)) * 100,
            "ssim_A_rot5": m_ssim(A, rotate(A, 5)) * 100,
            "ink_pct": float((A > 0).mean() * 100),
            "iou_AB": m_iou(A, B),
            "iou_A_blank": m_iou(A, blank),
            "dice_AB": m_dice(A, B),
            "chamfer_AB": m_chamfer(A, B),
        }
        rows.append(r)
        print(f"{case:9s} {r['reported_pct']:8.2f}% {r['ssim_AB']:9.2f}% "
              f"{r['ssim_A_blank']:13.2f}% {r['ink_pct']:5.2f}% "
              f"{r['iou_AB']:9.3f} {r['iou_A_blank']:13.3f} {r['chamfer_AB']:8.2f}")

    print()
    for r in rows:
        gap = r["ssim_AB"] - r["ssim_A_blank"]
        print(f"{r['case']:9s} drawing scores {gap:+.2f} points above an empty canvas; "
              f"a correct drawing shifted 5 px scores {r['ssim_A_scribbleshift']:.2f}% "
              f"({'below' if r['ssim_A_scribbleshift'] < r['ssim_A_blank'] else 'above'} the empty canvas)")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "paper_replication.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {RESULTS}/paper_replication.json")


if __name__ == "__main__":
    main()
