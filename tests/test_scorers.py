"""Hand-checked cases for the scoring code.   python3 -m unittest discover -s tests -v

Run from the repository root. Every expected value below is worked out by hand in the comment above it.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import analyze            # noqa: E402
import extra_metrics as em  # noqa: E402
import ssim_floor as sf   # noqa: E402


def canvas(*pixels, n=32):
    c = np.zeros((n, n), np.uint8)
    for y, x in pixels:
        c[y, x] = 255
    return c


class Overlap(unittest.TestCase):
    def test_empty_empty_is_one(self):
        z = np.zeros((16, 16), np.uint8)
        self.assertEqual(sf.m_iou(z, z), 1.0)
        self.assertEqual(sf.m_dice(z, z), 1.0)

    def test_blank_vs_drawing_is_zero(self):
        self.assertEqual(sf.m_iou(canvas((10, 10)), np.zeros((32, 32), np.uint8)), 0.0)
        self.assertEqual(sf.m_dice(canvas((10, 10)), np.zeros((32, 32), np.uint8)), 0.0)

    def test_identical_is_one(self):
        a = canvas((5, 5), (6, 7))
        self.assertEqual(sf.m_iou(a, a), 1.0)

    def test_two_blocks_one_column_apart_r0(self):
        # A = 2x2 block, B = same block moved one column right. |A&B| = 2, |A|B| = 6.
        # IoU = 2/6 = 1/3, Dice = 2*2/(4+4) = 1/2.
        a = canvas((10, 10), (10, 11), (11, 10), (11, 11))
        b = canvas((10, 11), (10, 12), (11, 11), (11, 12))
        self.assertAlmostEqual(sf.m_iou(a, b, r=0), 1 / 3)
        self.assertAlmostEqual(sf.m_dice(a, b, r=0), 0.5)

    def test_far_apart_beyond_tolerance_is_zero(self):
        # two single pixels 20 px apart, radius 2 discs cannot touch
        self.assertEqual(sf.m_iou(canvas((5, 5)), canvas((5, 25))), 0.0)

    def test_within_tolerance_overlaps(self):
        self.assertGreater(sf.m_iou(canvas((5, 5)), canvas((5, 7))), 0.0)


class Chamfer(unittest.TestCase):
    def test_empty_is_diagonal(self):
        z = np.zeros((32, 32), np.uint8)
        self.assertAlmostEqual(sf.m_chamfer(canvas((3, 3)), z), float(np.hypot(32, 32)))

    def test_identical_is_zero(self):
        a = canvas((3, 3), (9, 9))
        self.assertEqual(sf.m_chamfer(a, a), 0.0)

    def test_two_pixels_three_apart(self):
        # each pixel is 3 px from the other: exact symmetric mean 3. ssim_floor.m_chamfer uses
        # cv2.distanceTransform(DIST_L2, mask 3), a 3x3-mask approximation that returns 2.865 here
        # (-4.5%). Found by this test; it moves Chamfer values, not any ranking in the paper.
        self.assertAlmostEqual(sf.m_chamfer(canvas((10, 10)), canvas((10, 13))), 3.0, delta=0.15)


class Ssim(unittest.TestCase):
    def test_identical_and_blank_blank_are_one(self):
        a = canvas((5, 5), (6, 6), (7, 7))
        self.assertAlmostEqual(sf.m_ssim(a, a), 1.0)
        z = np.zeros_like(a)
        self.assertAlmostEqual(sf.m_ssim(z, z), 1.0)

    def test_empty_windows_dominate(self):
        # one lit pixel in a 32x32 canvas; a blank canvas still scores > 0.9 (the paper's mechanism)
        a = canvas((16, 16))
        self.assertGreater(sf.m_ssim(a, np.zeros_like(a)), 0.9)

    def test_own_implementation_matches_scikit_image(self):
        rng = np.random.default_rng(0)
        a = (rng.random((40, 40)) > 0.9).astype(np.uint8) * 255
        b = (rng.random((40, 40)) > 0.9).astype(np.uint8) * 255
        self.assertAlmostEqual(em.m_ssim_own(a, b), sf.m_ssim(a, b), places=9)

    def test_msssim_identical_is_one(self):
        a = (np.random.default_rng(1).random((256, 256)) > 0.95).astype(np.uint8) * 255
        self.assertAlmostEqual(em.m_msssim(a, a), 1.0, places=9)


class AucAndIntervals(unittest.TestCase):
    def test_auc_hand_case(self):
        # pos [3,4], neg [1,4]: 3>1 win, 3<4 loss, 4>1 win, 4=4 tie (0.5): (1+0+1+0.5)/4 = 0.625
        self.assertAlmostEqual(analyze.auc([3, 4], [1, 4]), 0.625)

    def test_auc_ties_only_is_half(self):
        self.assertAlmostEqual(analyze.auc([1, 1], [1, 1]), 0.5)

    def test_wilson_zero_of_ten(self):
        # k=0: lower 0, upper = z^2/(n+z^2) = 3.841459/13.841459 = 0.27753
        lo, hi = em.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 0.277534, places=5)

    def test_wilson_61_of_65_matches_released_file(self):
        lo, hi = em.wilson(61, 65)
        self.assertAlmostEqual(lo, 0.8521785, places=6)
        self.assertAlmostEqual(hi, 0.9758109, places=6)

    def test_ties_are_not_inversions(self):
        # blank ties the condition at 0 on 2 of 3 refs and is strictly better on 1: k=1, ties=2
        by = {f"r{i}": {"blank_canvas": {"iou": v}, "c": {"iou": 0.0}} for i, v in enumerate([0.0, 0.0, 0.1])}
        x = em._signed(by, list(by), "iou", ["blank_canvas", "c"])
        self.assertEqual(int((x[:, 0] > x[:, 1]).sum()), 1)
        self.assertEqual(int((x[:, 0] == x[:, 1]).sum()), 2)


if __name__ == "__main__":
    unittest.main()
