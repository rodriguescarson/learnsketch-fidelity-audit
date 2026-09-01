# Does SSIM have a floor on line art?

Reproduction code for **"Fidelity Assessment for Digital Sketching: Why SSIM Fails on Line Art,
and What to Use Instead."**

The paper reports a negative result: the Structural Similarity Index, deployed as the fidelity
score in a mobile sketching tutor, cannot reliably rank a correct drawing above an **empty
canvas**. Over 65 references a blank canvas averages **78.9% SSIM** and reaches **98.0%**,
outranking a correct drawing displaced by two pixels on 93.8% of references. A learner who draws
nothing is told they scored well.

Everything in the paper comes from the scripts here. Nothing is hand-computed.

## Run it

```bash
pip install -r requirements.txt

python3 ssim_floor.py        # the corpus + 14-condition ladder, n=65    (~1 min)
python3 learned_metrics.py   # LPIPS + DISTS over the same grid + cost   (~3 min)
python3 analyze.py           # mechanism, AUC, ordering failures, figure
python3 edge_benchmark.py    # the four edge operators in one harness
python3 replicate_paper.py   # the manuscript's own three test figures
python3 robustness.py        # dilation sweep + sensitivity checks
```

Run `learned_metrics.py` before `analyze.py` and the AUC table gains LPIPS and DISTS
automatically; rows join on `(reference, condition)` because both scripts walk the same seeded
corpus in the same order.

**The corpus is not distributed, because it does not need to be.** It is 25 Canny maps of images
bundled with scikit-image (each CC0 or public domain) plus 40 procedurally generated line drawings
from a generator seeded at **20260801**, so `ssim_floor.py` reconstructs it exactly. References are
resized to 256x256 and converted to grayscale before Canny at thresholds (100, 200) with a 3x3
Sobel aperture.

## What it finds

**SSIM has no floor on sparse line art.** A blank canvas is not the worst case; it sits
mid-ranking. Share of the 65 references where a blank canvas *outscores* the stated condition:

| Compared against | SSIM | IoU (dilated, r=2px) |
|---|---|---|
| Correct drawing shifted 2 px | 93.8% | **0%** |
| Correct drawing shifted 5 px | 96.9% | **0%** |
| A drawing of the wrong reference | 96.9% | **0%** |
| A random scribble | 98.5% | **0%** |

**The cause is sparsity, and it is measurable.** Over windows empty in both images SSIM's
luminance and contrast terms both approach 1, so those windows report agreement regardless of what
was drawn. Blank-canvas SSIM tracks reference ink density at **r = -0.952** (n = 65): on references
with under 2% ink, which is typical of a line drawing, drawing nothing scores 91.5% on average.

**Six metrics, measured on the same a priori split.** AUC for ranking a roughly-correct drawing
above no drawing, the wrong drawing, or a scribble, with cost per 256x256 comparison:

| Metric | AUC | Blank above a wrong drawing | Above a scribble | ms/pair |
|---|---|---|---|---|
| SSIM (7x7) | 0.766 | 96.9% | 98.5% | 2.76 |
| SSIM (11x11) | 0.787 | n/a | n/a | n/a |
| **IoU, dilated r=2px** | **0.984** | **0.0%** | **0.0%** | **0.10** |
| Dice, dilated r=2px | 0.984 | 0.0% | 0.0% | n/a |
| Chamfer distance | 0.989 | 0.0% | 0.0% | n/a |
| LPIPS (AlexNet) | **0.995** | 24.6% | 32.3% | 10.63 |
| DISTS | **0.994** | 13.8% | 3.1% | 190.37 |

The learned metrics do **not** inherit SSIM's failure and edge past IoU on AUC. Two things still
make IoU the recommendation: they invert the ordering on a quarter and an eighth of references
where IoU never does, and they cost 110x and 1,900x more while returning an unbounded distance
rather than a number a learner can read as a percentage.

**Robustness.** Dropping the perfect copy moves IoU 0.984 to 0.981 and SSIM 0.766 to 0.719.
Folding the five excluded middle conditions in as acceptable, the most demanding version of the
test, gives 0.952 against 0.707. Sweeping the dilation radius from 1 to 5 px gives 0.986, 0.984,
0.982, 0.980, 0.979: the 2 px radius the paper uses is **not** the maximising value, which is the
evidence it was not selected on the evaluation set.

## Reproducibility notes

Deterministic: fixed seed 20260801, CPU inference in evaluation mode, offline at run time after
the first fetch of pretrained weights. `results/` holds every number the paper reports, including
the raw per-pair scores for all 910 learned-metric comparisons.

Two traps recorded so nobody repeats them:

- `piq.DISTS()` defaults to `reduction="mean"` and silently returns **one number for the whole
  batch**. The first run looked plausible and was wrong. `learned_metrics.py` passes
  `reduction="none"` and asserts the returned length matches the number of conditions.
- The `replicate_paper.py` figures are low-resolution panels recovered from a Word document. The
  square reproduces to within 0.45 points; the flower and circle differ by about 20, because the
  crop and scale-normalisation the application applied cannot be recovered from the panels. The
  audit does not depend on those three numbers, since it runs on the 65-reference corpus instead.

## Limitation, stated plainly

**No human participants.** These results establish which metric better separates good drawings
from bad ones under controlled degradation. They do not establish that either tracks how a *person*
rates a drawing, which is how SSIM itself was validated in Wang et al. 2004. That study is the
natural follow-up.

## License

MIT, see `LICENSE`.
