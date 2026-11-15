# Experiment log

Reconstructed on 2026-10-30 from the version history of the authors' working repository, the
results files and the scripts. Dates are commit dates unless stated. Nothing here was written at
the time of the runs, so a date is the latest one the record allows, not a lab-notebook entry.

## Status of the analysis

The audit tests one question: does SSIM on sparse edge maps have a usable floor, that is, does an
empty canvas score below a correct drawing? It is exploratory. No analysis plan was fixed before
the results: `ssim_floor.py`, `analyze.py` and their outputs first entered version control
together on 2026-08-27; a note dated 2026-08-01 already reports the first floor result, so the
first run was on or before that date. The
acceptable and unacceptable groups, the SSIM window of 7, the 2 px dilation radius and Canny
thresholds 100/200 were all set on the same 65-reference corpus that gives the headline. There is
no held-out set.

## Runs

| Date | Run (script) | Result file in `results/` | Used in the paper for |
|---|---|---|---|
| 2026-08-27 | Metric floor, 65 references x 14 conditions = 910 comparisons, seed 20260801 (`ssim_floor.py`, `analyze.py`) | `floor_raw.json`, `floor_summary.json`, `analysis.json` | Headline: blank canvas 78.9%, inversion rates, AUC 0.766 against 0.984, r = -0.952 |
| 2026-08-27 | Edge operator timing, 18 images x 50 repetitions (`edge_benchmark.py`) | `edge_benchmark.json` | Table I |
| 2026-08-27 | Re-score of the three published test figures (`replicate_paper.py`) | `paper_replication.json` | Pilot-score table: flower and circle do not reproduce |
| 2026-08-27 | Acceptable-group variants (4) and dilation radius 1 to 5 px (`robustness.py`) | `robustness.json` | The four sensitivity checks |
| 2026-08-31 | LPIPS and DISTS over the same 910 pairs, plus cost (`learned_metrics.py`; weights fetched on first run) | `learned_raw.json`, `learned_timing.json` | Learned-metric AUCs, inversion rates and timings |
| 2026-10-23 | Counts, Wilson intervals, reference-level bootstrap (10,000 draws, seed 20261021), ties, AUC without the blank canvas, duplicate check, per-half results (`camera_numbers.py`) | `camera_numbers.json` | Intervals, duplicate disclosure, bootstrap interval for IoU minus SSIM |
| 2026-10-23 | Timing re-run on the same machine (outputs not in this release) | none | The "moved by up to 38%" disclosure |
| 2026-10-30 | Extra comparators and sweeps (`extra_metrics.py`, seed 20261030; 10,000 bootstrap draws for the main run, 2,000 per sweep cell): SSIM on dilated maps, SSIM on blurred maps (sigma 2 px), MS-SSIM (3 scales), MSE; Canny 30/90, 50/150, 100/200, 150/300, 200/400; sizes 128, 256 and 512 px with a fixed and a scaled dilation radius | `extra_metrics.json` | Tolerance controls, sweeps, within-reference AUCs |
| 2026-10-30 | Scorer tests, 18 hand-checked cases (`tests/test_scorers.py`) | none (pass or fail) | Found that the Chamfer distance via OpenCV's 3x3 mask reads 4.5% low on a 3 px gap; this moves Chamfer values, not rankings |
| 2026-10-30 | Crop and masking checks (`crop_and_masked.py`, main-corpus seed, 10,000 draws): the described bounding-box crop and rescale re-implemented (drawing only, and drawing plus reference; blank canvas kept whole or scored 0), SSIM over windows with ink in either image, Chamfer with an exact distance transform | `crop_and_masked.json` | Crop and inked-window sentences, Chamfer note |
| 2026-10-30 | Five fresh seeds (`fresh_seed.py`, 20261031 to 20261104), settings frozen, same 25 photograph maps | `fresh_seed.json` | Fresh-seed sentence |

Analyses beyond the headline, all reported in the paper or here: 4 acceptable-group variants,
5 dilation radii, an 11x11 window, 1 deduplicated corpus, 2 corpus halves, 4 extra comparators,
5 Canny settings, 3 sizes x 2 radius rules, 5 crop and mask variants, 1 exact Chamfer, 5 fresh
seeds. None was dropped after its result was seen.

## Not run, and why

- **CatSIM**: no implementation could be installed on the machine used, and a re-implementation
  from the paper would not be CatSIM. It remains the closest untested prior metric.
- **MicroSSIM, Boundary IoU, clDice**: not run.
- **Real hand-drawn sketches and human ratings**: not available for this study.
- **The deployed application's scoring path** (its crop and scale step): cannot be recovered from
  the published panels.

## Human ratings

A rating page was built on 2026-08-31. It has no ethics approval, and no response from it is used,
cited or analysed in the paper or in any file in this repository.

## Dropped or changed arms

- A synthetic colour wheel (an 18th scikit-image image) was dropped because its Canny map is empty.
  Disclosed in the paper.
- The flower and circle rows of the pilot-score table do not reproduce in the harness. Disclosed.
- 2026-10-30: the first full run of `extra_metrics.py` drew separate bootstrap resamples for each
  metric, which widened the IoU-minus-metric intervals. It was fixed to shared resamples
  (`boot_idx`) before any value was used; nothing from the first run is reported.
