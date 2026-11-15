#!/usr/bin/env python3
"""
Fresh-seed rerun. CPU only, offline, about 1 minute.

Settings frozen at the values of the main run (window 7, dilation 2 px, Canny 100/200, 256 px,
acceptable set unchanged). Only the seed changes: the 40 procedural drawings and every scribble /
ink-drop draw are regenerated. The 25 photograph edge maps are the SAME images (no new image
collection is available offline), so this is a fresh draw of the synthetic half and of the
degradations, not a held-out image set.

    python3 fresh_seed.py     # writes results/fresh_seed.json
"""
import json
import os
import warnings

import extra_metrics as em
import ssim_floor as sf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [20261031, 20261101, 20261102, 20261103, 20261104]
B = 2000


def main():
    metrics = {"ssim": lambda x, y: sf.m_ssim(x, y), "iou": lambda x, y: sf.m_iou(x, y)}
    out = {"frozen": "window 7, dilation 2 px, Canny 100/200, 256 px, acceptable set unchanged",
           "photograph_edge_maps": "same 25 images as the main run", "seeds": {}}
    orig = sf.SEED
    for seed in SEEDS:
        sf.SEED = seed
        corpus = sf.build_corpus()
        by = em.score_rows(corpus, metrics, seed=seed)
        refs = list(by)
        res = {"n_refs": len(refs)}
        for m in metrics:
            s, _ = em.summarise_metric(by, refs, m, B)
            res[m] = {"auc_pooled": s["auc_pooled"], "auc_within_ref": s["auc_within_ref"],
                      "inversions_k": {c: s["inversions"][c]["k"] for c in em.INV_CONDS}}
        out["seeds"][str(seed)] = res
        print(seed, res["n_refs"], {m: (round(res[m]["auc_pooled"], 3), round(res[m]["auc_within_ref"], 3),
                                       list(res[m]["inversions_k"].values())) for m in metrics})
    sf.SEED = orig
    with open(os.path.join(HERE, "results", "fresh_seed.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
