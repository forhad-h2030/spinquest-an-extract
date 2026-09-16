#!/usr/bin/env python3
"""Raw per-class A_N vs probability threshold, with an MC-purity overlay.

    python -m an.an_vs_threshold
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # lets this run as `python3 an_vs_threshold.py` too, not just `-m`

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from an import asymmetry as asy
from an import purity as pur


def an_at_threshold(dimu_px, spin, qual_mask, probs_by_boot, class_idx, t):
    ans, das = [], []
    for probs in probs_by_boot.values():
        sel = qual_mask & (probs.argmax(1) == class_idx) & (probs[:, class_idx] >= t)
        uL = int((sel & (spin == 1) & (dimu_px > 0)).sum())
        uR = int((sel & (spin == 1) & (dimu_px <= 0)).sum())
        dL = int((sel & (spin == 0) & (dimu_px > 0)).sum())
        dR = int((sel & (spin == 0) & (dimu_px <= 0)).sum())
        a_, da_ = asy.an(uL, uR, dL, dR)
        ans.append(a_); das.append(da_)
    stat = float(np.nanmean(das))
    model = float(np.nanstd(ans, ddof=1)) if len(ans) > 1 else 0.0
    return float(np.nanmean(ans)), stat, model


def main():
    dimu_px, spin, _pt, qual_mask, probs, _boots = asy.load_exp_data(
        asy.LO, asy.HI, asy.X_CUT, asy.Y_CUT, asy.PZ_CUT)
    purity_table = pur.compute_purity_table(asy.LO, asy.HI)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for k, (c, ax) in enumerate(zip(pur.CLS, axes.flat)):
        an_m, st_m, pu_m = [], [], []
        for t in pur.THR:
            a_, s_, _ = an_at_threshold(dimu_px, spin, qual_mask, probs, k, t)
            an_m.append(a_); st_m.append(s_)
            pu_m.append(purity_table[c][float(t)][0] * 100)

        ax.errorbar(pur.THR, an_m, yerr=st_m, fmt="o-", ms=4, capsize=3, label="A_N (raw, tag)")
        ax.axhline(0, ls="--", c="gray", lw=1)
        ax.set_title(c); ax.set_xlabel("prob threshold"); ax.set_ylabel("A_N")
        ax.grid(alpha=0.3, ls="--")

        ax2 = ax.twinx()
        ax2.plot(pur.THR, pu_m, "k--", label="MC purity")
        ax2.set_ylabel("purity [%]"); ax2.set_ylim(0, 105)

        if k == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8)

    fig.suptitle("Raw per-class A_N vs threshold (MC purity overlay)")
    fig.tight_layout()
    out = REPO / "output" / "an_vs_threshold_by_class_2_6_pyroot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
