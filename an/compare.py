#!/usr/bin/env python3
"""Compare the two independent A_N extraction methods on one plot: ML-tag
background-corrected (asymmetry.py) vs RooFit mass fit (massfit.py,
no ML model). massfit needs PyROOT.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # lets this run as `python3 compare.py` too, not just `-m`

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from an import asymmetry, massfit


def main():
    an_result = asymmetry.extract_corrected_an(verbose=False)
    mf_result = massfit.extract_massfit_an()

    labels = ["ML tag\n(corrected)", "Mass fit"]
    vals = [an_result["A_N_corr"], mf_result["A_N"]]
    errs = [an_result["dA_N_corr_stat"], mf_result["dA_N"]]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar([0, 1], vals, yerr=errs, fmt="o", ms=10, capsize=6, lw=2)
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(r"$A_N\ (J/\psi)$")
    ax.set_title(r"$A_N$: ML-tag correction vs mass fit")
    ax.grid(alpha=0.3, ls="--")
    for x, v, e in zip([0, 1], vals, errs):
        ax.annotate(f"{v:+.3f} +/- {e:.3f}", (x, v), textcoords="offset points",
                    xytext=(12, 0), va="center")

    out = REPO / "output" / "an_method_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
