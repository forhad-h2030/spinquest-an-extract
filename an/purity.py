#!/usr/bin/env python3
"""Purity vs probability threshold, per class -- from the same DATA_PRIOR-
resampled synthetic MC sample mc_purity.py uses for f_k. Writes
output/purity_vs_threshold.txt.

    python -m an.purity
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets this run as `python3 purity.py` too, not just `-m`

import numpy as np

from an._lib import mc_purity as mcc
from an._lib.mc_purity import CLS

THR = np.concatenate([[0.0], np.round(np.arange(0.30, 0.905, 0.05), 3)])


def purity_at(bundle: dict, class_idx: int, t: float) -> float:
    argmax = bundle["pr"].argmax(1)
    tag = (argmax == class_idx) & (bundle["pr"][:, class_idx] >= t)
    n = int(tag.sum())
    if n == 0:
        return float("nan")
    return float((tag & (bundle["yt"] == class_idx)).sum()) / n


def compute_purity_table(lo: float = mcc.MASS_LO, hi: float = mcc.MASS_HI) -> dict:
    """{class: {threshold: (mean, std)}}, averaged over locally-synced MC seeds."""
    mcc.set_mass_window(lo, hi)
    boots = mcc.seeds()
    table = {c: {} for c in CLS}
    for k, c in enumerate(CLS):
        for t in THR:
            purs = [purity_at(mcc._bundle(b), k, float(t)) for b in boots]
            mean = float(np.nanmean(purs))
            std = float(np.nanstd(purs, ddof=1)) if len(purs) > 1 else 0.0
            table[c][float(t)] = (mean, std)
    return table


def main():
    table = compute_purity_table()
    out = Path(__file__).resolve().parents[1] / "output" / "purity_vs_threshold.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        fp.write("# purity vs threshold, resampled to EXP argmax prior\n")
        fp.write("# class threshold purity_mean purity_std\n")
        for c in CLS:
            for t, (mean, std) in table[c].items():
                fp.write(f"{c} {t:.3f} {mean:.6f} {std:.6f}\n")
                print(f"  {c:5s} thr={t:.2f}  purity={mean*100:.2f}% (+/-{std*100:.2f})")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
