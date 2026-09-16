#!/usr/bin/env python3
"""Per-seed J/psi tagged signal counts and raw A_N, one row per checkpoint
seed baked into the tagged EXP file -- plus the mean +/- sigma (seed-to-seed
spread) row used as the "model" systematic elsewhere in the pipeline.

    python3 -m an.per_seed_table
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets this run as `python3 per_seed_table.py` too

import numpy as np

from an import asymmetry as asy

JPSI_IDX = asy.CLS.index("jpsi")


def build_table(lo: float = asy.LO, hi: float = asy.HI,
                x_cut: float | None = asy.X_CUT, y_cut: float | None = asy.Y_CUT,
                pz_cut: float | None = asy.PZ_CUT) -> list[dict]:
    dimu_px, spin, pt, qual_mask, probs, exp_boots = asy.load_exp_data(lo, hi, x_cut, y_cut, pz_cut)
    rows = []
    for boot in exp_boots:
        counts = asy.quadrant_counts(dimu_px, spin, qual_mask, probs[boot], JPSI_IDX)
        n_sig = sum(counts)
        A_N, dA_N = asy.an(*counts)
        rows.append(dict(seed=boot, n_up_left=counts[0], n_up_right=counts[1],
                         n_down_left=counts[2], n_down_right=counts[3],
                         n_sig=n_sig, A_N=A_N, dA_N=dA_N))
    return rows


def main():
    rows = build_table()
    A_vals = np.array([r["A_N"] for r in rows])
    dA_vals = np.array([r["dA_N"] for r in rows])
    mean = float(np.mean(A_vals))
    # Seeds are NOT independent trials (same EXP data, correlated tagging) --
    # so stat is the mean of each seed's own error (not reduced by sqrt(N)),
    # and the seed-to-seed spread is a separate "model" systematic. Matches
    # asymmetry.py's A_N_raw/dA_N_raw_stat/dA_N_raw_model exactly.
    stat = float(np.mean(dA_vals))
    model = float(np.std(A_vals, ddof=1))
    total = float(np.sqrt(stat**2 + model**2))

    print(f"{'seed':>10} {'N_uL':>6} {'N_uR':>6} {'N_dL':>6} {'N_dR':>6} {'A_N_raw':>10} {'+/- stat':>10}")
    for r in rows:
        print(f"{r['seed']:>10} {r['n_up_left']:>6} {r['n_up_right']:>6} {r['n_down_left']:>6} "
              f"{r['n_down_right']:>6} {r['A_N']:>+10.4f} {r['dA_N']:>10.4f}")
    print(f"{'mean':>10} {'':>6} {'':>6} {'':>6} {'':>6} {mean:>+10.4f}")
    print(f"{'stat':>10} {'':>6} {'':>6} {'':>6} {'':>6} {'':>10} {stat:>10.4f}")
    print(f"{'model':>10} {'':>6} {'':>6} {'':>6} {'':>6} {'':>10} {model:>10.4f}")
    print(f"{'total':>10} {'':>6} {'':>6} {'':>6} {'':>6} {'':>10} {total:>10.4f}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "per_seed_jpsi_table.txt"
    with open(txt_path, "w") as fp:
        fp.write(f"{'seed':>10} {'N_uL':>6} {'N_uR':>6} {'N_dL':>6} {'N_dR':>6} "
                 f"{'A_N_raw':>10} {'dA_N_raw_stat':>14}\n")
        for r in rows:
            fp.write(f"{r['seed']:>10} {r['n_up_left']:>6} {r['n_up_right']:>6} {r['n_down_left']:>6} "
                     f"{r['n_down_right']:>6} {r['A_N']:>+10.4f} {r['dA_N']:>14.4f}\n")
        fp.write(f"mean={mean:+.4f}  stat={stat:.4f}  model={model:.4f}  total={total:.4f}\n")
    print(f"\n[SAVED] {txt_path}")

    tex_path = out_dir / "per_seed_jpsi_table.tex"
    with open(tex_path, "w") as fp:
        fp.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        fp.write("seed & $N_{\\uparrow L}$ & $N_{\\uparrow R}$ & $N_{\\downarrow L}$ & $N_{\\downarrow R}$ & "
                 "$A_N^{\\rm raw}$ & stat.\\ err. \\\\\n\\midrule\n")
        for r in rows:
            seed_tex = r["seed"].replace("_", "\\_")
            fp.write(f"{seed_tex} & {r['n_up_left']} & {r['n_up_right']} & {r['n_down_left']} & "
                     f"{r['n_down_right']} & {r['A_N']:+.4f} & {r['dA_N']:.4f} \\\\\n")
        fp.write("\\midrule\n")
        fp.write(f"mean & & & & & {mean:+.4f} & \\\\\n")
        fp.write(f"$\\delta_{{\\rm stat}}$ (mean of seed stat.\\ err.) & & & & & & {stat:.4f} \\\\\n")
        fp.write(f"$\\delta_{{\\rm model}}$ (seed-to-seed $\\sigma$) & & & & & & {model:.4f} \\\\\n")
        fp.write(f"total ($\\sqrt{{\\rm stat^2+model^2}}$) & & & & & & {total:.4f} \\\\\n")
        fp.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[SAVED] {tex_path}")


if __name__ == "__main__":
    main()
