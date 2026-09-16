#!/usr/bin/env python3
"""QA plot: dimuon mass overlay, tagged into jpsi/psip/dy/comb. matplotlib
only, no PyROOT needed.

    python scripts/plot_mass_overlay.py
"""
from pathlib import Path
import operator
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "exp_tagged_data" / "exp_tagged_tgt_data_multiclass_12_feat_resnet4_oldjpsipsip_20seed.root"

MODE = "prob"   # "argmax", "prob", or "second_max"

MASS_LO, MASS_HI = 1.5, 6.0
NBINS = 45
X_CUT, Y_CUT, PZ_CUT = 25.0, 3.0, 5.0

CLS = ["jpsi", "psip", "dy", "comb"]
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}

PROB_CUTS: dict[str, list[tuple[str, str, float]]] = {
    "jpsi": [("jpsi", ">=", 0.5)],
    "psip": [("psip", ">=", 0.5)],
    "dy":   [("dy", ">=", 0.8)],
    "comb": [],
}

SECOND_MAX_VETO = 0.2


def tag_mask(qual_mask, probs, class_idx):
    argmax = probs.argmax(1)
    sel = qual_mask & (argmax == class_idx)
    if MODE == "prob":
        for prob_cls, op, thr in PROB_CUTS.get(CLS[class_idx]) or []:
            sel = sel & OPS[op](probs[:, CLS.index(prob_cls)], thr)
    elif MODE == "second_max" and CLS[class_idx] != "comb":
        second_highest = np.sort(probs, axis=1)[:, -2]
        sel = sel & (second_highest <= SECOND_MAX_VETO)
    return sel


def main():
    tree = uproot.open(f"{INPUT_FILE}:tree")

    mass   = tree["mass"].array(library="np").astype(np.float64)
    p_jpsi = tree["ml_p_jpsi"].array(library="np").astype(np.float64)
    p_psip = tree["ml_p_psip"].array(library="np").astype(np.float64)
    p_dy   = tree["ml_p_dy"].array(library="np").astype(np.float64)
    p_comb = tree["ml_p_comb"].array(library="np").astype(np.float64)
    x_pos  = tree["rec_track_pos_x_st1"].array(library="np").astype(np.float64)
    x_neg  = tree["rec_track_neg_x_st1"].array(library="np").astype(np.float64)
    y_pos  = tree["rec_track_pos_y_st1"].array(library="np").astype(np.float64)
    y_neg  = tree["rec_track_neg_y_st1"].array(library="np").astype(np.float64)
    pz_pos = tree["rec_dimu_mu_pos_pz"].array(library="np").astype(np.float64)
    pz_neg = tree["rec_dimu_mu_neg_pz"].array(library="np").astype(np.float64)

    qual_mask = (mass >= MASS_LO) & (mass <= MASS_HI)
    qual_mask &= (x_pos < X_CUT) & (x_neg < X_CUT)
    qual_mask &= (np.abs(y_pos) > Y_CUT) & (np.abs(y_neg) > Y_CUT)
    qual_mask &= (pz_pos > PZ_CUT) & (pz_neg > PZ_CUT)

    probs = np.column_stack([p_jpsi, p_psip, p_dy, p_comb])

    jpsi_mask = tag_mask(qual_mask, probs, 0)
    psip_mask = tag_mask(qual_mask, probs, 1)
    dy_mask   = tag_mask(qual_mask, probs, 2)
    comb_mask = tag_mask(qual_mask, probs, 3)

    if MODE == "argmax":
        title_tag = "argmax only, no probability threshold"
        out_name = "mass_overlay_argmax.png"
    elif MODE == "prob":
        comb_mask = comb_mask | (qual_mask & ~jpsi_mask & ~psip_mask & ~dy_mask)
        title_tag = "argmax + prob cuts (leftovers -> comb): " + "; ".join(
            f"{c}:[{', '.join(f'p_{pc}{op}{t:g}' for pc, op, t in PROB_CUTS[c])}]"
            if PROB_CUTS[c] else f"{c}:[none]" for c in CLS)
        out_name = "mass_overlay_prob.png"
    else:
        comb_mask = comb_mask | (qual_mask & ~jpsi_mask & ~psip_mask & ~dy_mask)
        title_tag = f"argmax + 2nd-max prob veto (ambiguous -> comb): p_2nd>{SECOND_MAX_VETO:g}"
        out_name = "mass_overlay_second_max.png"

    mass_all  = mass[qual_mask]
    mass_jpsi = mass[jpsi_mask]
    mass_psip = mass[psip_mask]
    mass_dy   = mass[dy_mask]
    mass_comb = mass[comb_mask]

    print(f"N total={len(mass_all):,}  jpsi={len(mass_jpsi):,}  psip={len(mass_psip):,}  "
          f"dy={len(mass_dy):,}  comb={len(mass_comb):,}  ({title_tag})")

    bins = np.linspace(MASS_LO, MASS_HI, NBINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    h_tot, _  = np.histogram(mass_all, bins=bins)
    h_jpsi, _ = np.histogram(mass_jpsi, bins=bins)
    h_psip, _ = np.histogram(mass_psip, bins=bins)
    h_dy, _   = np.histogram(mass_dy, bins=bins)
    h_comb, _ = np.histogram(mass_comb, bins=bins)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(centers, h_tot, yerr=np.sqrt(np.clip(h_tot, 1, None)), fmt="o",
               ms=4, capsize=3, color="black", label=f"All data  (N={len(mass_all):,})")
    ax.step(centers, h_jpsi, where="mid", color="#2471A3", lw=2.5,
            label=f"J/psi  (N={len(mass_jpsi):,})")
    ax.step(centers, h_psip, where="mid", color="#E74C3C", lw=2.5, ls="--",
            label=f"psi(2S)  (N={len(mass_psip):,})")
    ax.step(centers, h_dy, where="mid", color="#27AE60", lw=2.0, ls="-.",
            label=f"DY  (N={len(mass_dy):,})")
    ax.step(centers, h_comb, where="mid", color="#8E44AD", lw=2.0, ls=":",
            label=f"Comb  (N={len(mass_comb):,})")

    binw = bins[1] - bins[0]
    ax.set_xlabel(r"Dimuon mass [GeV/$c^2$]")
    ax.set_ylabel(fr"Events / {binw:g} GeV/$c^2$")
    ax.legend()
    ax.grid(alpha=0.3, ls="--")

    out = ROOT_DIR / "output" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
