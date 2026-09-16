#!/usr/bin/env python3
"""Background-only mass overlay, ML-tagged: dy and comb, argmax only
(tag_k = argmax==k, no probability cut) -- pared-down bkg view of
mass_overlay.py, which also plots jpsi/psip and supports prob/second_max
tag modes.
"""
from pathlib import Path
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE.parents[0] / "exp_tagged_data" / "exp_tagged_tgt_data_20063581.root"

MASS_LO, MASS_HI = 1.5, 6.0
NBINS = 45
X_CUT, Y_CUT, PZ_CUT = 25.0, 3.0, 5.0

CLS = ["jpsi", "psip", "dy", "comb"]


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
    argmax = probs.argmax(1)

    bkg_mask = qual_mask & ((argmax == CLS.index("dy")) | (argmax == CLS.index("comb")))

    mass_all = mass[qual_mask]
    mass_bkg = mass[bkg_mask]

    print(f"N total={len(mass_all):,}  bkg(dy+comb)={len(mass_bkg):,}  (argmax only)")

    bins = np.linspace(MASS_LO, MASS_HI, NBINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    h_tot, _ = np.histogram(mass_all, bins=bins)
    h_bkg, _ = np.histogram(mass_bkg, bins=bins)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(centers, h_tot, yerr=np.sqrt(np.clip(h_tot, 1, None)), fmt="o",
               ms=4, capsize=3, color="black", label=f"All data  (N={len(mass_all):,})")
    ax.step(centers, h_bkg, where="mid", color="#8E44AD", lw=2.0, ls="-.",
            label=f"DY + Comb  (N={len(mass_bkg):,})")

    binw = bins[1] - bins[0]
    ax.set_xlabel(r"Dimuon mass [GeV/$c^2$]")
    ax.set_ylabel(fr"Events / {binw:g} GeV/$c^2$")
    ax.set_title("argmax only, no probability threshold (dy + comb combined)")
    ax.legend()
    ax.grid(alpha=0.3, ls="--")

    out = HERE / "output" / "mass_overlay_bkg_argmax.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
