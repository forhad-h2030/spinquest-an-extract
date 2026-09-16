#!/usr/bin/env python3
"""Background-corrected J/psi TSSA A_N:

    A_N^corr = ( A_N^tag(jpsi) - sum_{k != jpsi} f_k * A_N^tag(k) ) / f_jpsi

f_k (contamination fractions) come from mc_purity's DATA_PRIOR-resampled
synthetic MC sample. MODE="argmax": tag_k = argmax==k. MODE="prob": also
ANDs each class's PROB_CUTS on top of argmax==k.

    ROOTPY=/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python
    $ROOTPY -m an.asymmetry
"""
from __future__ import annotations

import sys
import math
import operator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets this run as `python3 asymmetry.py` too, not just `-m`

import numpy as np
import uproot

from an._lib import mc_purity as mcc
from an._lib.mc_purity import CLS, DATA_FILE
from an.config import ETA, F, POL, X_CUT, Y_CUT, PZ_CUT

MODE = "argmax"   # "argmax" or "prob"

PREF = 1.0 / (ETA * F * POL)

LO, HI = mcc.MASS_LO, mcc.MASS_HI

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}

# MODE="prob" only -- list of (prob_class, op, threshold) cuts per tagged
# class, all ANDed together on top of argmax==k. Empty list = argmax only.
PROB_CUTS: dict[str, list[tuple[str, str, float]]] = {
    "jpsi": [("dy", "<=", 0.2)],
    "psip": [("psip", ">=", 0.91)],
    "dy":   [("dy", ">=", 0.64)],
    "comb": [],
}


def an(N_up_left, N_up_right, N_down_left, N_down_right):
    if min(N_up_left, N_up_right, N_down_left, N_down_right) == 0:
        return float("nan"), float("nan")
    A = math.sqrt(N_up_left * N_down_right)
    B = math.sqrt(N_down_left * N_up_right)
    den = A + B
    return (PREF * (A - B) / den,
            PREF * (A * B / den**2) * math.sqrt(
                1/N_up_left + 1/N_up_right + 1/N_down_left + 1/N_down_right))


def _cuts_text(class_key: str) -> str:
    cuts = PROB_CUTS.get(class_key)
    if not cuts:
        return "none"
    return ", ".join(f"p_{c}{op}{t:g}" for c, op, t in cuts)


def tag_mask(qual_mask: np.ndarray, class_probs: np.ndarray, class_idx: int) -> np.ndarray:
    argmax = class_probs.argmax(1)
    sel = qual_mask & (argmax == class_idx)
    if MODE == "prob":
        for prob_cls, op, thr in PROB_CUTS.get(CLS[class_idx]) or []:
            sel = sel & OPS[op](class_probs[:, CLS.index(prob_cls)], thr)
    return sel


def quadrant_counts(dimu_px, spin, qual_mask, class_probs, class_idx):
    sel = tag_mask(qual_mask, class_probs, class_idx)
    N_up_left    = int((sel & (spin == 1) & (dimu_px > 0)).sum())
    N_up_right   = int((sel & (spin == 1) & (dimu_px <= 0)).sum())
    N_down_left  = int((sel & (spin == 0) & (dimu_px > 0)).sum())
    N_down_right = int((sel & (spin == 0) & (dimu_px <= 0)).sum())
    return N_up_left, N_up_right, N_down_left, N_down_right


def exp_seeds() -> list[str]:
    """Boot tags actually baked into DATA_FILE (ml_p_jpsi_boot_* branches) --
    independent of mc_purity.seeds() (checkpoint boot dirs synced locally):
    the tagged ROOT file already has all seeds' scores baked in."""
    with uproot.open(f"{DATA_FILE}:tree") as tree:
        keys = tree.keys()
    return sorted({k[len("ml_p_jpsi_"):] for k in keys if k.startswith("ml_p_jpsi_boot_")})


def load_exp_data(lo: float, hi: float, x_cut: float | None, y_cut: float | None,
                  pz_cut: float | None = None):
    boots = exp_seeds()
    with uproot.open(f"{DATA_FILE}:tree") as tree:
        keys = set(tree.keys())
        missing = [f"ml_p_{c}_{boot}" for boot in boots for c in CLS
                  if f"ml_p_{c}_{boot}" not in keys]
        if missing:
            raise SystemExit(
                f"[ERROR] {DATA_FILE.name} is missing per-seed prob branches "
                f"(e.g. {missing[0]}) -- retag it with an.tag first.")

        need = (["mass", "dimuon_px", "spin_up", "pt",
                 "rec_track_pos_x_st1", "rec_track_neg_x_st1",
                 "rec_track_pos_y_st1", "rec_track_neg_y_st1",
                 "rec_dimu_mu_pos_pz", "rec_dimu_mu_neg_pz"]
                + [f"ml_p_{c}_{boot}" for boot in boots for c in CLS])
        a = tree.arrays(need, library="np")
    dimu_px = a["dimuon_px"].astype(np.float64)
    spin    = a["spin_up"].astype(np.int32)
    pt      = a["pt"].astype(np.float64)

    qual_mask = (a["mass"] >= lo) & (a["mass"] <= hi)
    if x_cut is not None:
        qual_mask &= (a["rec_track_pos_x_st1"] < x_cut) & (a["rec_track_neg_x_st1"] < x_cut)
    if y_cut is not None:
        qual_mask &= (np.abs(a["rec_track_pos_y_st1"]) > y_cut) & \
                     (np.abs(a["rec_track_neg_y_st1"]) > y_cut)
    if pz_cut is not None:
        qual_mask &= (a["rec_dimu_mu_pos_pz"] > pz_cut) & (a["rec_dimu_mu_neg_pz"] > pz_cut)

    probs = {}
    for boot in boots:
        probs[boot] = np.column_stack(
            [a[f"ml_p_{c}_{boot}"] for c in CLS]).astype(np.float64)
    return dimu_px, spin, pt, qual_mask, probs, boots


def configure_prior(lo: float, hi: float, x_cut: float | None,
                    y_cut: float | None) -> np.ndarray:
    """DATA_PRIOR = EXP ml_class (argmax) counts under the selection, set as
    a side effect of set_mass_window()."""
    mcc.set_mass_window(lo, hi, x_cut, y_cut)
    prior = mcc.DATA_PRIOR
    if np.any(prior < 0) or prior.sum() <= 0:
        raise SystemExit(f"[ERROR] bad DATA_PRIOR: {prior}")
    return prior


def sum_counts(count_list):
    return tuple(int(sum(c[i] for c in count_list)) for i in range(4))


def contamination_fractions(boot: str, tag_idx: int, lo: float, hi: float):
    """Share f_k of the class-tag_idx tag from each true class (one seed),
    on the DATA_PRIOR-resampled synthetic MC sample -- plain counts."""
    m = mcc._bundle(boot)
    qual_mc = (m["mass"] >= lo) & (m["mass"] <= hi)
    tag = tag_mask(qual_mc, m["pr"], tag_idx)
    tot = int(tag.sum())
    if tot <= 0:
        return {c: np.nan for c in CLS}, np.nan
    fk = {c: int((tag & (m["yt"] == k)).sum()) / tot for k, c in enumerate(CLS)}
    return fk, 1.0 - fk[CLS[tag_idx]]


def extract_corrected_an(lo: float = LO, hi: float = HI, x_cut: float | None = X_CUT,
                         y_cut: float | None = Y_CUT, pz_cut: float | None = PZ_CUT,
                         verbose: bool = True) -> dict:
    """Run the full pipeline and return a result dict (same fields written
    to output/an_result*.txt by main())."""
    mc_local_seeds = mcc.seeds()
    prior = configure_prior(lo, hi, x_cut, y_cut)
    dimu_px, spin, pt, qual_mask, probs, exp_boots = load_exp_data(lo, hi, x_cut, y_cut, pz_cut)

    fk_per_boot = [contamination_fractions(b, 0, lo, hi)[0] for b in mc_local_seeds]
    fk = {c: float(np.nanmean([f[c] for f in fk_per_boot])) for c in CLS}

    A_raw_seeds, dA_raw_seeds = [], []
    A_comp_seeds, dA_comp_seeds = [], []
    A_comb_seeds, dA_comb_seeds = [], []
    A_tag_seeds  = {c: [] for c in CLS}
    dA_tag_seeds = {c: [] for c in CLS}
    for boot in exp_boots:
        A_tag, dA_tag, tag_counts = {}, {}, {}
        for k, c in enumerate(CLS):
            counts = quadrant_counts(dimu_px, spin, qual_mask, probs[boot], k)
            tag_counts[c] = counts
            A_tag[c], dA_tag[c] = an(*counts)

        bg_counts = sum_counts([tag_counts[c] for c in CLS if c != "jpsi"])
        A_bg_all, dA_bg_all = an(*bg_counts)

        bg_sum  = sum(fk[c] * A_tag[c] for c in CLS if c != "jpsi")
        A_comp  = (A_tag["jpsi"] - bg_sum) / fk["jpsi"]
        dA_comp = math.sqrt(dA_tag["jpsi"] ** 2 +
                            sum((fk[c] * dA_tag[c]) ** 2
                                for c in CLS if c != "jpsi")) / fk["jpsi"]

        f_bg = 1.0 - fk["jpsi"]
        A_comb = (A_tag["jpsi"] - f_bg * A_bg_all) / fk["jpsi"]
        dA_comb = math.sqrt(dA_tag["jpsi"] ** 2 + (f_bg * dA_bg_all) ** 2) / fk["jpsi"]

        for c in CLS:
            A_tag_seeds[c].append(A_tag[c])
            dA_tag_seeds[c].append(dA_tag[c])
        A_raw_seeds.append(A_tag["jpsi"]); dA_raw_seeds.append(dA_tag["jpsi"])
        A_comp_seeds.append(A_comp); dA_comp_seeds.append(dA_comp)
        A_comb_seeds.append(A_comb); dA_comb_seeds.append(dA_comb)

        if verbose:
            print(f"{boot:>10} {A_tag['jpsi']:>+9.3f} {A_bg_all:>+9.3f} "
                  f"{fk['jpsi']:>8.4f} {f_bg:>8.4f} "
                  f"{A_comp:>+9.3f} {dA_comp:>7.3f} {A_comb:>+9.3f} {dA_comb:>7.3f}")

    A_raw = float(np.mean(A_raw_seeds))
    stat_raw = float(np.mean(dA_raw_seeds))
    model_raw = float(np.std(A_raw_seeds, ddof=1))
    dA_raw = math.sqrt(stat_raw**2 + model_raw**2)

    A_comp = float(np.mean(A_comp_seeds))
    stat_comp = float(np.mean(dA_comp_seeds))
    model_comp = float(np.std(A_comp_seeds, ddof=1))
    A_comb = float(np.mean(A_comb_seeds))
    stat_comb = float(np.mean(dA_comb_seeds))
    model_comb = float(np.std(A_comb_seeds, ddof=1))
    method = abs(A_comp - A_comb)
    syst_comp = math.sqrt(model_comp**2 + method**2)

    class_A     = {c: float(np.mean(A_tag_seeds[c])) for c in CLS}
    class_stat  = {c: float(np.mean(dA_tag_seeds[c])) for c in CLS}
    class_model = {c: float(np.std(A_tag_seeds[c], ddof=1)) for c in CLS}

    sel_pt = tag_mask(qual_mask, probs[exp_boots[0]], 0)
    pt_sel = pt[sel_pt]

    return dict(
        mode=MODE, data_file=str(DATA_FILE), mass_lo=lo, mass_hi=hi,
        eta=ETA, f=F, P=POL,
        n_exp_seeds=len(exp_boots), n_mc_seeds=len(mc_local_seeds),
        fk=fk, prior=dict(zip(CLS, prior.tolist())),
        A_N_raw=A_raw, dA_N_raw_stat=stat_raw, dA_N_raw_model=model_raw, dA_N_raw=dA_raw,
        A_N_corr=A_comp, dA_N_corr_stat=stat_comp, dA_N_corr_model=model_comp,
        dA_N_corr_method=method, dA_N_corr_syst=syst_comp,
        A_N_comb_bkg=A_comb, dA_N_comb_bkg_stat=stat_comb,
        f_jpsi=fk["jpsi"],
        pt_mean=float(pt_sel.mean()), pt_std=float(pt_sel.std()),
        class_A=class_A, class_stat=class_stat, class_model=class_model,
    )


def main():
    print(f"[INFO] MODE={MODE}")
    print(f"[INFO] data : {DATA_FILE}")
    tag_desc = ("pure argmax (no probability cut)" if MODE == "argmax" else
                "argmax + prob cuts: " +
                "; ".join(f"{c}:[{_cuts_text(c)}]" for c in CLS))
    print(f"\n=== corrected J/psi A_N  ({LO:g}-{HI:g} GeV, {tag_desc}, "
          f"x_st1<{X_CUT:g}, |y_st1|>{Y_CUT:g}, pz>{PZ_CUT}) ===")
    print(f"{'seed':>10} {'A_jpsi':>9} {'A_bg(all)':>9} {'f_jpsi':>8} {'f_bg':>8} "
          f"{'A_comp':>9} {'stat':>7} {'A_comb':>9} {'stat':>7}")

    r = extract_corrected_an()

    print("-" * 78)
    print(f"f_k (from {r['n_mc_seeds']} locally-synced MC seed(s)): "
          + "  ".join(f"{c}={r['fk'][c]:.4f}" for c in CLS))
    print()
    print(f"  {'raw per-class A_N (tag, uncorrected)':<64}{'corrected A_N':<20}")
    for i, c in enumerate(CLS):
        left = (f"A_N^tag({c:<4s}) = {r['class_A'][c]:+.4f} +/- {r['class_stat'][c]:.4f} (stat)"
                f" +/- {r['class_model'][c]:.4f} (model)")
        right = (f"A_N^corr(J/psi) = {r['A_N_corr']:+.4f} +/- {r['dA_N_corr_stat']:.4f} (stat)"
                 if i == 0 else "")
        print(f"  {left:<64}{right}")
    print(f"\n  component-wise:")
    print(f"    A_N^corr(J/psi) = {r['A_N_corr']:+.4f}  +/- {r['dA_N_corr_stat']:.4f} (stat)"
          f"  +/- {r['dA_N_corr_model']:.4f} (model)")
    print(f"  combined background:")
    print(f"    A_N^corr(J/psi) = {r['A_N_comb_bkg']:+.4f}  +/- {r['dA_N_comb_bkg_stat']:.4f} (stat)")
    print(f"  nominal with method systematic:")
    print(f"    A_N^corr(J/psi) = {r['A_N_corr']:+.4f}  +/- {r['dA_N_corr_stat']:.4f} (stat)"
          f"  +/- {r['dA_N_corr_model']:.4f} (model)  +/- {r['dA_N_corr_method']:.4f} (method)")
    print(f"  method systematic = |component-wise - combined background|")
    print(f"  (eta={ETA}, f={F}, P={POL}; DATA_PRIOR = argmax counts)")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / ("an_result.txt" if MODE == "argmax" else "an_result_prob.txt")
    with open(result_file, "w") as fp:
        fp.write(f"A_N_corr={r['A_N_corr']:+.6f}\n")
        fp.write(f"dA_N_corr_stat={r['dA_N_corr_stat']:.6f}\n")
    print(f"\n[SAVED] {result_file}")


if __name__ == "__main__":
    main()
