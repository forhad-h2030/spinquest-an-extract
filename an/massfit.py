#!/usr/bin/env python3
"""RooFit mass-shape-fit A_N: fits J/psi + psi' Gaussians + exponential bkg
simultaneously across the 4 spin/side panels of the *untagged* flattened EXP
data (no ML model involved at all -- just `mass` and `dimuon_px`).

    ROOTPY=/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python
    $ROOTPY -m an.massfit
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from an.config import ETA, F, POL as P_AVG, MASS_LO as MASS_MIN, MASS_HI as MASS_MAX, X_CUT, Y_CUT, PZ_CUT

try:
    import ROOT
    from ROOT import (
        RooRealVar, RooFormulaVar, RooDataSet, RooGaussian, RooExponential,
        RooAddPdf, RooSimultaneous, RooCategory,
        RooFit, RooArgSet, RooArgList, TCanvas, TH1F, TLatex, TLegend,
    )
    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
except ImportError as e:
    raise SystemExit(f"[ERROR] ROOT / PyROOT not available: {e}")

REPO = Path(__file__).resolve().parents[1]
DEF_UP   = REPO / "exp_tagged_data" / "exp_tgt_data_june30_up.root"
DEF_DOWN = REPO / "exp_tagged_data" / "exp_tgt_data_june30_down.root"
TREE_NAME = "tree"

N_BINS = 60
RATIO  = 3.686 / 3.097   # M_psi' / M_J/psi (PDG)

PANELS = [
    ("up",   "right", 1, "Spin Up, Right (p_{x} < 0)"),
    ("up",   "left",  2, "Spin Up, Left (p_{x} > 0)"),
    ("down", "right", 3, "Spin Down, Right (p_{x} < 0)"),
    ("down", "left",  4, "Spin Down, Left (p_{x} > 0)"),
]


def panel_suffix(spin, side):
    return f"{spin}_{side}"


def _load_panels_from_root(up_file: Path, down_file: Path) -> dict[str, np.ndarray]:
    cols = ["mass", "dimuon_px",
            "rec_track_pos_x_st1", "rec_track_neg_x_st1",
            "rec_track_pos_y_st1", "rec_track_neg_y_st1",
            "rec_dimu_mu_pos_pz", "rec_dimu_mu_neg_pz"]

    cuts = [
        ROOT.TCut(f"mass>={MASS_MIN:g} && mass<={MASS_MAX:g}"),
        ROOT.TCut(f"rec_track_pos_x_st1<{X_CUT:g} && rec_track_neg_x_st1<{X_CUT:g}"),
        ROOT.TCut(f"TMath::Abs(rec_track_pos_y_st1)>{Y_CUT:g} && TMath::Abs(rec_track_neg_y_st1)>{Y_CUT:g}"),
        ROOT.TCut(f"rec_dimu_mu_pos_pz>{PZ_CUT:g} && rec_dimu_mu_neg_pz>{PZ_CUT:g}"),
    ]
    qual_cut = sum(cuts, ROOT.TCut())

    results = {}
    for spin, path in (("up", up_file), ("down", down_file)):
        d = ROOT.RDataFrame(TREE_NAME, str(path)).Filter(qual_cut.GetTitle()).AsNumpy(cols)
        M  = np.asarray(d["mass"]).astype(np.float64)
        px = np.asarray(d["dimuon_px"]).astype(np.float64)
        results[f"{spin}_left"]  = M[px > 0]
        results[f"{spin}_right"] = M[px <= 0]
    for key, arr in results.items():
        print(f"  [{key}]  {len(arr):,} events in mass window")
    return results


def _compute_an(N_uL, N_uR, N_dL, N_dR, eta=ETA, f=F, P=P_AVG):
    if min(N_uL, N_uR, N_dL, N_dR) <= 0:
        return None
    A = math.sqrt(N_uL * N_dR)
    B = math.sqrt(N_dL * N_uR)
    den = A + B
    A_raw = (A - B) / den
    return A_raw, A_raw / (eta * f * P)


def An_error(N_uL, dN_uL, N_uR, dN_uR, N_dL, dN_dL, N_dR, dN_dR, eta=ETA, f=F, P=P_AVG):
    A = math.sqrt(N_uL * N_dR)
    B = math.sqrt(N_dL * N_uR)
    term_A = (B / A) ** 2 * ((N_dR * dN_uL) ** 2 + (N_uL * dN_dR) ** 2)
    term_B = (A / B) ** 2 * ((N_uR * dN_dL) ** 2 + (N_dL * dN_uR) ** 2)
    return math.sqrt(term_A + term_B) / (eta * f * P * (A + B) ** 2)


def build_panel_model(suffix, mass, mean1, sigma1, mean2, sigma2):
    tau   = RooRealVar(f"tau_{suffix}",   "Bkg slope", -2.0, -6.0,   0.0)
    nsig1 = RooRealVar(f"nsig1_{suffix}", "N J/#psi",   50,    0,  5000)
    nsig2 = RooRealVar(f"nsig2_{suffix}", "N #psi'",    20,    0,  2000)
    nbkg  = RooRealVar(f"nbkg_{suffix}",  "N bkg",     100,    0, 10000)

    sig1 = RooGaussian(f"sig1_{suffix}", "J/#psi PDF", mass, mean1, sigma1)
    sig2 = RooGaussian(f"sig2_{suffix}", "#psi' PDF",  mass, mean2, sigma2)
    bkg  = RooExponential(f"bkg_{suffix}", "Bkg PDF",  mass, tau)
    mdl  = RooAddPdf(f"mdl_{suffix}", "Total PDF",
                     RooArgList(sig1, sig2, bkg),
                     RooArgList(nsig1, nsig2, nbkg))
    return mdl, (tau, nsig1, nsig2, nbkg), (sig1, sig2, bkg)


def draw_panel(pad, mass, ds, mdl, suffix, nsig1, chi2ndf, label):
    bw = (MASS_MAX - MASS_MIN) / N_BINS
    frame = mass.frame(RooFit.Title(label))
    frame.GetYaxis().SetTitle(f"Events / ({bw:.3f} GeV)")
    frame.GetXaxis().SetTitle("Dimuon mass [GeV]")

    ds.plotOn(frame, RooFit.MarkerSize(0.8))
    mdl.plotOn(frame, RooFit.LineColor(ROOT.kBlue),      RooFit.LineWidth(2), RooFit.Name("fit_curve"))
    mdl.plotOn(frame, RooFit.Components(f"sig1_{suffix}"),
               RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen + 1),
               RooFit.LineWidth(2), RooFit.Name("jpsi_curve"))
    mdl.plotOn(frame, RooFit.Components(f"sig2_{suffix}"),
               RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kMagenta),
               RooFit.LineWidth(2), RooFit.Name("psi2_curve"))
    mdl.plotOn(frame, RooFit.Components(f"bkg_{suffix}"),
               RooFit.LineStyle(ROOT.kDotted), RooFit.LineColor(ROOT.kRed),
               RooFit.LineWidth(2), RooFit.Name("bkg_curve"))

    pad.cd()
    pad.SetLeftMargin(0.14); pad.SetRightMargin(0.05)
    pad.SetTopMargin(0.10);  pad.SetBottomMargin(0.14)
    frame.Draw()

    leg = TLegend(0.62, 0.55, 0.93, 0.88)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.038)
    leg.AddEntry(frame.findObject("fit_curve"),  "Fit",    "L")
    leg.AddEntry(frame.findObject("jpsi_curve"), "J/#psi", "L")
    leg.AddEntry(frame.findObject("psi2_curve"), "#psi'",  "L")
    leg.AddEntry(frame.findObject("bkg_curve"),  "Bckg",   "L")
    leg.Draw()

    lat = TLatex(); lat.SetNDC(); lat.SetTextSize(0.038); lat.SetTextAlign(13)
    lat.DrawLatex(0.63, 0.53, f"#chi^{{2}}/ndf = {chi2ndf:.2f}")
    lat.DrawLatex(0.63, 0.47, f"no. J/#psi = {nsig1.getVal():.2f} #pm {nsig1.getError():.2f}")

    pad.Update()
    return frame, leg, lat


def draw_histograms(mass_arrays, output_path):
    bw = (MASS_MAX - MASS_MIN) / N_BINS
    canvas = TCanvas("c_hist", "Mass histograms", 1400, 1100)
    canvas.Divide(2, 2)
    keep = []

    for spin, side, pad_idx, label in PANELS:
        suf  = panel_suffix(spin, side)
        vals = mass_arrays[suf].astype(np.float64)

        h = TH1F(f"h_{suf}", label, N_BINS, MASS_MIN, MASS_MAX)
        h.FillN(len(vals), vals, np.ones(len(vals)))
        h.GetXaxis().SetTitle("Dimuon mass [GeV]")
        h.GetYaxis().SetTitle(f"Events / ({bw:.3f} GeV)")
        h.SetLineWidth(2)

        pad = canvas.GetPad(pad_idx)
        pad.cd()
        pad.SetLeftMargin(0.14); pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.10);  pad.SetBottomMargin(0.14)

        ROOT.gStyle.SetOptStat("nemr")
        h.Draw("E")
        pad.Update()

        st = h.FindObject("TPaveStats")
        if st:
            st.SetX1NDC(0.62); st.SetX2NDC(0.93)
            st.SetY1NDC(0.65); st.SetY2NDC(0.88)
            st.Draw()
        pad.Update()
        keep.append(h)

    canvas.Update()
    canvas.SaveAs(output_path)


def fit_and_save(mass_arrays, output_path):
    """Simultaneous J/psi fit across the 4 panels. Returns (fit_result, panel_pars, keep_alive)."""
    mass = RooRealVar("rec_dimu_M", "Dimuon mass [GeV]", MASS_MIN, MASS_MAX)
    mass.setBins(N_BINS)

    mean1  = RooRealVar("mean1",  "J/#psi mean",   3.10, 2.90, 3.40)
    sigma1 = RooRealVar("sigma1", "J/#psi #sigma", 0.10, 0.04, 0.30)
    mean2  = RooFormulaVar("mean2",  "#psi' mean",   f"@0*{RATIO:.8f}", RooArgList(mean1))
    sigma2 = RooFormulaVar("sigma2", "#psi' #sigma", f"@0*{RATIO:.8f}", RooArgList(sigma1))

    datasets = {}
    for spin, side, _, _ in PANELS:
        suf  = panel_suffix(spin, side)
        vals = mass_arrays[suf]
        print(f"  [{suf}] {len(vals)} events in fit window")
        if len(vals) < 10:
            raise RuntimeError(f"Too few events in {suf}: {len(vals)}")
        ds = RooDataSet(f"ds_{suf}", f"ds_{suf}", RooArgSet(mass))
        for v in vals:
            mass.setVal(float(v))
            ds.add(RooArgSet(mass))
        datasets[suf] = ds

    models, panel_pars, keep_shapes = {}, {}, []
    for spin, side, _, _ in PANELS:
        suf = panel_suffix(spin, side)
        mdl, pars, shapes = build_panel_model(suf, mass, mean1, sigma1, mean2, sigma2)
        models[suf], panel_pars[suf] = mdl, pars
        keep_shapes.extend(shapes)

    sample = RooCategory("sample", "sample")
    for spin, side, _, _ in PANELS:
        sample.defineType(panel_suffix(spin, side))

    combData = RooDataSet(
        "combData", "Combined data", RooArgSet(mass, sample),
        RooFit.Index(sample),
        RooFit.Import("up_right",   datasets["up_right"]),
        RooFit.Import("up_left",    datasets["up_left"]),
        RooFit.Import("down_right", datasets["down_right"]),
        RooFit.Import("down_left",  datasets["down_left"]),
    )

    simPdf = RooSimultaneous("simPdf", "Simultaneous PDF", sample)
    for spin, side, _, _ in PANELS:
        suf = panel_suffix(spin, side)
        simPdf.addPdf(models[suf], suf)

    print("\n-- scanning initial conditions --")
    best_chi2, best_state = float("inf"), None
    for m1 in [2.95, 3.00, 3.05, 3.10, 3.15, 3.20, 3.25]:
        for tau_init in [-4.0, -2.0, -1.0]:
            mean1.setVal(m1); sigma1.setVal(0.10)
            for spin, side, _, _ in PANELS:
                tau, ns1, ns2, nb = panel_pars[panel_suffix(spin, side)]
                tau.setVal(tau_init); ns1.setVal(50); ns2.setVal(20); nb.setVal(100)

            res = simPdf.fitTo(combData, RooFit.PrintLevel(-1), RooFit.Save())
            if res.status() != 0:
                continue

            total_chi2 = 0
            for spin, side, _, _ in PANELS:
                suf = panel_suffix(spin, side)
                fr = mass.frame()
                datasets[suf].plotOn(fr, RooFit.Invisible())
                models[suf].plotOn(fr, RooFit.Invisible())
                total_chi2 += fr.chiSquare()

            if total_chi2 < best_chi2:
                best_chi2 = total_chi2
                best_state = {
                    "mean1": mean1.getVal(), "sigma1": sigma1.getVal(),
                    "panels": {panel_suffix(s, d): tuple(p.getVal() for p in panel_pars[panel_suffix(s, d)])
                              for s, d, _, _ in PANELS},
                }

    if best_state is None:
        print("  WARNING: no converged fit in scan; using last values")
    else:
        mean1.setVal(best_state["mean1"])
        sigma1.setVal(best_state["sigma1"])
        for spin, side, _, _ in PANELS:
            suf = panel_suffix(spin, side)
            for p, v in zip(panel_pars[suf], best_state["panels"][suf]):
                p.setVal(v)

    print("\n-- final simultaneous fit --")
    fit_result = simPdf.fitTo(combData, RooFit.PrintLevel(-1), RooFit.Save(True))
    fit_status = int(fit_result.status())
    fit_cov_qual = int(fit_result.covQual())
    print(f"  fit status={fit_status}  covariance quality={fit_cov_qual}")
    if fit_status != 0 or fit_cov_qual < 3:
        raise RuntimeError(f"Cannot quote fitted-yield uncertainties: status={fit_status}, covQual={fit_cov_qual}")
    print(f"  J/psi mean  = {mean1.getVal():.4f} +/- {mean1.getError():.4f} GeV")
    print(f"  J/psi sigma = {sigma1.getVal():.4f} +/- {sigma1.getError():.4f} GeV")

    canvas = TCanvas("c_jpsi", "J/psi fit", 1400, 1100)
    canvas.Divide(2, 2)
    keep_alive = [combData, simPdf, sample, mean1, sigma1, mean2, sigma2] + keep_shapes

    for spin, side, pad_idx, label in PANELS:
        suf = panel_suffix(spin, side)
        _, nsig1_p, *_ = panel_pars[suf]
        fr = mass.frame()
        datasets[suf].plotOn(fr, RooFit.Invisible())
        models[suf].plotOn(fr, RooFit.Invisible())
        chi2ndf = fr.chiSquare()
        print(f"  [{suf}]  chi2/ndf={chi2ndf:.2f}  N(J/psi)={nsig1_p.getVal():.0f} +/- {nsig1_p.getError():.0f}")

        frame, leg, lat = draw_panel(canvas.GetPad(pad_idx), mass, datasets[suf], models[suf],
                                     suf, nsig1_p, chi2ndf, label)
        keep_alive.extend([datasets[suf], frame, leg, lat])

    canvas.Update()
    canvas.SaveAs(output_path)
    draw_histograms(mass_arrays, output_path.replace(".png", "_hist.png"))
    return fit_result, panel_pars, keep_alive


def extract_massfit_an(up_file: Path = DEF_UP, down_file: Path = DEF_DOWN,
                       out_dir: Path | None = None) -> dict:
    for lbl, path in (("up_file", up_file), ("down_file", down_file)):
        if not path.is_file():
            raise SystemExit(f"[ERROR] {lbl} not found: {path}")
    if out_dir is None:
        out_dir = REPO / "output" / "massfit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  mass {MASS_MIN:g}-{MASS_MAX:g} GeV, x_st1<{X_CUT}, |y_st1|>{Y_CUT}, pz>{PZ_CUT}")
    mass_arrays = _load_panels_from_root(up_file, down_file)

    fit_plot = out_dir / "fit_4panel.png"
    fit_result, panel_pars, _ = fit_and_save(mass_arrays, str(fit_plot))
    fit_status = int(fit_result.status())
    fit_cov_qual = int(fit_result.covQual())
    mean1_val = fit_result.floatParsFinal().find("mean1").getVal()
    sigma1_val = fit_result.floatParsFinal().find("sigma1").getVal()

    yields = {}
    print(f"\n  {'panel':>12s}  {'N_sig':>8s}  {'err':>8s}")
    for spin, side, _, _ in PANELS:
        suf = panel_suffix(spin, side)
        nsig1 = panel_pars[suf][1]
        yields[f"{spin}_{side}"] = (nsig1.getVal(), nsig1.getError())
        print(f"  {suf:>12s}  {nsig1.getVal():>10.3f}  {nsig1.getError():>10.3f}")

    N_uL, dN_uL = yields["up_left"]
    N_uR, dN_uR = yields["up_right"]
    N_dL, dN_dL = yields["down_left"]
    N_dR, dN_dR = yields["down_right"]

    res = _compute_an(N_uL, N_uR, N_dL, N_dR, ETA, F, P_AVG)
    if res is None:
        raise RuntimeError("zero or negative yield in at least one panel -- cannot compute A_N")
    A_raw, A_N = res
    dA_N = An_error(N_uL, dN_uL, N_uR, dN_uR, N_dL, dN_dL, N_dR, dN_dR, ETA, F, P_AVG)

    result_file = out_dir / "an_result.txt"
    with open(result_file, "w") as fp:
        fp.write(f"A_N   = {A_N:+.6f}\n")
        fp.write(f"dA_N  = {dA_N:.6f}\n")

    fit_params = {
        "mean1": mean1_val, "sigma1": sigma1_val, "ratio": float(RATIO),
        "mass_min": float(MASS_MIN), "mass_max": float(MASS_MAX),
        "panels": {}, "A_N": A_N, "dA_N": dA_N, "A_raw": A_raw,
        "eta": ETA, "f": F, "P": P_AVG,
        "fit_status": fit_status, "fit_cov_qual": fit_cov_qual,
    }
    for spin, side, _, _ in PANELS:
        suf = panel_suffix(spin, side)
        tau_p, nsig1_p, nsig2_p, nbkg_p = panel_pars[suf]
        fit_params["panels"][suf] = {
            "N_sig1": nsig1_p.getVal(), "dN_sig1": nsig1_p.getError(),
            "N_sig2": nsig2_p.getVal(), "dN_sig2": nsig2_p.getError(),
            "N_bkg": nbkg_p.getVal(), "dN_bkg": nbkg_p.getError(),
            "tau": tau_p.getVal(),
        }
    with open(out_dir / "fit_params.json", "w") as fp:
        json.dump(fit_params, fp, indent=2)

    print(f"\n  A_raw = {A_raw:+.4f}")
    print(f"  A_N   = {A_N:+.4f} +/- {dA_N:.4f}   (eta={ETA}, f={F}, P={P_AVG})")
    print(f"  saved -> {result_file}")
    return fit_params


def main():
    extract_massfit_an()


if __name__ == "__main__":
    main()
