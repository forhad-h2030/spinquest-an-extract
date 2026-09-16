#!/usr/bin/env python3
"""Flattens a raw jagged ktracker tree into one row per surviving dimuon
candidate (momenta, vertices, station-1 vars, mass, pt).

Selection: vz_pos>-600 & vz_neg>-600, |y_st1_pos|>3 & |y_st1_neg|>3,
py_st1_pos*py_st1_neg<0 (opposite station-1 slopes). No mass window or
dimuon-momentum box cuts here -- left to downstream consumers.

  ROOTPY=/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python
  $ROOTPY post_processing/flatten_exp.py --input <raw.root> --output <flat.root>

e.g. to (re)build the flat files tssa_fit/extract_an.py reads (run from repo root):
  $ROOTPY post_processing/flatten_exp.py \
      --input data/exp_data_up_June30_refit_2026.root \
      --output exp_tagged_data/exp_tgt_data_june30_up.root
(swap up->down for the other spin file; see build_tuning_input_multiclass.sh)
"""
import argparse
import ROOT

MUON_MASS = 0.105658  # GeV/c^2
VZ_MIN = -600.0
Y_ST1_MIN = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tree", default="tree")
    args = ap.parse_args()

    import numpy as np
    fin = ROOT.TFile(args.input, "READ")
    tin = fin.Get(args.tree)

    fout = ROOT.TFile(args.output, "RECREATE")
    tout = ROOT.TTree("tree", "Muon 4-vectors and vertices")

    buf = {n: np.zeros(1, dtype=np.float64) for n in [
        "mass", "pt",
        "rec_dimu_mu_pos_px", "rec_dimu_mu_pos_py", "rec_dimu_mu_pos_pz",
        "rec_track_pos_vx", "rec_track_pos_vy", "rec_track_pos_vz",
        "rec_dimu_mu_neg_px", "rec_dimu_mu_neg_py", "rec_dimu_mu_neg_pz",
        "rec_track_neg_vx", "rec_track_neg_vy", "rec_track_neg_vz",
        "dimuon_px", "dimuon_py", "dimuon_pz",
        "rec_track_pos_px_st1", "rec_track_pos_x_st1", "rec_track_pos_y_st1",
        "rec_track_neg_px_st1", "rec_track_neg_x_st1", "rec_track_neg_y_st1",
        "rec_track_pos_py_st1", "rec_track_neg_py_st1",
    ]}
    for n, a in buf.items():
        tout.Branch(n, a, f"{n}/D")

    n_in = 0
    n_out = 0
    for event in tin:
        n_in += 1
        n_dim = len(event.rec_dimuon_px_pos_tgt)
        if n_dim == 0:
            continue
        for i in range(n_dim):
            vz1 = event.rec_dimuon_z_pos_vtx[i]
            vz2 = event.rec_dimuon_z_neg_vtx[i]
            # NOTE: |vz_pos - vz_neg| < 200 requirement intentionally dropped.
            if not (VZ_MIN < vz1 and VZ_MIN < vz2):
                continue

            y1 = event.rec_dimuon_y_pos_st1[i]
            y2 = event.rec_dimuon_y_neg_st1[i]
            if not (abs(y1) > Y_ST1_MIN and abs(y2) > Y_ST1_MIN):
                continue

            py1 = event.rec_dimuon_py_pos_st1[i]
            py2 = event.rec_dimuon_py_neg_st1[i]
            # station-1 opposite-vertical-slope topology cut (TSSA reference).
            if not (py1 * py2 < 0):
                continue

            buf["rec_dimu_mu_pos_px"][0] = event.rec_dimuon_px_pos_tgt[i]
            buf["rec_dimu_mu_pos_py"][0] = event.rec_dimuon_py_pos_tgt[i]
            buf["rec_dimu_mu_pos_pz"][0] = event.rec_dimuon_pz_pos_tgt[i]
            buf["rec_dimu_mu_neg_px"][0] = event.rec_dimuon_px_neg_tgt[i]
            buf["rec_dimu_mu_neg_py"][0] = event.rec_dimuon_py_neg_tgt[i]
            buf["rec_dimu_mu_neg_pz"][0] = event.rec_dimuon_pz_neg_tgt[i]

            buf["rec_track_pos_px_st1"][0] = event.rec_dimuon_px_pos_st1[i]
            buf["rec_track_pos_x_st1"][0]  = event.rec_dimuon_x_pos_st1[i]
            buf["rec_track_pos_y_st1"][0]  = event.rec_dimuon_y_pos_st1[i]
            buf["rec_track_neg_px_st1"][0] = event.rec_dimuon_px_neg_st1[i]
            buf["rec_track_neg_x_st1"][0]  = event.rec_dimuon_x_neg_st1[i]
            buf["rec_track_neg_y_st1"][0]  = event.rec_dimuon_y_neg_st1[i]
            buf["rec_track_pos_py_st1"][0] = py1
            buf["rec_track_neg_py_st1"][0] = py2

            tp = ROOT.TLorentzVector(); tn = ROOT.TLorentzVector()
            tp.SetXYZM(buf["rec_dimu_mu_pos_px"][0], buf["rec_dimu_mu_pos_py"][0],
                       buf["rec_dimu_mu_pos_pz"][0], MUON_MASS)
            tn.SetXYZM(buf["rec_dimu_mu_neg_px"][0], buf["rec_dimu_mu_neg_py"][0],
                       buf["rec_dimu_mu_neg_pz"][0], MUON_MASS)
            dimu = tp + tn
            buf["mass"][0] = dimu.M(); buf["pt"][0] = dimu.Pt()
            buf["dimuon_px"][0] = dimu.Px(); buf["dimuon_py"][0] = dimu.Py()
            buf["dimuon_pz"][0] = dimu.Pz()

            buf["rec_track_pos_vx"][0] = event.rec_dimuon_x_pos_vtx[i]
            buf["rec_track_pos_vy"][0] = event.rec_dimuon_y_pos_vtx[i]
            buf["rec_track_pos_vz"][0] = event.rec_dimuon_z_pos_vtx[i]
            buf["rec_track_neg_vx"][0] = event.rec_dimuon_x_neg_vtx[i]
            buf["rec_track_neg_vy"][0] = event.rec_dimuon_y_neg_vtx[i]
            buf["rec_track_neg_vz"][0] = event.rec_dimuon_z_neg_vtx[i]

            tout.Fill()
            n_out += 1

    fout.cd(); tout.Write(); fout.Close(); fin.Close()
    print(f"[flatten] {args.input}")
    print(f"[flatten] {n_in:,} raw events -> {n_out:,} flat dimuon candidates")
    print(f"[flatten] wrote {args.output}")


if __name__ == "__main__":
    main()
