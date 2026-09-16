#!/usr/bin/env python3
"""Flattens a raw jagged ktracker EXP tree into one row per surviving dimuon
candidate. Selection: vz>-600 (both), |y_st1|>3 (both), opposite station-1
py slopes. No mass window here -- left to tag().

    ROOTPY=/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python
    $ROOTPY -m an.flatten --input <raw.root> --output <flat.root>
"""
from __future__ import annotations

import argparse
from pathlib import Path

MUON_MASS = 0.105658  # GeV/c^2
VZ_MIN = -600.0
Y_ST1_MIN = 3.0


def flatten_raw_exp(input_path: Path, output_path: Path, tree_name: str = "tree") -> tuple[int, int]:
    """Flatten a raw EXP ROOT file. Returns (n_raw_events, n_flat_candidates)."""
    import numpy as np
    import ROOT

    fin = ROOT.TFile(str(input_path), "READ")
    tin = fin.Get(tree_name)
    if not tin.GetListOfBranches().FindObject("rec_dimuon_px_pos_tgt"):
        raise RuntimeError(
            f"{input_path} has no 'rec_dimuon_px_pos_tgt' branch -- this isn't a "
            "raw ktracker file (maybe already flattened/tagged?).")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fout = ROOT.TFile(str(output_path), "RECREATE")
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
            if not (VZ_MIN < vz1 and VZ_MIN < vz2):
                continue

            y1 = event.rec_dimuon_y_pos_st1[i]
            y2 = event.rec_dimuon_y_neg_st1[i]
            if not (abs(y1) > Y_ST1_MIN and abs(y2) > Y_ST1_MIN):
                continue

            py1 = event.rec_dimuon_py_pos_st1[i]
            py2 = event.rec_dimuon_py_neg_st1[i]
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
    return n_in, n_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--tree", default="tree")
    args = ap.parse_args()

    n_in, n_out = flatten_raw_exp(args.input, args.output, args.tree)
    print(f"[flatten] {args.input}")
    print(f"[flatten] {n_in:,} raw events -> {n_out:,} flat dimuon candidates")
    print(f"[flatten] wrote {args.output}")


if __name__ == "__main__":
    main()
