"""The 12 features the production model (12-feat resnet4) trains on,
computed from a flattened EXP tree row, in the checkpoint's own
`feature_names` order.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

MUON_MASS_GEV = 0.1056

REQUIRED_BRANCHES = [
    "rec_dimu_mu_pos_px", "rec_dimu_mu_pos_py", "rec_dimu_mu_pos_pz",
    "rec_dimu_mu_neg_px", "rec_dimu_mu_neg_py", "rec_dimu_mu_neg_pz",
    "rec_track_pos_vz", "rec_track_neg_vz",
]

FEATURE_NAMES = [
    "rec_dimu_y", "rec_dimu_eta", "rec_dimu_E", "rec_dimu_pz", "rec_dimu_M",
    "rec_mu_theta_pos", "rec_mu_theta_neg", "rec_mu_open_angle",
    "rec_mu_Epos", "rec_mu_Eneg", "rec_dz_vtx", "rec_mu_deltaR",
]
MASS_IDX = FEATURE_NAMES.index("rec_dimu_M")

CLASS_CODE = {"jpsi": 1, "psip": 2, "dy": 3, "comb": 4}
CLASS_NAME_TO_KEY = {"J/psi": "jpsi", "psi(2S)": "psip", "DY": "dy", "Combinatoric": "comb"}


def _delta_phi(phi1: float, phi2: float) -> float:
    dphi = phi1 - phi2
    return (dphi + math.pi) % (2.0 * math.pi) - math.pi


def require_branches(tree, names) -> None:
    br_list = tree.GetListOfBranches()
    missing = [b for b in names if not br_list.FindObject(b)]
    if missing:
        raise RuntimeError(f"Missing required branches in tree '{tree.GetName()}': {missing}")


def compute_features(event) -> Optional[np.ndarray]:
    """The 12 features (float32, FEATURE_NAMES order) for one flattened-tree
    row (after `tree.GetEntry(i)`). None if invalid (degenerate momenta)."""
    import ROOT

    pxp = float(getattr(event, "rec_dimu_mu_pos_px"))
    pyp = float(getattr(event, "rec_dimu_mu_pos_py"))
    pzp = float(getattr(event, "rec_dimu_mu_pos_pz"))
    pxn = float(getattr(event, "rec_dimu_mu_neg_px"))
    pyn = float(getattr(event, "rec_dimu_mu_neg_py"))
    pzn = float(getattr(event, "rec_dimu_mu_neg_pz"))

    mu_pos = ROOT.TLorentzVector()
    mu_neg = ROOT.TLorentzVector()
    mu_pos.SetXYZM(pxp, pyp, pzp, MUON_MASS_GEV)
    mu_neg.SetXYZM(pxn, pyn, pzn, MUON_MASS_GEV)

    dimu = mu_pos + mu_neg
    m = float(dimu.M())

    vpos = mu_pos.Vect()
    vneg = mu_neg.Vect()
    denom = float(vpos.Mag() * vneg.Mag())
    if denom <= 0:
        return None
    cos_open = float(vpos.Dot(vneg) / denom)
    open_angle = float(math.acos(max(-1.0, min(1.0, cos_open))))

    theta_pos = np.arctan(mu_pos.Px() / mu_pos.Pz())
    theta_neg = np.arctan(mu_neg.Px() / mu_neg.Pz())

    zpos = float(getattr(event, "rec_track_pos_vz"))
    zneg = float(getattr(event, "rec_track_neg_vz"))
    dz_vtx = float(zpos - zneg)

    d_eta = float(mu_pos.Eta() - mu_neg.Eta())
    d_phi = _delta_phi(mu_pos.Phi(), mu_neg.Phi())
    deltaR = float(math.sqrt(d_eta * d_eta + d_phi * d_phi))

    return np.array([
        dimu.Rapidity(), dimu.Eta(), dimu.E(), dimu.Pz(), m,
        theta_pos, theta_neg, open_angle,
        mu_pos.E(), mu_neg.E(), dz_vtx, deltaR,
    ], dtype=np.float32)
