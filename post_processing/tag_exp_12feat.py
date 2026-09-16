#!/usr/bin/env python3
"""
tag_exp_12feat.py — self-contained 12-feat multiclass EXP tagger.

Tags a flat EXP ROOT tree with the production 12-feat AdamW gap-stop ensemble
(job 19485623, 20 bootstrap seeds). Adds branches:
  ml_class, spin_up, ml_p_{jpsi,psip,dy,comb}   (ensemble average)
  ml_p_{jpsi,psip,dy,comb}_{boot_NNN}            (per-seed)
  ml_feat_*                                        (18 raw features)

All model and feature logic is inlined — no local utils/ imports required.

Example (run from /Users/spin/spinquest-an-extract):
  PYROOT=/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python
  $PYROOT post_processing/tag_exp_12feat.py --spin up
  $PYROOT post_processing/tag_exp_12feat.py --spin down
"""
from __future__ import annotations

from pathlib import Path
import math
import array
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ROOT
from ROOT import TLorentzVector

# ── paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
REPO      = HERE.parent
CKPT_DIR  = (REPO / "checkpoints" /
             "outputs_12feat_adamw_ep120_maxbest80_20063581" /
             "multiclass_dnn_12feat_adamw_ep120_nogapstop_20seed")
CKPT_NAME = "multiclass_dnn_12feat_adamw_ep120_nogapstop_20seed.best.pth"

# ── 18 training features (order matches training convention) ──────────────────
TRAIN_FEATURES = [
    "rec_dimu_y", "rec_dimu_eta", "rec_dimu_E", "rec_dimu_pz", "rec_dimu_M",
    "rec_mu_theta_pos", "rec_mu_theta_neg", "rec_mu_open_angle",
    "rec_mu_dpt", "rec_dimu_mT", "rec_mu_Epos", "rec_mu_Eneg",
    "rec_track_pos_x_st1", "rec_track_neg_x_st1",
    "rec_track_pos_px_st1", "rec_track_neg_px_st1",
    "rec_dz_vtx", "rec_mu_deltaR",
]

# short branch-name aliases (used for ml_feat_* branches)
FEATURE_NAMES = [
    "dimu_y", "dimu_eta", "dimu_E", "dimu_pz", "rec_dimu_M",
    "theta_pos", "theta_neg", "open_angle",
    "dpt", "dimu_mT", "Epos", "Eneg",
    "st1_x_pos", "st1_x_neg", "st1_px_pos", "st1_px_neg",
    "dz_vtx", "deltaR",
]

REQUIRED_BRANCHES = [
    "rec_dimu_mu_pos_px", "rec_dimu_mu_pos_py", "rec_dimu_mu_pos_pz",
    "rec_dimu_mu_neg_px", "rec_dimu_mu_neg_py", "rec_dimu_mu_neg_pz",
    "rec_track_pos_x_st1", "rec_track_neg_x_st1",
    "rec_track_pos_px_st1", "rec_track_neg_px_st1",
    "rec_track_pos_vz", "rec_track_neg_vz",
]

CLASS_CODE        = {"jpsi": 1, "psip": 2, "dy": 3, "comb": 4}
CLASS_NAME_TO_KEY = {"J/psi": "jpsi", "psi(2S)": "psip", "DY": "dy", "Combinatoric": "comb"}
MUON_MASS_GEV     = 0.1056


# ── model definition (inlined from core_train_multiclass_featsel.py) ──────────
class ParticleClassifierMulticlass(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512,
                 num_layers=4, dropout_rate=0.3, flat=False):
        super().__init__()
        layers, in_dim, h = [], input_dim, hidden_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, h), nn.ReLU(),
                       nn.BatchNorm1d(h), nn.Dropout(dropout_rate)]
            in_dim = h
            if not flat:
                h = max(h // 2, 8)
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ── feature computation ───────────────────────────────────────────────────────
def _delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + math.pi) % (2.0 * math.pi) - math.pi


def compute_18_features(event) -> Optional[np.ndarray]:
    pxp = float(getattr(event, "rec_dimu_mu_pos_px"))
    pyp = float(getattr(event, "rec_dimu_mu_pos_py"))
    pzp = float(getattr(event, "rec_dimu_mu_pos_pz"))
    pxn = float(getattr(event, "rec_dimu_mu_neg_px"))
    pyn = float(getattr(event, "rec_dimu_mu_neg_py"))
    pzn = float(getattr(event, "rec_dimu_mu_neg_pz"))

    mu_pos, mu_neg = TLorentzVector(), TLorentzVector()
    mu_pos.SetXYZM(pxp, pyp, pzp, MUON_MASS_GEV)
    mu_neg.SetXYZM(pxn, pyn, pzn, MUON_MASS_GEV)
    dimu = mu_pos + mu_neg
    m = float(dimu.M())

    vpos, vneg = mu_pos.Vect(), mu_neg.Vect()
    denom = float(vpos.Mag() * vneg.Mag())
    if denom <= 0:
        return None
    cos_open   = float(vpos.Dot(vneg) / denom)
    open_angle = float(math.acos(max(-1.0, min(1.0, cos_open))))
    dimu_pt    = float(dimu.Pt())
    dimu_mT    = float(math.sqrt(m * m + dimu_pt * dimu_pt))
    theta_pos  = float(np.arctan(mu_pos.Px() / mu_pos.Pz()))
    theta_neg  = float(np.arctan(mu_neg.Px() / mu_neg.Pz()))
    d_eta      = float(mu_pos.Eta() - mu_neg.Eta())
    d_phi      = float(_delta_phi(mu_pos.Phi(), mu_neg.Phi()))
    deltaR     = float(math.sqrt(d_eta * d_eta + d_phi * d_phi))

    st1_x_pos  = float(getattr(event, "rec_track_pos_x_st1"))
    st1_x_neg  = float(getattr(event, "rec_track_neg_x_st1"))
    st1_px_pos = float(getattr(event, "rec_track_pos_px_st1"))
    st1_px_neg = float(getattr(event, "rec_track_neg_px_st1"))
    dz_vtx     = float(getattr(event, "rec_track_pos_vz")) - float(getattr(event, "rec_track_neg_vz"))

    return np.array([
        float(dimu.Rapidity()), float(dimu.Eta()), float(dimu.E()), float(dimu.Pz()), m,
        theta_pos, theta_neg, open_angle,
        float(mu_pos.Pt() - mu_neg.Pt()), dimu_mT, float(mu_pos.E()), float(mu_neg.E()),
        st1_x_pos, st1_x_neg, st1_px_pos, st1_px_neg,
        dz_vtx, deltaR,
    ], dtype=np.float32)


def apply_scaler(x, scaler):
    if scaler is None:
        return x
    mu = np.asarray(scaler["mean"], dtype=np.float32).reshape(-1)
    sd = np.asarray(scaler["std"],  dtype=np.float32).reshape(-1)
    return ((x - mu) / sd).astype(np.float32)


@torch.no_grad()
def softmax_probs(model, X, device, batch_size=8192):
    out = []
    for s in range(0, len(X), batch_size):
        xb = torch.tensor(X[s:s + batch_size], dtype=torch.float32, device=device)
        out.append(F.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def require_branches(tree, names):
    br_list = tree.GetListOfBranches()
    missing = [b for b in names if not br_list.FindObject(b)]
    if missing:
        raise RuntimeError(f"Missing required branches: {missing}")


# ── checkpoint loading ────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: Path, device: str):
    """Returns (model, scaler, class_names, keep_idx)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "input_dim" in ckpt):
        raise RuntimeError(f"{ckpt_path} is not a valid multiclass checkpoint.")

    input_dim   = int(ckpt["input_dim"])
    num_classes = int(ckpt["num_classes"])
    class_names = list(ckpt["class_names"])
    cfg   = ckpt.get("cfg", {})
    model = ParticleClassifierMulticlass(
        input_dim=input_dim, num_classes=num_classes,
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        num_layers=int(cfg.get("num_layers", 4)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        flat=bool(cfg.get("flat", False)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    fnames   = ckpt.get("feature_names")
    keep_idx = None
    if fnames:
        missing = [n for n in fnames if n not in TRAIN_FEATURES]
        if missing:
            raise RuntimeError(f"{ckpt_path}: unknown feature_names: {missing}")
        keep_idx = np.array([TRAIN_FEATURES.index(n) for n in fnames], dtype=int)
        if len(keep_idx) != input_dim:
            raise RuntimeError(f"{ckpt_path}: feature_names/input_dim mismatch")
        print(f"[INFO] {ckpt_path.parent.name}: {len(keep_idx)}-feat subset {list(keep_idx)}")

    return model, ckpt.get("scaler", None), class_names, keep_idx


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse

    p = argparse.ArgumentParser(
        description="12-feat multiclass EXP tagger (production ensemble, job 19485623).",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input",  type=Path,
                   default=REPO / "exp_tagged_data" / "exp_tgt_data_june30_up.root")
    p.add_argument("--output", type=Path,
                   default=REPO / "exp_tagged_data" /
                           "exp_tagged_tgt_data_outputs_final_12feat_classweighted_19485623.root")
    p.add_argument("--tree",      default="tree")
    p.add_argument("--mass-min",  type=float, default=0.2)
    p.add_argument("--mass-max",  type=float, default=8.0)
    p.add_argument("--ckpt-dir",  type=Path,  default=CKPT_DIR,
                   help="directory containing boot_*/<ckpt>.best.pth checkpoints")
    p.add_argument("--ckpt-name", default=CKPT_NAME,
                   help="checkpoint filename to look for inside each boot_* dir")
    p.add_argument("--spin", choices=["up", "down", "unknown"], default="unknown")
    args = p.parse_args()

    spin_code = {"up": 1, "down": 0, "unknown": -1}[args.spin]

    # collect ensemble checkpoints
    ckpts = sorted(args.ckpt_dir.glob(f"boot_*/{args.ckpt_name}"))
    if not ckpts:
        raise SystemExit(f"[ERROR] no checkpoints found in {args.ckpt_dir}/boot_*/{args.ckpt_name}")
    print(f"[INFO] ensemble: {len(ckpts)} checkpoint(s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    models, class_names_ref, keep_idx_ref = [], None, None
    for c in ckpts:
        model, scaler, cnames, keep_idx = load_checkpoint(c, device)
        if class_names_ref is None:
            class_names_ref, keep_idx_ref = cnames, keep_idx
        else:
            if cnames != class_names_ref:
                raise SystemExit(f"[ERROR] class_names mismatch: {c}")
            same = (keep_idx is None and keep_idx_ref is None) or \
                   (keep_idx is not None and keep_idx_ref is not None and
                    np.array_equal(keep_idx, keep_idx_ref))
            if not same:
                raise SystemExit(f"[ERROR] feature subset mismatch: {c}")
        models.append((model, scaler))

    keys = [CLASS_NAME_TO_KEY.get(c) for c in class_names_ref]
    if None in keys or set(keys) != set(CLASS_CODE):
        raise SystemExit(f"[ERROR] unexpected class_names {class_names_ref}")
    col             = {k: i for i, k in enumerate(keys)}
    argmax_to_code  = np.array([CLASS_CODE[k] for k in keys], dtype=np.int32)
    model_tags      = [c.parent.name for c in ckpts]

    if not args.input.exists():
        raise SystemExit(f"[ERROR] --input not found: {args.input}")

    fin = ROOT.TFile.Open(str(args.input), "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Could not open {args.input}")
    tin = fin.Get(args.tree)
    if not tin:
        raise RuntimeError(f"Tree '{args.tree}' not found")
    require_branches(tin, REQUIRED_BRANCHES)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fout = ROOT.TFile.Open(str(args.output), "RECREATE")
    tout = tin.CloneTree(0)

    b_class = array.array("i", [0])
    b_spin  = array.array("i", [spin_code])
    b_pjpsi = array.array("f", [0.0])
    b_ppsip = array.array("f", [0.0])
    b_pdy   = array.array("f", [0.0])
    b_pcomb = array.array("f", [0.0])
    tout.Branch("ml_class",  b_class, "ml_class/I")
    tout.Branch("spin_up",   b_spin,  "spin_up/I")
    tout.Branch("ml_p_jpsi", b_pjpsi, "ml_p_jpsi/F")
    tout.Branch("ml_p_psip", b_ppsip, "ml_p_psip/F")
    tout.Branch("ml_p_dy",   b_pdy,   "ml_p_dy/F")
    tout.Branch("ml_p_comb", b_pcomb, "ml_p_comb/F")

    b_feats = [array.array("f", [0.0]) for _ in range(18)]
    for i, fname in enumerate(FEATURE_NAMES):
        tout.Branch(f"ml_feat_{fname}", b_feats[i], f"ml_feat_{fname}/F")

    b_pseed = []
    for tag in model_tags:
        d = {k: array.array("f", [0.0]) for k in CLASS_CODE}
        for k in CLASS_CODE:
            tout.Branch(f"ml_p_{k}_{tag}", d[k], f"ml_p_{k}_{tag}/F")
        b_pseed.append(d)

    # pass 1: features + mass window
    n_in = int(tin.GetEntries())
    kept_idx, kept_feats, skipped_zero = [], [], 0
    for i in range(n_in):
        tin.GetEntry(i)
        feats = compute_18_features(tin)
        if feats is None:
            skipped_zero += 1
            continue
        m = float(feats[4])
        if m < args.mass_min or m > args.mass_max:
            continue
        kept_idx.append(i)
        kept_feats.append(feats)

    if not kept_feats:
        raise SystemExit("[ERROR] no events survived the mass window")

    X    = np.vstack(kept_feats).astype(np.float32)
    Xsub = X[:, keep_idx_ref] if keep_idx_ref is not None else X

    probs_per_model = [softmax_probs(model, apply_scaler(Xsub, scaler), device)
                       for model, scaler in models]
    probs      = np.mean(probs_per_model, axis=0)
    pred_code  = argmax_to_code[probs.argmax(axis=1)]

    # pass 2: fill output tree
    for row, i in enumerate(kept_idx):
        tin.GetEntry(i)
        for j in range(18):
            b_feats[j][0] = float(X[row, j])
        b_pjpsi[0] = float(probs[row, col["jpsi"]])
        b_ppsip[0] = float(probs[row, col["psip"]])
        b_pdy[0]   = float(probs[row, col["dy"]])
        b_pcomb[0] = float(probs[row, col["comb"]])
        for im, pm in enumerate(probs_per_model):
            for k in CLASS_CODE:
                b_pseed[im][k][0] = float(pm[row, col[k]])
        b_class[0] = int(pred_code[row])
        tout.Fill()

    tout.Write()
    fout.Close()
    fin.Close()

    fracs = {k: int((pred_code == CLASS_CODE[k]).sum()) for k in CLASS_CODE}
    summary = {
        "input":  str(args.input),
        "output": str(args.output),
        "n_models": len(models),
        "spin": args.spin,
        "mass_min": args.mass_min,
        "mass_max": args.mass_max,
        "n_entries_input":   n_in,
        "n_entries_skipped": skipped_zero,
        "n_entries_written": len(kept_idx),
        "feature_subset_cols": (list(map(int, keep_idx_ref)) if keep_idx_ref is not None else None),
        "predicted_class_counts": fracs,
    }
    print(json.dumps(summary, indent=2))
    print("[DONE]")


if __name__ == "__main__":
    main()
