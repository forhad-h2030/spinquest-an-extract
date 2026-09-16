"""MC test-bundle loader/resampler: loads each bootstrap seed's held-out MC
test set, resamples it to the EXP data's own class mixture (DATA_PRIOR), and
hands back per-seed contamination fractions for asymmetry.py.

Single checkpoint by design (20063581, 12-feat AdamW fixed-stop ep≤80)
-- tagging, purity resampling, and A_N extraction all use it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import uproot

from an.config import MASS_LO, MASS_HI  # noqa: F401 (re-exported for asymmetry.py)

REPO = Path(__file__).resolve().parents[2]

CKPT_DIR  = REPO / "checkpoints" / "outputs_12feat_adamw_ep120_maxbest80_20063581" / "multiclass_dnn_12feat_adamw_ep120_nogapstop_20seed"
RUN_NAME  = "multiclass_dnn_12feat_adamw_ep120_nogapstop_20seed"
DATA_FILE = REPO / "exp_tagged_data" / "exp_tagged_tgt_data_20063581.root"

CLS      = ["jpsi", "psip", "dy", "comb"]              # canonical key order
NAME2KEY = {"J/psi": "jpsi", "psi(2S)": "psip", "DY": "dy", "Combinatoric": "comb"}
ML_CLASS_CODE = {"jpsi": 1, "psip": 2, "dy": 3, "comb": 4}
RESAMPLE_SEED = 42
N_DRAW = 20_000   # synthetic MC sample size per seed, split by DATA_PRIOR proportions


def compute_data_prior(lo: float, hi: float, x_cut: float | None = 25.0,
                       y_cut: float | None = 3.0, pz_cut: float | None = 5.0) -> np.ndarray:
    """EXP-side class-mixture prior: argmax-tagged event counts under the
    mass + station1 + pz selection (must match asymmetry.py's PZ_CUT so the
    prior reflects the same population the raw A_N counts are drawn from)."""
    with uproot.open(f"{DATA_FILE}:tree") as t:
        keys = set(t.keys())
        need = ["mass", "rec_track_pos_x_st1", "rec_track_neg_x_st1",
                "rec_track_pos_y_st1", "rec_track_neg_y_st1",
                "rec_dimu_mu_pos_pz", "rec_dimu_mu_neg_pz"]
        use_ml_class = "ml_class" in keys
        if use_ml_class:
            a = t.arrays(need + ["ml_class"], library="np")
        else:
            pcols = [f"ml_p_{c}" for c in CLS]
            if not all(c in keys for c in pcols):
                raise RuntimeError("tagged data has neither ml_class nor ml_p_<class> branches")
            a = t.arrays(need + pcols, library="np")

    sel = (a["mass"] >= lo) & (a["mass"] <= hi)
    if x_cut is not None:
        sel &= (a["rec_track_pos_x_st1"] < x_cut) & (a["rec_track_neg_x_st1"] < x_cut)
    if y_cut is not None:
        sel &= (np.abs(a["rec_track_pos_y_st1"]) > y_cut) & \
               (np.abs(a["rec_track_neg_y_st1"]) > y_cut)
    if pz_cut is not None:
        sel &= (a["rec_dimu_mu_pos_pz"] > pz_cut) & (a["rec_dimu_mu_neg_pz"] > pz_cut)

    if use_ml_class:
        return np.array([(sel & (a["ml_class"] == ML_CLASS_CODE[c])).sum() for c in CLS], float)
    arg = np.column_stack([a[f"ml_p_{c}"] for c in CLS]).astype(float).argmax(axis=1)
    return np.array([(sel & (arg == k)).sum() for k in range(len(CLS))], float)


try:
    DATA_PRIOR = compute_data_prior(MASS_LO, MASS_HI)
except (FileNotFoundError, OSError):
    DATA_PRIOR = None

def set_mass_window(lo: float, hi: float, x_cut: float | None = 25.0,
                    y_cut: float | None = 3.0, pz_cut: float | None = 5.0) -> None:
    global MASS_LO, MASS_HI, DATA_PRIOR
    MASS_LO, MASS_HI = lo, hi
    DATA_PRIOR = compute_data_prior(lo, hi, x_cut, y_cut, pz_cut)
    _bundle.cache_clear()


def seeds() -> list[str]:
    """Boot dirs with both .best.pth and .test_bundle.npz present."""
    complete = []
    for p in sorted(CKPT_DIR.glob("boot_*")):
        if not p.is_dir():
            continue
        base = p / RUN_NAME
        if Path(f"{base}.best.pth").exists() and Path(f"{base}.test_bundle.npz").exists():
            complete.append(p.name)
    return complete


@lru_cache(maxsize=None)
def _bundle(boot: str) -> dict:
    """One seed's MC test bundle, resampled to DATA_PRIOR's class mixture."""
    import torch
    base = CKPT_DIR / boot / RUN_NAME
    b  = np.load(f"{base}.test_bundle.npz", allow_pickle=True)
    ck = torch.load(f"{base}.best.pth", map_location="cpu", weights_only=False)

    class_names = [str(c) for c in b["class_names"]]
    order = [class_names.index(n) for n in NAME2KEY]     # trained col -> CLS order
    yt_raw = b["y_test"].astype(int)
    remap  = np.empty(4, int)
    for new_k, old_k in enumerate(order):
        remap[old_k] = new_k
    yt = remap[yt_raw]
    pr = b["y_proba"].astype(np.float64)[:, order]

    sc = ck["scaler"]
    mn = np.array(sc["mean"]).flatten()
    sd = np.array(sc["std"]).flatten()
    feat_key = "feature_names" if "feature_names" in b else "kept_feature_names"
    feat_names = [str(f) for f in b[feat_key]]
    c_mass = feat_names.index("rec_dimu_M")
    mass = b["X_test"][:, c_mass] * sd[c_mass] + mn[c_mass]

    inw = (mass >= MASS_LO) & (mass <= MASS_HI)
    frac = DATA_PRIOR / DATA_PRIOR.sum()
    rng = np.random.default_rng(RESAMPLE_SEED)
    idx_draw = []
    for k in range(4):
        pool = np.nonzero(inw & (yt == k))[0]
        n_k = int(round(N_DRAW * frac[k]))
        if n_k > len(pool):
            raise RuntimeError(f"[{boot}] class {CLS[k]}: need {n_k} events but "
                              f"pool only has {len(pool)} -- lower N_DRAW")
        idx_draw.append(rng.choice(pool, size=n_k, replace=False))
    idx_draw = np.concatenate(idx_draw)
    return dict(yt=yt[idx_draw], pr=pr[idx_draw], mass=mass[idx_draw])
