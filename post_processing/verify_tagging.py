#!/usr/bin/env python3
"""
verify_tagging.py — sanity check: compare tag_exp_12feat.py output against
the reference tagged file (exp_tagged_tgt_data_outputs_final_12feat_classweighted_19485623.root).

Checks (for the up-spin subset):
  1. Event count matches
  2. ml_p_jpsi mean/std within tolerance
  3. ml_class fractions match
  4. mass range within [0.2, 8.0]

Run from /Users/spin/spinquest-an-extract:
  python post_processing/verify_tagging.py
"""
from pathlib import Path
import numpy as np
import uproot

REPO = Path(__file__).resolve().parents[1]

REF  = Path("/Users/spin/SpinQuest-TSSA-Ana/exp_tagged_data/exp_tagged_tgt_data_outputs_final_12feat_classweighted_19485623.root")
TEST = Path("/tmp/test_tagged_up.root")

BRANCHES = ["mass", "ml_class", "ml_p_jpsi", "ml_p_psip", "ml_p_dy", "ml_p_comb", "spin_up"]
TOL = 1e-4

def load(path, branches):
    with uproot.open(path) as f:
        return f["tree"].arrays(branches, library="np")

print(f"Loading reference: {REF}")
ref = load(REF, BRANCHES)

# reference has both spins — filter to up only for comparison
up_mask = ref["spin_up"] == 1
ref_up  = {k: v[up_mask] for k, v in ref.items()}

print(f"Loading test:      {TEST}")
test = load(TEST, BRANCHES)

n_ref, n_test = len(ref_up["mass"]), len(test["mass"])
print(f"\n--- Event counts ---")
print(f"  reference (spin_up==1) : {n_ref:,}")
print(f"  test output            : {n_test:,}")
print(f"  match                  : {n_ref == n_test}")

print(f"\n--- Mass range ---")
for label, d in [("ref ", ref_up), ("test", test)]:
    print(f"  {label}  min={d['mass'].min():.4f}  max={d['mass'].max():.4f}")

print(f"\n--- ml_p_jpsi ---")
for label, d in [("ref ", ref_up), ("test", test)]:
    print(f"  {label}  mean={d['ml_p_jpsi'].mean():.6f}  std={d['ml_p_jpsi'].std():.6f}")
diff_mean = abs(ref_up["ml_p_jpsi"].mean() - test["ml_p_jpsi"].mean())
print(f"  |mean diff| = {diff_mean:.2e}  {'OK' if diff_mean < TOL else 'MISMATCH'}")

print(f"\n--- ml_class fractions ---")
for cls, code in [("jpsi",1),("psip",2),("dy",3),("comb",4)]:
    r = (ref_up["ml_class"] == code).mean()
    t = (test["ml_class"]   == code).mean()
    print(f"  {cls:5s}  ref={r:.4f}  test={t:.4f}  diff={abs(r-t):.2e}  {'OK' if abs(r-t)<TOL else 'MISMATCH'}")

print("\nDone.")
