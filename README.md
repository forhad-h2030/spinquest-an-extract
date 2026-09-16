# spinquest-an-extract

Extracts the transverse single-spin asymmetry (A_N) from SpinQuest EXP data
using a multiclass DNN ensemble, and compares with a mass-fit baseline.

---

## Environment setup

### Mac (local)

No module loading needed. Tested with:

| library | version |
|---|---|
| `uproot` | 5.6.0 |
| `numpy` | 2.0.2 |
| `matplotlib` | 3.9.4 |
| `torch` | 2.6.0 (step 1 only) |
| `ROOT` / PyROOT | 6.34.04 (step 1 flatten + step 3 mass fit) |

### Rivanna (UVA HPC)

Checkpoint and raw EXP data are stored at:

```
/project/ptgroup/spinquest/spinquest-tssa-multiclass-checkpoint/multiclass_12feat_20seed_final/
/project/ptgroup/spinquest/spinquest_tssa_exp_data/exp_data_up_June30_refit_2026.root
/project/ptgroup/spinquest/spinquest_tssa_exp_data/exp_data_down_June30_refit_2026.root
```

**One-time setup** (after cloning the repo):

```bash
source setup.sh   # loads gcc/11.4.0, openmpi/4.1.4, python/3.11.4, root/6.32.06

# Python packages not available as modules
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user uproot
```

Run `source setup.sh` at the start of every new session.

---

## Extraction steps

> **Note:** run `source setup.sh` (Rivanna) or activate your environment before each session.

### Step 1 — Flatten + tag EXP data

Converts raw SpinQuest EXP ROOT files into a flat tagged file with ML class probabilities.
Re-run only when the raw data or checkpoint changes.

```bash
# Mac (uses hardcoded default paths)
bash post_processing/build_tagged_data.sh

# Rivanna (pass data paths explicitly)
export CKPT_DIR=/project/ptgroup/spinquest/spinquest-tssa-multiclass-checkpoint/multiclass_12feat_20seed_final
export PYROOT=$(which python3)
export HADD=$(which hadd)

bash post_processing/build_tagged_data.sh \
  /project/ptgroup/spinquest/spinquest_tssa_exp_data/exp_data_up_June30_refit_2026.root \
  /project/ptgroup/spinquest/spinquest_tssa_exp_data/exp_data_down_June30_refit_2026.root
```

Output: `exp_tagged_data/exp_tagged_tgt_data_20063581.root`

Model: production 12-feat AdamW fixed-stop ep≤80, job **20063581**, 20 bootstrap seeds.
Tagging mass window: `[1.5, 6.0]` GeV/c²; extraction window: `[2.0, 6.0]` GeV/c².

> Raw EXP data produced by the ReVertex module:
> https://github.com/forhad-h2030/e1039-dimu-ana-exp/tree/main/ReVertex

---

### Steps 2–4 — Extract A_N and compare

Runs the ML-tag corrected A_N (step 2), mass-fit baseline (step 3), and
comparison plot (step 4). Requires only `uproot`, `numpy`, and `matplotlib` —
no ROOT or torch needed.

```bash
python3 run.py
```

Outputs written to `output/`:
- `an_result.txt` — corrected A_N (ML-tag method)
- `massfit/an_result.txt` — A_N from mass fit
- `an_method_comparison.png` — comparison plot

> Steps 2–4 can also be run individually: `python3 -m an.asymmetry`, `python3 -m an.massfit`, `python3 -m an.compare`.

---

## Optional diagnostics

| command | output | description |
|---|---|---|
| `python3 -m an.purity` | `output/purity_vs_threshold.txt` | purity vs threshold (MC resampled to EXP prior) |
| `python3 -m an.an_vs_threshold` | `output/an_vs_threshold.png` | raw A_N vs threshold scan |

---

## Requirements

| library | used by | notes |
|---|---|---|
| `uproot` | asymmetry.py, mc_purity.py, purity.py | read tagged ROOT files |
| `numpy` | all scripts | |
| `matplotlib` | compare.py, an_vs_threshold.py | plotting |
| `torch` | post_processing/tag_exp_12feat.py | DNN inference (step 1 only); CPU-only build sufficient |
| `ROOT` / PyROOT | post_processing/flatten_exp_single.py, massfit.py | step 1 flatten + step 3 mass-fit baseline |

---

## Layout

| file | role |
|---|---|
| `setup.sh` | Rivanna environment setup (modules + PYTHONPATH) |
| `run.py` | entry point: runs steps 2–4 with preflight check |
| `post_processing/build_tagged_data.sh` | full step 1 pipeline: flatten + tag + hadd |
| `post_processing/flatten_exp_single.py` | raw ktracker tree → flat dimuon candidates |
| `post_processing/tag_exp_12feat.py` | 12-feat ensemble tagger |
| `an/config.py` | shared physical constants and selection cuts |
| `an/asymmetry.py` | corrected A_N (ML-tag method) |
| `an/massfit.py` | A_N via mass-shape fit (no ML) |
| `an/compare.py` | both methods, one comparison plot |
| `an/purity.py` | purity vs threshold |
| `an/an_vs_threshold.py` | A_N vs threshold scan |
| `an/_lib/mc_purity.py` | MC resampling, contamination fractions (internal) |
