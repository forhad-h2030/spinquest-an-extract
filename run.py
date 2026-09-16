#!/usr/bin/env python3
"""Entry point for the SpinQuest J/psi A_N extraction pipeline.

Step 1 (flatten + tag EXP data) must be run separately before this script:

    bash post_processing/build_tagged_data.sh

This script runs steps 2-4:
    2. ML-tag background-corrected A_N  (an/asymmetry.py)
    3. Mass-fit baseline A_N            (an/massfit.py, requires PyROOT)
    4. Side-by-side comparison plot     (an/compare.py)

Outputs written to output/:
    an_result.txt              corrected A_N (ML-tag method)
    massfit/an_result.txt      A_N from mass fit
    an_method_comparison.png   comparison plot

Usage:
    python run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from an._lib.mc_purity import DATA_FILE
from an import compare


def main() -> None:
    if not DATA_FILE.exists():
        sys.exit(
            f"[ERROR] Tagged data file not found:\n  {DATA_FILE}\n\n"
            f"Run step 1 first:\n  bash post_processing/build_tagged_data.sh"
        )
    print(f"[OK] Tagged data: {DATA_FILE.name}")
    compare.main()


if __name__ == "__main__":
    main()
