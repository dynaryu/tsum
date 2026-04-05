"""
Estimate P(failure) for IEEE 118-bus using monotone classifier + importance sampling.

Instead of enumerating reference states (which cover negligible probability
for 711 components), this script:
  1. Trains a monotone classifier on (component-state, system-state) samples
  2. Refines it via active learning near the decision boundary
  3. Estimates P(failure) via importance sampling with the TRUE system function

Reference: Chan et al. (2024), Table 2: p_f ~ 1.0e-4

Usage:
    python run_classifier.py
    python run_classifier.py --n-initial 10000 --n-is 200000 --n-workers 48
"""

import sys
import os
import argparse
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import json

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from dcopt import make_dcopt_sfun
from tsum.classifier import estimate_failure_probability


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classifier-based P(failure) estimation for IEEE 118-bus")
    parser.add_argument("--n-initial", type=int, default=5000,
                        help="Initial training samples (default: 5000)")
    parser.add_argument("--n-active-rounds", type=int, default=5,
                        help="Active learning rounds (default: 5)")
    parser.add_argument("--n-active-samples", type=int, default=2000,
                        help="Samples per active learning round (default: 2000)")
    parser.add_argument("--n-active-candidates", type=int, default=50000,
                        help="Candidate pool for active selection (default: 50000)")
    parser.add_argument("--n-is", type=int, default=100000,
                        help="Importance sampling samples (default: 100000)")
    parser.add_argument("--is-shift", type=float, default=3.0,
                        help="IS shift factor (default: 3.0)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers for sfun evaluation (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="results_classifier",
                        help="Output directory (default: results_classifier)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Classifier-based P(failure) estimation")
    print("IEEE 118-bus DC-OPF")
    print("=" * 60)

    # Load input data
    data_dir = HERE / "case118_tsum_bus"
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)

    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())

    n_vars = len(row_names)
    n_gen = sum(1 for n in row_names if n.startswith("vbus")
                and len(probs_dict[n]) == 4)
    n_bin = n_vars - n_gen

    print(f"\n  Components: {n_vars} ({n_gen} generators 4-state, {n_bin} binary)")
    print(f"  Max states: {n_state}")

    # Build system function
    print("\nInitialising DC-OPF system function...")
    sfun = make_dcopt_sfun(
        case_path=str(HERE / "case118.m"),
        blackout_threshold=26.1,
        alpha=2.0,
    )

    # Sanity checks
    all_ok = {}
    for name in row_names:
        all_ok[name] = max(int(s) for s in probs_dict[name].keys())
    fval, sys_st, _ = sfun(all_ok)
    print(f"  All operational: blackout={fval:.4f}%, sys_st={sys_st}")

    all_fail = {name: 0 for name in row_names}
    fval, sys_st, _ = sfun(all_fail)
    print(f"  All failed:      blackout={fval:.4f}%, sys_st={sys_st}")

    # Run estimation
    output_dir = str(HERE / args.output_dir)
    result = estimate_failure_probability(
        sfun=sfun,
        probs_dict=probs_dict,
        row_names=row_names,
        n_state=n_state,
        n_initial_samples=args.n_initial,
        n_active_rounds=args.n_active_rounds,
        n_active_samples_per_round=args.n_active_samples,
        n_active_candidates=args.n_active_candidates,
        n_is_samples=args.n_is,
        is_shift_factor=args.is_shift,
        n_workers=args.n_workers,
        seed=args.seed,
        output_dir=output_dir,
    )

    print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")
    print(f"  Classifier IS estimate:          p_f = {result.p_failure:.4e} "
          f"± {result.p_failure_se:.4e}")


if __name__ == "__main__":
    main()
