"""
Variable reduction for ACTIVSg2000: fix irrelevant components, run TSUM on generators only.

The ACTIVSg2000 failure structure is similar to IEEE 118-bus: failures are
concentrated in a small number of critical generator buses (primarily Houston
area buses 7255, 4073, 4041, 4042, 4040). With a ~3% blackout threshold,
max single-component impact is 1.35%, so failures require combinations of
2-3 degraded generators.

This script:
  1. Selects components to model (default: all 485 generators)
  2. Fixes everything else at best (operational) state
  3. Wraps the sfun so fixed components are injected automatically
  4. Optionally seeds with failure rules from fixed-k search
  5. Runs TSUM on the reduced problem (~485 variables instead of 5206)

Usage:
    # Generators only (485 variables, 4-state)
    python run_variable_reduction.py --n-workers 48 --output-dir results_reduced

    # With fixed-k seed rules
    python run_variable_reduction.py --seed-rules results_fixedk/seed_rules_fail.json \
        --n-workers 48 --output-dir results_reduced_seeded

    # Custom component selection: frequency-based from fixed-k results
    python run_variable_reduction.py --mode frequency --failures-dir results_fixedk --min-freq 50 \
        --n-workers 48 --output-dir results_reduced_top

    # Custom threshold
    python run_variable_reduction.py --blackout-threshold 2.0 --n-workers 48
"""

import sys
import os
import argparse
import json
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from dcopt import make_dcopt_sfun
from tsum.variable_reduction import (
    select_components, run_reduced_tsum,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Variable reduction TSUM for ACTIVSg2000")

    # Component selection
    parser.add_argument("--mode", type=str, default="multistate",
                        choices=["multistate", "frequency", "custom"],
                        help="Component selection mode (default: multistate)")
    parser.add_argument("--failures-dir", type=str, default=None,
                        help="Directory with failures_k*.json for frequency-based selection")
    parser.add_argument("--min-freq", type=int, default=1,
                        help="Minimum failure frequency to include a component (for --mode frequency)")
    parser.add_argument("--components", type=str, default="",
                        help="Comma-separated component names (for --mode custom)")

    # Seed rules
    parser.add_argument("--seed-rules", type=str, default=None,
                        help="Path to seed_rules_fail.json for seeding TSUM")

    # DC-OPF parameters
    parser.add_argument("--blackout-threshold", type=float, default=3.0,
                        help="Blackout threshold %% (default: 3.0)")

    # TSUM parameters
    parser.add_argument("--unk-prob-thres", type=float, default=1e-5,
                        help="TSUM convergence threshold (default: 1e-5)")
    parser.add_argument("--bias-factor", type=float, default=0.0,
                        help="Bias factor for TSUM discovery sampling (0=off)")
    parser.add_argument("--bias-rounds", type=int, default=0,
                        help="Use biased sampling for first N rounds (0=all)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--devices", type=str, default="",
                        help="Comma-separated GPU devices for TSUM")
    parser.add_argument("--output-dir", type=str, default="results_reduced",
                        help="Output directory (default: results_reduced)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Variable Reduction TSUM for ACTIVSg2000 DC-OPF")
    print("=" * 60)

    # Load input data
    data_dir = HERE / "case2000_tsum_bus"
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)

    # Select components
    custom_list = ([c.strip() for c in args.components.split(",") if c.strip()]
                   if args.components else None)
    selected, fixed = select_components(
        probs_dict,
        mode=args.mode,
        failures_dir=args.failures_dir,
        min_freq=args.min_freq,
        custom_list=custom_list,
    )

    # Build sfun
    print("\nInitialising DC-OPF system function (precomputed solver)...")
    base_sfun = make_dcopt_sfun(
        case_path=str(HERE / "case_ACTIVSg2000.m"),
        blackout_threshold=args.blackout_threshold,
        alpha=2.0,
    )

    # Load seed rules if provided
    seed_rules = None
    if args.seed_rules:
        seed_path = Path(args.seed_rules)
        if seed_path.exists():
            with open(seed_path) as f:
                seed_rules = json.load(f)
        else:
            print(f"Warning: seed rules file not found: {seed_path}")

    # Parse devices
    device_list = ([d.strip() for d in args.devices.split(",") if d.strip()]
                   if args.devices else None)

    # Run
    run_reduced_tsum(
        sfun=base_sfun,
        probs_dict=probs_dict,
        selected=selected,
        fixed=fixed,
        seed_rules=seed_rules,
        unk_prob_thres=args.unk_prob_thres,
        bias_factor=args.bias_factor,
        bias_rounds=args.bias_rounds,
        n_workers=args.n_workers,
        devices=device_list,
        output_dir=args.output_dir,
    )

    print(f"\n  Reference (Chan et al. Table 2): p_f ~ 2.7e-3")


if __name__ == "__main__":
    main()
