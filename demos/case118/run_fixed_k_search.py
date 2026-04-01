"""
Fixed-k search for IEEE 118-bus: find short failure rules, then seed TSUM.

Phase 1: Randomly sample combinations of k degraded components (worst-state)
          and check if the system fails. Or load pre-computed failures.
Phase 2: Minimize discovered failures into minimal rules.
Phase 3: Seed TSUM with the discovered rules and continue MCS rule extraction.

Usage:
    # Discovery only
    python run_fixed_k_search.py --k 2 3 --n-samples 100000 --n-workers 48

    # Full pipeline: discover + minimize + run TSUM
    python run_fixed_k_search.py --k 2 3 4 --n-samples 500000 --n-workers 192 --run-tsum
    python run_fixed_k_search.py --k 3 4 5 --n-samples 500000 --n-workers 192 --run-tsum --bias-factor 10

    # Load pre-computed failures from directory and seed TSUM (skip Phase 1)
    python run_fixed_k_search.py --load-failures results_fixedk --run-tsum --n-workers 192 --output-dir results_seeded
    python run_fixed_k_search.py --load-failures results_fixedk --run-tsum --bias-factor 10 --output-dir results_seeded_bf10
"""

import sys
import os
import argparse
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import torch

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from dcopt import make_dcopt_sfun
from tsum.fixed_k_search import load_tsum_inputs, run_fixed_k_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-k search + TSUM seeding (IEEE 118-bus)")
    # Fixed-k search parameters
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3],
                        help="Number of components to degrade (e.g. --k 2 3 4)")
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="Random k-combinations to test per k (default: 100000)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--priority", type=str, default="",
                        help="Comma-separated priority components")
    parser.add_argument("--no-worst-state", action="store_true",
                        help="Sample degraded states by probability instead of worst state")
    parser.add_argument("--load-failures", type=str, default=None,
                        help="Directory containing failures_k*.json files (skip Phase 1)")
    # TSUM parameters
    parser.add_argument("--run-tsum", action="store_true",
                        help="After discovery, seed TSUM and run MCS extraction")
    parser.add_argument("--unk-prob-thres", type=float, default=1e-5,
                        help="TSUM convergence threshold (default: 1e-5)")
    parser.add_argument("--bias-factor", type=float, default=0.0,
                        help="Bias factor for TSUM discovery sampling (0=off)")
    parser.add_argument("--bias-rounds", type=int, default=0,
                        help="Use biased sampling for first N rounds (0=all)")
    parser.add_argument("--devices", type=str, default="",
                        help="Comma-separated GPU devices for TSUM")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory (default: results_fixedk)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Fixed-k search on IEEE 118-bus DC-OPF")
    print("=" * 60)

    # Load input data
    device_list = ([d.strip() for d in args.devices.split(",") if d.strip()]
                   if args.devices else [])
    device = torch.device(device_list[0] if device_list else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = HERE / "case118_tsum_bus"
    row_names, n_state, probs_tensor, probs_dict = load_tsum_inputs(
        data_dir, device=device)

    print(f"\n  Components:  {len(row_names)} total")
    print(f"  Max states:  {n_state}")
    print(f"  Threshold:   13.8% blackout (Scenario 1)")

    # Build system function
    print("\nInitialising DC-OPF system function...")
    sfun = make_dcopt_sfun(
        case_path=str(HERE / "case118.m"),
        blackout_threshold=13.8,
        alpha=2.0,
    )

    output_dir = args.output_dir if args.output_dir else str(HERE / "results_fixedk")
    priority = ([c.strip() for c in args.priority.split(",") if c.strip()]
                if args.priority else None)

    # Run pipeline
    rules = run_fixed_k_pipeline(
        sfun=sfun,
        row_names=row_names,
        n_state=n_state,
        probs_tensor=probs_tensor,
        k_values=args.k,
        n_samples=args.n_samples,
        n_workers=args.n_workers,
        priority_components=priority,
        worst_state=not args.no_worst_state,
        load_failures=args.load_failures,
        run_tsum=args.run_tsum,
        unk_prob_thres=args.unk_prob_thres,
        bias_factor=args.bias_factor,
        bias_rounds=args.bias_rounds,
        devices=device_list or None,
        output_dir=output_dir,
    )

    if rules:
        p_fail = None
        metrics_path = Path(output_dir) / "metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                rounds = [json.loads(line) for line in f if line.strip()]
            if rounds:
                p_fail = rounds[-1].get('p_failure', 0)
                print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")
                print(f"  TSUM estimate:                   p_f ~ {p_fail:.2e}")


if __name__ == "__main__":
    main()
