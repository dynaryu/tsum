"""
Run TSUM on IEEE 300-bus DC-OPF with degradation-ranked unknown selection.

Standard TSUM sampling discovers unknown states; unknowns are then ranked by
total component degradation (most degraded first) so that failure-prone states
are evaluated before survival-dominated ones.  This exploits system coherence
(monotonicity) without any ML training.

Component weights start uniform and adapt as failure rules are discovered:
components appearing in more failure rules get higher weight.  Optionally
seeds the weights from pre-found failure rules (e.g. from k-fixed search).

Components: 711 total (300 buses + 411 branches)
System function: DC-OPF, blackout_threshold=26.1%
Reference: Chan et al. (2024), Table 2: p_f ~ 1.0e-4

Usage:
    python run_classifier.py
    python run_classifier.py --unk-prob-thres 1e-4 --n-workers 48
    python run_classifier.py --devices cuda:0,cuda:1
"""

import sys
import os
import time
import argparse
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import json
import torch

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from dcopt import make_dcopt_sfun
from tsum import tsum


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSUM with degradation-ranked unknown selection on IEEE 300-bus DC-OPF")
    parser.add_argument("--unk-prob-thres", type=float, default=1e-3,
                        help="Convergence threshold for unknown probability (default: 1e-3)")
    parser.add_argument("--n-sample", type=int, default=10_000_000,
                        help="Samples per round for probability estimation (default: 10000000)")
    parser.add_argument("--sample-batch-size", type=int, default=100_000,
                        help="Batch size for GPU sampling (default: 100000)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers for sfun evaluation (default: 1)")
    parser.add_argument("--devices", type=str, default="",
                        help="Comma-separated GPU devices, e.g. 'cuda:0,cuda:1'")
    parser.add_argument("--seed-rules", type=str, default="",
                        help="Path to JSON file with failure rules to seed component weights (from k-fixed search)")
    parser.add_argument("--degradation-alpha", type=float, default=3.0,
                        help="Exponential scaling factor for component weights (default: 3.0)")
    parser.add_argument("--gen-weight", type=float, default=2.0,
                        help="Initial weight for generator buses (default: 2.0)")
    parser.add_argument("--branch-weight", type=float, default=1.5,
                        help="Initial weight for branches (default: 1.5)")
    parser.add_argument("--bus-weight", type=float, default=1.0,
                        help="Initial weight for ordinary buses (default: 1.0)")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="Disable sensitivity pre-screen (enabled by default)")
    parser.add_argument("--no-diversity", action="store_true",
                        help="Use deterministic top-k instead of probabilistic selection")
    parser.add_argument("--output-dir", type=str, default="results_degradation",
                        help="Output directory (default: results_degradation)")
    return parser.parse_args()


def main():
    args = parse_args()
    device_list = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else []
    multi_devices = device_list if len(device_list) > 1 else None

    print("=" * 60)
    print("TSUM rule extraction with degradation-ranked unknown selection")
    print("IEEE 300-bus DC-OPF")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load input data
    # ---------------------------------------------------------------
    data_dir = HERE / "case300_tsum_bus"
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)

    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())

    n_gen_bus = sum(1 for n in row_names if n.startswith("vbus")
                    and len(probs_dict[n]) == 4)
    n_ord_bus = sum(1 for n in row_names if n.startswith("vbus")
                    and len(probs_dict[n]) == 2)
    n_branch = sum(1 for n in row_names if n.startswith("br"))

    print(f"\n  Components:  {len(row_names)} total")
    print(f"    Generator buses: {n_gen_bus} (4-state)")
    print(f"    Ordinary buses:  {n_ord_bus} (2-state)")
    print(f"    Branches:        {n_branch} (2-state)")
    print(f"  Max states:  {n_state}")

    # ---------------------------------------------------------------
    # 2. Build probability tensor (padded to n_state)
    # ---------------------------------------------------------------
    device = torch.device(device_list[0] if device_list else ("cuda" if torch.cuda.is_available() else "cpu"))
    probs_list = []
    for name in row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0
               for s in range(n_state)]
        probs_list.append(row)

    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)
    print(f"  Device:      {device}")

    # ---------------------------------------------------------------
    # 3. Build system function
    # ---------------------------------------------------------------
    print("\nInitialising DC-OPF system function...")
    sfun = make_dcopt_sfun(
        case_path=str(HERE / "case300.m"),
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

    # ---------------------------------------------------------------
    # 4. Run rule extraction with degradation-ranked unknown selection
    # ---------------------------------------------------------------
    # Load seed failure rules if provided
    seed_rules = None
    if args.seed_rules:
        seed_path = Path(args.seed_rules)
        if not seed_path.is_absolute():
            seed_path = HERE / seed_path
        with open(seed_path) as f:
            seed_rules = json.load(f)
        print(f"\n  Seed rules:  {len(seed_rules)} failure rules from {seed_path.name}")

    # Build per-component initial weights by type
    comp_weights_init = {}
    for name in row_names:
        if name.startswith("vbus") and len(probs_dict[name]) == 4:
            comp_weights_init[name] = args.gen_weight
        elif name.startswith("br"):
            comp_weights_init[name] = args.branch_weight
        else:
            comp_weights_init[name] = args.bus_weight


    output_dir = HERE / args.output_dir
    print(f"\n  Output:      {output_dir}")
    print(f"  Samples:     {args.n_sample:,} per round (batch {args.sample_batch_size:,})")
    print(f"  Convergence: unk_prob < {args.unk_prob_thres:.0e}")
    print(f"  Ranking:     weighted degradation (most degraded unknowns first)")
    print(f"  Alpha:       {args.degradation_alpha}")
    print(f"  Sensitivity: {'disabled' if args.no_sensitivity else 'enabled'}")
    print(f"  Diversity:   {'disabled (top-k)' if args.no_diversity else 'enabled (probabilistic)'}")
    if multi_devices:
        print(f"  Devices:     {multi_devices}")
    print(f"\nStarting rule extraction...\n", flush=True)

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        unk_prob_thres=args.unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=args.n_sample,
        sample_batch_size=args.sample_batch_size,
        n_workers=args.n_workers,
        devices=multi_devices,
        rank_by_degradation=True,
        degradation_alpha=args.degradation_alpha,
        comp_weights_init=comp_weights_init,
        sensitivity_prescreen=not args.no_sensitivity,
        degradation_diversity=not args.no_diversity,
        classifier_seed_rules=seed_rules,
        output_dir=str(output_dir),
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")

    # ---------------------------------------------------------------
    # 5. Summary
    # ---------------------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            rounds = [json.loads(line) for line in f if line.strip()]
        last = rounds[-1]
        print(f"\n--- Summary ---")
        print(f"  Rounds:      {len(rounds)}")
        print(f"  Surv rules:  {last.get('n_rules_surv', '?')}")
        print(f"  Fail rules:  {last.get('n_rules_fail', '?')}")
        print(f"  Unk prob:    {last.get('p_unknown', '?')}")
        print(f"  P(survival): {last.get('p_survival', '?')}")
        print(f"  P(failure):  {last.get('p_failure', '?')}")
        p_fail = last.get('p_failure', 0)
        print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")
        print(f"  TSUM estimate:                   p_f ~ {p_fail:.2e}")


if __name__ == "__main__":
    main()
