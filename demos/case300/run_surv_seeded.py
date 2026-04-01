"""
Run TSUM on IEEE 300-bus DC-OPF, seeded with survival rules from fixed-k search.

The full-space case300 problem (711 variables) gets stuck because MCS cannot
discover useful survival rules — the rules it finds are too long (~58 conditions
out of 69 generators) and cover negligible probability. This script seeds TSUM
with survival rules found via fixed_k_survival_search, which identifies the
minimal set of generators that must stay operational.

Prerequisites:
    Run the survival seed generation first:
        python -c "
        import sys; sys.path.insert(0, '.')
        from dcopt import make_dcopt_sfun
        from tsum.fixed_k_search import load_tsum_inputs, run_fixed_k_pipeline
        row_names, n_state, probs_tensor, probs_dict = load_tsum_inputs('case300_tsum_bus')
        sfun = make_dcopt_sfun(case_path='case300.m', blackout_threshold=26.1, alpha=2.0)
        gen_names = [n for n in row_names if n.startswith('vbus') and len(probs_dict[n]) == 4]
        run_fixed_k_pipeline(sfun=sfun, row_names=row_names, n_state=n_state,
            probs_tensor=probs_tensor, k_values=[38, 40, 42, 45], n_samples=1000,
            survival=True, target_components=gen_names, output_dir='results_surv_seed')
        "

Usage:
    python run_surv_seeded.py --seed-rules results_surv_seed/seed_rules_surv.json \
        --n-workers 48 --output-dir results_surv_seeded
    python run_surv_seeded.py --seed-rules results_surv_seed/seed_rules_surv.json \
        --bias-factor 5 --n-workers 48 --output-dir results_surv_seeded_bf5
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
        description="TSUM on IEEE 300-bus seeded with survival rules")
    parser.add_argument("--seed-rules", type=str, required=True,
                        help="Path to seed_rules_surv.json")
    parser.add_argument("--unk-prob-thres", type=float, default=1e-5,
                        help="Convergence threshold (default: 1e-5)")
    parser.add_argument("--bias-factor", type=float, default=0.0,
                        help="Bias factor for discovery sampling (0=off)")
    parser.add_argument("--bias-rounds", type=int, default=0,
                        help="Biased sampling for first N rounds (0=all)")
    parser.add_argument("--walk-every", type=int, default=0,
                        help="Boundary walks every N rounds (0=off)")
    parser.add_argument("--walk-count", type=int, default=1,
                        help="Walks per walk round (default: 1)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--devices", type=str, default="",
                        help="Comma-separated GPU devices")
    parser.add_argument("--output-dir", type=str, default="results_surv_seeded",
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    device_list = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else []
    multi_devices = device_list if len(device_list) > 1 else None

    print("=" * 60)
    print("TSUM on IEEE 300-bus (seeded with survival rules)")
    print("=" * 60)

    # Load data
    data_dir = HERE / "case300_tsum_bus"
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)

    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())

    n_gen = sum(1 for n in row_names if n.startswith("vbus") and len(probs_dict[n]) == 4)
    n_ord = sum(1 for n in row_names if n.startswith("vbus") and len(probs_dict[n]) == 2)
    n_br = sum(1 for n in row_names if n.startswith("br"))

    print(f"\n  Components:  {len(row_names)} total")
    print(f"    Generator buses: {n_gen} (4-state)")
    print(f"    Ordinary buses:  {n_ord} (2-state)")
    print(f"    Branches:        {n_br} (2-state)")

    # Probs tensor
    device = torch.device(device_list[0] if device_list else ("cuda" if torch.cuda.is_available() else "cpu"))
    probs_list = []
    for name in row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0 for s in range(n_state)]
        probs_list.append(row)
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)

    # Load seed survival rules
    seed_path = Path(args.seed_rules)
    if not seed_path.exists():
        print(f"ERROR: seed rules not found: {seed_path}")
        sys.exit(1)
    with open(seed_path) as f:
        seed_surv_rules = json.load(f)
    print(f"  Seed rules:  {len(seed_surv_rules)} survival rules from {seed_path}")

    # Rule length distribution
    from collections import Counter
    lengths = [sum(1 for k in r if k != 'sys') for r in seed_surv_rules]
    dist = Counter(lengths)
    print(f"  Rule lengths: min={min(lengths)}, max={max(lengths)}, "
          f"median={sorted(lengths)[len(lengths)//2]}")

    # Build sfun
    print("\nInitialising DC-OPF system function...")
    sfun = make_dcopt_sfun(
        case_path=str(HERE / "case300.m"),
        blackout_threshold=26.1,
        alpha=2.0,
    )

    # Discovery probs
    disc_probs = None
    if args.bias_factor > 0:
        disc_probs = tsum.make_discovery_probs(probs_tensor, bias_factor=args.bias_factor)
        print(f"  Bias factor: {args.bias_factor}")

    # Run
    output_dir = Path(args.output_dir)
    print(f"\n  Output:      {output_dir}")
    print(f"  Convergence: unk_prob < {args.unk_prob_thres:.0e}")
    if args.walk_every > 0:
        print(f"  Walk:        every {args.walk_every} rounds, {args.walk_count} per round")
    if args.n_workers > 1:
        print(f"  Workers:     {args.n_workers}")
    print(f"\nStarting rule extraction...\n", flush=True)

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        rules_surv=seed_surv_rules,
        unk_prob_thres=args.unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=1_000_000,
        sample_batch_size=100_000,
        discovery_probs=disc_probs,
        bias_rounds=args.bias_rounds,
        walk_every=args.walk_every,
        walk_count=args.walk_count,
        n_workers=args.n_workers,
        devices=multi_devices,
        output_dir=output_dir,
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")

    # Summary
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            rounds = [json.loads(line) for line in f if line.strip()]
        last = rounds[-1]
        print(f"\n--- Summary ---")
        print(f"  Rounds:      {len(rounds)}")
        print(f"  Surv rules:  {last.get('n_rules_surv', '?')}")
        print(f"  Fail rules:  {last.get('n_rules_fail', '?')}")
        print(f"  P(survival): {last.get('p_survival', '?')}")
        print(f"  P(failure):  {last.get('p_failure', '?')}")
        print(f"  P(unknown):  {last.get('p_unknown', '?')}")
        print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")


if __name__ == "__main__":
    main()
