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
import time
import argparse
from pathlib import Path
from collections import Counter

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import json
import torch

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from sfun_dcopt import make_dcopt_sfun
from tsum import tsum


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
    print(f"Fixed-k search on IEEE 118-bus DC-OPF")
    print("=" * 60)

    # Load input data
    data_dir = HERE / "case118_tsum_bus"
    with open(data_dir / "edges.json") as f:
        edges = json.load(f)
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)

    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())

    print(f"\n  Components:  {len(row_names)} total")
    print(f"  Max states:  {n_state}")
    print(f"  Threshold:   13.8% blackout (Scenario 1)")

    # Build probability tensor
    device_list = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else []
    device = torch.device(device_list[0] if device_list else ("cuda" if torch.cuda.is_available() else "cpu"))
    multi_devices = device_list if len(device_list) > 1 else None

    probs_list = []
    for name in row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0 for s in range(n_state)]
        probs_list.append(row)
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)

    # Build system function
    print("\nInitialising DC-OPF system function...")
    sfun = make_dcopt_sfun(
        case_path=str(HERE / "case118.m"),
        blackout_threshold=13.8,
        alpha=2.0,
    )

    output_dir = Path(args.output_dir) if args.output_dir else HERE / "results_fixedk"
    output_dir.mkdir(parents=True, exist_ok=True)

    priority = [c.strip() for c in args.priority.split(",") if c.strip()] if args.priority else None

    # Build max states lookup
    max_states = {}
    for i, name in enumerate(row_names):
        row = probs_tensor[i]
        nonzero = (row > 0).nonzero(as_tuple=True)[0]
        max_states[name] = int(nonzero[-1].item()) if len(nonzero) > 0 else 0

    all_failures = []  # list of (comps_st, fval, sys_st)
    t0 = time.time()

    # ==================================================================
    # Phase 1
    # ==================================================================
    if args.load_failures:
        print(f"\n{'='*60}")
        print("Phase 1: Loading pre-computed failures")
        print(f"{'='*60}")

        fail_dir = Path(args.load_failures)
        fail_files = sorted(fail_dir.glob("failures_k*.json"))
        if not fail_files:
            print(f"  No failures_k*.json files found in {fail_dir}")
            return
        for fpath in fail_files:
            data = json.load(open(fpath))
            print(f"  {fpath.name}: {len(data)} entries")
            for entry in data:
                state = dict(max_states)
                for comp, val in entry["degraded"].items():
                    state[comp] = val
                fval, sys_st, _ = sfun(state)
                if sys_st < 1:
                    all_failures.append((state, fval, sys_st))
        print(f"Loaded {len(all_failures)} verified failures")
    else:
        print(f"\n{'='*60}")
        print("Phase 1: Fixed-k search for short failure modes")
        print(f"{'='*60}")

        for k in args.k:
            print(f"\n--- k={k} ---")
            failures = tsum.fixed_k_search(
                sfun=sfun,
                row_names=row_names,
                n_state=n_state,
                sys_surv_st=1,
                probs=probs_tensor,
                k=k,
                n_samples=args.n_samples,
                n_workers=args.n_workers,
                priority_components=priority,
                worst_state=not args.no_worst_state,
            )
            all_failures.extend(failures)

            # Save per-k results
            failures_out = []
            for comps_st, fval, sys_st in failures:
                degraded = {kn: v for kn, v in comps_st.items()
                            if v < max_states.get(kn, 0)}
                failures_out.append({
                    "degraded": degraded,
                    "n_conditions": len(degraded),
                    "blackout_pct": round(fval, 3),
                })
            with open(output_dir / f"failures_k{k}.json", "w") as f:
                json.dump(failures_out, f, indent=2)
            print(f"  Saved {len(failures_out)} failures to failures_k{k}.json")

    t_search = time.time() - t0
    print(f"\nPhase 1 complete: {len(all_failures)} failures in {t_search:.1f}s")

    # ==================================================================
    # Phase 2: Minimize failures into minimal rules
    # ==================================================================
    seed_rules_path = output_dir / "seed_rules_fail.json"
    t_minimize = 0.0

    if seed_rules_path.exists():
        print(f"\n{'='*60}")
        print("Phase 2: Loading pre-minimized rules")
        print(f"{'='*60}")
        with open(seed_rules_path) as f:
            unique_rules = json.load(f)
        print(f"  Loaded {len(unique_rules)} rules from {seed_rules_path}")
    elif not all_failures:
        print("No failures found and no seed_rules_fail.json. Exiting.")
        return
    else:
        print(f"\n{'='*60}")
        print(f"Phase 2: Minimizing {len(all_failures)} failures into rules")
        print(f"{'='*60}")

        t1 = time.time()
        seed_rules = []

        for i, (comps_st, fval, sys_st) in enumerate(all_failures):
            min_rule, info = tsum.minimise_fail_states_random(
                comps_st, sfun, max_state=n_state - 1,
                sys_fail_st=0, fval=fval)
            seed_rules.append(min_rule)
            if (i + 1) % 50 == 0 or i == len(all_failures) - 1:
                n_conds = sum(1 for kn in min_rule if kn != 'sys')
                print(f"  Minimized {i+1}/{len(all_failures)} "
                      f"(last: {n_conds} conditions)", flush=True)

        t_minimize = time.time() - t1

        # Deduplicate
        unique_rules = []
        seen_keys = set()
        for rule in seed_rules:
            key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                               for k, v in rule.items()))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_rules.append(rule)

        print(f"\nPhase 2 complete: {len(unique_rules)} unique rules "
              f"(from {len(seed_rules)}) in {t_minimize:.1f}s")

        with open(seed_rules_path, "w") as f:
            json.dump(unique_rules, f, indent=2)

    lengths = [sum(1 for kn in r if kn != 'sys') for r in unique_rules]
    dist = Counter(lengths)
    print(f"\nSeed rule length distribution:")
    for l in sorted(dist):
        print(f"  {l} conditions: {dist[l]} rules")

    if not args.run_tsum:
        print("\nDone. Use --run-tsum to continue with TSUM rule extraction.")
        return

    # ==================================================================
    # Phase 3: Seed TSUM and run MCS rule extraction
    # ==================================================================
    print(f"\n{'='*60}")
    print("Phase 3: TSUM rule extraction (seeded with fixed-k rules)")
    print(f"{'='*60}")

    # Extract critical components from seed rules
    critical = tsum.get_critical_components(unique_rules, min_frequency=0.3)
    if critical:
        print(f"  Critical components: {', '.join(critical)}")

    disc_probs = None
    if args.bias_factor > 0:
        disc_probs = tsum.make_discovery_probs(
            probs_tensor, bias_factor=args.bias_factor,
            row_names=row_names, critical_components=critical)
        print(f"  Bias factor: {args.bias_factor}"
              f" (critical: {args.bias_factor * 10})"
              if critical else f"  Bias factor: {args.bias_factor}")

    print(f"  Seed rules:  {len(unique_rules)} failure rules")
    print(f"  Convergence: unk_prob < {args.unk_prob_thres:.0e}")
    print(f"  Device:      {device}")
    if args.n_workers > 1:
        print(f"  Workers:     {args.n_workers}")
    print(f"\nStarting rule extraction...\n", flush=True)

    t2 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        rules_fail=unique_rules,
        unk_prob_thres=args.unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=1_000_000,
        sample_batch_size=100_000,
        discovery_probs=disc_probs,
        bias_rounds=args.bias_rounds,
        n_workers=args.n_workers,
        devices=multi_devices,
        output_dir=str(output_dir),
    )
    t_tsum = time.time() - t2

    print(f"\nTSUM completed in {t_tsum:.1f}s")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"Results saved to: {output_dir}")

    # Summary
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            rounds = [json.loads(line) for line in f if line.strip()]
        last = rounds[-1]
        print(f"\n--- Summary ---")
        print(f"  Phase 1:      {t_search:.1f}s")
        print(f"  Minimization: {t_minimize:.1f}s")
        print(f"  TSUM rounds:  {len(rounds)} ({t_tsum:.1f}s)")
        print(f"  Surv rules:   {last.get('n_rules_surv', '?')}")
        print(f"  Fail rules:   {last.get('n_rules_fail', '?')}")
        print(f"  P(survival):  {last.get('p_survival', '?')}")
        print(f"  P(failure):   {last.get('p_failure', '?')}")
        print(f"  P(unknown):   {last.get('p_unknown', '?')}")
        p_fail = last.get('p_failure', 0)
        print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")
        print(f"  TSUM estimate:                   p_f ~ {p_fail:.2e}")


if __name__ == "__main__":
    main()
