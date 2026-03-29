"""
Fixed-k search for IEEE 118-bus: find short failure rules by sampling.

Randomly samples combinations of k degraded components (weighted by failure
probability) and checks if the system fails. This directly targets the
short failure rules that dominate p_f.

For case118 (304 components):
  k=2: 100K samples covers the space well (minutes)
  k=3: 100K-1M samples explores critical combinations (hours on cluster)

Usage:
    python run_fixed_k_search.py --k 2
    python run_fixed_k_search.py --k 3 --n-samples 500000 --n-workers 48
    python run_fixed_k_search.py --k 3 --n-workers 192 --priority vbus59,vbus80,vbus77
    python run_fixed_k_search.py --k 2 --k 3 --k 4 --n-samples 100000 --n-workers 48
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
sys.path.insert(0, str(HERE))

from sfun_dcopt import make_dcopt_sfun
from tsum import tsum


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-k search for short failure rules (IEEE 118-bus)")
    parser.add_argument("--k", type=int, nargs="+", default=[2],
                        help="Number of components to degrade (can specify multiple, e.g. --k 2 3 4)")
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="Number of random k-combinations to test per k (default: 100000)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers for sfun evaluation (default: 1)")
    parser.add_argument("--priority", type=str, default="",
                        help="Comma-separated priority components (at least one "
                             "per combo must be from this list)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory (default: results_fixedk)")
    parser.add_argument("--no-worst-state", action="store_true",
                        help="Sample degraded states by probability instead "
                             "of always using worst state (state 0)")
    parser.add_argument("--minimize", action="store_true",
                        help="Minimize discovered failures into rules")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Fixed-k search (k={args.k}) on IEEE 118-bus DC-OPF")
    print("=" * 60)

    # Load input data
    data_dir = HERE / "case118_tsum_bus"
    with open(data_dir / "edges.json") as f:
        edges = json.load(f)
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
    print(f"  Threshold:   13.8% blackout (Scenario 1)")

    # Build probability tensor
    probs_list = []
    for name in row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0
               for s in range(n_state)]
        probs_list.append(row)
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32)

    # Build system function
    print("\nInitialising DC-OPF system function...")
    case_path = str(HERE / "case118.m")
    sfun = make_dcopt_sfun(
        case_path=case_path,
        blackout_threshold=13.8,
        alpha=2.0,
    )

    # Parse priority components
    priority = None
    if args.priority:
        priority = [c.strip() for c in args.priority.split(",") if c.strip()]
        print(f"  Priority components: {priority}")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else HERE / "results_fixedk"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_failures = {}
    t_total = 0

    for k in args.k:
        print(f"\n{'='*60}")
        print(f"Searching k={k}")
        print(f"{'='*60}")

        t0 = time.time()
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
        t_search = time.time() - t0
        t_total += t_search

        print(f"\nk={k}: {len(failures)} failures in {t_search:.1f}s")

        # Build max states for output
        max_states = {}
        for i, name in enumerate(row_names):
            row = probs_tensor[i]
            nonzero = (row > 0).nonzero(as_tuple=True)[0]
            max_states[name] = int(nonzero[-1].item()) if len(nonzero) > 0 else 0

        # Save raw failures
        failures_out = []
        for comps_st, fval, sys_st in failures:
            degraded = {k_name: v for k_name, v in comps_st.items()
                        if v < max_states.get(k_name, 0)}
            failures_out.append({
                "degraded": degraded,
                "n_conditions": len(degraded),
                "blackout_pct": round(fval, 3),
            })
        all_failures[k] = failures_out

        out_file = output_dir / f"failures_k{k}.json"
        with open(out_file, "w") as f:
            json.dump(failures_out, f, indent=2)
        print(f"Saved to {out_file}")

        # Minimize failures into rules if requested
        if args.minimize and failures:
            print(f"\nMinimizing {len(failures)} failures into rules...")
            t1 = time.time()
            rules = []
            for i, (comps_st, fval, sys_st) in enumerate(failures):
                min_rule, info = tsum.minimise_fail_states_random(
                    comps_st, sfun, max_state=n_state - 1,
                    sys_fail_st=0, fval=fval)
                n_conds = sum(1 for k_name, v in min_rule.items()
                              if k_name != 'sys')
                rules.append(min_rule)
                if (i + 1) % 10 == 0 or i == len(failures) - 1:
                    print(f"  Minimized {i+1}/{len(failures)}", flush=True)

            rules_file = output_dir / f"rules_k{k}.json"
            with open(rules_file, "w") as f:
                json.dump(rules, f, indent=2)
            print(f"Rules saved to {rules_file} ({time.time()-t1:.1f}s)")

        # Summary for this k
        if failures_out:
            lengths = [f["n_conditions"] for f in failures_out]
            print(f"\n  k={k} summary:")
            print(f"    Failures: {len(failures_out)}")
            print(f"    Conditions: min={min(lengths)}, max={max(lengths)}, "
                  f"avg={sum(lengths)/len(lengths):.1f}")

            sorted_f = sorted(failures_out, key=lambda x: x["blackout_pct"],
                              reverse=True)
            print(f"    Top failures:")
            for f in sorted_f[:5]:
                comps = ", ".join(f"{k_name}={v}"
                                 for k_name, v in f["degraded"].items())
                print(f"      {f['blackout_pct']:.1f}%: {comps}")
        else:
            print(f"\n  No failures found with k={k}.")

    # Overall summary
    print(f"\n{'='*60}")
    print(f"Overall Summary")
    print(f"{'='*60}")
    print(f"  Total time: {t_total:.1f}s")
    for k in args.k:
        n = len(all_failures.get(k, []))
        print(f"  k={k}: {n} failures found")


if __name__ == "__main__":
    main()
