"""
Fixed-k search: find short failure/survival rules by sampling k components.

A general-purpose tool for discovering short rules that dominate system
probability. Can be used standalone for discovery, or as a seeding step
before TSUM rule extraction.

**Failure mode** (default):
  Phase 1: Randomly sample combinations of k degraded components (worst-state)
            and check if the system fails. Or load pre-computed failures.
  Phase 2: Minimize discovered failures into minimal failure rules.

**Survival mode** (--survival):
  Phase 1: Keep k components operational, degrade everything else, check if
            the system survives. Finds "what minimal set keeps the system alive?"
  Phase 2: Minimize discovered survivals into minimal survival rules.

Phase 3 (optional): Seed TSUM with discovered rules and continue MCS extraction.

Usage as a library:
    from tsum.fixed_k_search import load_tsum_inputs, run_fixed_k_pipeline

    row_names, n_state, probs_tensor, probs_dict = load_tsum_inputs("data_dir")
    rules = run_fixed_k_pipeline(sfun, row_names, n_state, probs_tensor, k_values=[2, 3])

    # Survival mode:
    rules = run_fixed_k_pipeline(sfun, row_names, n_state, probs_tensor,
                                 k_values=[10, 15], survival=True)

Usage as a CLI:
    # Failure discovery (DC-OPF example)
    python -m tsum.fixed_k_search \\
        --sfun-module demos/case118/sfun_dcopt.py \\
        --sfun-func make_dcopt_sfun \\
        --sfun-args '{"case_path": "demos/case118/case118.m", "blackout_threshold": 13.8}' \\
        --data-dir demos/case118/case118_tsum_bus \\
        --k 2 3 --n-samples 100000

    # Survival discovery
    python -m tsum.fixed_k_search --survival \\
        --sfun-module demos/case300/sfun_dcopt.py \\
        --sfun-func make_dcopt_sfun \\
        --sfun-args '{"case_path": "demos/case300/case300.m", "blackout_threshold": 26.1}' \\
        --data-dir demos/case300/case300_tsum_bus \\
        --k 10 15 20 --n-samples 100000

    # Load pre-computed failures from a directory and seed TSUM
    python -m tsum.fixed_k_search \\
        --sfun-module demos/case118/sfun_dcopt.py \\
        --sfun-func make_dcopt_sfun \\
        --sfun-args '{"case_path": "demos/case118/case118.m", "blackout_threshold": 13.8}' \\
        --data-dir demos/case118/case118_tsum_bus \\
        --load-failures results_fixedk --run-tsum --output-dir results_seeded

See demos/case118/run_fixed_k_search.py for a case-specific wrapper.

The --sfun-module must be a Python file with a factory function (named by
--sfun-func, default: make_sfun) that returns a TSUM-compatible sfun:

    def make_sfun(**kwargs):
        # kwargs passed from --sfun-args (JSON string)
        ...
        return sfun  # callable: dict[str,int] -> (fval, sys_st, info)
"""

import sys
import os
import time
import json
import argparse
import importlib.util
from pathlib import Path
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from tsum import tsum


def load_tsum_inputs(data_dir, device=None):
    """Load standard TSUM inputs (edges.json, probs.json) from a directory.

    Args:
        data_dir: path to directory containing edges.json and probs.json
        device: torch device for the probability tensor (default: cpu)

    Returns:
        row_names: list of component names
        n_state: maximum number of states
        probs_tensor: (n_var, n_state) probability tensor
        probs_dict: raw probability dictionary
    """
    data_dir = Path(data_dir)
    with open(data_dir / "edges.json") as f:
        edges = json.load(f)
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)

    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())

    probs_list = []
    for name in row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0
               for s in range(n_state)]
        probs_list.append(row)

    if device is None:
        device = torch.device("cpu")
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)

    return row_names, n_state, probs_tensor, probs_dict


def _build_max_states(row_names, probs_tensor):
    """Build dict of best (max) state index per component."""
    max_states = {}
    for i, name in enumerate(row_names):
        row = probs_tensor[i]
        nonzero = (row > 0).nonzero(as_tuple=True)[0]
        max_states[name] = int(nonzero[-1].item()) if len(nonzero) > 0 else 0
    return max_states


def load_failures_from_dir(
    fail_dir,
    sfun,
    max_states,
):
    """Load pre-computed failures from a directory of failures_k*.json files.

    Each file contains entries with "degraded" dicts. States are reconstructed
    and re-evaluated to verify they are still failures.

    Args:
        fail_dir: directory containing failures_k*.json files
        sfun: system function for verification
        max_states: dict of component name -> best state index

    Returns:
        list of (full_state_dict, fval, sys_st) tuples
    """
    fail_dir = Path(fail_dir)
    fail_files = sorted(fail_dir.glob("failures_k*.json"))
    if not fail_files:
        print(f"  No failures_k*.json files found in {fail_dir}")
        return []

    all_failures = []
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
    return all_failures


def run_fixed_k_pipeline(
    sfun,
    row_names,
    n_state,
    probs_tensor,
    k_values,
    n_samples=100_000,
    n_workers=1,
    priority_components=None,
    worst_state=True,
    load_failures=None,
    run_tsum=False,
    unk_prob_thres=1e-5,
    bias_factor=0.0,
    bias_rounds=0,
    devices=None,
    output_dir="results_fixedk",
    survival=False,
    target_components=None,
):
    """Run the full fixed-k search pipeline.

    Args:
        sfun: system function
        row_names: component names
        n_state: max states per component
        probs_tensor: (n_var, n_state) probability tensor
        k_values: list of k values to search
        n_samples: samples per k
        n_workers: parallel workers
        priority_components: priority component list
        worst_state: use worst state (True) or sample by probability
        load_failures: path to directory with failures_k*.json (skip Phase 1)
        run_tsum: whether to run TSUM after discovery
        unk_prob_thres: TSUM convergence threshold
        bias_factor: TSUM bias factor
        bias_rounds: TSUM bias rounds
        devices: GPU device list
        output_dir: output directory
        survival: if True, search for survival rules (keep k operational,
            degrade rest) instead of failure rules
        target_components: for survival mode, restrict search to these
            components (e.g., only generators). Others stay at best state.

    Returns:
        list of unique minimized rules, or None if no matches found
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = probs_tensor.device
    multi_devices = devices if devices and len(devices) > 1 else None

    max_states = _build_max_states(row_names, probs_tensor)
    rule_type = "survival" if survival else "failure"

    all_hits = []  # failures or survivals depending on mode
    t0 = time.time()

    # ==================================================================
    # Phase 1: Fixed-k search (or load pre-computed)
    # ==================================================================
    if load_failures and not survival:
        print(f"\n{'='*60}")
        print("Phase 1: Loading pre-computed failures")
        print(f"{'='*60}")
        all_hits = load_failures_from_dir(load_failures, sfun, max_states)
        print(f"Loaded {len(all_hits)} verified failures")
    elif survival:
        print(f"\n{'='*60}")
        print("Phase 1: Fixed-k survival search (keep k operational)")
        print(f"{'='*60}")

        for k in k_values:
            print(f"\n--- k={k} (keep {k} operational, degrade rest) ---")
            survivals = tsum.fixed_k_survival_search(
                sfun=sfun,
                row_names=row_names,
                n_state=n_state,
                sys_surv_st=1,
                probs=probs_tensor,
                k=k,
                n_samples=n_samples,
                n_workers=n_workers,
                priority_components=priority_components,
                target_components=target_components,
            )
            all_hits.extend(survivals)

            # Save per-k results
            surv_out = []
            for comps_st, fval, sys_st in survivals:
                kept = {kn: v for kn, v in comps_st.items()
                        if v >= max_states.get(kn, 0) and max_states.get(kn, 0) > 0}
                surv_out.append({
                    "kept_operational": kept,
                    "n_kept": len(kept),
                    "blackout_pct": round(fval, 3),
                })
            with open(output_dir / f"survivals_k{k}.json", "w") as f:
                json.dump(surv_out, f, indent=2)
            print(f"  Saved {len(surv_out)} survivals to survivals_k{k}.json")
    else:
        print(f"\n{'='*60}")
        print("Phase 1: Fixed-k search for short failure modes")
        print(f"{'='*60}")

        for k in k_values:
            print(f"\n--- k={k} ---")
            failures = tsum.fixed_k_search(
                sfun=sfun,
                row_names=row_names,
                n_state=n_state,
                sys_surv_st=1,
                probs=probs_tensor,
                k=k,
                n_samples=n_samples,
                n_workers=n_workers,
                priority_components=priority_components,
                worst_state=worst_state,
            )
            all_hits.extend(failures)

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
    print(f"\nPhase 1 complete: {len(all_hits)} total {rule_type}s in {t_search:.1f}s")

    # ==================================================================
    # Phase 2: Minimize into minimal rules (cached)
    # ==================================================================
    if survival:
        seed_rules_path = output_dir / "seed_rules_surv.json"
    else:
        seed_rules_path = output_dir / "seed_rules_fail.json"
    t_minimize = 0.0

    if seed_rules_path.exists():
        print(f"\n{'='*60}")
        print("Phase 2: Loading pre-minimized rules")
        print(f"{'='*60}")
        with open(seed_rules_path) as f:
            unique_rules = json.load(f)
        print(f"  Loaded {len(unique_rules)} rules from {seed_rules_path}")
    elif not all_hits:
        print(f"No {rule_type}s found and no {seed_rules_path.name}. Exiting.")
        return None
    else:
        print(f"\n{'='*60}")
        print(f"Phase 2: Minimizing {len(all_hits)} {rule_type}s into rules")
        print(f"{'='*60}")

        t1 = time.time()
        seed_rules = []

        for i, (comps_st, fval, sys_st) in enumerate(all_hits):
            if survival:
                min_rule, info = tsum.minimise_surv_states_random(
                    comps_st, sfun, sys_surv_st=1, fval=fval)
            else:
                min_rule, info = tsum.minimise_fail_states_random(
                    comps_st, sfun, max_state=n_state - 1,
                    sys_fail_st=0, fval=fval)
            seed_rules.append(min_rule)
            if (i + 1) % 50 == 0 or i == len(all_hits) - 1:
                n_conds = sum(1 for kn in min_rule if kn != 'sys')
                print(f"  Minimized {i+1}/{len(all_hits)} "
                      f"(last: {n_conds} conditions)", flush=True)

        t_minimize = time.time() - t1

        # Deduplicate rules
        unique_rules = []
        seen_keys = set()
        for rule in seed_rules:
            key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                               for k, v in rule.items()))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_rules.append(rule)

        print(f"\nPhase 2 complete: {len(unique_rules)} unique {rule_type} rules "
              f"(from {len(seed_rules)}) in {t_minimize:.1f}s")

        with open(seed_rules_path, "w") as f:
            json.dump(unique_rules, f, indent=2)

    # Show distribution
    lengths = [sum(1 for kn in r if kn != 'sys') for r in unique_rules]
    dist = Counter(lengths)
    print(f"\nSeed rule length distribution:")
    for l in sorted(dist):
        print(f"  {l} conditions: {dist[l]} rules")

    if not run_tsum:
        print("\nDone. Use --run-tsum to continue with TSUM rule extraction.")
        return unique_rules

    # ==================================================================
    # Phase 3: Seed TSUM and run MCS rule extraction
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"Phase 3: TSUM rule extraction (seeded with {rule_type} rules)")
    print(f"{'='*60}")

    # Extract critical components from seed rules
    critical = tsum.get_critical_components(unique_rules, min_frequency=0.3)
    if critical:
        print(f"  Critical components: {', '.join(critical)}")

    disc_probs = None
    if bias_factor > 0:
        disc_probs = tsum.make_discovery_probs(
            probs_tensor, bias_factor=bias_factor,
            row_names=row_names, critical_components=critical)
        if critical:
            print(f"  Bias factor: {bias_factor}"
                  f" (critical: {bias_factor * 10})")
        else:
            print(f"  Bias factor: {bias_factor}")

    if survival:
        print(f"  Seed rules:  {len(unique_rules)} survival rules")
    else:
        print(f"  Seed rules:  {len(unique_rules)} failure rules")
    print(f"  Convergence: unk_prob < {unk_prob_thres:.0e}")
    print(f"  Device:      {device}")
    if n_workers > 1:
        print(f"  Workers:     {n_workers}")
    print(f"\nStarting rule extraction...\n", flush=True)

    t2 = time.time()
    seed_kwargs = {}
    if survival:
        seed_kwargs['rules_surv'] = unique_rules
    else:
        seed_kwargs['rules_fail'] = unique_rules

    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        unk_prob_thres=unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=1_000_000,
        sample_batch_size=100_000,
        discovery_probs=disc_probs,
        bias_rounds=bias_rounds,
        n_workers=n_workers,
        devices=multi_devices,
        output_dir=str(output_dir),
        **seed_kwargs,
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

    return unique_rules


def _load_sfun_module(module_path, func_name="make_sfun", sfun_args=None):
    """Dynamically load a module and call its sfun factory function.

    Args:
        module_path: path to a Python file
        func_name: name of the factory function (default: make_sfun)
        sfun_args: JSON string of kwargs, or None
    """
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"sfun module not found: {module_path}")

    # Add module's directory to sys.path so its local imports work
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("sfun_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, func_name):
        available = [a for a in dir(mod) if not a.startswith('_')]
        raise AttributeError(
            f"{module_path} has no function '{func_name}'. "
            f"Available: {available}")

    factory = getattr(mod, func_name)
    kwargs = json.loads(sfun_args) if sfun_args else {}
    return factory(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-k search + TSUM seeding (generic)")

    # System function
    parser.add_argument("--sfun-module", type=str, required=True,
                        help="Path to Python module with sfun factory function")
    parser.add_argument("--sfun-func", type=str, default="make_sfun",
                        help="Name of the factory function in sfun-module (default: make_sfun)")
    parser.add_argument("--sfun-args", type=str, default="",
                        help='JSON dict of kwargs for the sfun factory '
                             '(e.g. \'{"case_path": "case14.m", "blackout_threshold": 54.8}\')')
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing edges.json and probs.json")

    # Fixed-k search parameters
    parser.add_argument("--survival", action="store_true",
                        help="Search for survival rules (keep k operational) instead of failure rules")
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3],
                        help="Number of components to degrade/keep (e.g. --k 2 3 4)")
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="Random k-combinations to test per k (default: 100000)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--priority", type=str, default="",
                        help="Comma-separated priority components")
    parser.add_argument("--no-worst-state", action="store_true",
                        help="Sample degraded states by probability instead of worst state")
    parser.add_argument("--target-components", type=str, default="",
                        help="Comma-separated components to target (survival mode: only these are "
                             "degraded, rest stay operational). Use 'multistate' to auto-select "
                             "multi-state components (e.g., generators).")
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
    parser.add_argument("--output-dir", type=str, default="results_fixedk",
                        help="Output directory (default: results_fixedk)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Fixed-k search pipeline")
    print("=" * 60)

    # Load TSUM inputs
    device_list = ([d.strip() for d in args.devices.split(",") if d.strip()]
                   if args.devices else [])
    device = torch.device(device_list[0] if device_list else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    row_names, n_state, probs_tensor, probs_dict = load_tsum_inputs(
        args.data_dir, device=device)

    print(f"\n  Data dir:    {args.data_dir}")
    print(f"  Components:  {len(row_names)} total")
    print(f"  Max states:  {n_state}")
    print(f"  Device:      {device}")

    # Load system function
    print(f"\nLoading system function from {args.sfun_module}:{args.sfun_func}...")
    sfun = _load_sfun_module(args.sfun_module, args.sfun_func,
                             args.sfun_args or None)

    priority = ([c.strip() for c in args.priority.split(",") if c.strip()]
                if args.priority else None)

    # Parse target components
    target_comps = None
    if args.target_components:
        if args.target_components == "multistate":
            target_comps = [n for n in row_names
                            if len(probs_dict[n]) > 2]
            print(f"  Target:      {len(target_comps)} multi-state components")
        else:
            target_comps = [c.strip() for c in args.target_components.split(",")
                            if c.strip()]
            print(f"  Target:      {len(target_comps)} specified components")

    # Run pipeline
    if args.survival:
        print(f"  Mode:        survival (keep k operational)")
    run_fixed_k_pipeline(
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
        output_dir=args.output_dir,
        survival=args.survival,
        target_components=target_comps,
    )


if __name__ == "__main__":
    main()
