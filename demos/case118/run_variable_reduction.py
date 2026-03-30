"""
Variable reduction for IEEE 118-bus: fix irrelevant components, run TSUM on generators only.

The idea: fixed-k search shows that system failures are driven almost entirely
by generator bus degradation.  All 9 k=3 failure rules involve only 8 generators.
Binary components (186 branches + 64 ordinary buses) have individual failure
probabilities of ~10^-3 or less; multi-component branch failures contribute
negligibly to system risk compared to generator degradation.

This script:
  1. Selects components to model (default: all 54 generators)
  2. Fixes everything else at best (operational) state
  3. Wraps the sfun so fixed components are injected automatically
  4. Optionally seeds with failure rules from fixed-k search
  5. Runs TSUM on the reduced problem (~54 variables instead of 304)

Usage:
    # Generators only (54 variables, 4-state)
    python run_variable_reduction.py --n-workers 48 --output-dir results_reduced

    # With fixed-k seed rules
    python run_variable_reduction.py --seed-rules results_fixedk/seed_rules_fail.json \
        --n-workers 48 --output-dir results_reduced_seeded

    # Custom component selection: top-19 from k=4 frequency analysis
    python run_variable_reduction.py --select-from results_fixedk --min-freq 50 \
        --n-workers 48 --output-dir results_reduced_top19

    # Include all components appearing in any failure rule
    python run_variable_reduction.py --select-from results_fixedk --min-freq 1 \
        --n-workers 48 --output-dir results_reduced_all_fail
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path
from collections import Counter

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import torch

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from sfun_dcopt import make_dcopt_sfun
from tsum import tsum


def parse_args():
    parser = argparse.ArgumentParser(
        description="Variable reduction TSUM for IEEE 118-bus")

    # Component selection
    parser.add_argument("--mode", type=str, default="generators",
                        choices=["generators", "frequency", "custom"],
                        help="Component selection mode (default: generators)")
    parser.add_argument("--select-from", type=str, default=None,
                        help="Directory with failures_k*.json for frequency-based selection")
    parser.add_argument("--min-freq", type=int, default=1,
                        help="Minimum failure frequency to include a component (for --mode frequency)")
    parser.add_argument("--components", type=str, default="",
                        help="Comma-separated component names (for --mode custom)")

    # Seed rules
    parser.add_argument("--seed-rules", type=str, default=None,
                        help="Path to seed_rules_fail.json for seeding TSUM")

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


def select_components(mode, probs_dict, select_from=None, min_freq=1, custom_list=None):
    """Select which components to model (the rest are fixed at best state).

    Returns:
        selected: list of component names to include in TSUM
        fixed: dict mapping fixed component names to their best state index
    """
    all_names = list(probs_dict.keys())

    if mode == "generators":
        # All multi-state (generator) components
        selected = [k for k in all_names if len(probs_dict[k]) > 2]

    elif mode == "frequency":
        if not select_from:
            raise ValueError("--select-from required for frequency mode")
        # Count how often each component appears in failure rules
        freq = Counter()
        fail_dir = Path(select_from)
        for fpath in sorted(fail_dir.glob("failures_k*.json")):
            data = json.load(open(fpath))
            for entry in data:
                for comp in entry["degraded"]:
                    freq[comp] += 1
        selected = [comp for comp, cnt in freq.most_common() if cnt >= min_freq]
        if not selected:
            raise ValueError(f"No components with frequency >= {min_freq}")

    elif mode == "custom":
        if not custom_list:
            raise ValueError("--components required for custom mode")
        selected = [c.strip() for c in custom_list.split(",") if c.strip()]
        # Validate
        for c in selected:
            if c not in probs_dict:
                raise ValueError(f"Unknown component: {c}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Build fixed components: everything not selected, at their best state
    selected_set = set(selected)
    fixed = {}
    for name in all_names:
        if name not in selected_set:
            # Best state = highest state index with nonzero probability
            states = probs_dict[name]
            best = max(int(s) for s in states.keys())
            fixed[name] = best

    return sorted(selected), fixed


def make_reduced_sfun(sfun, fixed_states):
    """Wrap sfun to inject fixed component states.

    The wrapper accepts a state dict with only the selected (variable) components,
    merges in the fixed states, and calls the original sfun.
    """
    def reduced_sfun(comps_st):
        full_state = dict(fixed_states)
        full_state.update(comps_st)
        return sfun(full_state)

    return reduced_sfun


def filter_seed_rules(seed_rules, selected_set):
    """Filter seed rules to only include conditions on selected components.

    Rules with conditions on fixed components: those conditions are always
    satisfied (the component is at best state, which is >= any threshold),
    so we can drop them. Wait -- actually, failure rules have conditions like
    comp <= threshold. If a fixed component is at best state, it does NOT
    satisfy a failure condition, so the rule can never fire. We need to
    discard rules that require a fixed component to be degraded.

    Returns filtered rules (only those whose conditions are all on selected components).
    """
    filtered = []
    for rule in seed_rules:
        # Check if all non-sys conditions are on selected components
        conditions_on_fixed = False
        for k in rule:
            if k == 'sys':
                continue
            if k not in selected_set:
                conditions_on_fixed = True
                break
        if not conditions_on_fixed:
            filtered.append(rule)
    return filtered


def main():
    args = parse_args()

    print("=" * 60)
    print("Variable Reduction TSUM for IEEE 118-bus DC-OPF")
    print("=" * 60)

    # Load input data
    data_dir = HERE / "case118_tsum_bus"
    with open(data_dir / "edges.json") as f:
        edges = json.load(f)
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)

    all_names = list(probs_dict.keys())
    n_state_full = max(len(v) for v in probs_dict.values())

    print(f"\n  Full model:  {len(all_names)} components, {n_state_full} max states")

    # Select components
    selected, fixed = select_components(
        mode=args.mode,
        probs_dict=probs_dict,
        select_from=args.select_from,
        min_freq=args.min_freq,
        custom_list=args.components,
    )

    n_state = max(len(probs_dict[name]) for name in selected)

    print(f"  Selected:    {len(selected)} variable components")
    print(f"  Fixed:       {len(fixed)} components at best state")
    print(f"  Max states:  {n_state}")
    print(f"  Mode:        {args.mode}")

    # Categorise selected components
    gens = [c for c in selected if len(probs_dict[c]) > 2]
    binary = [c for c in selected if len(probs_dict[c]) == 2]
    if gens:
        print(f"    Generators: {len(gens)}")
    if binary:
        print(f"    Binary:     {len(binary)}")

    # Build probability tensor for selected components only
    device_list = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else []
    device = torch.device(device_list[0] if device_list else ("cuda" if torch.cuda.is_available() else "cpu"))
    multi_devices = device_list if len(device_list) > 1 else None

    probs_list = []
    for name in selected:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0 for s in range(n_state)]
        probs_list.append(row)
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)

    # Compute probability mass that is fixed (always at best state)
    # This is the product of P(best state) for all fixed components
    import math
    log_p_fixed = 0.0
    for name, best_st in fixed.items():
        p = probs_dict[name]
        p_best = p[str(best_st)]["p"]
        log_p_fixed += math.log(p_best)
    p_fixed = math.exp(log_p_fixed)
    print(f"\n  P(all fixed at best): {p_fixed:.6f}")
    print(f"  P(any fixed degraded): {1 - p_fixed:.6f}")
    print(f"  Note: ignoring {1 - p_fixed:.4%} of probability space where")
    print(f"        fixed components are degraded.")

    # Build sfun
    print(f"\nInitialising DC-OPF system function...")
    base_sfun = make_dcopt_sfun(
        case_path=str(HERE / "case118.m"),
        blackout_threshold=13.8,
        alpha=2.0,
    )
    sfun = make_reduced_sfun(base_sfun, fixed)

    # Verify: test all-best-state
    test_state = {name: max(int(s) for s in probs_dict[name].keys())
                  for name in selected}
    fval, sys_st, _ = sfun(test_state)
    print(f"  Verification (all operational): blackout={fval:.2f}%, sys_st={sys_st}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reduction metadata
    meta = {
        "mode": args.mode,
        "n_selected": len(selected),
        "n_fixed": len(fixed),
        "n_state": n_state,
        "selected_components": selected,
        "p_fixed_at_best": p_fixed,
        "p_any_fixed_degraded": 1 - p_fixed,
    }
    with open(output_dir / "reduction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Load and filter seed rules if provided
    seed_rules = None
    if args.seed_rules:
        seed_path = Path(args.seed_rules)
        if seed_path.exists():
            with open(seed_path) as f:
                all_seed_rules = json.load(f)
            selected_set = set(selected)
            seed_rules = filter_seed_rules(all_seed_rules, selected_set)
            print(f"\n  Seed rules: {len(seed_rules)} applicable "
                  f"(from {len(all_seed_rules)} total)")
            if seed_rules:
                lengths = Counter(sum(1 for k in r if k != 'sys') for r in seed_rules)
                for l in sorted(lengths):
                    print(f"    {l} conditions: {lengths[l]} rules")
        else:
            print(f"\n  Warning: seed rules file not found: {seed_path}")

    # Setup discovery probs
    disc_probs = None
    if args.bias_factor > 0:
        critical = None
        if seed_rules:
            critical = tsum.get_critical_components(seed_rules, min_frequency=0.3)
            if critical:
                print(f"  Critical components: {', '.join(critical)}")
        disc_probs = tsum.make_discovery_probs(
            probs_tensor, bias_factor=args.bias_factor,
            row_names=selected, critical_components=critical)
        print(f"  Bias factor: {args.bias_factor}")

    # Run TSUM
    print(f"\n{'='*60}")
    print(f"TSUM rule extraction ({len(selected)} variables, {n_state} states)")
    print(f"{'='*60}")
    print(f"  Convergence: unk_prob < {args.unk_prob_thres:.0e}")
    print(f"  Device:      {device}")
    if args.n_workers > 1:
        print(f"  Workers:     {args.n_workers}")
    print(f"\nStarting rule extraction...\n", flush=True)

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=selected,
        n_state=n_state,
        sys_surv_st=1,
        rules_fail=seed_rules,
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
    t_total = time.time() - t0

    print(f"\nTSUM completed in {t_total:.1f}s")
    print(f"Results saved to: {output_dir}")

    # Summary
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            rounds = [json.loads(line) for line in f if line.strip()]
        last = rounds[-1]
        print(f"\n--- Summary ---")
        print(f"  Components:   {len(selected)} (reduced from {len(all_names)})")
        print(f"  TSUM rounds:  {len(rounds)} ({t_total:.1f}s)")
        print(f"  Surv rules:   {last.get('n_rules_surv', '?')}")
        print(f"  Fail rules:   {last.get('n_rules_fail', '?')}")
        print(f"  P(survival):  {last.get('p_survival', '?')}")
        print(f"  P(failure):   {last.get('p_failure', '?')}")
        print(f"  P(unknown):   {last.get('p_unknown', '?')}")

        p_fail = last.get('p_failure', 0)
        p_unk = last.get('p_unknown', 0)
        print(f"\n  Note: these probabilities are conditional on all {len(fixed)}")
        print(f"  fixed components being at best state (P={p_fixed:.6f}).")
        print(f"  Unconditional P(failure) >= {p_fail * p_fixed:.2e}")
        print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")


if __name__ == "__main__":
    main()
