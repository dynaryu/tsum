"""
Variable reduction: fix irrelevant components at best state, run TSUM on a subset.

Reduces the TSUM search space by screening out components that contribute
negligibly to system failure. The full system function is still evaluated
(fixed components are injected at their best state), so the physical model
is preserved — only the rule search space shrinks.

Three component selection modes:
  - "multistate": all multi-state (>2 states) components (e.g. generators)
  - "frequency":  components appearing in failure rules above a threshold
  - "custom":     user-specified list

The result is a lower bound on P(failure), conditional on all fixed
components being at their best state.

Usage as a library:
    from tsum.variable_reduction import select_components, make_reduced_sfun, run_reduced_tsum

    selected, fixed = select_components(probs_dict, mode="multistate")
    reduced_sfun = make_reduced_sfun(base_sfun, fixed)
    run_reduced_tsum(reduced_sfun, probs_dict, selected, ...)

Usage as a CLI:
    # Multistate mode (select all >2-state components)
    python -m tsum.variable_reduction \\
        --sfun-module demos/case118/sfun_dcopt.py \\
        --sfun-func make_dcopt_sfun \\
        --sfun-args '{"case_path": "demos/case118/case118.m", "blackout_threshold": 13.8}' \\
        --data-dir demos/case118/case118_tsum_bus \\
        --mode multistate --output-dir results_reduced

    # Frequency mode (select components from failure data)
    python -m tsum.variable_reduction \\
        --sfun-module demos/case118/sfun_dcopt.py \\
        --sfun-func make_dcopt_sfun \\
        --sfun-args '{"case_path": "demos/case118/case118.m", "blackout_threshold": 13.8}' \\
        --data-dir demos/case118/case118_tsum_bus \\
        --mode frequency --failures-dir results_fixedk --min-freq 50

See demos/case118/run_variable_reduction.py for a case-specific wrapper.
"""

import sys
import os
import time
import json
import math
from pathlib import Path
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from tsum import tsum


def load_tsum_inputs(data_dir):
    """Load standard TSUM inputs (edges.json, probs.json) from a directory."""
    data_dir = Path(data_dir)
    with open(data_dir / "edges.json") as f:
        edges = json.load(f)
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)
    return edges, probs_dict


def select_components(
    probs_dict: Dict[str, Dict],
    mode: str = "multistate",
    failures_dir: Optional[str] = None,
    min_freq: int = 1,
    custom_list: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Select which components to model; the rest are fixed at best state.

    Args:
        probs_dict: component name -> {state_idx_str: {"p": float}}
        mode: "multistate" (all >2-state components), "frequency" (from failure
              data), or "custom" (explicit list)
        failures_dir: directory with failures_k*.json (required for "frequency")
        min_freq: minimum failure frequency to include (for "frequency")
        custom_list: component names (for "custom")

    Returns:
        selected: sorted list of component names to include in TSUM
        fixed: dict mapping fixed component names to their best state index
    """
    all_names = list(probs_dict.keys())

    if mode == "multistate":
        selected = [k for k in all_names if len(probs_dict[k]) > 2]

    elif mode == "frequency":
        if not failures_dir:
            raise ValueError("--failures-dir required for frequency mode")
        freq = Counter()
        fail_dir = Path(failures_dir)
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
            raise ValueError("custom_list required for custom mode")
        for c in custom_list:
            if c not in probs_dict:
                raise ValueError(f"Unknown component: {c}")
        selected = list(custom_list)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Build fixed components: everything not selected, at their best state
    selected_set = set(selected)
    fixed = {}
    for name in all_names:
        if name not in selected_set:
            states = probs_dict[name]
            best = max(int(s) for s in states.keys())
            fixed[name] = best

    return sorted(selected), fixed


def make_reduced_sfun(
    sfun: Callable,
    fixed_states: Dict[str, int],
) -> Callable:
    """Wrap sfun to inject fixed component states.

    The wrapper accepts a state dict with only the selected (variable)
    components, merges in the fixed states, and calls the original sfun.
    """
    def reduced_sfun(comps_st):
        full_state = dict(fixed_states)
        full_state.update(comps_st)
        return sfun(full_state)

    return reduced_sfun


def filter_seed_rules(
    seed_rules: List[Dict],
    selected: List[str],
) -> List[Dict]:
    """Filter seed rules to only those whose conditions are all on selected components.

    Failure rules with conditions on fixed components can never fire (the fixed
    component is at best state), so they must be discarded.

    Returns:
        Filtered list of applicable rules.
    """
    selected_set = set(selected)
    filtered = []
    for rule in seed_rules:
        if all(k in selected_set or k == 'sys' for k in rule):
            filtered.append(rule)
    return filtered


def compute_fixed_probability(
    probs_dict: Dict[str, Dict],
    fixed: Dict[str, int],
) -> float:
    """Compute P(all fixed components at best state)."""
    log_p = 0.0
    for name, best_st in fixed.items():
        p_best = probs_dict[name][str(best_st)]["p"]
        log_p += math.log(p_best)
    return math.exp(log_p)


def build_reduced_probs(
    probs_dict: Dict[str, Dict],
    selected: List[str],
    device: torch.device = None,
) -> Tuple[torch.Tensor, int]:
    """Build probability tensor for selected components only.

    Returns:
        probs_tensor: (n_selected, n_state) tensor
        n_state: maximum number of states across selected components
    """
    if device is None:
        device = torch.device("cpu")

    n_state = max(len(probs_dict[name]) for name in selected)

    probs_list = []
    for name in selected:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0
               for s in range(n_state)]
        probs_list.append(row)

    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)
    return probs_tensor, n_state


def run_reduced_tsum(
    sfun: Callable,
    probs_dict: Dict[str, Dict],
    selected: List[str],
    fixed: Dict[str, int],
    *,
    seed_rules: Optional[List[Dict]] = None,
    unk_prob_thres: float = 1e-5,
    bias_factor: float = 0.0,
    bias_rounds: int = 0,
    n_workers: int = 1,
    devices: Optional[List[str]] = None,
    output_dir: str = "results_reduced",
    n_sample: int = 1_000_000,
    sample_batch_size: int = 100_000,
) -> Optional[Dict[str, Any]]:
    """Run TSUM on a reduced set of components.

    Args:
        sfun: the *original* (full) system function — will be wrapped internally
        probs_dict: full probability dictionary (all components)
        selected: list of component names to model
        fixed: dict mapping fixed component names to their best state index
        seed_rules: optional failure rules to seed TSUM with (will be filtered)
        unk_prob_thres: convergence threshold
        bias_factor: bias factor for discovery sampling (0=off)
        bias_rounds: use biased sampling for first N rounds (0=all)
        n_workers: parallel workers
        devices: GPU device list
        output_dir: output directory
        n_sample: samples per round
        sample_batch_size: batch size for sampling

    Returns:
        TSUM result dict, or None on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_list = devices or []
    device = torch.device(device_list[0] if device_list else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    multi_devices = device_list if len(device_list) > 1 else None

    # Build reduced probability tensor
    probs_tensor, n_state = build_reduced_probs(probs_dict, selected, device)

    # Compute probability mass of the conditioning event
    p_fixed = compute_fixed_probability(probs_dict, fixed)

    all_names = list(probs_dict.keys())
    n_multistate = sum(1 for c in selected if len(probs_dict[c]) > 2)
    n_binary = sum(1 for c in selected if len(probs_dict[c]) == 2)

    print(f"\n  Full model:  {len(all_names)} components")
    print(f"  Selected:    {len(selected)} variable components")
    print(f"  Fixed:       {len(fixed)} components at best state")
    print(f"  Max states:  {n_state}")
    if n_multistate:
        print(f"    Multi-state: {n_multistate}")
    if n_binary:
        print(f"    Binary:      {n_binary}")
    print(f"\n  P(all fixed at best): {p_fixed:.6f}")
    print(f"  P(any fixed degraded): {1 - p_fixed:.6f}")

    # Wrap sfun
    reduced_sfun = make_reduced_sfun(sfun, fixed)

    # Verify: all operational
    test_state = {name: max(int(s) for s in probs_dict[name].keys())
                  for name in selected}
    fval, sys_st, _ = reduced_sfun(test_state)
    print(f"\n  Verification (all operational): blackout={fval:.2f}%, sys_st={sys_st}")

    # Filter seed rules
    filtered_rules = None
    if seed_rules:
        filtered_rules = filter_seed_rules(seed_rules, selected)
        print(f"\n  Seed rules: {len(filtered_rules)} applicable "
              f"(from {len(seed_rules)} total)")
        if filtered_rules:
            lengths = Counter(sum(1 for k in r if k != 'sys') for r in filtered_rules)
            for l in sorted(lengths):
                print(f"    {l} conditions: {lengths[l]} rules")
        if not filtered_rules:
            filtered_rules = None

    # Discovery probs
    disc_probs = None
    if bias_factor > 0:
        critical = None
        if filtered_rules:
            critical = tsum.get_critical_components(filtered_rules, min_frequency=0.3)
            if critical:
                print(f"  Critical components: {', '.join(critical)}")
        disc_probs = tsum.make_discovery_probs(
            probs_tensor, bias_factor=bias_factor,
            row_names=selected, critical_components=critical)
        print(f"  Bias factor: {bias_factor}")

    # Save reduction metadata
    meta = {
        "n_full": len(all_names),
        "n_selected": len(selected),
        "n_fixed": len(fixed),
        "n_state": n_state,
        "selected_components": selected,
        "p_fixed_at_best": p_fixed,
        "p_any_fixed_degraded": 1 - p_fixed,
    }
    with open(output_dir / "reduction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Run TSUM
    print(f"\n{'='*60}")
    print(f"TSUM rule extraction ({len(selected)} variables, {n_state} states)")
    print(f"{'='*60}")
    print(f"  Convergence: unk_prob < {unk_prob_thres:.0e}")
    print(f"  Device:      {device}")
    if n_workers > 1:
        print(f"  Workers:     {n_workers}")
    print(f"\nStarting rule extraction...\n", flush=True)

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=reduced_sfun,
        probs=probs_tensor,
        row_names=selected,
        n_state=n_state,
        sys_surv_st=1,
        rules_fail=filtered_rules,
        unk_prob_thres=unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=n_sample,
        sample_batch_size=sample_batch_size,
        discovery_probs=disc_probs,
        bias_rounds=bias_rounds,
        n_workers=n_workers,
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
        print(f"\n  Note: these probabilities are conditional on all {len(fixed)}")
        print(f"  fixed components being at best state (P={p_fixed:.6f}).")
        print(f"  Unconditional P(failure) >= {p_fail * p_fixed:.2e}")

    return result


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Variable reduction TSUM (generic)")

    # System function
    parser.add_argument("--sfun-module", type=str, required=True,
                        help="Path to Python module with sfun factory function")
    parser.add_argument("--sfun-func", type=str, default="make_sfun",
                        help="Name of the factory function (default: make_sfun)")
    parser.add_argument("--sfun-args", type=str, default="",
                        help='JSON dict of kwargs for the sfun factory')
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing edges.json and probs.json")

    # Component selection
    parser.add_argument("--mode", type=str, default="multistate",
                        choices=["multistate", "frequency", "custom"],
                        help="Component selection mode (default: multistate)")
    parser.add_argument("--failures-dir", type=str, default=None,
                        help="Directory with failures_k*.json (for --mode frequency)")
    parser.add_argument("--min-freq", type=int, default=1,
                        help="Minimum failure frequency (for --mode frequency)")
    parser.add_argument("--components", type=str, default="",
                        help="Comma-separated component names (for --mode custom)")

    # Seed rules
    parser.add_argument("--seed-rules", type=str, default=None,
                        help="Path to seed_rules_fail.json for seeding TSUM")

    # TSUM parameters
    parser.add_argument("--unk-prob-thres", type=float, default=1e-5,
                        help="TSUM convergence threshold (default: 1e-5)")
    parser.add_argument("--bias-factor", type=float, default=0.0,
                        help="Bias factor for discovery sampling (0=off)")
    parser.add_argument("--bias-rounds", type=int, default=0,
                        help="Use biased sampling for first N rounds (0=all)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--devices", type=str, default="",
                        help="Comma-separated GPU devices")
    parser.add_argument("--output-dir", type=str, default="results_reduced",
                        help="Output directory (default: results_reduced)")
    return parser.parse_args()


def main():
    args = parse_args()
    from tsum.fixed_k_search import _load_sfun_module

    print("=" * 60)
    print("Variable Reduction TSUM")
    print("=" * 60)

    # Load data
    _, probs_dict = load_tsum_inputs(args.data_dir)

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

    # Load sfun
    print(f"\nLoading system function from {args.sfun_module}:{args.sfun_func}...")
    sfun = _load_sfun_module(args.sfun_module, args.sfun_func,
                             args.sfun_args or None)

    # Load seed rules
    seed_rules = None
    if args.seed_rules:
        seed_path = Path(args.seed_rules)
        if seed_path.exists():
            with open(seed_path) as f:
                seed_rules = json.load(f)
        else:
            print(f"Warning: seed rules not found: {seed_path}")

    # Parse devices
    device_list = ([d.strip() for d in args.devices.split(",") if d.strip()]
                   if args.devices else None)

    run_reduced_tsum(
        sfun=sfun,
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


if __name__ == "__main__":
    main()
