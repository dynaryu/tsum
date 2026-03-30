"""
Fixed-k search: find short failure rules by sampling k degraded components.

A general-purpose tool for discovering short failure rules that dominate
system failure probability. Can be used standalone for discovery, or as
a seeding step before TSUM rule extraction.

Phase 1: Randomly sample combinations of k degraded components (worst-state)
          and check if the system fails.
Phase 2: Minimize discovered failures into minimal rules.
Phase 3 (optional): Seed TSUM with discovered rules and continue MCS extraction.

Usage:
    # Discovery only
    python -m tsum.fixed_k_search --sfun-module path/to/sfun_module.py \\
        --data-dir path/to/tsum_data --k 2 3 --n-samples 100000

    # Load pre-computed failures and seed TSUM
    python -m tsum.fixed_k_search --sfun-module path/to/sfun_module.py \\
        --data-dir path/to/tsum_data --load-failures results/failures_k3.json \\
        --run-tsum --output-dir results_seeded

See demos/case118/run_fixed_k_search.py for a case-specific example.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

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

    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())

    probs_list = []
    for name in row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0
               for s in range(n_state)]
        probs_list.append(row)
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32)

    return row_names, n_state, probs_tensor, probs_dict


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
        load_failures: list of paths to pre-computed failure JSON files to load
        run_tsum: whether to run TSUM after discovery
        unk_prob_thres: TSUM convergence threshold
        bias_factor: TSUM bias factor
        bias_rounds: TSUM bias rounds
        devices: GPU device list
        output_dir: output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = probs_tensor.device
    multi_devices = devices if devices and len(devices) > 1 else None

    # Build max states for output
    max_states = {}
    for i, name in enumerate(row_names):
        row = probs_tensor[i]
        nonzero = (row > 0).nonzero(as_tuple=True)[0]
        max_states[name] = int(nonzero[-1].item()) if len(nonzero) > 0 else 0

    all_failures = []
    t0 = time.time()

    # ==================================================================
    # Phase 1: Fixed-k search (or load pre-computed)
    # ==================================================================
    if load_failures:
        print(f"\n{'='*60}")
        print("Phase 1: Loading pre-computed failures")
        print(f"{'='*60}")
        for fpath in load_failures:
            data = json.load(open(fpath))
            print(f"  {fpath}: {len(data)} failures")
            for entry in data:
                # Reconstruct full state dict
                state = dict(max_states)
                for comp, val in entry["degraded"].items():
                    state[comp] = val
                # We don't have fval/sys_st from file, re-evaluate
                fval, sys_st, _ = sfun(state)
                if sys_st < 1:  # verify it's still a failure
                    all_failures.append((state, fval, sys_st))
        print(f"Loaded {len(all_failures)} verified failures")
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

    t_search = time.time() - t0
    print(f"\nPhase 1 complete: {len(all_failures)} total failures in {t_search:.1f}s")

    if not all_failures:
        print("No failures found. Exiting.")
        return

    # ==================================================================
    # Phase 2: Minimize failures into minimal rules
    # ==================================================================
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

    # Deduplicate rules
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

    with open(output_dir / "seed_rules_fail.json", "w") as f:
        json.dump(unique_rules, f, indent=2)

    # Show distribution
    from collections import Counter
    lengths = [sum(1 for kn in r if kn != 'sys') for r in unique_rules]
    dist = Counter(lengths)
    print(f"\nSeed rule length distribution:")
    for l in sorted(dist):
        print(f"  {l} conditions: {dist[l]} rules")

    if not run_tsum:
        print("\nDone. Use --run-tsum to continue with TSUM rule extraction.")
        return unique_rules

    # ==================================================================
    # Phase 3: Seed TSUM
    # ==================================================================
    print(f"\n{'='*60}")
    print("Phase 3: TSUM rule extraction (seeded with fixed-k rules)")
    print(f"{'='*60}")

    disc_probs = None
    if bias_factor > 0:
        disc_probs = tsum.make_discovery_probs(probs_tensor, bias_factor=bias_factor)
        print(f"  Bias factor: {bias_factor}")

    print(f"  Seed rules:  {len(unique_rules)} failure rules")
    print(f"  Convergence: unk_prob < {unk_prob_thres:.0e}")
    if n_workers > 1:
        print(f"  Workers:     {n_workers}")
    print(f"\nStarting rule extraction...\n", flush=True)

    t2 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        rules_fail=unique_rules,
        unk_prob_thres=unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=1_000_000,
        sample_batch_size=100_000,
        discovery_probs=disc_probs,
        bias_rounds=bias_rounds,
        n_workers=n_workers,
        devices=multi_devices,
        output_dir=str(output_dir),
    )
    t_tsum = time.time() - t2

    print(f"\nTSUM completed in {t_tsum:.1f}s")
    print(f"Total time: {time.time() - t0:.1f}s")

    # Summary
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            rounds = [json.loads(line) for line in f if line.strip()]
        last = rounds[-1]
        print(f"\n--- Summary ---")
        print(f"  Fixed-k/load: {t_search:.1f}s")
        print(f"  Minimization: {t_minimize:.1f}s")
        print(f"  TSUM rounds:  {len(rounds)} ({t_tsum:.1f}s)")
        print(f"  Surv rules:   {last.get('n_rules_surv', '?')}")
        print(f"  Fail rules:   {last.get('n_rules_fail', '?')}")
        print(f"  P(survival):  {last.get('p_survival', '?')}")
        print(f"  P(failure):   {last.get('p_failure', '?')}")
        print(f"  P(unknown):   {last.get('p_unknown', '?')}")

    return unique_rules
