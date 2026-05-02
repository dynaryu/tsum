"""
Hierarchical Subset Simulation for TSUM rule extraction.

Phase 1: Partition the network into zones and run SuS on zone groups
         (other zones fixed at best state).  For weakly-coupled networks
         (e.g. case300) single-zone runs suffice.  For tightly-coupled
         networks (e.g. case118) multi-zone groups may be needed because
         no single zone can cause failure alone.

Phase 2: Combine zone-local rules and run full-model SuS with seeds.

Zone partitioning:
  - If MATPOWER case has multiple zones (mpc.bus column 11), use those.
  - Otherwise, use spectral bisection on the bus admittance graph.

Usage:
    # Phase 1: per-zone runs (case300 — weakly coupled, single zones work)
    python run_hierarchical_sus.py --case case300 --phase 1
    python run_hierarchical_sus.py --case case300 --phase 1 --zone 2

    # Phase 1: multi-zone group (case118 — needs zone pairs for failure)
    python run_hierarchical_sus.py --case case118 --phase 1 --zone 2,3
    python run_hierarchical_sus.py --case case118 --phase 1 --zone 0,2

    # Phase 1: auto-detect minimum zone combos that can cause failure
    python run_hierarchical_sus.py --case case118 --phase 1 --auto-combos

    # Phase 2: combine and run full model
    python run_hierarchical_sus.py --case case300 --phase 2
"""
import sys
import os
import re
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch


# =====================================================================
# Network partitioning
# =====================================================================

def parse_matpower_zones(case_path):
    """Extract bus zones from MATPOWER .m file."""
    with open(case_path) as f:
        content = f.read()

    bus_match = re.search(r'mpc\.bus\s*=\s*\[(.*?)\];', content, re.DOTALL)
    bus_zone = {}
    bus_load = {}
    for line in bus_match.group(1).strip().split('\n'):
        line = line.strip().rstrip(';')
        if not line or line.startswith('%'):
            continue
        vals = line.split()
        if len(vals) >= 11:
            bid = int(vals[0])
            bus_zone[bid] = int(vals[10])
            bus_load[bid] = float(vals[2])

    # Parse gen capacity
    gen_match = re.search(r'mpc\.gen\s*=\s*\[(.*?)\];', content, re.DOTALL)
    gen_cap = defaultdict(float)
    gen_buses = set()
    for line in gen_match.group(1).strip().split('\n'):
        line = line.strip().rstrip(';')
        if not line or line.startswith('%'):
            continue
        vals = line.split()
        bid = int(vals[0])
        gen_buses.add(bid)
        gen_cap[bid] += float(vals[8])

    # Parse branches
    branch_match = re.search(r'mpc\.branch\s*=\s*\[(.*?)\];', content, re.DOTALL)
    branch_buses = []
    for line in branch_match.group(1).strip().split('\n'):
        line = line.strip().rstrip(';')
        if not line or line.startswith('%'):
            continue
        vals = line.split()
        branch_buses.append((int(vals[0]), int(vals[1]), float(vals[3])))

    return bus_zone, bus_load, gen_buses, gen_cap, branch_buses


def spectral_partition(case_path, n_partitions=4):
    """Partition network using recursive spectral bisection."""
    bus_zone, bus_load, gen_buses, gen_cap, branch_buses = parse_matpower_zones(case_path)
    bus_ids = sorted(bus_zone.keys())
    n = len(bus_ids)
    bus_idx = {b: i for i, b in enumerate(bus_ids)}

    # Build weighted Laplacian (susceptance-weighted)
    adj = defaultdict(dict)
    for fb, tb, x in branch_buses:
        w = 1.0 / x if x > 0 else 1000.0
        adj[fb][tb] = w
        adj[tb][fb] = w

    L = np.zeros((n, n))
    for b in bus_ids:
        i = bus_idx[b]
        for nb, w in adj[b].items():
            j = bus_idx[nb]
            L[i, j] = -w
            L[i, i] += w

    # Recursive bisection
    n_levels = int(np.ceil(np.log2(n_partitions)))

    def bisect(indices):
        if len(indices) < 4:
            return [indices]
        sub_L = np.zeros((len(indices), len(indices)))
        sub_idx = {s: k for k, s in enumerate(indices)}
        for s in indices:
            b = bus_ids[s]
            for nb, w in adj[b].items():
                j = bus_idx[nb]
                if j in sub_idx:
                    sub_L[sub_idx[s], sub_idx[j]] = -w
                    sub_L[sub_idx[s], sub_idx[s]] += w
        evals, evecs = np.linalg.eigh(sub_L)
        fiedler = evecs[:, 1]
        median = np.median(fiedler)
        left = [i for i, v in zip(indices, fiedler) if v <= median]
        right = [i for i, v in zip(indices, fiedler) if v > median]
        if not left or not right:
            return [indices]
        return left, right

    # Recursive partitioning
    parts = [list(range(n))]
    for _ in range(n_levels):
        if len(parts) >= n_partitions:
            break
        new_parts = []
        for p in parts:
            result = bisect(p)
            if isinstance(result, list) and isinstance(result[0], list):
                new_parts.extend(result)
            elif isinstance(result, tuple):
                new_parts.extend(result)
            else:
                new_parts.append(result)
        parts = new_parts

    # Convert to bus_zone dict (zone = partition index)
    new_zone = {}
    for zi, part in enumerate(parts):
        for idx in part:
            new_zone[bus_ids[idx]] = zi

    return new_zone


def get_zones(case_path, n_partitions=4):
    """Get zone partition — use MATPOWER zones if multi-zone, else spectral."""
    bus_zone, bus_load, gen_buses, gen_cap, branch_buses = parse_matpower_zones(case_path)

    zones = set(bus_zone.values())
    if len(zones) > 1:
        print(f"Using MATPOWER zones: {len(zones)} zones")
        return bus_zone, bus_load, gen_buses, gen_cap, branch_buses
    else:
        print(f"Single MATPOWER zone — using spectral partitioning into {n_partitions} groups")
        new_zone = spectral_partition(case_path, n_partitions)
        return new_zone, bus_load, gen_buses, gen_cap, branch_buses


# =====================================================================
# Map zones to TSUM components
# =====================================================================

def map_components_to_zones(edges_path, probs_path, bus_zone):
    """Map TSUM component IDs to zones."""
    with open(edges_path) as f:
        edges = json.load(f)
    with open(probs_path) as f:
        probs_dict = json.load(f)

    comp_zone = {}
    gen_comps = set()

    for eid, edata in edges.items():
        ct = edata.get('component_type', '')
        if eid.startswith('vbus'):
            bid = int(eid.replace('vbus', ''))
            if bid in bus_zone:
                comp_zone[eid] = bus_zone[bid]
            if ct == 'generator_bus':
                gen_comps.add(eid)
        elif eid.startswith('br'):
            f_bus = edata['from'].replace('_int', '').replace('bus', '')
            t_bus = edata['to'].replace('_int', '').replace('bus', '')
            try:
                fb, tb = int(f_bus), int(t_bus)
                fz = bus_zone.get(fb)
                tz = bus_zone.get(tb)
                if fz == tz and fz is not None:
                    comp_zone[eid] = fz
                elif fz is not None and tz is not None:
                    # Cross-zone branch: assign to "cross" zone
                    comp_zone[eid] = f"cross_{min(fz,tz)}_{max(fz,tz)}"
                elif fz is not None:
                    comp_zone[eid] = fz
                elif tz is not None:
                    comp_zone[eid] = tz
            except ValueError:
                pass

    return comp_zone, gen_comps, probs_dict


def print_zone_summary(comp_zone, probs_dict, bus_zone, bus_load, gen_buses, gen_cap):
    """Print zone composition summary."""
    zone_info = defaultdict(lambda: {'gen': 0, 'bus': 0, 'br': 0, 'comps': []})
    for eid, z in comp_zone.items():
        zone_info[z]['comps'].append(eid)
        if eid.startswith('vbus') and len(probs_dict[eid]) == 4:
            zone_info[z]['gen'] += 1
        elif eid.startswith('vbus'):
            zone_info[z]['bus'] += 1
        elif eid.startswith('br'):
            zone_info[z]['br'] += 1

    print(f"\n{'Zone':<12} {'Gen':>5} {'Bus':>5} {'Branch':>7} {'Total':>6}")
    print("-" * 40)
    for z in sorted(zone_info.keys(), key=str):
        zi = zone_info[z]
        total = zi['gen'] + zi['bus'] + zi['br']
        print(f"{str(z):<12} {zi['gen']:>5} {zi['bus']:>5} {zi['br']:>7} {total:>6}")

    # Zone load/capacity from bus-level data
    zone_load = defaultdict(float)
    zone_gcap = defaultdict(float)
    for bid, z in bus_zone.items():
        zone_load[z] += bus_load.get(bid, 0)
        if bid in gen_buses:
            zone_gcap[z] += gen_cap.get(bid, 0)

    print(f"\n{'Zone':<12} {'Load(MW)':>10} {'GenCap(MW)':>12} {'Reserve':>10}")
    print("-" * 48)
    for z in sorted(set(bus_zone.values())):
        load = zone_load[z]
        gcap = zone_gcap[z]
        reserve = (gcap - load) / load * 100 if load > 0 else float('inf')
        print(f"{str(z):<12} {load:>10.0f} {gcap:>12.0f} {reserve:>9.1f}%")

    return zone_info


# =====================================================================
# Feasibility analysis: which zone combos can cause failure?
# =====================================================================

def find_feasible_combos(bus_zone, bus_load, gen_buses, gen_cap,
                         blackout_threshold_pct, max_combo_size=3):
    """Find minimum zone combinations that can cause system failure.

    A zone combo can cause failure if losing ALL generation in those zones
    sheds more than blackout_threshold_pct of total load.  This is a
    necessary (not sufficient) condition — actual failure depends on
    network topology and branch capacity — but it identifies combos that
    are worth running.

    Returns list of (combo_tuple, shed_pct) sorted by shed_pct descending.
    """
    from itertools import combinations

    total_load = sum(bus_load.values())
    threshold_mw = total_load * blackout_threshold_pct / 100.0

    zone_gcap = defaultdict(float)
    for bid in gen_buses:
        z = bus_zone.get(bid)
        if z is not None:
            zone_gcap[z] += gen_cap.get(bid, 0)

    all_zones = sorted(set(bus_zone.values()))
    total_gen = sum(zone_gcap[z] for z in all_zones)

    feasible = []
    for size in range(1, min(max_combo_size, len(all_zones)) + 1):
        for combo in combinations(all_zones, size):
            lost_gen = sum(zone_gcap[z] for z in combo)
            remaining_gen = total_gen - lost_gen
            shed = max(0, total_load - remaining_gen)
            shed_pct = shed / total_load * 100
            if shed_pct >= blackout_threshold_pct:
                # Check it's minimal: no proper subset already causes failure
                is_minimal = True
                for prev_combo, _ in feasible:
                    if set(prev_combo) < set(combo):
                        is_minimal = False
                        break
                if is_minimal:
                    feasible.append((combo, shed_pct))

    feasible.sort(key=lambda x: len(x[0]))  # smallest combos first
    return feasible


def print_feasibility(bus_zone, bus_load, gen_buses, gen_cap,
                      blackout_threshold_pct, comp_zone):
    """Print feasibility analysis and return feasible combos."""
    total_load = sum(bus_load.values())

    combos = find_feasible_combos(
        bus_zone, bus_load, gen_buses, gen_cap,
        blackout_threshold_pct, max_combo_size=3)

    print(f"\n{'='*60}")
    print(f"Feasibility: which zone combos can cause failure?")
    print(f"  System load: {total_load:.0f} MW")
    print(f"  Blackout threshold: {blackout_threshold_pct}%"
          f" = {total_load * blackout_threshold_pct / 100:.0f} MW shed")
    print(f"{'='*60}")

    if not combos:
        print("  No combos up to size 3 can cause failure!")
        return []

    # Count components per combo
    for combo, shed_pct in combos:
        n_comps = sum(1 for eid, z in comp_zone.items()
                      if z in combo or
                      (isinstance(z, str) and z.startswith("cross") and
                       any(str(zz) in z for zz in combo)))
        zones_str = "+".join(str(z) for z in combo)
        print(f"  Zones {zones_str:>8}: {n_comps:>4} comps, "
              f"worst-case shed {shed_pct:.1f}%"
              f"  {'<-- minimal' if len(combo) == min(len(c) for c, _ in combos) else ''}")

    return combos


# =====================================================================
# Phase 1: Per-zone SuS
# =====================================================================

def run_phase1_zone(zone_ids_group, zone_comps, probs_dict, all_row_names,
                    n_state, sfun, device, args):
    """Run SuS on one or more zones with other components fixed at best state.

    Args:
        zone_ids_group: single zone ID or list of zone IDs to vary together.
        zone_comps: list of component IDs in the variable set.
    """
    if not isinstance(zone_ids_group, (list, tuple)):
        zone_ids_group = [zone_ids_group]

    group_label = "+".join(str(z) for z in zone_ids_group)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tsum import tsum

    # Build probs tensor for FULL model (all components)
    probs_list = []
    for name in all_row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0 for s in range(n_state)]
        probs_list.append(row)
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)

    # For zone-local run: fix non-zone components at best state
    # We do this by setting their probabilities to 1.0 at best state
    zone_comp_set = set(zone_comps)
    fixed_probs = probs_tensor.clone()
    for i, name in enumerate(all_row_names):
        if name not in zone_comp_set:
            fixed_probs[i] = 0.0
            best_state = len(probs_dict[name]) - 1
            fixed_probs[i, best_state] = 1.0

    output_dir = Path(args.output_dir) / f"zone_{group_label}"

    # Load seed rules if provided
    seed_fail_rules = []
    if args.seed_rules:
        seed_path = Path(args.seed_rules)
        if seed_path.exists():
            with open(seed_path) as f:
                all_seeds = json.load(f)
            # Filter to rules whose components are all within the zone group
            for r in all_seeds:
                comps = {k for k in r if k != "sys"}
                if comps <= zone_comp_set:
                    seed_fail_rules.append(r)
            if seed_fail_rules:
                print(f"  Seed fail rules for zone {group_label}: "
                      f"{len(seed_fail_rules)}")

    n_var = len(zone_comps)
    print(f"\n  Running SuS on zone(s) {group_label}: "
          f"{n_var} variable components")
    print(f"  Fixed at best: {len(all_row_names) - n_var} components")
    print(f"  Output: {output_dir}")

    # Full-model run (no components fixed at best): zone-conditional caveats
    # don't apply, so honour the user's --sus-surv-mc-samples and
    # --unk-prob-thres. Per-zone runs still hard-override to 0 since zone-local
    # surv rules / unk probabilities aren't valid for the full model.
    is_full_model = (n_var == len(all_row_names))
    if is_full_model:
        eff_surv_mc_samples = args.sus_surv_mc_samples
        eff_unk_prob_thres = args.unk_prob_thres
        print(f"  Full-model run: surv mining ON "
              f"(sus_surv_mc_samples={eff_surv_mc_samples}), "
              f"unk_prob_thres={eff_unk_prob_thres}")
    else:
        eff_surv_mc_samples = 0
        eff_unk_prob_thres = 0

    device_list = ([d.strip() for d in args.devices.split(",") if d.strip()]
                   if args.devices else [])
    multi_devices = device_list if len(device_list) > 1 else None

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=fixed_probs,
        row_names=all_row_names,
        n_state=n_state,
        sys_surv_st=1,
        rules_fail=seed_fail_rules if seed_fail_rules else None,
        unk_prob_thres=eff_unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=args.n_sample,
        sample_batch_size=args.sample_batch_size,
        max_rounds=args.max_rounds,
        max_stale_rounds=args.max_stale_rounds,
        p_fail_rel_tol=args.p_fail_rel_tol,
        p_fail_k_sigma=args.p_fail_k_sigma,
        p_fail_window=args.p_fail_window,
        p_fail_stale_rounds=args.p_fail_stale_rounds,
        n_workers=args.n_workers,
        devices=multi_devices,
        prob_update_every=args.prob_update_every,
        # Subset Simulation
        use_subset_sim=True,
        sus_n_per_level=args.sus_n_per_level,
        sus_p0=args.sus_p0,
        sus_max_levels=args.sus_max_levels,
        sus_n_flip_mean=args.sus_n_flip_mean,
        sus_surv_mc_samples=eff_surv_mc_samples,
        output_dir=str(output_dir),
    )
    elapsed = time.time() - t0

    print(f"\n  Zone(s) {group_label} completed in {elapsed:.1f}s")
    return result


# =====================================================================
# Phase 2: Combine and run full model
# =====================================================================

def run_phase2(zone_dir_names, probs_dict, all_row_names, n_state, sfun,
               device, args):
    """Combine zone fail rules and run full-model plain MCS.

    Phase 1 (SuS per zone) finds failure rules efficiently.
    Phase 2 uses plain MCS on the full model — prior MC samples are mostly
    survival states, so this naturally discovers survival rules while also
    picking up any failure rules the zones missed.
    """

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tsum import tsum

    # Collect fail rules from all zone directories
    combined_fail = []
    base_dir = Path(args.output_dir)

    for zdir_name in zone_dir_names:
        zone_dir = base_dir / zdir_name
        fail_path = zone_dir / "rules_leq_0.json"

        if fail_path.exists():
            with open(fail_path) as f:
                rules = json.load(f)
            print(f"  {zdir_name}: {len(rules)} fail rules")
            combined_fail.extend(rules)

        # Note: surv rules from zone runs are NOT valid for the full model
        # (they assumed other zones at best state). We skip them.

    # Also load any additional seed rules
    if args.seed_rules:
        seed_path = Path(args.seed_rules)
        if seed_path.exists():
            with open(seed_path) as f:
                extra_seeds = json.load(f)
            print(f"  Extra seed rules: {len(extra_seeds)}")
            combined_fail.extend(extra_seeds)

    print(f"\n  Combined fail rules (seeds): {len(combined_fail)}")
    print(f"  Surv rules: starting from scratch (zone surv rules not valid)")

    # Build full probs tensor
    probs_list = []
    for name in all_row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0 for s in range(n_state)]
        probs_list.append(row)
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)

    output_dir = base_dir / "full_model"

    device_list = ([d.strip() for d in args.devices.split(",") if d.strip()]
                   if args.devices else [])
    multi_devices = device_list if len(device_list) > 1 else None

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=all_row_names,
        n_state=n_state,
        sys_surv_st=1,
        rules_fail=combined_fail,
        unk_prob_thres=args.unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=args.n_sample,
        sample_batch_size=args.sample_batch_size,
        max_rounds=args.max_rounds,
        p_fail_rel_tol=args.p_fail_rel_tol,
        p_fail_k_sigma=args.p_fail_k_sigma,
        p_fail_window=args.p_fail_window,
        p_fail_stale_rounds=args.p_fail_stale_rounds,
        n_workers=args.n_workers,
        devices=multi_devices,
        prob_update_every=args.prob_update_every,
        # Plain MCS — no SuS. Prior MC samples are mostly survival states,
        # so this efficiently discovers survival rules. Failure rules from
        # zone seeds are already loaded; new ones found when MC hits rare
        # failure states.
        use_subset_sim=False,
        output_dir=str(output_dir),
    )
    elapsed = time.time() - t0

    print(f"\n  Full model completed in {elapsed:.1f}s")
    return result


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hierarchical SuS for TSUM rule extraction")
    parser.add_argument("--case", type=str, required=True,
                        choices=["case118", "case300"],
                        help="Which case to run")
    parser.add_argument("--phase", type=int, required=True,
                        choices=[1, 2],
                        help="Phase 1 (per-zone) or Phase 2 (full model)")
    parser.add_argument("--zone", type=str, default=None,
                        help="Zone(s) for Phase 1. Single: '2'. "
                             "Multi-zone group: '2,3'. "
                             "Multiple groups: '2,3;0,2' (default: all single zones)")
    parser.add_argument("--auto-combos", action="store_true",
                        help="Auto-detect minimal zone combos that can cause failure "
                             "and run each combo as a Phase 1 group")
    parser.add_argument("--n-partitions", type=int, default=4,
                        help="Number of spectral partitions if no MATPOWER zones (default: 4)")
    parser.add_argument("--unk-prob-thres", type=float, default=1e-3,
                        help="Convergence threshold (default: 1e-3)")
    parser.add_argument("--n-sample", type=int, default=10_000_000)
    parser.add_argument("--sample-batch-size", type=int, default=100_000)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--max-stale-rounds", type=int, default=10,
                        help="Phase 1: stop after N rounds with no new fail rules (0=disabled)")
    parser.add_argument("--p-fail-rel-tol", type=float, default=0.0,
                        help="Coverage-plateau termination: stop when relative p_failure "
                             "change over a refresh window stays below this. "
                             "0 = disabled. Try 0.05.")
    parser.add_argument("--p-fail-k-sigma", type=float, default=2.0,
                        help="Noise floor multiplier for the plateau check (default: 2.0).")
    parser.add_argument("--p-fail-window", type=int, default=10,
                        help="Refresh-windows used for the cumulative p_failure delta "
                             "(default: 10).")
    parser.add_argument("--p-fail-stale-rounds", type=int, default=3,
                        help="Consecutive plateau windows required to terminate "
                             "(default: 3).")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--devices", type=str, default="")
    parser.add_argument("--seed-rules", type=str, default="",
                        help="Path to seed failure rules JSON")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results_hierarchical)")
    # Subset Simulation parameters
    parser.add_argument("--sus-n-per-level", type=int, default=2000,
                        help="SuS samples per level (default: 2000)")
    parser.add_argument("--sus-p0", type=float, default=0.1,
                        help="SuS conditional probability per level (default: 0.1)")
    parser.add_argument("--sus-max-levels", type=int, default=10,
                        help="SuS max levels (default: 10)")
    parser.add_argument("--sus-n-flip-mean", type=float, default=5.0,
                        help="SuS mean component flips per MCMC step (default: 5.0)")
    parser.add_argument("--sus-surv-mc-samples", type=int, default=1_000_000,
                        help="Extra prior-MC samples per round for survival rule mining (default: 1000000)")
    parser.add_argument("--prob-update-every", type=int, default=500,
                        help="Unbiased probability refresh interval (default: 500)")
    return parser.parse_args()


def main():
    args = parse_args()

    HERE = Path(__file__).parent
    case_dir = HERE / args.case

    # Paths
    case_m = case_dir / f"{args.case}.m"
    data_dir = case_dir / f"{args.case}_tsum_bus"
    edges_path = data_dir / "edges.json"
    probs_path = data_dir / "probs.json"

    if args.output_dir is None:
        args.output_dir = str(case_dir / "results_hierarchical")

    print("=" * 60)
    print(f"Hierarchical SuS — {args.case} — Phase {args.phase}")
    print("=" * 60)

    # 1. Get zone partition
    bus_zone, bus_load, gen_buses, gen_cap, branch_buses = get_zones(
        str(case_m), args.n_partitions)

    # 2. Map TSUM components to zones
    comp_zone, gen_comps, probs_dict = map_components_to_zones(
        str(edges_path), str(probs_path), bus_zone)

    zone_info = print_zone_summary(
        comp_zone, probs_dict, bus_zone, bus_load, gen_buses, gen_cap)

    # 3. Setup
    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())

    device_list = ([d.strip() for d in args.devices.split(",") if d.strip()]
                   if args.devices else [])
    device = torch.device(
        device_list[0] if device_list
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build sfun
    sys.path.insert(0, str(case_dir))
    from sfun_dcopt import make_dcopt_sfun

    blackout_thres = 13.8 if args.case == "case118" else 10.0  # adjust per case
    sfun = make_dcopt_sfun(
        case_path=str(case_m),
        blackout_threshold=blackout_thres,
        alpha=2.0,
    )
    print(f"\n  sfun ready, blackout_threshold={blackout_thres}%")

    # 4. Feasibility check
    all_zone_ids = sorted(set(z for z in comp_zone.values()
                               if not isinstance(z, str) or not z.startswith("cross")))

    feasible_combos = print_feasibility(
        bus_zone, bus_load, gen_buses, gen_cap,
        blackout_thres, comp_zone)

    # Helper to collect components for a group of zone IDs
    def collect_zone_comps(zone_group):
        """Get all components belonging to the given zones, including
        cross-zone branches that connect any two zones in the group."""
        zset = set(zone_group)
        comps = [eid for eid, z in comp_zone.items() if z in zset]
        # Include cross-zone branches where BOTH endpoints are in the group
        for eid, z in comp_zone.items():
            if isinstance(z, str) and z.startswith("cross"):
                # z is like "cross_1_2" — extract the two zone IDs
                parts = z.split("_")[1:]
                try:
                    z1, z2 = int(parts[0]), int(parts[1])
                    if z1 in zset and z2 in zset and eid not in comps:
                        comps.append(eid)
                except (ValueError, IndexError):
                    pass
        return comps

    # 5. Run
    if args.phase == 1:
        # Determine which zone groups to run
        groups_to_run = []

        if args.auto_combos:
            # Run all feasible minimal combos
            if not feasible_combos:
                print("\nNo feasible combos found — try individual zones or "
                      "increase --n-partitions")
                return
            for combo, shed_pct in feasible_combos:
                groups_to_run.append(list(combo))

        elif args.zone is not None:
            # Parse --zone argument: "2,3" or "2,3;0,2"
            for group_str in args.zone.split(";"):
                group_str = group_str.strip()
                if not group_str:
                    continue
                zone_group = []
                for z_str in group_str.split(","):
                    z_str = z_str.strip()
                    try:
                        zone_group.append(int(z_str))
                    except ValueError:
                        zone_group.append(z_str)
                groups_to_run.append(zone_group)
        else:
            # Default: run each zone individually
            for zid in all_zone_ids:
                groups_to_run.append([zid])

        # Warn about infeasible single-zone runs
        for group in groups_to_run:
            if len(group) == 1:
                zid = group[0]
                is_feasible = any(zid in combo for combo, _ in feasible_combos
                                  if len(combo) == 1)
                if not is_feasible:
                    print(f"\n  WARNING: Zone {zid} alone cannot cause failure.")
                    print(f"  Consider using --zone with zone pairs "
                          f"(e.g., --zone 2,3) or --auto-combos")

        # Run each group
        print(f"\n  Phase 1: {len(groups_to_run)} zone group(s) to run")
        for group in groups_to_run:
            zone_comps = collect_zone_comps(group)
            run_phase1_zone(
                group, zone_comps, probs_dict, row_names, n_state,
                sfun, device, args)

    elif args.phase == 2:
        # Scan output_dir for all zone_* directories
        base_dir = Path(args.output_dir)
        zone_dirs = sorted(base_dir.glob("zone_*"))
        if not zone_dirs:
            print(f"No zone results found in {base_dir}")
            return

        print(f"\n  Found {len(zone_dirs)} zone result(s):")
        for zd in zone_dirs:
            print(f"    {zd.name}")

        run_phase2(
            [zd.name for zd in zone_dirs],
            probs_dict, row_names, n_state,
            sfun, device, args)


if __name__ == "__main__":
    main()
