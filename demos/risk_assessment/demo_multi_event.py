#!/usr/bin/env python3
"""
Multi-event risk assessment demo using TSUM.

Demonstrates the key idea: compute reference states ONCE (Stage 1),
then reuse them to estimate system reliability under many different
hazard scenarios (Stage 2), each with its own edge failure probabilities.

This demo:
  1. Builds a small grid graph (4x4 = 16 nodes, 24 edges)
  2. Runs Stage 1 to find reference states for single-OD connectivity
  3. Generates 10 hazard scenarios with varying edge failure probabilities
  4. Runs Stage 2 for all 10 scenarios using evaluate_event_probs
  5. Compares timing: Stage 1 (once) vs Stage 2 (per event)

No external dependencies beyond torch and networkx.
"""

import time
import json
import torch
import networkx as nx
import numpy as np
from pathlib import Path

from tsum import tsum


# ---------------------------------------------------------------------------
# 1. Build a 4x4 grid graph
# ---------------------------------------------------------------------------

def build_sparse_graph(seed=0):
    """
    Build a series-parallel graph with bridge edges and parallel sections.

    Structure (series of parallel pairs):

        s --e0-- n1 --e2-- n3 --e5-- n5 --e8-- n7 --e11-- t
                  |         |         |         |
                 e1        e4        e7        e10
                  |         |         |         |
                 n2 --e3-- n4 --e6-- n6 --e9-- n8 --e12-- t2 --e13-- t

    This creates bridge edges (e.g. s-n1) and parallel sections where
    either of two edges suffices. Results in meaningful reference states:
    - Survival rules: specific path combinations through parallel sections
    - Failure rules: bridge edges failing, or both edges in a parallel section

    ~14 edges, ~12 nodes. Min cut = 1 (at source bridge), giving rich
    reference state structure.
    """
    G = nx.Graph()

    # Build a ladder-like graph with bridge at entrance
    # Layer 0: source bridge
    G.add_edge("s", "n1")           # e0: bridge

    # Layer 1: parallel pair
    G.add_edge("n1", "n2")          # e1
    G.add_edge("n1", "n3")          # e2
    G.add_edge("n2", "n4")          # e3

    # Layer 2: parallel pair
    G.add_edge("n3", "n4")          # e4
    G.add_edge("n3", "n5")          # e5
    G.add_edge("n4", "n6")          # e6

    # Layer 3: parallel pair
    G.add_edge("n5", "n6")          # e7
    G.add_edge("n5", "n7")          # e8
    G.add_edge("n6", "n8")          # e9

    # Layer 4: converge to sink
    G.add_edge("n7", "n8")          # e10
    G.add_edge("n7", "t")           # e11
    G.add_edge("n8", "t")           # e12

    edge_names = [f"e{i}" for i in range(G.number_of_edges())]

    pos = {
        "s":  np.array([0.0, 1.0]),
        "n1": np.array([1.0, 1.0]),
        "n2": np.array([2.0, 0.0]),
        "n3": np.array([2.0, 2.0]),
        "n4": np.array([3.0, 0.0]),
        "n5": np.array([3.0, 2.0]),
        "n6": np.array([4.0, 0.0]),
        "n7": np.array([4.0, 2.0]),
        "n8": np.array([5.0, 0.0]),
        "t":  np.array([5.0, 2.0]),
    }

    return G, edge_names, pos


def make_sfun(G, edge_names, origin, destination):
    """Create a system function: is origin connected to destination?"""

    def sfun(comps_st):
        """
        comps_st: dict {edge_name: 0 or 1}
        Returns: (fval, sys_state, min_comps_st)
        """
        # Build subgraph of functioning edges
        edges = list(G.edges())
        active_edges = []
        for ename, (u, v) in zip(edge_names, edges):
            if comps_st.get(ename, 1) >= 1:
                active_edges.append((u, v))

        subG = nx.Graph()
        subG.add_nodes_from(G.nodes())
        subG.add_edges_from(active_edges)

        connected = nx.has_path(subG, origin, destination)
        sys_st = 1 if connected else 0

        # For survival, find the shortest path as minimal rule
        min_comps_st = None
        if connected:
            path = nx.shortest_path(subG, origin, destination)
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Find edge name
                for ename, (eu, ev) in zip(edge_names, edges):
                    if (eu == u and ev == v) or (eu == v and ev == u):
                        path_edges.add(ename)
                        break
            min_comps_st = {e: ('>=', 1) for e in path_edges}
            min_comps_st['sys'] = ('>=', 1)

        return None, sys_st, min_comps_st

    return sfun


# ---------------------------------------------------------------------------
# 2. Generate hazard scenarios
# ---------------------------------------------------------------------------

def generate_hazard_scenarios(n_edges, n_scenarios=10, seed=42):
    """
    Generate n_scenarios different edge failure probability vectors.

    Simulates varying hazard intensities (e.g., different earthquake
    scenarios) where each scenario assigns different failure probabilities
    to each edge.

    Returns:
        list of (n_edges, 2) tensors: [p_fail, p_survive] per edge
        scenario_descriptions: list of strings describing each scenario
    """
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scenarios = []
    descriptions = []

    for i in range(n_scenarios):
        if i == 0:
            # Scenario 0: baseline (low uniform failure)
            p_fail = np.full(n_edges, 0.05)
            desc = "Baseline: p_fail=0.05"
        elif i == 1:
            # Scenario 1: moderate uniform
            p_fail = np.full(n_edges, 0.20)
            desc = "Moderate: p_fail=0.20"
        elif i == 2:
            # Scenario 2: severe uniform
            p_fail = np.full(n_edges, 0.40)
            desc = "Severe: p_fail=0.40"
        elif i == 3:
            # Scenario 3: extreme uniform
            p_fail = np.full(n_edges, 0.60)
            desc = "Extreme: p_fail=0.60"
        elif i == 4:
            # Scenario 4: very extreme
            p_fail = np.full(n_edges, 0.80)
            desc = "Very extreme: p_fail=0.80"
        elif i <= 7:
            # Scenarios 5-7: localised damage (some edges hit hard)
            p_fail = np.full(n_edges, 0.10)
            n_damaged = max(3, n_edges // 3)
            damaged = rng.choice(n_edges, size=n_damaged, replace=False)
            severity = 0.5 + 0.2 * (i - 5)  # 0.5, 0.7, 0.9
            p_fail[damaged] = severity
            desc = f"Localised: {n_damaged} edges at {severity:.1f}, rest 0.10"
        else:
            # Scenarios 8-9: random heterogeneous
            p_fail = rng.beta(a=1.5 + i * 0.5, b=3.0, size=n_edges)
            p_fail = np.clip(p_fail, 0.01, 0.99)
            desc = f"Random: mean={p_fail.mean():.2f}, std={p_fail.std():.2f}"

        probs = np.column_stack([p_fail, 1 - p_fail])  # (n_edges, 2)
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=device)
        scenarios.append(probs_tensor)
        descriptions.append(desc)

    return scenarios, descriptions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Build graph ---
    G, edge_names, pos = build_sparse_graph(seed=0)

    # OD pair: source to sink
    origin = "s"
    destination = "t"
    n_edges = len(edge_names)

    print(f"Graph: {G.number_of_nodes()} nodes, {n_edges} edges")
    print(f"OD pair: {origin} -> {destination}\n")

    sfun = make_sfun(G, edge_names, origin, destination)
    sys_surv_st = 1
    n_state = 2

    # --- Stage 1: Find reference states (run ONCE) ---
    print("=" * 60)
    print("STAGE 1: Finding reference states (one-time cost)")
    print("=" * 60)

    # Use high failure probs for rule discovery to explore the boundary well.
    # This spreads samples across both survival and failure regions.
    baseline_probs = torch.tensor(
        [[0.40, 0.60]] * n_edges, dtype=torch.float32, device=device
    )

    output_dir = Path(__file__).parent / "tsum_res"

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=baseline_probs,
        row_names=edge_names,
        n_state=n_state,
        sys_surv_st=sys_surv_st,
        unk_prob_thres=1e-3,
        max_rounds=500,
        n_sample=500_000,
        sample_batch_size=100_000,
        output_dir=str(output_dir),
    )
    stage1_time = time.time() - t0

    # Load the rules
    rules_surv = torch.load(
        result["rules_surv_pt_path"], map_location=device, weights_only=True
    )
    rules_fail = torch.load(
        result["rules_fail_pt_path"], map_location=device, weights_only=True
    )

    n_surv = rules_surv.shape[0] if rules_surv.ndim == 3 else 0
    n_fail = rules_fail.shape[0] if rules_fail.ndim == 3 else 0

    print(f"\nStage 1 complete:")
    print(f"  Time: {stage1_time:.1f}s")
    print(f"  Survival rules: {n_surv}")
    print(f"  Failure rules:  {n_fail}")

    # --- Stage 2: Evaluate 10 hazard scenarios ---
    print(f"\n{'=' * 60}")
    print("STAGE 2: Evaluating 10 hazard scenarios (reusing rules)")
    print("=" * 60)

    scenarios, descriptions = generate_hazard_scenarios(n_edges, n_scenarios=10)

    t0 = time.time()
    results = tsum.evaluate_event_probs(
        rules_mat_surv=rules_surv,
        rules_mat_fail=rules_fail,
        event_probs=scenarios,
        row_names=edge_names,
        s_fun=sfun,
        sys_surv_st=sys_surv_st,
        n_sample=500_000,
        sample_batch_size=100_000,
    )
    stage2_time = time.time() - t0

    # --- Print results ---
    print(f"\nStage 2 complete: {stage2_time:.1f}s for {len(scenarios)} scenarios")
    print(f"  Average per scenario: {stage2_time / len(scenarios):.2f}s\n")

    print(f"{'Scenario':<4} {'Description':<50} {'P(surv)':>10} {'P(fail)':>10} {'P(unk)':>10}")
    print("-" * 88)
    for i, (desc, r) in enumerate(zip(descriptions, results)):
        print(f"{i:<4} {desc:<50} {r['survival']:>10.4f} {r['failure']:>10.4f} {r['unknown']:>10.4f}")

    # --- Timing comparison ---
    print(f"\n{'=' * 60}")
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"  Stage 1 (rule discovery, one-time):  {stage1_time:>8.1f}s")
    print(f"  Stage 2 (10 scenarios, total):       {stage2_time:>8.1f}s")
    print(f"  Stage 2 (per scenario):              {stage2_time/len(scenarios):>8.2f}s")
    print(f"  Speedup vs running Stage 1 per event: {stage1_time * len(scenarios) / (stage1_time + stage2_time):.1f}x")

    # --- Save results ---
    output = {
        "stage1_time_s": stage1_time,
        "stage2_time_s": stage2_time,
        "n_surv_rules": n_surv,
        "n_fail_rules": n_fail,
        "scenarios": [
            {"description": d, **r}
            for d, r in zip(descriptions, results)
        ],
    }
    out_path = output_dir / "risk_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
