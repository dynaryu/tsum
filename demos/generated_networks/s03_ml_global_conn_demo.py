"""
Demo: ML-guided rule extraction for global connectivity.

Generates a random geometric graph (200 nodes, radius=0.14) that has:
  - A bridge edge (edge connectivity = 1) → clear bottleneck structure
  - ~1100 edges → survival rules (spanning trees, ~199 edges) are only ~18%
    of total edges, well under the 25% threshold for ML activation
  - Topology analysis predicts sparse rules → ML recommended

Runs global connectivity (k=1) with ML auto-enabled, showing:
  1. Topology analysis output
  2. ML activation/deactivation decisions per round
  3. Comparison of rule extraction speed vs random sampling

Usage:
    python s03_ml_global_conn_demo.py [--use-ml auto|true|false] [--use-igraph]

Requires:
    pip install tsum[ml]       # for scikit-learn (ML guidance)
    pip install python-igraph  # optional, 10-50x faster sfun
"""

import sys
from pathlib import Path
import json
import time

import networkx as nx
import torch

HOME = Path(__file__).parent
sys.path.append(str(HOME.joinpath("../../../network-datasets/")))
ROOT = HOME.resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ndtools.network_generator import GenConfig, generate_and_save
from ndtools.io import load_json
from ndtools.graphs import build_graph
from ndtools.fun_binary_graph import eval_global_conn_k

from tsum import tsum
from tsum.ml_guide import analyze_topology, _HAS_SKLEARN

try:
    from tsum.igraph_sfun import make_igraph_sfun_global_conn
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False


def main(use_ml: str = "auto", use_igraph: bool = False):
    out_base = HOME / "data"

    # ---------------------------------------------------------------
    # 1. Generate graph
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Generate random geometric graph")
    print("=" * 60)

    cfg = GenConfig(
        name="rg_ml_demo",
        generator="rg",
        description="n_nodes=200, radius=0.14, p_fail=0.05 (ML demo)",
        generator_params={"n_nodes": 200, "radius": 0.14, "p_fail": 0.05},
        seed=999,
    )
    ds_root = generate_and_save(out_base, cfg, draw_graph=True)

    nodes = load_json(ds_root / "data" / "nodes.json")
    edges = load_json(ds_root / "data" / "edges.json")
    probs = load_json(ds_root / "data" / "probs.json")
    G = build_graph(nodes, edges, probs)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"\nGraph: {n_nodes} nodes, {n_edges} edges")
    print(f"Connected: {nx.is_connected(G)}")

    # ---------------------------------------------------------------
    # 2. Topology analysis
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Topology analysis")
    print("=" * 60)

    topo = analyze_topology(G, n_edges)
    print(f"  Edge connectivity:      {topo['edge_connectivity']}")
    print(f"  Min-cut ratio:          {topo['min_cut_ratio']:.4f}")
    print(f"  Predicted rule density: {topo['predicted_rule_density']:.4f}")
    print(f"  ML recommended:         {topo['ml_recommended']}")
    print(f"  Reason: {topo['reason']}")

    if not _HAS_SKLEARN:
        print("\nWARNING: scikit-learn not installed. ML guidance disabled.")
        print("Install with: pip install tsum[ml]")

    # Spanning tree ratio
    span_ratio = (n_nodes - 1) / n_edges
    print(f"\n  Spanning tree / edges:  {n_nodes - 1}/{n_edges} = {span_ratio:.3f}")
    print(f"  (ML threshold: < 0.25)")

    # ---------------------------------------------------------------
    # 3. Build system function
    # ---------------------------------------------------------------
    target_g_conn = 1

    if use_igraph and HAS_IGRAPH:
        print("\nUsing igraph-accelerated system function")
        sys_func = make_igraph_sfun_global_conn(G, target_g_conn)
    elif use_igraph and not HAS_IGRAPH:
        print("\nWARNING: igraph not installed, falling back to NetworkX")
        def sys_func(comps_st):
            _, k, _ = eval_global_conn_k(comps_st, G)
            return k, (1 if k >= target_g_conn else 0), None
    else:
        def sys_func(comps_st):
            _, k, _ = eval_global_conn_k(comps_st, G)
            return k, (1 if k >= target_g_conn else 0), None

    # ---------------------------------------------------------------
    # 4. Run rule extraction
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Step 3: Run rule extraction (use_ml={use_ml})")
    print("=" * 60)

    _use_ml_map = {"auto": None, "true": True, "false": False}
    use_ml_bool = _use_ml_map.get(use_ml.lower(), None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row_names = list(edges.keys())
    n_state = 2
    probs_tensor = torch.tensor(
        [[probs[n]["0"]["p"], probs[n]["1"]["p"]] for n in row_names],
        dtype=torch.float32, device=device,
    )

    output_dir = ds_root / f"tsum_global_ml_{use_ml}"

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sys_func,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        unk_prob_thres=1e-3,
        n_sample=1_000_000,
        sample_batch_size=100_000,
        output_dir=output_dir,
        use_ml=use_ml_bool,
        graph=G,
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML-guided global connectivity demo")
    parser.add_argument(
        "--use-ml", default="auto", choices=["auto", "true", "false"],
        help="ML guidance: auto (default), true (force on), false (force off)",
    )
    parser.add_argument(
        "--use-igraph", action="store_true",
        help="Use igraph for faster connectivity evaluation",
    )
    args = parser.parse_args()
    main(use_ml=args.use_ml, use_igraph=args.use_igraph)
