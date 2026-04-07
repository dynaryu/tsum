"""
Run TSUM with Subset Simulation (Au & Beck 2001) on IEEE 118-bus DC-OPF.

Components: 304 total (118 buses + 186 branches)
  - 54 generator buses: 4-state (0=removed, 1=40%cap, 2=80%cap, 3=full)
  - 64 ordinary buses: 2-state (0=failed, 1=operational)
  - 186 branches: 2-state (0=failed, 1=operational)
System function: DC-OPF, blackout_threshold=13.8% (Scenario 1 from paper)
Reference: Chan et al. (2024), Table 2: p_f ~ 1.0e-4

Search-phase sampler is replaced by Subset Simulation with sticky-prior
component-wise Metropolis-Hastings. The unbiased probability estimate is
refreshed periodically via prior MC (controlled by --prob-update-every).

Usage:
    python run_subset.py
    python run_subset.py --sus-n-per-level 2000 --sus-p0 0.1
    python run_subset.py --n-workers 96 --devices cuda:0,cuda:1
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
    p = argparse.ArgumentParser(description="TSUM + Subset Simulation on IEEE 118-bus DC-OPF")
    p.add_argument("--unk-prob-thres", type=float, default=1e-5,
                   help="Convergence threshold for unknown probability (default: 1e-5)")
    p.add_argument("--devices", type=str, default="",
                   help="Comma-separated GPU devices, e.g. 'cuda:0,cuda:1'")
    p.add_argument("--n-workers", type=int, default=1,
                   help="CPU workers for parallel sfun + minimization (default: 1)")
    p.add_argument("--prob-update-every", type=int, default=5,
                   help="Refresh unbiased unk_prob every N rounds (default: 5)")
    # Subset Simulation hyperparameters
    p.add_argument("--sus-p0", type=float, default=0.1,
                   help="Intermediate conditional probability p0 (default: 0.1)")
    p.add_argument("--sus-n-per-level", type=int, default=1000,
                   help="Samples per SuS level N (default: 1000)")
    p.add_argument("--sus-max-levels", type=int, default=10,
                   help="Cap on number of SuS levels (default: 10)")
    p.add_argument("--sus-n-flip-mean", type=float, default=5.0,
                   help="Average components perturbed per CWM-H step (default: 5.0)")
    p.add_argument("--output-dir", type=str, default="results_subset",
                   help="Output directory under demos/case118/ (default: results_subset)")
    return p.parse_args()


def main():
    args = parse_args()
    device_list = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else []
    multi_devices = device_list if len(device_list) > 1 else None

    print("=" * 60)
    print("TSUM + Subset Simulation on IEEE 118-bus DC-OPF")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load input data
    # ---------------------------------------------------------------
    data_dir = HERE / "case118_tsum_bus"
    with open(data_dir / "edges.json") as f:
        edges = json.load(f)
    with open(data_dir / "probs.json") as f:
        probs_dict = json.load(f)

    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())  # 4

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

    # ---------------------------------------------------------------
    # 2. Build probability tensor (padded to n_state=4)
    # ---------------------------------------------------------------
    device = torch.device(device_list[0] if device_list else ("cuda" if torch.cuda.is_available() else "cpu"))
    probs_list = []
    for name in row_names:
        p = probs_dict[name]
        row = [p[str(s)]["p"] if str(s) in p else 0.0
               for s in range(n_state)]
        probs_list.append(row)

    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)
    print(f"  Device:      {device}")

    # ---------------------------------------------------------------
    # 3. Build system function
    # ---------------------------------------------------------------
    print("\nInitialising DC-OPF system function...")
    case_path = str(HERE / "case118.m")
    sfun = make_dcopt_sfun(
        case_path=case_path,
        blackout_threshold=13.8,
        alpha=2.0,
    )

    # Sanity checks
    all_ok = {name: max(int(s) for s in probs_dict[name].keys()) for name in row_names}
    fval, sys_st, _ = sfun(all_ok)
    print(f"  All operational: blackout={fval:.4f}%, sys_st={sys_st}")
    all_fail = {name: 0 for name in row_names}
    fval, sys_st, _ = sfun(all_fail)
    print(f"  All failed:      blackout={fval:.4f}%, sys_st={sys_st}")

    # ---------------------------------------------------------------
    # 4. Run TSUM with Subset Simulation
    # ---------------------------------------------------------------
    output_dir = HERE / args.output_dir
    print(f"\n  Output:           {output_dir}")
    print(f"  Convergence:      unk_prob < {args.unk_prob_thres:.0e}")
    print(f"  CPU workers:      {args.n_workers}")
    if multi_devices:
        print(f"  Devices:          {multi_devices}")
    print(f"\n  --- Subset Simulation ---")
    print(f"  p0:               {args.sus_p0}")
    print(f"  N per level:      {args.sus_n_per_level}")
    print(f"  max levels:       {args.sus_max_levels}")
    print(f"  n_flip_mean:      {args.sus_n_flip_mean}")
    print(f"  prob_update_every: {args.prob_update_every} rounds")
    print(f"\n  severity_sign:    +1 (higher blackout %% = more failed)")
    print(f"\nStarting rule extraction...\n", flush=True)

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        unk_prob_thres=args.unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=1_000_000,
        sample_batch_size=100_000,
        n_workers=args.n_workers,
        devices=multi_devices,
        prob_update_every=args.prob_update_every,
        # Subset Simulation
        use_subset_sim=True,
        sus_p0=args.sus_p0,
        sus_n_per_level=args.sus_n_per_level,
        sus_max_levels=args.sus_max_levels,
        sus_n_flip_mean=args.sus_n_flip_mean,
        sus_severity_sign=+1,
        output_dir=output_dir,
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")

    # ---------------------------------------------------------------
    # 5. Summary
    # ---------------------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            rounds = [json.loads(line) for line in f if line.strip()]
        if rounds:
            last = rounds[-1]
            total_sus_calls = sum(r.get("sus_n_sfun_calls", 0) for r in rounds)
            print(f"\n--- Summary ---")
            print(f"  Rounds:           {len(rounds)}")
            print(f"  Surv rules:       {last.get('n_rules_surv', '?')}")
            print(f"  Fail rules:       {last.get('n_rules_fail', '?')}")
            print(f"  Unk prob:         {last.get('p_unknown', '?')}")
            print(f"  P(survival):      {last.get('p_survival', '?')}")
            print(f"  P(failure):       {last.get('p_failure', '?')}")
            print(f"  Total SuS sfun:   {total_sus_calls:,}")
            p_fail = last.get('p_failure', 0)
            print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")
            print(f"  TSUM estimate:                   p_f ~ {p_fail:.2e}")


if __name__ == "__main__":
    main()
