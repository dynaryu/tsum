"""
Run TSUM on IEEE 14-bus DC-OPF blackout model (branches only).

Components: 20 branches, 2-state (0=failed, 1=operational)
System function: DC-OPF, blackout_threshold=54.8% (Scenario 1 from paper)
Reference: Chan et al. (2024), Table 2: p_f ~ 1.1e-4

Usage:
    python run_case14.py
"""

import sys
import os
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import json
import torch

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from dcopt import make_dcopt_sfun
from tsum import tsum


def main():
    print("=" * 60)
    print("TSUM on IEEE 14-bus DC-OPF (branches only)")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load input data
    # ---------------------------------------------------------------
    with open(HERE / "edges.json") as f:
        edges = json.load(f)
    with open(HERE / "probs.json") as f:
        probs = json.load(f)

    row_names = list(edges.keys())
    n_state = 2
    n_edges = len(row_names)

    print(f"\n  Components:  {n_edges} branches")
    print(f"  States:      {n_state} per component (0=failed, 1=operational)")
    print(f"  Threshold:   54.8% blackout (Scenario 1)")

    # ---------------------------------------------------------------
    # 2. Build probability tensor
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs_tensor = torch.tensor(
        [[probs[n][str(s)]["p"] for s in range(n_state)] for n in row_names],
        dtype=torch.float32, device=device,
    )
    print(f"  Device:      {device}")

    # ---------------------------------------------------------------
    # 3. Build system function
    # ---------------------------------------------------------------
    print("\nInitialising DC-OPF system function...")
    sfun = make_dcopt_sfun(
        case_path='case14',
        blackout_threshold=54.8,
        alpha=2.0,
    )

    # Quick sanity check
    all_ok = {name: 1 for name in row_names}
    fval, sys_st, _ = sfun(all_ok)
    print(f"  All operational: blackout={fval:.4f}%, sys_st={sys_st}")

    # ---------------------------------------------------------------
    # 4. Run TSUM rule extraction
    # ---------------------------------------------------------------
    output_dir = HERE / "tsum_results"
    print(f"\n  Output:      {output_dir}")
    print(f"  Samples:     1,000,000 per round (batch 100,000)")
    print(f"  Convergence: unk_prob < 1e-5")
    print(f"\nStarting rule extraction...\n", flush=True)

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        unk_prob_thres=1e-5,
        unk_prob_opt='abs',
        n_sample=1_000_000,
        sample_batch_size=100_000,
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
        last = rounds[-1]
        print(f"\n--- Summary ---")
        print(f"  Rounds:      {len(rounds)}")
        print(f"  Surv rules:  {last.get('n_rules_surv', '?')}")
        print(f"  Fail rules:  {last.get('n_rules_fail', '?')}")
        print(f"  Unk prob:    {last.get('p_unknown', '?')}")
        print(f"  P(survival): {last.get('p_survival', '?')}")
        print(f"  P(failure):  {last.get('p_failure', '?')}")
        p_fail = last.get('p_failure', 0)
        print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.1e-4")
        print(f"  TSUM estimate:                   p_f ~ {p_fail:.2e}")


if __name__ == "__main__":
    main()
