"""
Run TSUM on IEEE 300-bus DC-OPF with classifier-guided boundary search.

Uses a monotone classifier to bias sampling toward the decision boundary,
accelerating discovery of unknown states in high-dimensional problems.

Components: 711 total (300 buses + 411 branches)
System function: DC-OPF, blackout_threshold=26.1%
Reference: Chan et al. (2024), Table 2: p_f ~ 1.0e-4

Usage:
    python run_classifier.py
    python run_classifier.py --unk-prob-thres 1e-4 --n-workers 48
    python run_classifier.py --devices cuda:0,cuda:1
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
sys.path.insert(0, str(HERE.parent))

from dcopt import make_dcopt_sfun
from tsum import tsum


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classifier-guided TSUM on IEEE 300-bus DC-OPF")
    parser.add_argument("--unk-prob-thres", type=float, default=1e-5,
                        help="Convergence threshold for unknown probability (default: 1e-5)")
    parser.add_argument("--n-sample", type=int, default=1_000_000,
                        help="Samples per round for probability estimation (default: 1000000)")
    parser.add_argument("--sample-batch-size", type=int, default=100_000,
                        help="Batch size for GPU sampling (default: 100000)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers for sfun evaluation (default: 1)")
    parser.add_argument("--devices", type=str, default="",
                        help="Comma-separated GPU devices, e.g. 'cuda:0,cuda:1'")
    # Classifier options
    parser.add_argument("--classifier-n-pretrain", type=int, default=5000,
                        help="Initial sfun evaluations for classifier pre-training (default: 5000)")
    parser.add_argument("--classifier-retrain-every", type=int, default=10,
                        help="Retrain classifier every N rounds (default: 10)")
    parser.add_argument("--classifier-shift-factor", type=float, default=3.0,
                        help="IS shift aggressiveness for boundary sampling (default: 3.0)")
    parser.add_argument("--classifier-mix-original", type=float, default=0.3,
                        help="Fraction of original distribution to mix in (default: 0.3)")
    parser.add_argument("--output-dir", type=str, default="results_classifier",
                        help="Output directory (default: results_classifier)")
    return parser.parse_args()


def main():
    args = parse_args()
    device_list = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else []
    multi_devices = device_list if len(device_list) > 1 else None

    print("=" * 60)
    print("Classifier-guided TSUM rule extraction")
    print("IEEE 300-bus DC-OPF")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load input data
    # ---------------------------------------------------------------
    data_dir = HERE / "case300_tsum_bus"
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

    # ---------------------------------------------------------------
    # 2. Build probability tensor (padded to n_state)
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
    sfun = make_dcopt_sfun(
        case_path=str(HERE / "case300.m"),
        blackout_threshold=26.1,
        alpha=2.0,
    )

    # Sanity checks
    all_ok = {}
    for name in row_names:
        all_ok[name] = max(int(s) for s in probs_dict[name].keys())
    fval, sys_st, _ = sfun(all_ok)
    print(f"  All operational: blackout={fval:.4f}%, sys_st={sys_st}")

    all_fail = {name: 0 for name in row_names}
    fval, sys_st, _ = sfun(all_fail)
    print(f"  All failed:      blackout={fval:.4f}%, sys_st={sys_st}")

    # ---------------------------------------------------------------
    # 4. Run classifier-guided rule extraction
    # ---------------------------------------------------------------
    output_dir = HERE / args.output_dir
    print(f"\n  Output:      {output_dir}")
    print(f"  Samples:     {args.n_sample:,} per round (batch {args.sample_batch_size:,})")
    print(f"  Convergence: unk_prob < {args.unk_prob_thres:.0e}")
    print(f"  Classifier:  pretrain={args.classifier_n_pretrain}, "
          f"retrain_every={args.classifier_retrain_every}, "
          f"shift={args.classifier_shift_factor}, "
          f"mix={args.classifier_mix_original}")
    if multi_devices:
        print(f"  Devices:     {multi_devices}")
    print(f"\nStarting classifier-guided rule extraction...\n", flush=True)

    t0 = time.time()
    result = tsum.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs_tensor,
        row_names=row_names,
        n_state=n_state,
        sys_surv_st=1,
        unk_prob_thres=args.unk_prob_thres,
        unk_prob_opt='abs',
        n_sample=args.n_sample,
        sample_batch_size=args.sample_batch_size,
        n_workers=args.n_workers,
        devices=multi_devices,
        classifier_guided=True,
        classifier_n_pretrain=args.classifier_n_pretrain,
        classifier_retrain_every=args.classifier_retrain_every,
        classifier_shift_factor=args.classifier_shift_factor,
        classifier_mix_original=args.classifier_mix_original,
        output_dir=str(output_dir),
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
        print(f"\n  Reference (Chan et al. Table 2): p_f ~ 1.0e-4")
        print(f"  TSUM estimate:                   p_f ~ {p_fail:.2e}")


if __name__ == "__main__":
    main()
