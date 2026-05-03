"""Merge failure rules from multiple TSUM runs into a single deduplicated JSON.

A failure rule is a sufficient condition for system failure, so the union of
two valid rule sets is itself valid. This merger collapses identical rules,
optionally drops Pareto-dominated rules (those whose precondition set is a
strict superset of another's), and writes a single seed file usable as
`--seed-rules` in run_hierarchical_sus.py.

Usage:
    python merge_fail_rules.py \\
        path/runA/zone_X/rules_leq_0.json \\
        path/runB/zone_X/rules_leq_0.json \\
        ... \\
        -o merged_seed_fail.json

    # Add Pareto trim (slower, O(n^2)):
    python merge_fail_rules.py ... -o out.json --pareto-trim
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def rule_to_key(rule: Dict) -> Tuple[Tuple[str, str, int], ...]:
    """Sorted tuple of (component, op, value) literals — the rule's antecedent
    in canonical form. Used as dict key for exact-match dedup."""
    items = []
    for k, v in rule.items():
        if k == "sys":
            continue
        items.append((k, v[0], int(v[1])))
    return tuple(sorted(items))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="Input rules_leq_*.json files")
    p.add_argument("-o", "--output", required=True,
                   help="Merged output JSON path")
    p.add_argument("--pareto-trim", action="store_true",
                   help="After exact dedup, drop rules whose precondition set "
                        "is a strict superset of another rule's. Slower (O(n^2)) "
                        "but produces a smaller, sharper rule set.")
    return p.parse_args()


def main():
    args = parse_args()

    all_rules = []
    per_file_counts = {}
    for path in args.inputs:
        with open(path) as f:
            rs = json.load(f)
        per_file_counts[path] = len(rs)
        all_rules.extend(rs)

    print(f"Loaded {sum(per_file_counts.values())} rules total "
          f"from {len(args.inputs)} file(s):")
    for path, n in per_file_counts.items():
        print(f"  {n:>9}  {path}")

    # 1. Exact dedup by canonical literal tuple.
    by_key: Dict[Tuple, Dict] = {}
    for r in all_rules:
        k = rule_to_key(r)
        if k not in by_key:
            by_key[k] = r
    n_dups = len(all_rules) - len(by_key)
    print(f"\nAfter exact dedup: {len(by_key)} unique rules "
          f"({n_dups} duplicates removed)")

    # 2. Optional Pareto trim: drop rules whose literal-set is a strict
    #    superset of any other rule's. Smaller (more general) rules dominate.
    if args.pareto_trim:
        keys = sorted(by_key.keys(), key=len)        # ascending by size
        sets = [set(k) for k in keys]
        keep = [True] * len(keys)
        for i in range(len(keys)):
            if not keep[i]:
                continue
            si = sets[i]
            # Only earlier (smaller-or-equal length) rules can dominate i.
            for j in range(i):
                if keep[j] and sets[j] < si:           # strict subset → j dominates i
                    keep[i] = False
                    break
        out_rules = [by_key[keys[i]] for i in range(len(keys)) if keep[i]]
        print(f"After Pareto trim: {len(out_rules)} rules "
              f"({len(keys) - len(out_rules)} dominated rules dropped)")
    else:
        out_rules = list(by_key.values())

    # 3. Length distribution summary.
    lens = [len(rule_to_key(r)) for r in out_rules]
    if lens:
        import statistics
        print(f"\nRule-length stats: "
              f"min={min(lens)}, mean={statistics.mean(lens):.2f}, "
              f"median={statistics.median(lens)}, max={max(lens)}")

    # 4. Write.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_rules, f)
    print(f"\nWrote {len(out_rules)} rules to {out_path}")


if __name__ == "__main__":
    main()
