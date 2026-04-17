#!/usr/bin/env python3
"""
Compare two benchmark JSON files and summarize tflops improvements/regressions.
Only compares entries where both have "success": true.
"""

import json
import sys
from typing import Dict, List, Tuple


def load_json(filepath: str) -> List[Dict]:
    """Load and parse a JSON benchmark file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_benchmark_key(config: Dict) -> Tuple:
    """Create a unique key for a benchmark configuration."""
    return (config["M"], config["N"], config["K"], config["operation"])


def compare_benchmarks(baseline_file: str, new_file: str) -> None:
    """Compare two benchmark files and print tflops improvements/regressions."""

    baseline_data = load_json(baseline_file)
    new_data = load_json(new_file)

    # Create lookup dictionaries
    baseline_dict = {get_benchmark_key(config): config for config in baseline_data}
    new_dict = {get_benchmark_key(config): config for config in new_data}

    # Find common benchmark configurations
    common_keys = set(baseline_dict.keys()) & set(new_dict.keys())

    if not common_keys:
        print("No common benchmark configurations found between the two files.")
        return

    print(f"\n{'=' * 100}")
    print(f"BENCHMARK COMPARISON: {baseline_file} vs {new_file}")
    print(f"{'=' * 100}\n")

    # Track statistics
    improvements = []
    regressions = []
    no_change = []

    # Sort keys for consistent output
    sorted_keys = sorted(common_keys)

    for key in sorted_keys:
        baseline_config = baseline_dict[key]
        new_config = new_dict[key]

        m, n, k, operation = key

        print(f"\n{'─' * 100}")
        print(f"Configuration: M={m}, N={n}, K={k}, operation={operation}")
        print(f"{'─' * 100}")

        # Compare each benchmark variant
        baseline_benchmarks = baseline_config.get("benchmarks", {})
        new_benchmarks = new_config.get("benchmarks", {})

        common_variants = set(baseline_benchmarks.keys()) & set(new_benchmarks.keys())

        if not common_variants:
            print("  No common benchmark variants found.")
            continue

        variant_results = []

        for variant in sorted(common_variants):
            baseline_variant = baseline_benchmarks[variant]
            new_variant = new_benchmarks[variant]

            # Only compare if both have success=true
            baseline_success = baseline_variant.get("success", False)
            new_success = new_variant.get("success", False)

            if not (baseline_success and new_success):
                status_msg = []
                if not baseline_success:
                    status_msg.append("baseline failed")
                if not new_success:
                    status_msg.append("new failed")
                print(f"  {variant:25s} - SKIPPED ({', '.join(status_msg)})")
                continue

            baseline_tflops = baseline_variant.get("tflops", 0)
            new_tflops = new_variant.get("tflops", 0)

            if baseline_tflops == 0:
                print(f"  {variant:25s} - SKIPPED (baseline tflops is 0)")
                continue

            # Calculate change
            delta_tflops = new_tflops - baseline_tflops
            percent_change = (delta_tflops / baseline_tflops) * 100

            # Determine status
            if abs(percent_change) < 0.01:  # Less than 0.01% change
                status = "→"
                color_code = ""
                no_change.append((key, variant, baseline_tflops, new_tflops, percent_change))
            elif percent_change > 0:
                status = "↑"
                color_code = "+"
                improvements.append((key, variant, baseline_tflops, new_tflops, percent_change))
            else:
                status = "↓"
                color_code = "-"
                regressions.append((key, variant, baseline_tflops, new_tflops, percent_change))

            variant_results.append(
                {
                    "variant": variant,
                    "status": status,
                    "baseline_tflops": baseline_tflops,
                    "new_tflops": new_tflops,
                    "delta_tflops": delta_tflops,
                    "percent_change": percent_change,
                }
            )

            print(
                f"  {variant:25s} {status} {baseline_tflops:10.2f} → {new_tflops:10.2f} TFLOPs "
                f"({color_code}{percent_change:+7.2f}%)"
            )

    # Print summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}\n")

    total_comparisons = len(improvements) + len(regressions) + len(no_change)

    print(f"Total comparisons:   {total_comparisons}")
    print(
        f"Improvements:        {len(improvements)} ({len(improvements) / total_comparisons * 100:.1f}%)"
        if total_comparisons > 0
        else "Improvements:        0"
    )
    print(
        f"Regressions:         {len(regressions)} ({len(regressions) / total_comparisons * 100:.1f}%)"
        if total_comparisons > 0
        else "Regressions:         0"
    )
    print(
        f"No change:           {len(no_change)} ({len(no_change) / total_comparisons * 100:.1f}%)"
        if total_comparisons > 0
        else "No change:           0"
    )

    if improvements:
        print(f"\n{'─' * 100}")
        print("TOP IMPROVEMENTS:")
        print(f"{'─' * 100}")
        # Sort by absolute improvement
        improvements.sort(key=lambda x: x[4], reverse=True)
        for i, (key, variant, baseline, new, pct) in enumerate(improvements[:10], 1):
            m, n, k, op = key
            print(
                f"{i:2d}. {variant:25s} M={m:6d} N={n:6d} K={k:6d}: {baseline:8.2f} → {new:8.2f} TFLOPs (+{pct:.2f}%)"
            )

    if regressions:
        print(f"\n{'─' * 100}")
        print("TOP REGRESSIONS:")
        print(f"{'─' * 100}")
        # Sort by absolute regression
        regressions.sort(key=lambda x: x[4])
        for i, (key, variant, baseline, new, pct) in enumerate(regressions[:10], 1):
            m, n, k, op = key
            print(f"{i:2d}. {variant:25s} M={m:6d} N={n:6d} K={k:6d}: {baseline:8.2f} → {new:8.2f} TFLOPs ({pct:.2f}%)")

    print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_benchmarks.py <baseline.json> <new.json>")
        print()
        print("Example:")
        print("  python compare_benchmarks.py baseline_results.json new_results.json")
        sys.exit(1)

    baseline_file = sys.argv[1]
    new_file = sys.argv[2]

    try:
        compare_benchmarks(baseline_file, new_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
