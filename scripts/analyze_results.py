#!/usr/bin/env python3
"""
Analyze and compare evaluation results.

Generates summary metrics and comparisons between baseline and variant results.
"""
import argparse
import json
from pathlib import Path
from typing import Any

from cxr_contrabench.common import load_result_payload
from cxr_contrabench.metrics import (
    build_metric_summary,
    build_variant_comparison,
    exact_match_rate,
    negation_contradiction_count,
)


def print_summary_table(summary: dict[str, Any]) -> None:
    """Print a summary as a formatted table."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        if isinstance(value, float):
            if "rate" in key.lower():
                print(f"  {key:.<45} {value:>8.2%}")
            else:
                print(f"  {key:.<45} {value:>8.4f}")
        else:
            print(f"  {key:.<45} {value:>8}")
    print("=" * 70)


def analyze_single_result(result_path: Path) -> None:
    """Analyze a single result file."""
    print(f"\nAnalyzing: {result_path.name}")
    print("-" * 70)

    payload = load_result_payload(result_path)
    records = payload["records"]
    summary = payload.get("summary", {})

    # Display original summary
    if summary:
        print_summary_table(summary)

    # Compute metrics
    metrics = build_metric_summary(records)

    print("\nMetrics:")
    print(f"  Total samples: {metrics['total']}")
    print(f"  Exact match accuracy: {metrics['exact_match']}/{metrics['total']} ({metrics['exact_match_rate']:.2%})")
    print(f"  Negation contradictions: {metrics['negation_contradiction_count']} ({metrics['negation_contradiction_rate']:.2%})")

    # Show sample errors if any
    errors = [r for r in records if not r.get("exact_match")]
    if errors:
        print(f"\nSample errors (showing first 3):")
        for error in errors[:3]:
            print(f"  Sample {error['sample_id']}:")
            print(f"    Question: {error['question']}")
            print(f"    Prediction: {error.get('pred_letter', 'N/A')}")
            print(f"    Gold: {error.get('gold_letter', 'N/A')}")


def compare_results(baseline_path: Path, variant_paths: list[Path]) -> None:
    """Compare multiple result files against a baseline."""
    print(f"\nComparing against baseline: {baseline_path.name}")
    print("=" * 70)

    baseline_payload = load_result_payload(baseline_path)
    baseline_records = baseline_payload["records"]

    variant_records = {}
    for path in variant_paths:
        variant_payload = load_result_payload(path)
        variant_records[path.stem] = variant_payload["records"]

    # Build comparison
    comparison = build_variant_comparison(baseline_records, variant_records)

    # Print table header
    print("\n{:<15} {:<8} {:<8} {:<12} {:<15} {:<12} {:<10}".format(
        "Variant", "Total", "Correct", "Accuracy", "Delta (pp)", "Pred Changes", "Improved"
    ))
    print("-" * 100)

    # Print baseline row
    baseline_exact = sum(1 for r in baseline_records if r.get("exact_match"))
    baseline_rate = baseline_exact / len(baseline_records) if baseline_records else 0.0
    print("{:<15} {:<8} {:<8} {:<12.2%} {:<15} {:<12} {:<10}".format(
        "BASELINE", len(baseline_records), baseline_exact, baseline_rate, "—", "—", "—"
    ))

    # Print variant rows
    for row in comparison:
        delta_pp = row["delta_exact_match_rate_pp"]
        sign = "+" if delta_pp > 0 else ""
        print("{:<15} {:<8} {:<8} {:<12.2%} {:<15} {:<12} {:<10}".format(
            row["variant"],
            row["total"],
            row["exact_match"],
            row["exact_match_rate"],
            f"{sign}{delta_pp:.2f}pp",
            row["prediction_changes"],
            row["improved"],
        ))

    print("-" * 100)
    print("\nDetailed comparison:")
    for row in comparison:
        print(f"\n{row['variant']}:")
        print(f"  Total: {row['total']}")
        print(f"  Correct: {row['exact_match']}/{row['total']} ({row['exact_match_rate']:.2%})")
        print(f"  Delta vs baseline: {row['delta_exact_match_vs_baseline']:+d} cases ({row['delta_exact_match_rate_pp']:+.2f}pp)")
        print(f"  Predictions changed: {row['prediction_changes']}")
        print(f"  Improved: {row['improved']}")
        print(f"  Worsened: {row['worsened']}")
        print(f"  Negation contradictions: {row['negation_contradiction_count']}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CXR-ContraBench evaluation results"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline results JSON file",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        help="Paths to variant results for comparison",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Single input file to analyze",
    )

    args = parser.parse_args()

    if args.input:
        analyze_single_result(args.input)
    elif args.baseline and args.compare:
        compare_results(args.baseline, args.compare)
    else:
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
