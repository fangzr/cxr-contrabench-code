#!/usr/bin/env python3
"""
Apply QCCV-Neg verification to evaluation results.

This script applies the deterministic polarity repair mechanism to correct
negation-attracted predictions without retraining.
"""
import argparse
from pathlib import Path

from cxr_contrabench.common import load_result_payload, save_result_payload
from cxr_contrabench.metrics import (
    negation_contradiction_count,
    exact_match_rate,
)
from cxr_contrabench.verifier import apply_m1_to_records


def apply_verification(
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Apply QCCV-Neg M1 repair to evaluation results.

    Args:
        input_path: Path to evaluation results JSON
        output_path: Path to save repaired results
    """
    # Load results
    print(f"Loading results from: {input_path}")
    payload = load_result_payload(input_path)
    records = payload["records"]
    original_summary = payload.get("summary", {})

    print(f"  Total records: {len(records)}")

    # Count pre-repair negation contradictions
    pre_count = negation_contradiction_count(records)
    print(f"  Negation contradictions (pre-repair): {pre_count}")

    # Apply M1 repair
    print("\nApplying M1 repair (presence-side negation trap correction)...")
    repaired_records = apply_m1_to_records(records)

    # Count post-repair negation contradictions
    post_count = negation_contradiction_count(repaired_records)
    print(f"  Negation contradictions (post-repair): {post_count}")

    # Calculate accuracy change
    pre_accuracy = exact_match_rate(records)
    post_accuracy = exact_match_rate(repaired_records)
    pre_correct = sum(1 for r in records if r.get("exact_match"))
    post_correct = sum(1 for r in repaired_records if r.get("exact_match"))

    print(f"\nAccuracy:")
    print(f"  Pre-repair: {pre_correct}/{len(records)} ({pre_accuracy:.2%})")
    print(f"  Post-repair: {post_correct}/{len(records)} ({post_accuracy:.2%})")
    print(f"  Delta: {post_correct - pre_correct} cases (+{100*(post_accuracy-pre_accuracy):.2f}pp)")

    # Count repairs applied
    repairs_applied = sum(1 for r in repaired_records if r.get("verifier_applied"))
    print(f"\nRepairs applied: {repairs_applied}")

    # Build new summary
    new_summary = dict(original_summary)
    new_summary.update({
        "model": original_summary.get("model", "unknown"),
        "protocol": original_summary.get("protocol", "unknown"),
        "verifier": "M1",
        "verifier_applied_count": repairs_applied,
        "total": len(repaired_records),
        "exact_match_pre_repair": pre_correct,
        "exact_match_post_repair": post_correct,
        "exact_match_rate_pre_repair": round(pre_accuracy, 4),
        "exact_match_rate_post_repair": round(post_accuracy, 4),
        "exact_match_rate_post_repair_percent": round(100.0 * post_accuracy, 2),
        "negation_contradiction_count_pre": pre_count,
        "negation_contradiction_count_post": post_count,
        "negation_contradictions_repaired": max(0, pre_count - post_count),
    })

    # Save repaired results
    print(f"\nSaving repaired results to: {output_path}")
    save_result_payload(output_path, new_summary, repaired_records)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Apply QCCV-Neg M1 repair to evaluation results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input results JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save repaired results JSON file",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    try:
        apply_verification(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
