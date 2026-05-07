#!/usr/bin/env python3
"""
Minimal example of using CXR-ContraBench.

This example shows:
1. Loading a protocol
2. Running inference on a model
3. Applying QCCV-Neg repair
4. Computing metrics
"""
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cxr_contrabench.common import option_map, resolve_gold_letter
from cxr_contrabench.datasets import load_protocol
from cxr_contrabench.metrics import exact_match_rate, negation_contradiction_count
from cxr_contrabench.verifier import apply_m1_to_records


def mock_model_inference(question: str, options: list[str]) -> str:
    """
    Mock model inference function.
    In real usage, this would call your actual VLM.
    """
    # For demo: randomly return an option letter
    import random
    return chr(65 + random.randint(0, len(options) - 1))


def example_1_load_and_evaluate():
    """Example 1: Load protocol and run evaluation."""
    print("\n" + "=" * 70)
    print("Example 1: Load Protocol and Run Evaluation")
    print("=" * 70)

    # Load example protocol
    protocol_file = Path(__file__).parent / "example_protocol.json"
    print(f"\nLoading protocol: {protocol_file}")

    # Create a minimal example protocol if it doesn't exist
    if not protocol_file.exists():
        print("Creating example protocol...")
        example_protocol = {
            "records": [
                {
                    "sample_id": "example_001",
                    "study_id": "study_001",
                    "ordinal": 0,
                    "question": "Which finding is present on this chest X-ray?",
                    "options": ["Consolidation", "Pleural Effusion", "No consolidation"],
                    "gold_letter": "A",
                    "gold_option": "Consolidation",
                    "image_paths": ["path/to/image1.jpg"],
                    "metadata": {"task_name": "presence_probe"},
                },
                {
                    "sample_id": "example_002",
                    "study_id": "study_002",
                    "ordinal": 0,
                    "question": "Which finding is absent on this chest X-ray?",
                    "options": ["Pneumothorax", "Atelectasis", "No pneumothorax"],
                    "gold_letter": "C",
                    "gold_option": "No pneumothorax",
                    "image_paths": ["path/to/image2.jpg"],
                    "metadata": {"task_name": "absence_probe"},
                },
            ]
        }
        protocol_file.parent.mkdir(parents=True, exist_ok=True)
        with protocol_file.open("w") as f:
            json.dump(example_protocol, f, indent=2)
        print(f"Created {protocol_file}")

    # Load protocol
    try:
        records = load_protocol(protocol_file)
        print(f"Loaded {len(records)} records")
    except Exception as e:
        print(f"Could not load protocol: {e}")
        return

    # Run inference on each sample
    print("\nRunning inference...")
    result_records = []
    for i, record in enumerate(records):
        question = str(record.get("question") or "")
        options = record.get("options", [])

        # Get ground truth
        gold_letter = resolve_gold_letter(record, options)

        # Get prediction
        pred_letter = mock_model_inference(question, options)

        # Check if correct
        is_correct = (pred_letter == gold_letter)

        # Build result
        result = dict(record)
        result.update({
            "pred_letter": pred_letter,
            "gold_letter": gold_letter,
            "exact_match": is_correct,
        })
        result_records.append(result)

        status = "✓" if is_correct else "✗"
        print(f"  {status} Sample {record['sample_id']}: Predicted {pred_letter}, Gold {gold_letter}")

    # Display accuracy
    accuracy = exact_match_rate(result_records)
    print(f"\nAccuracy (before repair): {accuracy:.2%}")

    return result_records


def example_2_apply_verifier(records: list):
    """Example 2: Apply QCCV-Neg repair to results."""
    print("\n" + "=" * 70)
    print("Example 2: Apply QCCV-Neg M1 Repair")
    print("=" * 70)

    if not records:
        print("No records to repair")
        return

    # Count negation contradictions before repair
    pre_contradictions = negation_contradiction_count(records)
    print(f"\nNegation contradictions (before repair): {pre_contradictions}")

    # Apply M1 repair
    print("Applying M1 repair...")
    repaired_records = apply_m1_to_records(records)

    # Count negation contradictions after repair
    post_contradictions = negation_contradiction_count(repaired_records)
    print(f"Negation contradictions (after repair): {post_contradictions}")

    # Show accuracy change
    pre_accuracy = exact_match_rate(records)
    post_accuracy = exact_match_rate(repaired_records)

    print(f"\nAccuracy change:")
    print(f"  Before repair: {pre_accuracy:.2%}")
    print(f"  After repair: {post_accuracy:.2%}")
    print(f"  Delta: +{(post_accuracy - pre_accuracy) * 100:.2f}pp")

    # Show what was repaired
    repairs = [r for r in repaired_records if r.get("verifier_applied")]
    if repairs:
        print(f"\nRepairs applied ({len(repairs)} samples):")
        for repair in repairs[:3]:  # Show first 3
            print(f"  {repair['sample_id']}: {repair.get('b0_pred_letter')} → {repair['pred_letter']}")

    return repaired_records


def example_3_metrics(records: list):
    """Example 3: Compute and display metrics."""
    print("\n" + "=" * 70)
    print("Example 3: Compute Metrics")
    print("=" * 70)

    from cxr_contrabench.metrics import build_metric_summary

    metrics = build_metric_summary(records, subset_name="all_samples")

    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if "rate" in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CXR-ContraBench Minimal Example")
    print("=" * 70)

    # Example 1: Load and evaluate
    result_records = example_1_load_and_evaluate()

    if result_records:
        # Example 2: Apply verifier
        repaired = example_2_apply_verifier(result_records)

        # Example 3: Metrics
        if repaired:
            example_3_metrics(repaired)

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
