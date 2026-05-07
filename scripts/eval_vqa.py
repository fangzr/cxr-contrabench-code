#!/usr/bin/env python3
"""
Simple VQA evaluation script for CXR-ContraBench.

This is a minimal example showing how to:
1. Load a protocol
2. Run inference on a model
3. Save results

For production use, adapt this to your specific model's inference API.
"""
import argparse
import json
from pathlib import Path
from typing import Any

from cxr_contrabench.common import option_map, resolve_gold_letter, save_result_payload
from cxr_contrabench.datasets import load_protocol


def run_vqa_evaluation(
    protocol_path: Path,
    model_name: str,
    inference_fn,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run VQA evaluation on a protocol using a provided inference function.

    Args:
        protocol_path: Path to protocol JSON file
        model_name: Name of model being evaluated
        inference_fn: Function that takes (question, options, image_paths) and returns predicted letter
        output_path: Optional path to save results

    Returns:
        Dictionary with summary and records
    """
    records = load_protocol(protocol_path)
    result_records = []

    total = len(records)
    correct = 0

    for i, record in enumerate(records):
        if (i + 1) % max(1, total // 20) == 0:
            print(f"  [{i+1}/{total}] samples processed...")

        question = str(record.get("question") or "")
        options = record.get("options", [])
        image_paths = record.get("image_paths", [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Get gold letter
        try:
            gold_letter = resolve_gold_letter(record, options)
        except Exception as e:
            print(f"Warning: Could not resolve gold answer for sample {record.get('sample_id')}: {e}")
            continue

        # Run inference (mock)
        try:
            pred_letter = inference_fn(question, options, image_paths)
        except Exception as e:
            print(f"Warning: Inference failed for sample {record.get('sample_id')}: {e}")
            pred_letter = "A"  # Default to first option

        # Check correctness
        is_correct = (pred_letter == gold_letter)
        if is_correct:
            correct += 1

        # Build result record
        options_map = option_map(options)
        result_record = dict(record)
        result_record.update({
            "pred_letter": pred_letter,
            "gold_letter": gold_letter,
            "pred_raw": pred_letter,
            "exact_match": is_correct,
            "options": [str(opt) for opt in options],
        })
        result_records.append(result_record)

    # Build summary
    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "model": model_name,
        "protocol": str(protocol_path.name),
        "total": total,
        "exact_match": correct,
        "exact_match_rate": round(accuracy, 4),
        "exact_match_rate_percent": round(100.0 * accuracy, 2),
    }

    print(f"\nResults:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")

    # Save if requested
    if output_path:
        save_result_payload(output_path, summary, result_records)
        print(f"\nResults saved to: {output_path}")

    return {"summary": summary, "records": result_records}


def mock_inference_fn(question: str, options: list[str], image_paths: list[str]) -> str:
    """
    Mock inference function that returns a random option.
    Replace this with your actual model inference.
    """
    import random
    return chr(65 + random.randint(0, len(options) - 1))


def main():
    parser = argparse.ArgumentParser(
        description="Run VQA evaluation on a CXR-ContraBench protocol"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Path to protocol JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock-model",
        help="Model name/identifier",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for results JSON",
    )

    args = parser.parse_args()

    if not args.protocol.exists():
        print(f"Error: Protocol file not found: {args.protocol}")
        exit(1)

    print(f"Loading protocol from: {args.protocol}")
    print(f"Model: {args.model}")

    # Run evaluation with mock inference
    # In practice, replace mock_inference_fn with your actual inference function
    results = run_vqa_evaluation(
        protocol_path=args.protocol,
        model_name=args.model,
        inference_fn=mock_inference_fn,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
