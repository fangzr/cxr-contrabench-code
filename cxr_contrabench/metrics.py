"""
Evaluation metrics for CXR-ContraBench.
"""
from __future__ import annotations

from typing import Any

from .common import option_map
from .verifier import (
    ORIGINAL_NEGATION_TRIGGERS,
    find_m1_replacement,
)


def exact_match_rate(records: list[dict[str, Any]]) -> float:
    """Calculate exact match accuracy."""
    if not records:
        return 0.0
    correct = sum(1 for record in records if bool(record.get("exact_match")))
    return correct / len(records)


def invalid_option_count(records: list[dict[str, Any]]) -> int:
    """Count predictions that selected an invalid option letter."""
    total = 0
    for record in records:
        mapping = option_map(record.get("options", []))
        if str(record.get("pred_letter") or "").upper() not in mapping:
            total += 1
    return total


def negation_contradiction_count(
    records: list[dict[str, Any]],
    *,
    triggers: tuple[str, ...] = ORIGINAL_NEGATION_TRIGGERS,
) -> int:
    """
    Count predictions that would be corrected by M1 repair.

    This counts presence-side negation traps: cases where a presence question
    is answered with a negated option.
    """
    total = 0
    for record in records:
        pred_letter = str(record.get("pred_letter") or "").upper()
        if find_m1_replacement(
            record,
            pred_letter,
            triggers=triggers,
        ):
            total += 1
    return total


def compare_records(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    """
    Compare predictions between baseline and candidate.

    Returns:
        (prediction_changes, correctness_changes, improved, worsened)
    """
    baseline_by_id = {str(record["sample_id"]): record for record in baseline_records}
    prediction_changes = 0
    correctness_changes = 0
    improved = 0
    worsened = 0
    for candidate in candidate_records:
        baseline = baseline_by_id[str(candidate["sample_id"])]
        if str(baseline.get("pred_letter") or "") != str(candidate.get("pred_letter") or ""):
            prediction_changes += 1
        if bool(baseline.get("exact_match")) != bool(candidate.get("exact_match")):
            correctness_changes += 1
        if not bool(baseline.get("exact_match")) and bool(candidate.get("exact_match")):
            improved += 1
        if bool(baseline.get("exact_match")) and not bool(candidate.get("exact_match")):
            worsened += 1
    return prediction_changes, correctness_changes, improved, worsened


def build_metric_summary(
    records: list[dict[str, Any]],
    subset_name: str = "full",
) -> dict[str, Any]:
    """
    Build a summary of metrics for a set of records.

    Args:
        records: List of evaluation records
        subset_name: Name of the subset being evaluated

    Returns:
        Dictionary with metric values
    """
    if not records:
        return {
            "subset": subset_name,
            "total": 0,
            "exact_match": 0,
            "exact_match_rate": 0.0,
            "negation_contradiction_count": 0,
            "invalid_option_count": 0,
        }

    exact = sum(1 for record in records if bool(record.get("exact_match")))
    rate = exact / len(records)
    neg_count = negation_contradiction_count(records)
    invalid_count = invalid_option_count(records)

    return {
        "subset": subset_name,
        "total": len(records),
        "exact_match": exact,
        "exact_match_rate": round(rate, 4),
        "exact_match_rate_percent": round(100.0 * rate, 2),
        "negation_contradiction_count": neg_count,
        "negation_contradiction_rate": round(neg_count / len(records) if records else 0, 4),
        "invalid_option_count": invalid_count,
    }


def build_variant_comparison(
    baseline_records: list[dict[str, Any]],
    variant_records: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Compare multiple variants against a baseline.

    Args:
        baseline_records: Records from baseline model (e.g., B0)
        variant_records: Mapping of variant name -> records

    Returns:
        List of comparison rows
    """
    rows: list[dict[str, Any]] = []
    baseline_exact = sum(1 for record in baseline_records if bool(record.get("exact_match")))
    baseline_rate = baseline_exact / len(baseline_records) if baseline_records else 0.0

    for variant_name, records in variant_records.items():
        variant_exact = sum(1 for record in records if bool(record.get("exact_match")))
        variant_rate = variant_exact / len(records) if records else 0.0

        pred_changes, corr_changes, improved, worsened = compare_records(baseline_records, records)

        rows.append({
            "variant": variant_name,
            "total": len(records),
            "exact_match": variant_exact,
            "exact_match_rate": round(variant_rate, 4),
            "exact_match_rate_percent": round(100.0 * variant_rate, 2),
            "delta_exact_match_vs_baseline": variant_exact - baseline_exact,
            "delta_exact_match_rate_pp": round(100.0 * (variant_rate - baseline_rate), 2),
            "prediction_changes": pred_changes,
            "correctness_changes": corr_changes,
            "improved": improved,
            "worsened": worsened,
            "negation_contradiction_count": negation_contradiction_count(records),
        })

    return rows
