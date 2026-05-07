"""
QCCV-Neg: Question-Conditioned Consistency Verifier for Negation.

This module implements the deterministic polarity repair mechanism described in
CXR-ContraBench. It identifies when a model has selected a negated option despite
the question asking for a present finding, and proposes a correction.
"""
from __future__ import annotations

import re
from typing import Any

from .common import normalize_text, option_map


# Patterns for detecting negation-related questions
ORIGINAL_NEGATION_TRIGGERS = (
    " absent",
    "not present",
    "not seen",
    "not visible",
    "not observed",
    "not demonstrated",
    "negative for",
    "free of",
    "without",
    "which finding is not",
    "which abnormality is not",
    "least likely present",
)

EXTENDED_NEGATION_TRIGGERS = ORIGINAL_NEGATION_TRIGGERS + (
    "missing",
    "not identified",
    "no evidence of",
    " no ",
)

NEGATIVE_OPTION_RE = re.compile(r"\b(no|normal|clear|absent|without|negative for)\b")
NEGATIVE_PREFIX_RE = re.compile(r"^(?:no|without|absent|negative for|free of|lack of)\s+")
QUESTION_PRESENT_RE = re.compile(
    r"\b(which of the following findings is present|which finding is present|finding is present)\b"
)
QUESTION_ABSENT_RE = re.compile(
    r"\b(absent|not present|not seen|not visible|not observed|negative for|free of|without|"
    r"least likely present|which finding is not|which abnormality is not)\b"
)

# Stopwords to ignore when comparing options
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "without", "and", "or",
    "is", "are", "was", "were", "this", "that", "these", "those", "what", "which", "following",
    "regarding", "related", "observed", "visible", "notable", "finding", "findings", "best",
    "most", "likely", "least", "present", "status", "assessment", "chest", "x", "ray",
    "images", "image", "based", "given", "no", "normal", "clear", "absent",
}


def has_negation_prompt(question: str, triggers: tuple[str, ...] = ORIGINAL_NEGATION_TRIGGERS) -> bool:
    """Check if question contains negation triggers."""
    normalized = f" {normalize_text(question)} "
    return any(trigger in normalized for trigger in triggers)


def infer_question_polarity(question: str, triggers: tuple[str, ...] = ORIGINAL_NEGATION_TRIGGERS) -> str | None:
    """
    Infer whether a question asks about presence or absence of a finding.

    Returns:
        "presence" if question asks what IS present
        "absence" if question asks what is ABSENT or NOT present
        None if unable to determine
    """
    normalized = normalize_text(question)
    if QUESTION_PRESENT_RE.search(normalized):
        return "presence"
    if QUESTION_ABSENT_RE.search(normalized) or has_negation_prompt(normalized, triggers=triggers):
        return "absence"
    return None


def is_negative_option(text: str) -> bool:
    """Check if an option is negated (e.g., 'No consolidation')."""
    normalized = f" {normalize_text(text)} "
    if " no " in normalized:
        return True
    return bool(NEGATIVE_OPTION_RE.search(normalized))


def strip_negative_prefix(text: str) -> str:
    """Remove negation prefixes from text."""
    normalized = normalize_text(text)
    normalized = NEGATIVE_PREFIX_RE.sub("", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def canonicalize_token(token: str) -> str:
    """Canonicalize token for comparison (handle plurals)."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def core_tokens(text: str) -> set[str]:
    """Extract core content tokens from text (excluding stopwords)."""
    return {
        canonicalize_token(token)
        for token in re.findall(r"[a-z]+", normalize_text(text))
        if token not in STOPWORDS
    }


def has_shared_core(option_texts: list[str]) -> bool:
    """Check if options share core semantic tokens."""
    token_sets = [core_tokens(text) for text in option_texts]
    if not token_sets or any(not tokens for tokens in token_sets):
        return False
    return bool(set.intersection(*token_sets))


def matching_positive_counterpart(options: dict[str, str], negative_letter: str) -> str | None:
    """
    Find the positive option that matches a negated option.

    For example, if option A is "No consolidation", find option that is "Consolidation".

    Args:
        options: Mapping of letter -> option text
        negative_letter: Letter of the negated option

    Returns:
        Letter of matching positive option if exactly one match, else None
    """
    negative_text = options.get(negative_letter, "")
    if not negative_text:
        return None
    target = strip_negative_prefix(negative_text)
    matches = [
        letter
        for letter, text in options.items()
        if letter != negative_letter and not is_negative_option(text) and strip_negative_prefix(text) == target
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def find_m1_replacement(
    record: dict[str, Any],
    current_letter: str,
    *,
    triggers: tuple[str, ...] = ORIGINAL_NEGATION_TRIGGERS,
) -> str | None:
    """
    Find M1 replacement: detect and repair presence-side negation traps.

    M1 repairs cases where:
    1. The question asks what finding IS present
    2. The model selected a negated option (e.g., "No consolidation")
    3. A unique positive counterpart exists

    Args:
        record: Evaluation record with question, options, prediction
        current_letter: The letter of the model's current prediction
        triggers: Negation trigger phrases to recognize

    Returns:
        Letter of replacement option if found, else None
    """
    question = str(record.get("question") or "")
    options = option_map(record.get("options", []))
    if current_letter not in options:
        return None
    if not is_negative_option(options[current_letter]):
        return None

    polarity = infer_question_polarity(question, triggers=triggers)
    if polarity != "presence":
        return None

    negative_letters = [letter for letter, text in options.items() if is_negative_option(text)]
    positive_letters = [letter for letter in options if letter not in negative_letters]

    if len(negative_letters) != 1 or current_letter != negative_letters[0]:
        return None
    if not positive_letters:
        return None

    return matching_positive_counterpart(options, current_letter)


def apply_m1_to_records(
    records: list[dict[str, Any]],
    *,
    triggers: tuple[str, ...] = ORIGINAL_NEGATION_TRIGGERS,
) -> list[dict[str, Any]]:
    """
    Apply M1 verification to a batch of records.

    This is the QCCV-Neg repair mechanism: deterministically corrects
    negation-attracted predictions without retraining.

    Args:
        records: List of evaluation records
        triggers: Negation trigger phrases to recognize

    Returns:
        List of records with repairs applied (if applicable)
    """
    updated_records: list[dict[str, Any]] = []
    for record in records:
        current_letter = str(record.get("pred_letter") or "").upper()
        replacement = find_m1_replacement(
            record,
            current_letter,
            triggers=triggers,
        )

        # Prepare updated record
        updated = dict(record)
        if "b0_pred_letter" not in updated:
            updated["b0_pred_letter"] = current_letter
        if "b0_exact_match" not in updated:
            updated["b0_exact_match"] = bool(record.get("exact_match"))

        if replacement and replacement != current_letter:
            # Apply correction
            updated["pred_letter"] = replacement
            updated["pred_raw"] = replacement
            updated["exact_match"] = replacement == str(updated.get("gold_letter") or "").upper()
            updated["verifier_applied"] = True
            updated["verifier_reason"] = "m1_presence_negation_trap"
        else:
            # No repair needed
            updated["verifier_applied"] = False
            updated["verifier_reason"] = "kept"

        updated["verifier_stage"] = "M1"
        updated_records.append(updated)

    return updated_records
