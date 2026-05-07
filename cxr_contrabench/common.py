"""
Common utilities for CXR-ContraBench.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalSample:
    """A single evaluation sample with question, options, and answer."""
    sample_id: str
    study_id: str
    ordinal: int
    question: str
    options: list[str]
    correct_answer: str
    image_paths: list[Path]
    metadata: dict[str, Any]
    image_view_positions: list[str] = field(default_factory=list)


def ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any) -> str:
    """Normalize text for comparison: lowercase, remove extra spaces and punctuation."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?\"'")


def slugify(value: Any) -> str:
    """Convert text to slug format."""
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "item"


def split_option(option: Any) -> tuple[str, str]:
    """Split lettered option into letter and text. E.g., 'A. Yes' -> ('A', 'Yes')."""
    match = re.match(r"^\s*([A-Z])[\.\):]\s*(.*)$", str(option or "").strip())
    if not match:
        return "", str(option or "").strip()
    return match.group(1).upper(), match.group(2).strip()


def render_lettered_options(options: Iterable[Any]) -> list[str]:
    """Render options with letters if not already present."""
    rendered: list[str] = []
    for index, option in enumerate(options):
        letter, text = split_option(option)
        if letter:
            rendered.append(str(option).strip())
        else:
            rendered.append(f"{chr(65 + index)}. {str(option).strip()}")
    return rendered


def option_map(options: Iterable[Any]) -> dict[str, str]:
    """Map letters to option text."""
    rendered = render_lettered_options(options)
    mapping: dict[str, str] = {}
    for option in rendered:
        letter, text = split_option(option)
        if letter:
            mapping[letter] = text
    return mapping


def parse_options(raw: Any) -> list[str]:
    """Parse options from various formats."""
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split("||") if str(part).strip()]


def resolve_gold_letter(record: dict[str, Any], options: Iterable[Any] | None = None) -> str:
    """Resolve the correct answer letter from a record."""
    rendered = render_lettered_options(options if options is not None else record.get("options", []))
    mapping = option_map(rendered)
    for key in ("gold_letter", "correct_answer", "gold_option"):
        candidate = str(record.get(key) or "").strip()
        if not candidate:
            continue
        if re.fullmatch(r"[A-Za-z]", candidate):
            return candidate.upper()
        norm = normalize_text(candidate)
        for letter, text in mapping.items():
            if normalize_text(text) == norm:
                return letter
    raise ValueError(f"Could not resolve gold answer for record {record.get('sample_id') or record.get('record_id')}")


def format_answer(letter: str, text: str, fallback: str = "") -> str:
    """Format an answer as 'Letter. Text'."""
    if letter and text:
        return f"{letter}. {text}"
    if letter:
        return letter
    return str(fallback or "").strip()


def pred_answer(record: dict[str, Any]) -> str:
    """Get the predicted answer from a result record."""
    mapping = option_map(record.get("options", []))
    letter = str(record.get("pred_letter") or "").upper()
    return format_answer(letter, mapping.get(letter, ""), fallback=str(record.get("pred_raw") or ""))


def gold_answer(record: dict[str, Any]) -> str:
    """Get the correct answer from a result record."""
    mapping = option_map(record.get("options", []))
    letter = str(record.get("gold_letter") or resolve_gold_letter(record, record.get("options", []))).upper()
    return format_answer(letter, mapping.get(letter, ""))


def stable_hash_int(*parts: Any) -> int:
    """Generate a stable hash as integer."""
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def deterministic_layout(
    sample_id: str,
    target_option: str,
    negative_option: str,
    distractor_option: str,
) -> list[str]:
    """Create a deterministic, shuffled layout of options based on sample_id."""
    layout = [target_option, distractor_option, negative_option]
    random.Random(str(sample_id).lower()).shuffle(layout)
    return render_lettered_options(layout)


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    """Save JSON to file."""
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_result_payload(path: Path) -> dict[str, Any]:
    """Load evaluation result payload."""
    payload = load_json(path)
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"Unexpected result payload in {path}")
    return payload


def save_result_payload(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Save evaluation result payload."""
    save_json(path, {"summary": summary, "records": records})


def load_protocol_records(path: Path) -> list[dict[str, Any]]:
    """Load protocol records from JSON."""
    payload = load_json(path)
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"Unexpected protocol payload in {path}")
    return list(payload["records"])


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """Write CSV file."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, text: str) -> None:
    """Write markdown file."""
    ensure_parent(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def relative_image_paths(paths: Iterable[Path]) -> list[str]:
    """Convert paths to relative strings."""
    return [str(path) for path in paths]


def task_name_from_record(record: dict[str, Any]) -> str:
    """Extract task name from record."""
    metadata = record.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("task_name"):
        return str(metadata["task_name"])
    if record.get("task_name"):
        return str(record["task_name"])
    return "Unknown"


def bool_to_yesno(value: bool) -> str:
    """Convert boolean to yes/no."""
    return "yes" if value else "no"
