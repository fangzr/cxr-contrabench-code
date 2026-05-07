"""
Dataset loaders for CXR-ContraBench.

Handles loading evaluation protocols from JSON files and building evaluation samples.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import EvalSample, load_json, option_map, resolve_gold_letter


def load_protocol(protocol_path: Path) -> list[dict[str, Any]]:
    """
    Load evaluation protocol from JSON file.

    Args:
        protocol_path: Path to protocol JSON file

    Returns:
        List of record dictionaries
    """
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    payload = load_json(protocol_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict at top level of {protocol_path}")

    # Handle both formats: {"records": [...]} and list of records
    if "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unexpected protocol format in {protocol_path}")


def record_to_eval_sample(record: dict[str, Any], image_root: Path | None = None) -> EvalSample:
    """
    Convert a protocol record to an EvalSample.

    Args:
        record: Protocol record dictionary
        image_root: Optional root directory for image paths

    Returns:
        EvalSample object
    """
    sample_id = str(record.get("sample_id") or record.get("record_id") or "")
    study_id = str(record.get("study_id") or "")
    ordinal = int(record.get("ordinal") or 0)
    question = str(record.get("question") or "")
    options = record.get("options", [])
    if isinstance(options, str):
        options = [opt.strip() for opt in options.split("||")]
    options = [str(opt) for opt in options]

    # Resolve correct answer
    gold_letter = resolve_gold_letter(record, options)

    # Get image paths
    image_paths_raw = record.get("image_paths", [])
    if isinstance(image_paths_raw, str):
        image_paths_raw = [image_paths_raw]

    image_paths = []
    for img_path in image_paths_raw:
        img_path = Path(img_path)
        if image_root and not img_path.is_absolute():
            img_path = image_root / img_path
        image_paths.append(img_path)

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return EvalSample(
        sample_id=sample_id,
        study_id=study_id,
        ordinal=ordinal,
        question=question,
        options=options,
        correct_answer=gold_letter,
        image_paths=image_paths,
        metadata=metadata,
    )


def load_protocol_samples(
    protocol_path: Path,
    image_root: Path | None = None,
) -> list[EvalSample]:
    """
    Load evaluation samples from a protocol file.

    Args:
        protocol_path: Path to protocol JSON file
        image_root: Optional root directory for image paths

    Returns:
        List of EvalSample objects
    """
    records = load_protocol(protocol_path)
    samples = []
    for record in records:
        try:
            sample = record_to_eval_sample(record, image_root)
            samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to load sample {record.get('sample_id')}: {e}")
            continue
    return samples


def save_protocol(path: Path, records: list[dict[str, Any]]) -> None:
    """
    Save evaluation records to a protocol file.

    Args:
        path: Output file path
        records: List of record dictionaries
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)


class ProtocolDataset:
    """
    Dataset class for loading and managing protocol records.

    Useful for iterating, filtering, and analyzing evaluation protocols.
    """

    def __init__(self, protocol_path: Path, image_root: Path | None = None):
        """
        Initialize dataset from protocol file.

        Args:
            protocol_path: Path to protocol JSON file
            image_root: Optional root directory for image paths
        """
        self.protocol_path = protocol_path
        self.image_root = image_root
        self.records = load_protocol(protocol_path)
        self.samples = load_protocol_samples(protocol_path, image_root)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> EvalSample:
        """Get sample by index."""
        return self.samples[idx]

    def filter(self, predicate) -> list[EvalSample]:
        """Filter samples by predicate function."""
        return [sample for sample in self.samples if predicate(sample)]

    def by_task(self, task_name: str) -> list[EvalSample]:
        """Get samples from a specific task."""
        return [
            sample for sample in self.samples
            if sample.metadata.get("task_name") == task_name
        ]

    def summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_samples": len(self.samples),
            "num_records": len(self.records),
            "protocol_path": str(self.protocol_path),
        }
