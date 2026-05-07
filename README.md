# CXR-ContraBench: Benchmarking Negated-Option Attraction in Medical VLMs

CXR-ContraBench is a reproducible benchmark for evaluating **negated-option attraction** in medical vision-language models (VLMs). The benchmark focuses on a clinically meaningful failure mode in chest X-ray question answering: a model selects a negated answer option, such as “No consolidation,” even when the image and question indicate that the finding is present.

This repository provides the public code release for CXR-ContraBench.

## Overview

Negated-option attraction is a polarity-confusion failure in generative medical VLMs. A typical presence-side error is:

- A chest X-ray shows a finding, such as consolidation.
- The question asks which finding is present.
- The model selects a negated option, such as “No consolidation.”
- The output therefore contradicts the visual evidence and the question intent.

CXR-ContraBench makes this failure mode measurable and auditable through:

1. **Direct probes**: controlled multiple-choice questions with positive and negated answer options.
2. **Retrospective audits**: large-scale evaluations using standard chest X-ray datasets and report-derived labels.
3. **QCCV-Neg repair**: a deterministic post-hoc verifier that remaps eligible negated predictions under strict conditions.
4. **Diagnostic metrics**: exact-match accuracy, negation-contradiction counts, invalid-option counts, and case-level changes.

## Key Features

- **Minimal and modular**: core code for protocol loading, evaluation, verification, and metric computation.
- **No medical images redistributed**: users must obtain datasets from the official sources.
- **Reproducible protocols**: deterministic protocol construction and deterministic verification logic.
- **Clinically motivated but non-clinical**: designed for research evaluation of medical VLM behavior, not for diagnosis or deployment.
- **Training-free repair analysis**: QCCV-Neg is a post-hoc consistency verifier and does not require model retraining or additional model calls.

## Installation

```bash
# Clone repository
git clone https://github.com/fangzr/cxr-contrabench-code.git
cd cxr-contrabench-code

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- Pillow
- pandas
- tqdm
- numpy

Depending on the evaluated VLM, additional model-specific dependencies may be required.

## Quick Start

### 1. Prepare Local Dataset Copies

Prepare local copies of CheXpert and/or OpenI. This repository does **not** redistribute dataset files.

```bash
python scripts/prepare_datasets.py \
  --chexpert /path/to/chexpert \
  --openi /path/to/openi
```

The script constructs protocol files under:

```bash
data/protocols/
```

Users are responsible for obtaining dataset access from the official sources and complying with the corresponding licenses, terms of use, and citation requirements.

### 2. Evaluate a Model

Run evaluation on a protocol file:

```bash
python scripts/eval_vqa.py \
  --model medgemma-4b-it \
  --protocol data/protocols/chexpert_presence_direct.json \
  --output results/medgemma_chexpert.json
```

The evaluation script records model predictions, selected options, normalized answers, and per-sample correctness information.

### 3. Apply QCCV-Neg

Apply the deterministic verifier to eligible predictions:

```bash
python scripts/apply_verifier.py \
  --input results/medgemma_chexpert.json \
  --output results/medgemma_chexpert_repaired.json
```

QCCV-Neg only modifies predictions when the protocol-defined conditions are satisfied, such as selecting exactly one negated option with a unique positive counterpart.

### 4. Analyze Results

Compare baseline and repaired outputs:

```bash
python scripts/analyze_results.py \
  --baseline results/medgemma_chexpert.json \
  --repaired results/medgemma_chexpert_repaired.json
```

The analysis script reports exact-match accuracy, contradiction counts, invalid-option counts, and improved or worsened cases.

## Benchmark Structure

CXR-ContraBench contains several protocol types.

| Protocol | Source | Type | Samples | Focus |
|----------|--------|------|---------|-------|
| Direct Presence Probe | CheXpert | Direct | 235 | Presence questions with negated options |
| CheXpert Presence Protocol | CheXpert | Large-scale | 135,754 | Training-split matched presence samples |
| OpenI Presence Audit | OpenI Reports | Retrospective | 396 | Report-derived presence assertions |
| OpenI Absence Audit | OpenI Reports | Retrospective | 979 | Report-derived absence assertions |
| ReXVQA Internal Slices | ReXVQA | Internal | 1k-10k | Additional question/option configurations |

The public code release focuses on protocols that can be reconstructed from datasets available to users under their respective terms. ReXVQA internal slices are not redistributed in this code release.

## Negated-Option Attraction

### Presence-Side Semantic Reversal

The primary failure mode studied in this benchmark occurs when the image contains a finding and the question asks for a present finding, but the model selects a negated answer option.

Example:

- Image finding: pneumothorax is present.
- Question: “Which finding is present?”
- Options:
  - A. Pneumothorax
  - B. Pleural effusion
  - C. No pneumothorax
- Model answer: C
- Error type: presence-side semantic reversal.

This is the main risk setting because the model output states the opposite of the visual evidence.

### Absence-Side Diagnostic Test

The benchmark also includes absence-side tests. These evaluate whether models copy negated surface forms or confuse the semantic target of absence questions.

Example:

- Image finding: edema is absent.
- Question: “Which finding is absent?”
- Options:
  - A. Edema
  - B. Pneumothorax
  - C. No edema
- Model answer: C
- Error type: answer-form contradiction under the protocol definition.

These cases are used as secondary diagnostics rather than the primary clinical-risk setting.

## QCCV-Neg Verifier

QCCV-Neg stands for **Question-Conditioned Consistency Verifier for Negation**. It is a deterministic post-hoc verifier for eligible polarity-confused predictions.

### Primary Rule: M1 Repair

QCCV-Neg applies the primary repair rule when all of the following conditions hold:

1. The question is a presence question.
2. The model selects a negated option.
3. The negated option has a safely identifiable positive counterpart.
4. The positive counterpart is unique among the answer options.
5. The prediction is otherwise a valid option selection.

When these conditions are met, QCCV-Neg remaps the selected negated option to its positive counterpart.

Example:

```text
Selected option: No pneumothorax
Positive counterpart: Pneumothorax
Verifier output: Pneumothorax
```

### Optional Fallback

An optional confidence-based fallback can be enabled for specific controlled settings. This fallback is disabled by default unless explicitly configured. It should be reported separately from the primary deterministic repair rule.

## Evaluation Metrics

### Core Metrics

- **Exact Match Accuracy**: proportion of predictions matching the protocol label.
- **Negation Contradiction Count**: number of presence questions answered with negated options.
- **Invalid Option Count**: number of predictions that do not correspond to any available option.
- **Changed Predictions**: number of predictions modified by QCCV-Neg.

### Comparison Metrics

When both baseline and repaired outputs are available, the analysis reports:

- **Delta Exact Match**: absolute change in exact-match accuracy.
- **Delta Exact Match Rate**: percentage-point change.
- **Improved Cases**: samples changed from incorrect to correct.
- **Worsened Cases**: samples changed from correct to incorrect.
- **Unchanged Cases**: samples unaffected by the verifier.

## Dataset Access

This repository does not redistribute medical images, clinical reports, or third-party dataset files. Users must obtain datasets from the official sources.

Official dataset pages:

- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
- OpenI Indiana Chest X-ray: https://openi.nlm.nih.gov/faq#collection

ReXVQA internal slices are not included in this code release. The released code supports reconstruction of public protocols from datasets that users obtain separately.

## Repository Structure

```text
cxr-contrabench-code/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── cxr_contrabench/
│   ├── __init__.py
│   ├── common.py
│   ├── datasets.py
│   ├── metrics.py
│   └── verifier.py
├── scripts/
│   ├── prepare_datasets.py
│   ├── build_protocols.py
│   ├── eval_vqa.py
│   ├── apply_verifier.py
│   └── analyze_results.py
├── examples/
│   ├── minimal_example.py
│   ├── example_protocol.json
│   └── example_results.json
└── docs/
    ├── BENCHMARK.md
    ├── DATASETS.md
    └── PROTOCOLS.md
```

## Core Modules

### `cxr_contrabench/common.py`

Shared utilities for text normalization, option parsing, answer canonicalization, and negation handling.

### `cxr_contrabench/datasets.py`

Protocol loading utilities and conversion into evaluation samples.

### `cxr_contrabench/verifier.py`

Implementation of QCCV-Neg, including negated-option detection, counterpart matching, and deterministic remapping.

### `cxr_contrabench/metrics.py`

Metric computation for exact match, contradiction counts, invalid predictions, and baseline-versus-repair comparisons.

## Example Protocol Format

A protocol file is a JSON list of evaluation samples. Each sample contains an image path, question, answer options, ground-truth answer, and metadata.

```json
{
  "id": "sample_000001",
  "image": "/path/to/image.png",
  "question": "Which finding is present?",
  "options": {
    "A": "Pneumothorax",
    "B": "Pleural effusion",
    "C": "No pneumothorax"
  },
  "answer": "A",
  "metadata": {
    "finding": "pneumothorax",
    "question_type": "presence",
    "source": "chexpert"
  }
}
```

The exact schema is described in `docs/PROTOCOLS.md`.

## Reproducibility Notes

To reproduce the benchmark results:

1. Obtain the required datasets from the official sources.
2. Prepare protocol files using the provided scripts.
3. Run model evaluation with fixed decoding settings.
4. Apply QCCV-Neg using the provided verifier.
5. Report baseline and repaired metrics separately.

Recommended deterministic decoding settings:

```text
temperature = 0
top_p = 1
num_beams = 1
```

For models requiring different inference APIs, users should keep decoding deterministic whenever possible and report any deviations.

## Medical and Ethical Disclaimer

This repository is intended for research evaluation only. It is not a medical device and must not be used for clinical diagnosis, treatment planning, triage, or patient-facing decision-making.

The benchmark evaluates model behavior on chest X-ray question-answering protocols. It does not validate clinical safety, diagnostic reliability, or deployment readiness of any medical AI system.

## License

The code and documentation in this repository are released under the Apache License 2.0. See [LICENSE](LICENSE).

This license applies only to the code and documentation provided in this repository. It does not grant any rights to third-party datasets, including CheXpert, OpenI, or ReXVQA. Users are responsible for obtaining dataset access and complying with the original dataset licenses, terms of use, and citation requirements.

No chest X-ray images or protected clinical reports are redistributed in this repository.


## Dataset Access

This repository contains only code and dataset-processing scripts. The full benchmark reconstruction requires users to obtain the underlying datasets, including CheXpert, OpenI, and ReXVQA, whose local storage footprint is approximately 360GB depending on preprocessing and image formats.

This repository does not redistribute medical images, clinical reports, or third-party dataset files. Users must obtain datasets from the official sources and comply with the corresponding licenses, terms of use, and citation requirements.
