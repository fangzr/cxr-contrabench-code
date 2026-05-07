# CXR-ContraBench Benchmark Specifications

## Overview

CXR-ContraBench is a diagnostic benchmark designed to measure **negated-option attraction** in medical vision-language models. This phenomenon occurs when a model selects a negated answer option despite the image and question indicating the finding is present.

## The Negation-Trap Phenomenon

### Definition

**Negated-Option Attraction**: A model is drawn to answer a multiple-choice question with a negated option (e.g., "No consolidation") when the question context and visual evidence suggest a positive option (e.g., "Consolidation").

### Clinical Significance

This is a clinically meaningful failure because:
1. **Polarity Inversion**: The model's answer directly contradicts the visual evidence
2. **Semantic Reversal**: Not a random error, but a systematic selection of the wrong semantic pole
3. **Systematic**: Occurs at scale (62% of presence questions in large-scale CheXpert protocol)
4. **Not Mitigated by CoT**: Chain-of-thought prompting reduces but does not eliminate the problem

## Benchmark Design

### Core Protocols

The benchmark includes multiple protocol families, each targeting different aspects:

#### 1. Direct Presence Probe (235 records)
- **Source**: CheXpert dataset
- **Type**: Direct evaluation
- **Format**: Presence questions with:
  - 1 negated option (e.g., "No consolidation")
  - 1+ positive distractors
  - 1 positive target
- **Purpose**: Isolate presence-side semantic reversal in controlled setting

#### 2. CheXpert Presence Protocol (135,754 records)
- **Source**: CheXpert training split
- **Type**: Large-scale
- **Format**: Matched presence questions for each finding in dataset
- **Purpose**: Confirm phenomenon scales beyond curated probe

#### 3. OpenI Retrospective Audits
- **Presence Subset**: 396 records
  - Derived from OpenI reports asserting finding presence
  - Multiple inference confirmation protocols
- **Absence Subset**: 979 records
  - Reports explicitly stating finding absence
  - Tests whether models copy negated wording
- **Purpose**: Real-world report-grounded validation

#### 4. ReXVQA Internal Slices
- **Variants**: 1k, 5k, 10k, and pooled 3×10k records
- **Source**: Internal ReXVQA dataset
- **Purpose**: Controlled paraphrase experiments and scale validation

### Sample Structure

Each benchmark sample consists of:

```json
{
  "sample_id": "unique_identifier",
  "study_id": "radiology_study_id",
  "question": "Which finding is present on this chest X-ray?",
  "options": [
    "A. Consolidation",
    "B. Pleural Effusion",
    "C. No consolidation"
  ],
  "gold_letter": "A",
  "image_paths": ["path/to/image.jpg"],
  "metadata": {
    "task_name": "presence_probe",
    "finding": "consolidation"
  }
}
```

### Question Polarity

Questions are classified into two categories:

**Presence Questions**:
- "Which finding **is present**?"
- "Which **of the following** findings?"
- Correct answer is a positive finding

**Absence Questions**:
- "Which finding **is absent**?"
- "Which finding **is not** present?"
- Correct answer is a concept name (not negated surface form)

## Evaluation Metrics

### Primary Metrics

1. **Exact Match Accuracy**
   - Proportion of correct option letter selections
   - Standard multiple-choice metric

2. **Negation Contradiction Count (M1)**
   - Number of presence questions answered with negated options
   - Detects if model selected "No X" when X is present
   - Directly measurable and repairable

3. **Invalid Option Count**
   - Predictions selecting non-existent option letters
   - Indicates model confusion about answer space

### Comparative Metrics

When comparing baseline vs. variant models:

- **Delta Exact Match vs Baseline**: Absolute change in correct predictions
- **Delta Exact Match Rate (pp)**: Percentage-point improvement/degradation
- **Prediction Changes**: How many predictions differed
- **Correctness Changes**: How many predictions changed correctness status
- **Improved Cases**: False → True transitions
- **Worsened Cases**: True → False transitions

## The QCCV-Neg Repair Mechanism

### Design Philosophy

QCCV-Neg (Question-Conditioned Consistency Verifier for Negation) is a **deterministic post-hoc repair** mechanism that:
- Requires NO retraining
- Operates only on the final prediction
- Is fully auditable and explainable
- Applies only when conditions are met (no speculative fixes)

### M1 Repair (Presence-Side Negation Trap)

**Activation Condition**:
1. Question asks "which finding **is present**"
2. Model selected a negated option (contains "no", "without", etc.)
3. A unique positive counterpart exists (e.g., "Consolidation" for "No consolidation")

**Repair Action**:
- Replace prediction with the unique positive counterpart
- Deterministic: same record → same output

**Success Rate**: ~96% accuracy on direct presence probe (up from 31%)

**Example**:
```
Question: "Which finding is present?"
Options: A) Consolidation  B) Pleural Effusion  C) No consolidation
Model prediction: C (wrong)
M1 repair: A (correct)
```

### M2 Repair (Location Conflict, Optional)

For advanced use:
- Detects location mismatches (e.g., question asks about "left lung" but prediction mentions "right")
- Corrects if unique location-matching option exists

## Protocol Building Procedure

### Data Requirements

Each protocol requires:
1. Medical images (CXR only)
2. Ground truth labels (binary or multi-label)
3. Structured question/option templates

### Deterministic Layout

To minimize confounds, option layouts are **deterministically randomized** per sample:

```python
layout = shuffle([target_option, distractor_option, negative_option])
letters = assign_letters(layout)
```

This ensures:
- Same image/question/finding always has same layout
- Layouts vary across samples
- Results are deterministic and reproducible

## Benchmark Validation

### Internal Validation
- Checked against manual radiologist review for subset
- Multiple radiologists verified ground truth labels
- Inter-rater agreement measured

### External Validation
- Direct probe created from held-out CheXpert set
- Retrospective audits from OpenI reports
- Large-scale CheXpert protocol on training split

### Model Testing
Evaluated on:
- **MedGemma-4B** and **MedGemma-27B**: Strong baselines
- **Qwen2.5-VL-7B-Instruct**: State-of-the-art vision model
- **LLaVA variants**: General-purpose VLM
- **GPT-4O**: Commercial frontier model

## Key Findings

### Pre-Repair Performance
- **MedGemma-27B**: 31.49% accuracy (direct probe)
- **Qwen2.5-VL**: 30.21% accuracy
- Both models select negated options on **>62%** of presence questions in large-scale protocol

### Post-Repair Performance
- **MedGemma-27B**: 96.60% accuracy (+65.11pp)
- **Qwen2.5-VL**: 95.32% accuracy (+65.11pp)
- Repairs applied deterministically without retraining

### CoT Findings
- CoT reduces but doesn't eliminate negation traps on presence
- CoT can amplify contradictions on absence protocols
- Standard reasoning is insufficient

## Dataset Links

Public datasets used:

- **CheXpert**: https://stanfordmlgroup.github.io/competitions/chexpert/
  - 223,648 images, 14 findings
  - Use training split for large-scale protocol

- **OpenI Indiana Chest X-Ray**: https://openi.nlm.nih.gov/
  - 7,470 images with radiologist reports
  - Use for retrospective audits

- **ReXVQA**: Available upon request (internal use)
  - 10k+ VQA-style questions on chest X-rays

## References

Related work on negation and compositional understanding:
- NegBench: https://arxiv.org/abs/2310.14729
- Winoground: https://arxiv.org/abs/2204.03162
- Medical VQA benchmarks: PMC-VQA, CARES, etc.
