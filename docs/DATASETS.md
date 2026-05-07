# Dataset Preparation Guide

This document describes how to download, prepare, and build evaluation protocols from public datasets.

## Public Datasets Used

### 1. CheXpert Dataset

**Homepage**: https://stanfordmlgroup.github.io/competitions/chexpert/

**Details**:
- 223,648 chest X-ray images
- 14 binary labels for common findings
- Public training set (224,316 reports, 191,229 images)
- Held-out test set

**Download**:
```bash
# Request access at: https://stanfordmlgroup.github.io/competitions/chexpert/
# After approval, download:
wget <chexpert-download-link>

# Extract
unzip chexpert-v1.0-small.zip  # Or full version if available
```

**Structure**:
```
chexpert/
├── train.csv           # Training set metadata
├── valid.csv           # Validation set metadata
└── CheXpert_v1.0-small/
    ├── train/
    │   └── patient00001/study1/...
    └── valid/
```

**Key Findings** (14 labels):
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion
- Pneumothorax
- (and others)

### 2. OpenI (Indiana Chest X-Ray Collection)

**Homepage**: https://openi.nlm.nih.gov/

**Details**:
- 7,470 chest X-ray images from Indiana University
- Paired with radiologist reports
- Public and freely available

**Download**:
```bash
# Via MIMIC-CXR interface or direct download
# Download indices and images
wget https://openi.nlm.nih.gov/...

# Or use provided download script
# See: https://github.com/KevinBro/Indiana-CXR-Report-Extraction
```

**Structure**:
```
openi/
├── images/
│   └── *.png  (7,470 images)
├── indiana_anatomy.txt  # Anatomy annotations
└── indiana_reports.xml  # Radiologist reports
```

**Report Analysis**:
Reports are free-text radiologist interpretations:
```xml
<REPORT>
"No acute cardiopulmonary process. No evidence of pneumothorax,
pleural effusion, or consolidation."
</REPORT>
```

### 3. ReXVQA (Internal/Upon Request)

**Availability**: Available upon request (internal use)

**Details**:
- 10,000+ VQA-style questions on chest X-rays
- Multiple question types and answer formats
- Used for internal validation

## Protocol Building Process

### Phase 1: Data Collection

#### For CheXpert Presence Protocol:

```python
import pandas as pd

# Load CheXpert metadata
df = pd.read_csv('chexpert/train.csv')

# Filter samples where findings are explicitly present
# (CheXpert labels: 1.0 = present, 0.0 = absent, -1.0 = uncertain)
presence_df = df[df['Consolidation'] == 1.0]

# Build question-option tuples
records = []
for idx, row in presence_df.iterrows():
    record = {
        'sample_id': f"chexpert_presence_{idx}",
        'study_id': row['Study'],
        'image_paths': [row['Path']],
        'question': "Which finding is present on this chest X-ray?",
        'options': [
            "Consolidation",
            "Pleural Effusion",
            "No consolidation"
        ],
        'gold_letter': "A",  # Consolidation
        'gold_finding': "Consolidation",
    }
    records.append(record)
```

#### For OpenI Retrospective Audit:

```python
import re
from xml.etree import ElementTree as ET

# Parse OpenI reports
tree = ET.parse('openi/indiana_reports.xml')
root = tree.getroot()

records = []
for report in root.findall('.//REPORT'):
    text = report.text.lower() if report.text else ""

    # Extract findings using patterns
    if re.search(r'\bno evidence of pneumothorax\b', text):
        # Finding is absent
        record = {
            'sample_id': f"openi_absence_{len(records)}",
            'image_paths': [...],
            'question': "Which finding is absent?",
            'options': ["Pneumothorax", "Atelectasis", "No pneumothorax"],
            'gold_letter': "C",  # No pneumothorax
        }
        records.append(record)

    elif re.search(r'\bpneumothorax\b', text) and not re.search(r'\bno.*pneumothorax\b', text):
        # Finding is present
        record = {
            'sample_id': f"openi_presence_{len(records)}",
            'image_paths': [...],
            'question': "Which finding is present?",
            'options': ["Pneumothorax", "Pleural Effusion", "No pneumothorax"],
            'gold_letter': "A",  # Pneumothorax
        }
        records.append(record)
```

### Phase 2: Option Layout Randomization

Ensure deterministic, varied layouts:

```python
import random
import hashlib

def deterministic_shuffle(sample_id, items):
    """Shuffle deterministically based on sample_id."""
    seed = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
    random.Random(seed).shuffle(items)
    return items

# For each sample, shuffle options deterministically
for record in records:
    # Original: [target, distractor, negative]
    options = [
        "Consolidation",      # target
        "Pleural Effusion",    # distractor
        "No consolidation"     # negative
    ]

    shuffled = deterministic_shuffle(record['sample_id'], options)
    record['options'] = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(shuffled)]

    # Update gold_letter to match new position
    for i, opt in enumerate(shuffled):
        if opt == "Consolidation":
            record['gold_letter'] = chr(65 + i)
            break
```

### Phase 3: Validation

Validate protocol before use:

```python
def validate_protocol(records):
    """Check protocol integrity."""
    errors = []

    for i, record in enumerate(records):
        # Check required fields
        required = ['sample_id', 'question', 'options', 'gold_letter']
        for field in required:
            if field not in record:
                errors.append(f"Record {i}: Missing {field}")

        # Check gold_letter is valid
        options = record.get('options', [])
        gold = record.get('gold_letter', '')
        num_options = len(options)

        if not (0 <= ord(gold) - 65 < num_options):
            errors.append(f"Record {i}: Invalid gold_letter {gold}")

        # Check uniqueness
        if len(set(record['sample_id'])) != len(records):
            errors.append(f"Duplicate sample_id detected")

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    print(f"✓ Protocol valid: {len(records)} records")
    return True
```

## Building CheXpert Presence Protocol

```bash
python scripts/build_protocols.py \
  --dataset chexpert \
  --split train \
  --findings Consolidation Pneumothorax "Pleural Effusion" \
  --question-type presence \
  --output data/protocols/chexpert_presence_full.json
```

This creates ~135k presence questions from CheXpert training split.

## Building OpenI Audits

```bash
python scripts/build_protocols.py \
  --dataset openi \
  --report-file openi/indiana_reports.xml \
  --extraction-type presence_absence \
  --output-presence data/protocols/openi_presence_audit.json \
  --output-absence data/protocols/openi_absence_audit.json
```

## Minimal Example Protocol

For testing and development, use a minimal protocol:

```json
{
  "records": [
    {
      "sample_id": "example_001",
      "study_id": "study_001",
      "question": "Which finding is present on this chest X-ray?",
      "options": ["A. Consolidation", "B. Pleural Effusion", "C. No consolidation"],
      "gold_letter": "A",
      "image_paths": ["path/to/image.jpg"],
      "metadata": {"task_name": "presence_probe"}
    },
    {
      "sample_id": "example_002",
      "study_id": "study_002",
      "question": "Which finding is absent?",
      "options": ["A. Edema", "B. Atelectasis", "C. No edema"],
      "gold_letter": "C",
      "image_paths": ["path/to/image2.jpg"],
      "metadata": {"task_name": "absence_probe"}
    }
  ]
}
```

## Storage and Organization

Recommended directory structure:

```
data/
├── raw/
│   ├── chexpert/              # Raw CheXpert dataset
│   ├── openi/                 # Raw OpenI dataset
│   └── rexvqa/                # Raw ReXVQA (if available)
├── processed/
│   ├── chexpert_metadata.csv  # Extracted metadata
│   └── openi_reports.json     # Parsed reports
└── protocols/
    ├── chexpert_presence_direct.json      # Direct probe
    ├── chexpert_presence_full.json        # Large-scale
    ├── openi_presence_audit.json          # Retrospective
    └── openi_absence_audit.json
```

## Memory Requirements

- CheXpert: ~300 GB (full resolution images)
- OpenI: ~5 GB
- CheXpert processed metadata: ~50 MB
- Protocol JSON files: ~50-500 MB each

## Notes

- Images are not included in this repository (too large)
- All protocols are stored as JSON for portability
- Image paths in protocols are relative or use environment variables
- Protocols can be shared without images by using only metadata
