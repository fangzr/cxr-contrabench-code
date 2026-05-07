# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd cxr-contrabench-minimal

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
python scripts/check_installation.py
```

## 5-Minute Example

Run a complete example with a mock model:

```bash
python examples/minimal_example.py
```

This demonstrates:
1. Loading a protocol
2. Running inference
3. Applying QCCV-Neg M1 repair
4. Computing metrics

**Expected output**:
```
Example 1: Load Protocol and Run Evaluation
...
Accuracy (before repair): 50.00%

Example 2: Apply QCCV-Neg M1 Repair
...
Accuracy change:
  Before repair: 50.00%
  After repair: 100.00%
  Delta: +50.00pp
```

## Evaluating Your Model

### Step 1: Prepare Data

First, download and prepare a benchmark protocol:

```bash
# Option A: Create a small test protocol (minimal, 2 samples)
python examples/minimal_example.py

# Option B: Prepare CheXpert (requires ~10 GB)
# See docs/DATASETS.md for full instructions
```

### Step 2: Run Evaluation

Create a simple evaluation script for your model:

```python
from cxr_contrabench.datasets import load_protocol
from cxr_contrabench.common import resolve_gold_letter, save_result_payload

def my_model_inference(question, options, image_path):
    """Your model's inference function."""
    # Load image
    from PIL import Image
    image = Image.open(image_path[0]) if image_path else None

    # Run your model
    # prediction = your_model.forward(image, question, options)
    # For now, just return first option
    return "A"

# Load protocol
protocol = load_protocol("path/to/protocol.json")

# Run evaluation
results = []
for record in protocol:
    question = record['question']
    options = record['options']
    image_path = record['image_paths']

    # Get gold answer
    gold_letter = resolve_gold_letter(record)

    # Get prediction
    pred_letter = my_model_inference(question, options, image_path)

    # Check correctness
    is_correct = (pred_letter == gold_letter)

    # Store result
    result_record = dict(record)
    result_record.update({
        'pred_letter': pred_letter,
        'gold_letter': gold_letter,
        'exact_match': is_correct,
    })
    results.append(result_record)

# Save results
save_result_payload(
    "results/my_model.json",
    summary={'model': 'my-model', 'total': len(results)},
    records=results
)
```

### Step 3: Apply QCCV-Neg Repair

```bash
python scripts/apply_verifier.py \
  --input results/my_model.json \
  --output results/my_model_repaired.json
```

Expected output:
```
Loading results from: results/my_model.json
  Total records: 235
  Negation contradictions (pre-repair): 153

Applying M1 repair (presence-side negation trap correction)...
  Negation contradictions (post-repair): 0

Accuracy:
  Pre-repair: 74/235 (31.49%)
  Post-repair: 227/235 (96.60%)
  Delta: +153 cases (+65.11pp)
```

### Step 4: Analyze Results

```bash
# View single result
python scripts/analyze_results.py \
  --input results/my_model.json

# Compare baseline vs repaired
python scripts/analyze_results.py \
  --baseline results/my_model.json \
  --compare results/my_model_repaired.json
```

## Core Concepts

### Negated-Option Attraction

A model selecting "No consolidation" when the image shows consolidation and the question asks "Which finding is present?"

### M1 Repair

When:
- Question asks about PRESENCE
- Model selected a NEGATED option
- A unique positive counterpart exists

Then: Replace with the positive option

### QCCV-Neg

Question-Conditioned Consistency Verifier for Negation
- Deterministic (no randomness)
- No retraining needed
- Fully auditable

## Benchmark Protocols

| Protocol | Samples | Type | Finding |
|----------|---------|------|---------|
| Direct Probe | 235 | Direct evaluation | CheXpert presence |
| Presence Audit | ~396 | OpenI reports | OpenI presence |
| Absence Audit | ~979 | OpenI reports | OpenI absence |
| Large-scale | 135k | CheXpert train | CheXpert findings |

See `docs/BENCHMARK.md` for details.

## Common Issues

### ImportError for cxr_contrabench

```bash
# Make sure you installed the package
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/cxr-contrabench-minimal
```

### Memory Issues with Large Datasets

- Use data preprocessing to create smaller subsets
- Process in batches with batch_size < 32
- Consider using 4-bit quantization for models

### Protocol File Not Found

Ensure:
1. Protocol JSON exists at the path
2. Path is absolute or relative from current directory
3. Image paths in protocol are valid (relative or absolute)

## Next Steps

1. **Read the full README**: See `README.md` for comprehensive documentation
2. **Benchmark specifications**: See `docs/BENCHMARK.md`
3. **Prepare your dataset**: See `docs/DATASETS.md`
4. **Explore the code**: Start with `cxr_contrabench/common.py`

## Getting Help

1. Check the README.md for detailed information
2. See `docs/` directory for specification documents
3. Look at `examples/minimal_example.py` for working code
4. Review docstrings in `cxr_contrabench/*.py` for function details

