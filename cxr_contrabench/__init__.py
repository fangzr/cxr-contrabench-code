"""
CXR-ContraBench: Benchmarking Negated-Option Attraction in Medical VLMs

A diagnostic benchmark for evaluating medical vision-language models (VLMs)
on their susceptibility to negated-option attraction - a failure mode where
models select negated answer options (e.g., "No consolidation") even when
the image and question indicate the finding is present.

This minimal benchmark package includes:
- Dataset loaders and builders for CheXpert, OpenI, and ReXVQA
- Evaluation runners and metrics
- QCCV-Neg (Question-Conditioned Consistency Verifier) for polarity repair
- Protocol builders for direct and retrospective evaluation
"""

__version__ = "0.1.0"
