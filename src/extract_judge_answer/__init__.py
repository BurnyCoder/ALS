"""Expose answer extraction and judging helpers for ALS evaluation.

The evaluation loop imports from this package so it can normalize model outputs
and compare them to dataset labels without depending on individual utility files.
"""

# Re-exporting these functions provides a compact public API for generation/evaluation modules.
from .utils import extract_answer, extract_true_answer, judge_answer
