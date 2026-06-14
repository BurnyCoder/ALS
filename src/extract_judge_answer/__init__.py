# Local: file src/extract_judge_answer/__init__.py provides first-party ALS source context. Global: exposes package helpers used across the ALS evaluation pipeline.
# Local: starts a multi-line text literal that Python treats as one value. Global: exposes package helpers used across the ALS evaluation pipeline.
"""
Extract and judge answers for evaluation.
"""
# Local: imports selected helpers from .utils. Global: exposes package helpers used across the ALS evaluation pipeline.
from .utils import extract_answer, extract_true_answer, judge_answer
