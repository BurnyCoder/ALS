"""Expose solver prompt builders as the public `prompts` package API."""

# Re-exporting these builders lets `data.py` import prompt functions from `prompts` directly.
from .solver_prompts import gsm8k_prompt, MATH_500_prompt, AIME_2024_prompt
