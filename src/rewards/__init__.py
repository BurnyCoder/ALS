# Local: file src/rewards/__init__.py provides first-party ALS source context. Global: exposes package helpers used across the ALS evaluation pipeline.
from .reward import RewardModel  # Local: imports selected helpers from .reward. Global: exposes package helpers used across the ALS evaluation pipeline.

__all__ = ["RewardModel"]  # Local: sets __all__ for later use in this scope. Global: exposes package helpers used across the ALS evaluation pipeline.
