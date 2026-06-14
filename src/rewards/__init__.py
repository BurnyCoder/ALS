"""Expose the verifier-backed reward wrapper as the rewards package API."""

# Re-exporting `RewardModel` lets callers import `rewards.RewardModel` or `from rewards.reward`.
from .reward import RewardModel

__all__ = ["RewardModel"]
