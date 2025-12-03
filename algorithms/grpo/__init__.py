"""
GRPO Algorithm: Per-PII binary actions with group-based rewards.
"""

from .grpo_policy import RulePolicy
from .grpo_train import (
    load_dataset_from_excel,
    rollout_batch,
    policy_gradient_update,
    evaluate_average_reward,
)

__all__ = [
    'RulePolicy',
    'load_dataset_from_excel',
    'rollout_batch',
    'policy_gradient_update',
    'evaluate_average_reward',
]

