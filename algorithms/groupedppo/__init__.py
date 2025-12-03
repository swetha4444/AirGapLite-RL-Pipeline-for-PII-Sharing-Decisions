"""
GroupedPPO Algorithm: Group-based actions (3 per group) with PPO updates.
"""

from .grpo_policy import RulePolicy
from .grpo_train import (
    load_dataset_from_excel,
    rollout_batch_grpo as rollout_batch,
    grpo_update as policy_gradient_update,
    evaluate_average_reward,
)

__all__ = [
    'RulePolicy',
    'load_dataset_from_excel',
    'rollout_batch',
    'policy_gradient_update',
    'evaluate_average_reward',
]

