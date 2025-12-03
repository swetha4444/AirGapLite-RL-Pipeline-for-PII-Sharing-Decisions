"""
Algorithm Registry: Plug-and-play interface for all RL algorithms.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Any, Type, Callable
import torch
import torch.optim as optim

from common.config import NUM_PII, NUM_SCENARIOS


class AlgorithmRegistry:
    """Registry for all RL algorithms."""
    
    _algorithms: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, policy_class: Type, config: Dict[str, Any]):
        """Register an algorithm."""
        cls._algorithms[name] = {
            'policy_class': policy_class,
            'config': config,
        }
    
    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        """Get algorithm configuration."""
        if name not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {name}. Available: {list(cls._algorithms.keys())}")
        return cls._algorithms[name]
    
    @classmethod
    def list_algorithms(cls) -> list:
        """List all registered algorithms."""
        return list(cls._algorithms.keys())
    
    @classmethod
    def create_policy(cls, name: str, **kwargs):
        """Create a policy instance for an algorithm."""
        algo = cls.get(name)
        policy_class = algo['policy_class']
        config = algo['config']
        
        # Merge config with kwargs
        init_kwargs = {**config.get('init_kwargs', {}), **kwargs}
        return policy_class(**init_kwargs)
    
    @classmethod
    def create_optimizer(cls, name: str, policy, learning_rate: float = 3e-4):
        """Create optimizer for an algorithm."""
        return optim.Adam(policy.parameters(), lr=learning_rate)


# Register all algorithms
def _register_algorithms():
    """Register all available algorithms."""
    from algorithms.grpo import RulePolicy as GRPOPolicy
    from algorithms.grpo import (
        load_dataset_from_excel as load_grpo,
        rollout_batch as rollout_grpo,
        policy_gradient_update as update_grpo,
        evaluate_average_reward as eval_grpo,
    )
    
    from algorithms.groupedppo import RulePolicy as GroupedPPOPolicy
    from algorithms.groupedppo import (
        load_dataset_from_excel as load_gppo,
        rollout_batch as rollout_gppo,
        policy_gradient_update as update_gppo,
        evaluate_average_reward as eval_gppo,
    )
    
    from algorithms.vanillarl import RulePolicy as VanillaRLPolicy
    from algorithms.vanillarl import (
        load_dataset_from_excel as load_vrl,
        rollout_batch as rollout_vrl,
        policy_gradient_update as update_vrl,
        evaluate_average_reward as eval_vrl,
    )
    
    # Register GRPO
    AlgorithmRegistry.register('grpo', GRPOPolicy, {
        'init_kwargs': {},
        'load_dataset': load_grpo,
        'rollout': rollout_grpo,
        'update': update_grpo,
        'evaluate': eval_grpo,
        'update_kwargs': {'epochs': 2},
    })
    
    # Register GroupedPPO
    AlgorithmRegistry.register('groupedppo', GroupedPPOPolicy, {
        'init_kwargs': {},
        'load_dataset': load_gppo,
        'rollout': rollout_gppo,
        'update': update_gppo,
        'evaluate': eval_gppo,
        'update_kwargs': {
            'ppo_epochs': 4,
            'clip_eps': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'kl_coef': 0.1,
        },
    })
    
    # Register VanillaRL
    AlgorithmRegistry.register('vanillarl', VanillaRLPolicy, {
        'init_kwargs': {},
        'load_dataset': load_vrl,
        'rollout': rollout_vrl,
        'update': update_vrl,
        'evaluate': eval_vrl,
        'update_kwargs': {'epochs': 3},
    })


# Auto-register on import
_register_algorithms()

