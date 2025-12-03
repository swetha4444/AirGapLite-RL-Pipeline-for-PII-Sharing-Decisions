"""
Unified Pipeline for RL Algorithms.

Easy to use, plug-and-play interface for training and testing any algorithm.
"""

from .algorithm_registry import AlgorithmRegistry
from .train import TrainingPipeline
from .test import EvaluationPipeline

__all__ = [
    'AlgorithmRegistry',
    'TrainingPipeline',
    'EvaluationPipeline',
]

