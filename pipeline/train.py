"""
Unified Training Pipeline - Switch between any learning method easily.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Dict, Any
import json

from pipeline.algorithm_registry import AlgorithmRegistry


class TrainingPipeline:
    """Unified training pipeline for any registered algorithm."""
    
    def __init__(self, algorithm: str, config: Dict[str, Any] = None):
        """
        Initialize training pipeline.
        
        Args:
            algorithm: Algorithm name (e.g., 'grpo', 'groupedppo', 'vanillarl')
            config: Optional configuration overrides
        """
        self.algorithm_name = algorithm.lower()
        self.algo_config = AlgorithmRegistry.get(self.algorithm_name)
        self.config = config or {}
        
        # Create policy
        self.policy = AlgorithmRegistry.create_policy(self.algorithm_name)
        
        # Create optimizer
        self.optimizer = AlgorithmRegistry.create_optimizer(
            self.algorithm_name,
            self.policy,
            learning_rate=self.config.get('learning_rate', 3e-4)
        )
        
        # Get algorithm-specific functions
        self.load_dataset = self.algo_config['config']['load_dataset']
        self.rollout = self.algo_config['config']['rollout']
        self.update = self.algo_config['config']['update']
        self.evaluate = self.algo_config['config']['evaluate']
        self.update_kwargs = self.algo_config['config']['update_kwargs'].copy()
        
        # Override update kwargs if provided
        if 'update_kwargs' in self.config:
            self.update_kwargs.update(self.config['update_kwargs'])
        
        self.dataset_rows = None
        self.training_history = {
            'iterations': [],
            'rewards': [],
        }
    
    def load_data(self, dataset_path: str):
        """Load dataset."""
        print(f"Loading dataset from: {dataset_path}")
        self.dataset_rows = self.load_dataset(dataset_path)
        print(f" Loaded {len(self.dataset_rows)} rows")
    
    def train(self, num_iters: int = None, batch_size: int = 64, log_every: int = 10, 
              convergence_threshold: float = 0.001, patience: int = 20, max_iters: int = 1000):
        """
        Train the policy until convergence or max iterations.
        
        Args:
            num_iters: Deprecated - use max_iters instead. If provided, trains for fixed iterations.
            batch_size: Batch size for rollouts
            log_every: Log every N iterations
            convergence_threshold: Minimum improvement to consider progress (default: 0.001)
            patience: Number of iterations without improvement before stopping (default: 20)
            max_iters: Maximum number of iterations (default: 1000)
        """
        if self.dataset_rows is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        # Support legacy num_iters parameter
        if num_iters is not None:
            max_iters = num_iters
            convergence_threshold = None  # Disable convergence detection
        
        use_convergence = convergence_threshold is not None
        
        print(f"\n{'='*70}")
        if use_convergence:
            print(f"Training {self.algorithm_name.upper()} until convergence (max {max_iters} iterations)")
            print(f"Convergence: threshold={convergence_threshold}, patience={patience}")
        else:
            print(f"Training {self.algorithm_name.upper()} for {max_iters} iterations")
        print(f"{'='*70}\n")
        
        best_reward = float('-inf')
        no_improvement_count = 0
        
        for it in range(1, max_iters + 1):
            # Rollout batch
            batch = self.rollout(self.policy, self.dataset_rows, batch_size=batch_size)
            
            # Update policy
            self.update(self.policy, self.optimizer, batch, **self.update_kwargs)
            
            # Evaluate and check convergence
            if it % log_every == 0:
                avg_reward = self.evaluate(
                    self.policy,
                    self.dataset_rows,
                    num_samples=self.config.get('eval_samples', 200)
                )
                self.training_history['iterations'].append(it)
                self.training_history['rewards'].append(avg_reward)
                print(f"[Iter {it:4d}] Avg reward: {avg_reward:.4f}", end='')
                
                # Check convergence
                if use_convergence:
                    improvement = avg_reward - best_reward
                    if improvement > convergence_threshold:
                        best_reward = avg_reward
                        no_improvement_count = 0
                        print(f" (↑ {improvement:.4f})")
                    else:
                        no_improvement_count += 1
                        print(f" (no improvement: {no_improvement_count}/{patience})")
                        
                        if no_improvement_count >= patience:
                            print(f"\n Converged at iteration {it} (no improvement for {patience} evaluations)")
                            break
                else:
                    print()
        
        if use_convergence and no_improvement_count < patience:
            print(f"\n Training completed (reached max iterations: {max_iters})")
        elif not use_convergence:
            print(f"\n Training completed!")
    
    def save_model(self, save_path: str):
        """Save trained model."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), save_path)
        print(f" Model saved to: {save_path}")
    
    def save_history(self, save_path: str):
        """Save training history."""
        with open(save_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f" Training history saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified Training Pipeline')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=AlgorithmRegistry.list_algorithms(),
                       help=f'Algorithm to use: {AlgorithmRegistry.list_algorithms()}')
    parser.add_argument('--dataset', type=str, default='690-Project-Dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--num_iters', type=int, default=None,
                       help='[Deprecated] Number of training iterations (use --max_iters instead)')
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for rollouts')
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log every N iterations')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--convergence_threshold', type=float, default=0.001,
                       help='Minimum improvement to consider progress (0 to disable)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Iterations without improvement before stopping')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = {
        'learning_rate': args.learning_rate,
        'eval_samples': 200,
    }
    
    # Initialize pipeline
    pipeline = TrainingPipeline(args.algorithm, config)
    
    # Load data
    dataset_path = args.dataset
    if not Path(dataset_path).exists():
        dataset_path = f"final_project/{args.dataset}"
    pipeline.load_data(dataset_path)
    
    # Train
    convergence_threshold = args.convergence_threshold if args.convergence_threshold > 0 else None
    pipeline.train(
        num_iters=args.num_iters,  # Legacy support
        batch_size=args.batch_size,
        log_every=args.log_every,
        convergence_threshold=convergence_threshold,
        patience=args.patience,
        max_iters=args.max_iters
    )
    
    # Save model and history
    model_path = output_dir / f"{args.algorithm}_model.pt"
    history_path = output_dir / f"{args.algorithm}_history.json"
    pipeline.save_model(str(model_path))
    pipeline.save_history(str(history_path))


if __name__ == "__main__":
    main()

