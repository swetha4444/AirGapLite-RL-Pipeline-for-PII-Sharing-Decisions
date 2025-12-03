"""
Compare all algorithms - generates comprehensive comparison graphs and tables.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json
import time

from pipeline import AlgorithmRegistry, TrainingPipeline, EvaluationPipeline
from common.config import NUM_PII, SCENARIO_NAME2ID
from common.mdp import build_state


class AlgorithmComparator:
    """Compare all registered algorithms."""
    
    def __init__(self, dataset_path: str):
        """Initialize with dataset."""
        self.dataset_path = dataset_path
        self.results = {}
        self.training_curves = {}
        
    def train_algorithm(self, algorithm: str, num_iters: int = 200, batch_size: int = 64, log_every: int = 10):
        """Train a single algorithm."""
        print(f"\n{'='*70}")
        print(f"TRAINING: {algorithm.upper()}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Create training pipeline
        pipeline = TrainingPipeline(algorithm)
        pipeline.load_data(self.dataset_path)
        pipeline.train(num_iters, batch_size, log_every)
        
        training_time = time.time() - start_time
        
        # Store results
        self.training_curves[algorithm] = {
            'iterations': pipeline.training_history['iterations'],
            'rewards': pipeline.training_history['rewards'],
            'training_time': training_time
        }
        
        # Final evaluation
        final_reward = pipeline.evaluate(pipeline.policy, pipeline.dataset_rows, num_samples=500)
        
        # Store model and results
        self.results[algorithm] = {
            'policy': pipeline.policy,
            'dataset_rows': pipeline.dataset_rows,
            'final_reward': final_reward,
            'training_time': training_time,
        }
        
        print(f" {algorithm.upper()} training completed in {training_time:.1f}s")
        print(f"   Final average reward: {final_reward:.4f}\n")
        
        return pipeline.policy
    
    def evaluate_utility_privacy(self, algorithm: str, domain: str, num_samples: int = 200, directive: str = 'balanced', threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate utility-privacy metrics (no dataset needed - uses model's derived regex)."""
        eval_pipeline = EvaluationPipeline(algorithm)
        eval_pipeline.policy = self.results[algorithm]['policy']
        # No need for dataset_rows - utility calculation uses fixed expected patterns
        return eval_pipeline.evaluate_utility_privacy(domain, num_samples, directive=directive, threshold=threshold)
    
    def evaluate_with_directives(self, algorithm: str, domain: str, num_samples: int = 200) -> Dict[str, Dict]:
        """
        Evaluate utility-privacy metrics for each directive.
        
        Utility = fraction of expected regex pattern that model shares
        Privacy = 1 - (extra shared / total disallowed) - penalty for sharing extra PII
        
        Expected regex patterns (fixed):
        - Restaurant: EMAIL, PHONE
        - Bank: CREDIT_CARD, DATE/DOB, EMAIL, PHONE, SSN
        """
        from common.config import PII_TYPES, NUM_PII, SCENARIO_NAME2ID
        from common.mdp import build_state
        
        policy = self.results[algorithm]['policy']
        scenario_id = SCENARIO_NAME2ID[domain]
        
        # Expected regex patterns (fixed)
        expected_patterns = {
            'restaurant': {'EMAIL', 'PHONE'},
            'bank': {'CREDIT_CARD', 'DATE/DOB', 'EMAIL', 'PHONE', 'SSN'}
        }
        expected_pii_set = expected_patterns[domain]
        expected_pii_list = sorted(expected_pii_set)
        
        # Get model's actual output pattern (all PII present, with directive)
        present_mask_all = [1] * NUM_PII
        state_all = build_state(present_mask_all, scenario_id)
        
        results = {}
        directives = ['strictly', 'balanced', 'accurately']
        
        for directive in directives:
            # Get model's actual regex pattern for this directive
            actions_all = policy.act(state_all, deterministic=True, threshold=0.5, directive=directive)
            model_pii_set = set([PII_TYPES[i] for i, a in enumerate(actions_all) if a == 1])
            model_pii_list = sorted(model_pii_set)
            
            # Utility = how much of expected pattern does model share?
            shared_expected = expected_pii_set & model_pii_set
            if expected_pii_set:
                utility = len(shared_expected) / len(expected_pii_set)
            else:
                utility = 1.0 if len(model_pii_set) == 0 else 0.0
            
            # Privacy = penalty for sharing extra (not expected) PII
            extra_shared = model_pii_set - expected_pii_set
            # Total disallowed = all PII types not in expected
            all_pii_set = set(PII_TYPES)
            disallowed_set = all_pii_set - expected_pii_set
            
            if disallowed_set:
                privacy = 1.0 - (len(extra_shared) / len(disallowed_set))
            else:
                privacy = 1.0 if len(extra_shared) == 0 else 0.0
            
            results[directive] = {
                'utility': utility,
                'privacy': privacy,
                'expected_pattern': expected_pii_list,
                'model_pattern': model_pii_list,
                'shared_expected': sorted(shared_expected),
                'extra_shared': sorted(extra_shared),
            }
        
        return results
    
    def plot_training_curves(self, save_path: str = "comparison_training_curves.png"):
        """Plot training curves for all algorithms."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'grpo': 'blue', 'groupedppo': 'orange', 'vanillarl': 'green'}
        markers = {'grpo': 'o', 'groupedppo': 's', 'vanillarl': '^'}
        
        for algo in AlgorithmRegistry.list_algorithms():
            if algo in self.training_curves:
                curve = self.training_curves[algo]
                ax.plot(
                    curve['iterations'],
                    curve['rewards'],
                    label=algo.upper(),
                    color=colors.get(algo, 'gray'),
                    marker=markers.get(algo, 'o'),
                    markersize=6,
                    linewidth=2,
                    alpha=0.8
                )
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('Training Curves: Algorithm Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved training curves to: {save_path}")
        plt.close()
    
    def plot_utility_privacy_comparison(self, save_path: str = "comparison_utility_privacy.png"):
        """
        Plot utility-privacy comparison for each algorithm across directives.
        
        For each algorithm, shows 3 points (strictly, balanced, accurately) 
        showing how threshold affects utility-privacy tradeoff.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        colors = {'grpo': 'blue', 'groupedppo': 'orange', 'vanillarl': 'green'}
        directive_markers = {'strictly': '^', 'balanced': 'o', 'accurately': 's'}
        directive_colors = {'strictly': 'red', 'balanced': 'gold', 'accurately': 'green'}
        directive_labels = {'strictly': 'STRICTLY (≥0.7)', 'balanced': 'BALANCED (0.5)', 'accurately': 'ACCURATELY (≤0.3)'}
        
        for domain_idx, domain in enumerate(['restaurant', 'bank']):
            ax = axes[domain_idx]
            
            for algo in AlgorithmRegistry.list_algorithms():
                if algo in self.results:
                    algo_color = colors.get(algo, 'gray')
                    directive_results = self.evaluate_with_directives(algo, domain)
                    
                    # Plot each directive point for this algorithm
                    for directive in ['strictly', 'balanced', 'accurately']:
                        metrics = directive_results[directive]
                        utility = metrics['utility']
                        privacy = metrics['privacy']
                        
                        ax.scatter(
                            utility,
                            privacy,
                            color=algo_color,
                            marker=directive_markers[directive],
                            s=300,
                            alpha=0.8,
                            edgecolors=directive_colors[directive],
                            linewidths=2.5,
                            label=f'{algo.upper()} - {directive_labels[directive]}' if algo == list(self.results.keys())[0] else '',
                            zorder=5
                        )
                    
                    # Draw line connecting directives for this algorithm (strictly → balanced → accurately)
                    utilities = [directive_results[d]['utility'] for d in ['strictly', 'balanced', 'accurately']]
                    privacies = [directive_results[d]['privacy'] for d in ['strictly', 'balanced', 'accurately']]
                    ax.plot(
                        utilities,
                        privacies,
                        color=algo_color,
                        linestyle='--',
                        alpha=0.4,
                        linewidth=2.0,
                        zorder=1
                    )
            
            ax.set_xlabel('Utility (Fraction of Expected PII Shared)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Privacy (1 - Extra Shared / Disallowed)', fontsize=12, fontweight='bold')
            ax.set_title(f'{domain.capitalize()} Domain: Utility vs Privacy by Directive', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='best', framealpha=0.95, ncol=1)
            ax.grid(True, alpha=0.3, zorder=0)
            ax.set_xlim([0, 1.1])
            ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved utility-privacy comparison to: {save_path}")
        plt.close()
    
    def plot_bank_utility_tradeoff(self, save_path: str = "bank_utility_tradeoff.png"):
        """
        Plot utility tradeoff for BANK domain only - focusing on utility changes.
        Privacy is always 100% so we focus on utility vs directive.
        """
        trained_algorithms = [algo for algo in AlgorithmRegistry.list_algorithms() if algo in self.results]
        
        if not trained_algorithms:
            print(" No trained algorithms found for utility tradeoff graph")
            return
        
        colors = {'grpo': 'blue', 'groupedppo': 'orange', 'vanillarl': 'green'}
        directive_colors = {'strictly': 'red', 'balanced': 'gold', 'accurately': 'green'}
        directive_markers = {'strictly': '^', 'balanced': 'o', 'accurately': 's'}
        
        fig, ax = plt.subplots(figsize=(12, 8))
        domain = 'bank'
        
        for algo in trained_algorithms:
            algo_color = colors.get(algo, 'gray')
            directive_results = self.evaluate_with_directives(algo, domain)
            
            # Get utilities in order (strictly → balanced → accurately)
            utilities = [directive_results[d]['utility'] for d in ['strictly', 'balanced', 'accurately']]
            directives = ['strictly', 'balanced', 'accurately']
            
            # Draw line connecting directives
            x_positions = [0, 1, 2]  # Positions for strictly, balanced, accurately
            ax.plot(
                x_positions,
                utilities,
                color=algo_color,
                linestyle='-',
                alpha=0.8,
                linewidth=4.0,
                marker='o',
                markersize=12,
                label=f'{algo.upper()}',
                zorder=2
            )
            
            # Plot each directive point with annotations
            for idx, directive in enumerate(directives):
                utility = utilities[idx]
                ax.scatter(
                    x_positions[idx],
                    utility,
                    color=algo_color,
                    marker=directive_markers[directive],
                    s=600,
                    alpha=0.9,
                    edgecolors=directive_colors[directive],
                    linewidths=4.0,
                    zorder=5
                )
                
                # Add utility value annotation
                ax.annotate(
                    f'{utility:.3f}',
                    xy=(x_positions[idx], utility),
                    xytext=(0, 15),
                    textcoords='offset points',
                    fontsize=11,
                    color=directive_colors[directive],
                    fontweight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                            edgecolor=directive_colors[directive], linewidth=2)
                )
        
        # Set x-axis labels
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['STRICTLY\n(≥0.7)', 'BALANCED\n(0.5)', 'ACCURATELY\n(≤0.3)'], 
                          fontsize=12, fontweight='bold')
        ax.set_xlabel('Directive (Threshold)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Utility (Fraction of Allowed PII Shared)', fontsize=14, fontweight='bold')
        ax.set_title('Bank Domain: Utility Tradeoff Across Directives\n(Privacy = 100% for all directives)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_ylim([0, 1.1])
        
        # Add annotation
        annotation_text = (
            'Tradeoff Explanation:\n'
            '• STRICTLY (≥0.7): LOW utility\n'
            '  → Only shares highly probable PII\n\n'
            '• BALANCED (0.5): MEDIUM utility\n'
            '  → Balanced sharing\n\n'
            '• ACCURATELY (≤0.3): HIGH utility\n'
            '  → Shares even minimally probable PII\n\n'
            'Privacy is 100% for all directives\n'
            '(model correctly protects disallowed PII)'
        )
        ax.text(0.02, 0.98, 
               annotation_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, 
                       edgecolor='navy', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved bank utility tradeoff graph to: {save_path}")
        plt.close()
    
    def plot_performance_comparison(self, save_path: str = "comparison_performance.png"):
        """Plot performance comparison bar chart."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        algorithms = AlgorithmRegistry.list_algorithms()
        colors = {'grpo': 'blue', 'groupedppo': 'orange', 'vanillarl': 'green'}
        
        # Filter to only algorithms that were trained
        trained_algorithms = [a for a in algorithms if a in self.results]
        
        # 1. Final Reward
        ax = axes[0, 0]
        rewards = [self.results[a]['final_reward'] for a in trained_algorithms]
        labels = [a.upper() for a in trained_algorithms]
        bars = ax.bar(labels, rewards, color=[colors.get(a, 'gray') for a in trained_algorithms], alpha=0.7)
        ax.set_ylabel('Average Reward', fontsize=11)
        ax.set_title('Final Average Reward', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, reward in zip(bars, rewards):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{reward:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Training Time
        ax = axes[0, 1]
        times = [self.results[a]['training_time'] for a in trained_algorithms]
        bars = ax.bar(labels, times, color=[colors.get(a, 'gray') for a in trained_algorithms], alpha=0.7)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title('Training Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)
        
        # 3. Restaurant Utility (balanced directive)
        ax = axes[1, 0]
        utilities = []
        for a in trained_algorithms:
            metrics = self.evaluate_utility_privacy(a, 'restaurant', directive='balanced')
            utilities.append(metrics['utility'])
        bars = ax.bar(labels, utilities, color=[colors.get(a, 'gray') for a in trained_algorithms], alpha=0.7)
        ax.set_ylabel('Utility', fontsize=11)
        ax.set_title('Restaurant: Utility (Balanced Directive)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, util in zip(bars, utilities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{util:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Bank Utility (balanced directive)
        ax = axes[1, 1]
        utilities = []
        for a in trained_algorithms:
            metrics = self.evaluate_utility_privacy(a, 'bank', directive='balanced')
            utilities.append(metrics['utility'])
        bars = ax.bar(labels, utilities, color=[colors.get(a, 'gray') for a in trained_algorithms], alpha=0.7)
        ax.set_ylabel('Utility', fontsize=11)
        ax.set_title('Bank: Utility (Balanced Directive)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, util in zip(bars, utilities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{util:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved performance comparison to: {save_path}")
        plt.close()
    
    def generate_comparison_table(self, save_path: str = "comparison_table.csv"):
        """Generate comparison table with metrics for all directives."""
        data = []
        
        for algo in AlgorithmRegistry.list_algorithms():
            if algo in self.results:
                row = {'Algorithm': algo.upper()}
                row['Final Reward'] = self.results[algo]['final_reward']
                row['Training Time (s)'] = self.results[algo]['training_time']
                
                # Restaurant metrics for each directive
                rest_directives = self.evaluate_with_directives(algo, 'restaurant')
                for directive in ['strictly', 'balanced', 'accurately']:
                    metrics = rest_directives[directive]
                    row[f'Restaurant Utility ({directive})'] = metrics['utility']
                    row[f'Restaurant Privacy ({directive})'] = metrics['privacy']
                
                # Bank metrics for each directive
                bank_directives = self.evaluate_with_directives(algo, 'bank')
                for directive in ['strictly', 'balanced', 'accurately']:
                    metrics = bank_directives[directive]
                    row[f'Bank Utility ({directive})'] = metrics['utility']
                    row[f'Bank Privacy ({directive})'] = metrics['privacy']
                
                data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f" Saved comparison table to: {save_path}")
        print("\n" + df.to_string(index=False))
        
        return df
    
    def save_results(self, save_path: str = "comparison_results.json"):
        """Save all results to JSON."""
        results_dict = {}
        
        for algo in AlgorithmRegistry.list_algorithms():
            if algo in self.results:
                results_dict[algo] = {
                    'final_reward': float(self.results[algo]['final_reward']),
                    'training_time': float(self.results[algo]['training_time']),
                    'training_curve': self.training_curves[algo],
                    'restaurant_metrics': self.evaluate_utility_privacy(algo, 'restaurant', directive='balanced'),
                    'bank_metrics': self.evaluate_utility_privacy(algo, 'bank', directive='balanced'),
                    'restaurant_directives': self.evaluate_with_directives(algo, 'restaurant'),
                    'bank_directives': self.evaluate_with_directives(algo, 'bank'),
                }
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f" Saved results to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare all RL algorithms')
    parser.add_argument('--dataset', type=str, default='690-Project-Dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--num_iters', type=int, default=200,
                       help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log every N iterations')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='Output directory')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       choices=AlgorithmRegistry.list_algorithms(),
                       default=AlgorithmRegistry.list_algorithms(),
                       help='Algorithms to compare')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Dataset path
    dataset_path = args.dataset
    if not Path(dataset_path).exists():
        dataset_path = f"final_project/{args.dataset}"
    
    # Initialize comparator
    comparator = AlgorithmComparator(dataset_path)
    
    # Train all algorithms
    for algo in args.algorithms:
        comparator.train_algorithm(
            algo,
            num_iters=args.num_iters,
            batch_size=args.batch_size,
            log_every=args.log_every
        )
    
    # Generate comparisons
    print(f"\n{'='*70}")
    print("GENERATING COMPARISONS")
    print(f"{'='*70}\n")
    
    comparator.plot_training_curves(output_dir / "training_curves.png")
    comparator.plot_utility_privacy_comparison(output_dir / "utility_privacy.png")
    comparator.plot_bank_utility_tradeoff(output_dir / "bank_utility_tradeoff.png")
    comparator.plot_performance_comparison(output_dir / "performance.png")
    comparator.generate_comparison_table(output_dir / "comparison_table.csv")
    comparator.save_results(output_dir / "results.json")
    
    print(f"\n All comparisons saved to: {output_dir}/")


if __name__ == "__main__":
    main()

