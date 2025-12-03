import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load all CSV files
results_dir = "output/results/mac_run"
csv_files = list(Path(results_dir).glob("*.csv"))

print(f"Found {len(csv_files)} CSV files")
print("="*70)

# Load the combined results if it exists
combined_file = Path(results_dir) / "results_all_models_combined_full.csv"
if combined_file.exists():
    df = pd.read_csv(combined_file)
    print(f"Loaded combined file: {len(df)} samples")
else:
    # Load and combine all individual files
    dfs = []
    for csv_file in csv_files:
        if 'combined' not in csv_file.name and 'summary' not in csv_file.name:
            temp_df = pd.read_csv(csv_file)
            dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(dfs)} files: {len(df)} samples")

# Clean model names for better display
df['model_short'] = df['model_name'].str.replace('mlx-community/', '').str.replace('-4bit', '')

print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Total samples: {len(df)}")
print(f"Models tested: {df['model_short'].nunique()}")
print(f"Samples per model: {df.groupby('model_short').size().to_dict()}")

# Calculate summary statistics
summary = df.groupby('model_short').agg({
    'privacy_score': ['mean', 'std'],
    'utility_score': ['mean', 'std'],
    'time': ['mean', 'sum']
}).round(3)

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(summary)

# Create output directory for plots
plot_dir = Path("output/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PLOT 1: Privacy vs Utility Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

model_stats = df.groupby('model_short').agg({
    'privacy_score': 'mean',
    'utility_score': 'mean'
}).reset_index()

models = model_stats['model_short']
privacy_scores = model_stats['privacy_score'] * 100
utility_scores = model_stats['utility_score'] * 100

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, privacy_scores, width, label='Privacy', alpha=0.8)
bars2 = ax.bar(x + width/2, utility_scores, width, label='Utility', alpha=0.8)

# Add paper benchmark line
ax.axhline(y=97, color='r', linestyle='--', alpha=0.5, label='Paper Privacy (97%)')
ax.axhline(y=87, color='b', linestyle='--', alpha=0.5, label='Paper Utility (87%)')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Privacy vs Utility Performance Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(plot_dir / "1_privacy_utility_comparison.png", dpi=300, bbox_inches='tight')
print(f"\nSaved: {plot_dir}/1_privacy_utility_comparison.png")
plt.close()

# ============================================================================
# PLOT 2: Privacy and Utility Distribution (Box Plot)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Privacy distribution
privacy_data = [df[df['model_short'] == model]['privacy_score'].values * 100 
                for model in df['model_short'].unique()]
bp1 = ax1.boxplot(privacy_data, labels=df['model_short'].unique(), patch_artist=True)
ax1.set_title('Privacy Score Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Privacy Score (%)', fontsize=12)
ax1.set_xlabel('Model', fontsize=12)
ax1.axhline(y=97, color='r', linestyle='--', alpha=0.5, label='Paper Benchmark (97%)')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Utility distribution
utility_data = [df[df['model_short'] == model]['utility_score'].values * 100 
                for model in df['model_short'].unique()]
bp2 = ax2.boxplot(utility_data, labels=df['model_short'].unique(), patch_artist=True)
ax2.set_title('Utility Score Distribution', fontsize=14, fontweight='bold')
ax2.set_ylabel('Utility Score (%)', fontsize=12)
ax2.set_xlabel('Model', fontsize=12)
ax2.axhline(y=87, color='b', linestyle='--', alpha=0.5, label='Paper Benchmark (87%)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Color the boxes
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
for patch in bp2['boxes']:
    patch.set_facecolor('lightgreen')

plt.tight_layout()
plt.savefig(plot_dir / "2_score_distributions.png", dpi=300, bbox_inches='tight')
print(f"Saved: {plot_dir}/2_score_distributions.png")
plt.close()

# ============================================================================
# PLOT 3: Privacy-Utility Trade-off Scatter
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

for model in df['model_short'].unique():
    model_data = df[df['model_short'] == model]
    ax.scatter(model_data['privacy_score'] * 100, 
               model_data['utility_score'] * 100,
               alpha=0.6, s=50, label=model)

# Add paper benchmark point
ax.scatter(97, 87, color='red', marker='*', s=500, 
           edgecolors='black', linewidths=2, 
           label='AirGapAgent Paper', zorder=5)

ax.set_xlabel('Privacy Score (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Utility Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Privacy-Utility Trade-off', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 105])
ax.set_ylim([0, 105])

# Add ideal quadrant shading
ax.axvspan(90, 105, alpha=0.1, color='green')
ax.axhspan(90, 105, alpha=0.1, color='green')

plt.tight_layout()
plt.savefig(plot_dir / "3_privacy_utility_tradeoff.png", dpi=300, bbox_inches='tight')
print(f"Saved: {plot_dir}/3_privacy_utility_tradeoff.png")
plt.close()

# ============================================================================
# PLOT 4: Performance Over Samples (Running Average)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

for model in df['model_short'].unique():
    model_data = df[df['model_short'] == model].sort_values('sample_id')
    
    # Running average with window of 20
    window = 20
    privacy_smooth = model_data['privacy_score'].rolling(window=window, min_periods=1).mean() * 100
    utility_smooth = model_data['utility_score'].rolling(window=window, min_periods=1).mean() * 100
    
    ax1.plot(model_data['sample_id'], privacy_smooth, label=model, linewidth=2)
    ax2.plot(model_data['sample_id'], utility_smooth, label=model, linewidth=2)

ax1.axhline(y=97, color='r', linestyle='--', alpha=0.5, label='Paper Benchmark')
ax1.set_title('Privacy Score Over Samples (20-sample moving average)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Privacy Score (%)', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

ax2.axhline(y=87, color='b', linestyle='--', alpha=0.5, label='Paper Benchmark')
ax2.set_title('Utility Score Over Samples (20-sample moving average)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Sample ID', fontsize=12)
ax2.set_ylabel('Utility Score (%)', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / "4_performance_over_samples.png", dpi=300, bbox_inches='tight')
print(f"Saved: {plot_dir}/4_performance_over_samples.png")
plt.close()

# ============================================================================
# PLOT 5: Heatmap of Model Performance
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = df.groupby('model_short')[['privacy_score', 'utility_score']].mean() * 100
heatmap_data.columns = ['Privacy', 'Utility']

sns.heatmap(heatmap_data.T, annot=True, fmt='.1f', cmap='RdYlGn', 
            vmin=0, vmax=100, cbar_kws={'label': 'Score (%)'}, ax=ax)
ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Metric', fontsize=12)

plt.tight_layout()
plt.savefig(plot_dir / "5_performance_heatmap.png", dpi=300, bbox_inches='tight')
print(f"Saved: {plot_dir}/5_performance_heatmap.png")
plt.close()

# ============================================================================
# PLOT 6: Model Ranking
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

model_stats = df.groupby('model_short').agg({
    'privacy_score': 'mean',
    'utility_score': 'mean'
}).reset_index()

# Calculate composite score (average of privacy and utility)
model_stats['composite_score'] = (model_stats['privacy_score'] + model_stats['utility_score']) / 2 * 100
model_stats = model_stats.sort_values('composite_score', ascending=True)

bars = ax.barh(model_stats['model_short'], model_stats['composite_score'], 
               color=plt.cm.viridis(model_stats['composite_score']/100))

ax.set_xlabel('Composite Score (%) = (Privacy + Utility) / 2', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Overall Model Ranking', fontsize=14, fontweight='bold')
ax.axvline(x=92, color='r', linestyle='--', alpha=0.5, label='Paper Avg (92%)')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(plot_dir / "6_model_ranking.png", dpi=300, bbox_inches='tight')
print(f"Saved: {plot_dir}/6_model_ranking.png")
plt.close()

# ============================================================================
# Generate Summary Report
# ============================================================================
report_file = plot_dir / "analysis_report.txt"
with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("AIRGAP MINIMIZER - ANALYSIS REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Total Samples Analyzed: {len(df)}\n")
    f.write(f"Models Tested: {df['model_short'].nunique()}\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("="*70 + "\n")
    f.write("MODEL PERFORMANCE SUMMARY\n")
    f.write("="*70 + "\n\n")
    
    for model in df['model_short'].unique():
        model_data = df[df['model_short'] == model]
        f.write(f"{model}:\n")
        f.write(f"  Samples: {len(model_data)}\n")
        f.write(f"  Privacy: {model_data['privacy_score'].mean()*100:.2f}% ± {model_data['privacy_score'].std()*100:.2f}%\n")
        f.write(f"  Utility: {model_data['utility_score'].mean()*100:.2f}% ± {model_data['utility_score'].std()*100:.2f}%\n")
        f.write(f"  Total Time: {model_data['time'].sum()/60:.1f} minutes\n")
        f.write(f"  Avg Time/Sample: {model_data['time'].mean():.2f}s\n\n")
    
    f.write("="*70 + "\n")
    f.write("COMPARISON TO AIRGAPAGENT PAPER\n")
    f.write("="*70 + "\n\n")
    f.write("Paper Benchmarks:\n")
    f.write("  Privacy: 97%\n")
    f.write("  Utility: 87%\n\n")
    
    f.write("Your Best Results:\n")
    best_privacy_model = df.groupby('model_short')['privacy_score'].mean().idxmax()
    best_utility_model = df.groupby('model_short')['utility_score'].mean().idxmax()
    
    f.write(f"  Best Privacy: {best_privacy_model} ({df[df['model_short']==best_privacy_model]['privacy_score'].mean()*100:.2f}%)\n")
    f.write(f"  Best Utility: {best_utility_model} ({df[df['model_short']==best_utility_model]['utility_score'].mean()*100:.2f}%)\n")

print(f"\nSaved: {report_file}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"All plots saved to: {plot_dir}")
print(f"Generated {len(list(plot_dir.glob('*.png')))} plots")
print(f"Analysis report: {report_file}")
