# RL Pipeline for PII Sharing Decisions

A clean, modular pipeline for training and testing RL algorithms for PII sharing decisions.

## Project Poster

**[View Project Poster PDF](690F_AirGapLite_Poster.pdf)**



##  Project Structure

```
final_project/
├── common/                    #  Shared code (plug-and-play)
│   ├── config.py             # Configuration constants
│   └── mdp.py                # MDP helpers and utilities
│
├── algorithms/                #  RL algorithms 
│   ├── grpo/                 # GRPO: Per-PII binary actions
│   ├── groupedppo/           # GroupedPPO: Group actions + PPO
│   └── vanillarl/            # VanillaRL: Group actions + REINFORCE
│
├── pipeline/                  #  Unified pipeline
│   ├── algorithm_registry.py # Algorithm registry
│   ├── train.py              # Training pipeline (with convergence)
│   └── test.py               # Testing pipeline
│
└── scripts/                   #  Analysis scripts
    └── compare_all.py        # Compare all algorithms
```

##  Quick Start

### Train Algorithm (Recommended Settings)

```bash
cd final_project
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

**Note**: All directives will show perfect matches (Utility=1.0, Privacy=1.0) with this dataset because SSN/CREDIT_CARD have 90.3% frequency, resulting in learned probabilities >0.98.

### Test Algorithm

```bash
cd final_project
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive accurately \
    --get-regex
```

### Compare All Algorithms

```bash
cd final_project
python scripts/compare_all.py \
    --algorithms grpo groupedppo vanillarl \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir results
```

##  Training Options

### Convergence Detection (Recommended)

```bash
python pipeline/train.py \
    --algorithm grpo \
    --convergence_threshold 0.001 \  # Minimum improvement
    --patience 20 \                  # Iterations without improvement
    --max_iters 1000                 # Safety limit
```

Training stops when:
- No improvement > threshold for `patience` evaluations, OR
- Reaches `max_iters` iterations

### Fixed Iterations (Recommended for Tradeoff Dataset)

```bash
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

##  Available Algorithms

All algorithms use **per-PII binary actions** (0=don't share, 1=share) for each of the 11 PII types:
- **grpo**: Per-PII binary actions with group-based rewards + PPO-style updates with KL regularization
- **groupedppo**: Per-PII binary actions with group-based rewards + PPO with clipping
- **vanillarl**: Per-PII binary actions with group-based rewards + REINFORCE (simplest)

### MDP State-Action Space

**State**: `[present_mask (11), scenario_one_hot (2)]` = 13 dimensions
- `present_mask`: Binary vector indicating which PII types are present (NAME, PHONE, EMAIL, DATE/DOB, company, location, IP, SSN, CREDIT_CARD, age, sex)
- `scenario_one_hot`: One-hot encoding of domain (restaurant=[1,0], bank=[0,1])
- **Important**: The model NEVER sees `allowed_mask` in the state - it must learn domain-specific patterns from rewards

**Action**: Binary vector of length 11
- Each element: 0 (don't share) or 1 (share) for each PII type
- Example: `[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]` means share PHONE and EMAIL only

**Reward**: Group-based reward computation
- Reward computed per PII group (identity, contact, financial, network, org, demographic)
- Formula: `R = α·utility + β·privacy - complexity_penalty`
- Domain weights: Restaurant (α=0.6, β=0.4), Bank (α=0.7, β=0.3)

## 📈 Outputs

### Training
- `models/{algorithm}_model.pt`: Trained model
- `models/{algorithm}_history.json`: Training history

### Testing
- `evaluation_results.json`: Detailed metrics

### Comparison
- `training_curves.png`: Learning progress
- `utility_privacy.png`: Tradeoff visualization
- `performance.png`: Bar charts
- `comparison_table.csv`: Numerical comparison

##  Adding a New Algorithm

1. Create `algorithms/my_algorithm/` with:
   - `policy.py`: Policy network
   - `train.py`: Training functions
   - `__init__.py`: Exports

2. Register in `pipeline/algorithm_registry.py`

3. Use: `python pipeline/train.py --algorithm my_algorithm`

##  Common Code

All algorithms share:
- `common/config.py`: PII types, groups, scenarios
- `common/mdp.py`: State building, rewards, actions

Update once, all algorithms benefit!

##  Documentation

- **`HOW_TO_RUN.md`**: Complete guide on running all code, scripts, and workflows
- **`ALGORITHM_EXPLANATION.md`**: Detailed explanation of MDP, algorithms, and training flow

##  Dataset

**Finalized Dataset**: `690-Project-Dataset-final.csv` (Recommended)
- **Size**: 15,805 rows
- **Purpose**: Complete dataset with proper PII frequencies for learning domain patterns
- **Key Features**:
  - EMAIL: 98.7% frequency → learned prob >0.99 (shared by all directives)
  - PHONE: 60.8% frequency → learned prob >0.99 (shared by all directives)
  - DATE/DOB: 56.7% frequency → learned prob >0.99 (shared by all directives)
  - SSN: 90.3% frequency → learned prob >0.98 (shared by all directives)
  - CREDIT_CARD: 90.3% frequency → learned prob >0.98 (shared by all directives)
  - **100% coverage**: All rows with SSN/CREDIT_CARD in ground_truth also have them in allowed_bank
- **Expected Results** (Bank Domain):
  - **STRICTLY** (≥0.7): Utility = 1.0, Privacy = 1.0 ✓ Perfect match
  - **BALANCED** (≥0.5): Utility = 1.0, Privacy = 1.0 ✓ Perfect match
  - **ACCURATELY** (≤0.3): Utility = 1.0, Privacy = 1.0 ✓ Perfect match
- **Expected Patterns**:
  - Restaurant: EMAIL, PHONE (all directives)
  - Bank: EMAIL, PHONE, DATE/DOB, SSN, CREDIT_CARD (all directives)

##  Latest Updates

### State-Action Space (All Algorithms)
- **State**: `[present_mask (11), scenario_one_hot (2)]` = 13 dimensions
  - `present_mask`: Binary vector for 11 PII types (NAME, PHONE, EMAIL, DATE/DOB, company, location, IP, SSN, CREDIT_CARD, age, sex)
  - `scenario_one_hot`: Domain encoding (restaurant=[1,0], bank=[0,1])
  - **No `allowed_mask` in state** - model learns patterns from rewards

- **Action**: Binary vector of length 11
  - Each element: 0 (don't share) or 1 (share) for each PII type
  - All algorithms (GRPO, GroupedPPO, VanillaRL) use the same per-PII binary action space

- **Reward**: Group-based computation
  - Computed per PII group (identity, contact, financial, network, org, demographic)
  - Formula: `R = α·utility + β·privacy - complexity_penalty`
  - Domain weights: Restaurant (α=0.6, β=0.4), Bank (α=0.7, β=0.3)

### Training Pipeline Features
- **Fixed Iterations**: Recommended 300 iterations with batch_size=64 for tradeoff dataset
- **Unified Interface**: Same training/test commands for all algorithms
- **Model Storage**: Models saved to `models/{algorithm}_model.pt`
- **Training History**: Saved to `models/{algorithm}_history.json`

### Testing Pipeline Features
- **Directive System**: Control utility-privacy tradeoff with `--directive` (strictly/balanced/accurately)
  - `strictly`: High threshold (≥0.7), lower utility, higher privacy
  - `balanced`: Default threshold (0.5), balanced tradeoff
  - `accurately`: Low threshold (≤0.3), higher utility, lower privacy
- **Regex Extraction**: Get learned patterns with `--get-regex` flag
- **Domain Evaluation**: Separate metrics for restaurant and bank domains
- **Utility-Privacy Metrics**: Comprehensive evaluation showing tradeoff across all directives
- **No Dataset Required**: Utility/privacy calculated from model's derived regex pattern (not from dataset)
