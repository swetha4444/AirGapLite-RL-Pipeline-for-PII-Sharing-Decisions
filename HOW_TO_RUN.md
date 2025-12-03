#  Complete Guide: How to Run All Code

This guide covers all scripts and pipelines in the project.

##  Quick Setup

**First time setup**:
```bash
# 1. Create and activate environment
conda create -n overthink python=3.10
conda activate overthink

# 2. Install dependencies
cd final_project
pip install -r requirements.txt

# 3. Install spaCy model
python -m spacy download en_core_web_sm

# 4. (Optional) Install MLX for Apple Silicon baseline
pip install mlx mlx-lm
```

##  Quick Start Commands

**Train a single algorithm**:
```bash
cd final_project
python pipeline/train.py --algorithm grpo --dataset 690-Project-Dataset-final.csv --num_iters 300 --batch_size 64 --output_dir models
```

**Test/Evaluate a trained model**:
```bash
cd final_project
python pipeline/test.py --algorithm grpo --model models/grpo_model.pt --directive accurately --get-regex
```

**Train and compare all algorithms**:
```bash
cd final_project
python scripts/compare_all.py --algorithms grpo groupedppo vanillarl --dataset 690-Project-Dataset-final.csv --num_iters 300 --batch_size 64 --output_dir results
```

---

##  Prerequisites

### Environment Setup

1. **Python Environment**: Python 3.7+ (Python 3.10 recommended)
   - Using conda (recommended):
     ```bash
     conda create -n overthink python=3.10
     conda activate overthink
     ```
   - Or using venv:
     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

2. **Install Core Dependencies**:
   ```bash
   cd final_project
   pip install -r requirements.txt
   ```

3. **Install spaCy Language Model** (required for PII extraction):
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Optional: MLX for Apple Silicon** (for baseline minimizer comparison):
   ```bash
   pip install mlx mlx-lm
   ```
   - This enables the MLX baseline minimizer which works on Apple Silicon without a GPU
   - If you skip this, the comparison script will run with RL pipeline only

5. **Optional: GPU Support** (for GPU-based baseline minimizer):
   ```bash
   pip install bitsandbytes
   ```
   - Requires CUDA-capable GPU
   - Only needed if you want to use GPU baseline instead of MLX

### Dataset and Models

- **Dataset**: `690-Project-Dataset-final.csv` (finalized dataset) in the `final_project/` directory
- **Model Path**: `models/{algorithm}_model.pt` (created after training)
- **Training Settings**: `--num_iters 300 --batch_size 64` (recommended)

##  1. Training Algorithms

### 1.1 Train GRPO (Recommended)

**With Convergence Detection (Recommended)**:
```bash
cd final_project
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --convergence_threshold 0.001 \
    --patience 20 \
    --max_iters 1000 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --output_dir models
```

**Fixed Iterations** (Recommended for tradeoff dataset):
```bash
python pipeline/train.py \
    --algorithm grpo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

**Outputs**:
- `models/grpo_model.pt`: Trained model weights
- `models/grpo_history.json`: Training history (iterations, rewards)

### 1.2 Train GroupedPPO

```bash
python pipeline/train.py \
    --algorithm groupedppo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

### 1.3 Train VanillaRL

```bash
python pipeline/train.py \
    --algorithm vanillarl \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir models
```

### 1.4 Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--algorithm` | Algorithm name (grpo/groupedppo/vanillarl) | Required |
| `--dataset` | Path to dataset CSV | `690-Project-Dataset-final.csv` |
| `--max_iters` | Maximum training iterations | `1000` |
| `--batch_size` | Batch size for rollouts | `64` |
| `--learning_rate` | Learning rate | `3e-4` |
| `--convergence_threshold` | Min improvement to consider progress | `0.001` |
| `--patience` | Iterations without improvement before stopping | `20` |
| `--log_every` | Log every N iterations | `10` |
| `--output_dir` | Output directory for models | `models` |

---

##  2. Testing/Evaluation

### 2.1 Full Evaluation

**Basic Evaluation (with balanced directive)**:
```bash
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive balanced \
    --dataset 690-Project-Dataset-final.csv \
    --output evaluation_results.json 
```

**Evaluation with specific directive**:
```bash
# Strictly directive (high threshold, less utility)
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive strictly \
    --dataset 690-Project-Dataset-final.csv

# Accurately directive (low threshold, more utility)
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive accurately \
    --dataset 690-Project-Dataset-final.csv
```

**Output**: Shows:
- Average reward (if dataset provided)
- Restaurant domain metrics (utility, privacy, breach rate) for the specified directive
- Bank domain metrics (utility, privacy, breach rate) for the specified directive
- **Summary table** showing utility-privacy tradeoff for ALL directives (strictly, balanced, accurately)

### 2.2 Get Learned Regex Patterns

**Get regex for a specific directive**:
```bash
python pipeline/test.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive accurately \
    --get-regex
```

**Directives** (with tradeoff dataset):
- `strictly`: High threshold (≥0.7), more privacy, less utility
  - Bank: Only EMAIL, PHONE → Utility = 0.4
- `balanced`: Default threshold (0.5), balanced tradeoff
  - Bank: EMAIL, PHONE, DATE/DOB → Utility = 0.6
- `accurately`: Low threshold (≤0.3), more utility, less privacy
  - Bank: All 5 PII types (EMAIL, PHONE, DATE/DOB, SSN, CREDIT_CARD) → Utility = 1.0


### 2.3 Test Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--algorithm` | Algorithm name | Required |
| `--model` | Path to trained model | Required |
| `--dataset` | Path to dataset CSV (optional, only for reward evaluation) | `690-Project-Dataset-final.csv` |
| `--directive` | Directive for regex (strictly/balanced/accurately) | `balanced` |
| `--get-regex` | Flag to extract regex patterns only | `False` |
| `--output` | Output file for results | `evaluation_results.json` |

---

##  3. Analysis Scripts

### 3.1 Analyze Dataset Probabilities

**Check PII type frequencies in dataset**:
```bash
python scripts/analyze_dataset_probabilities.py \
    --dataset 690-Project-Dataset-final.csv
```

**Output**: Table showing:
- Probability of each PII type in restaurant domain
- Probability of each PII type in bank domain
- Differences between domains


### 3.2 Analyze with Directives

**Compare utility-privacy tradeoff across directives**:
```bash
python scripts/analyze_with_directives.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --dataset 690-Project-Dataset-final.csv
```

**Output**: 
- Utility, privacy, and breach rate for each directive
- Comparison table


### 3.4 Get Regex by Directive

**Get regex pattern for specific directive**:
```bash
python scripts/get_regex_by_directive.py \
    --algorithm grpo \
    --model models/grpo_model.pt \
    --directive accurately \
    --domain bank
```

**Parameters**:
- `--algorithm`: Algorithm name
- `--model`: Path to trained model
- `--directive`: strictly/balanced/accurately
- `--domain`: restaurant/bank

---

## 4. Compare All Algorithms

**Train and compare all algorithms** (recommended):
```bash
cd final_project
python scripts/compare_all.py \
    --algorithms grpo groupedppo vanillarl \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir results
```

**Compare specific algorithms**:
```bash
# Compare only GRPO and GroupedPPO
python scripts/compare_all.py \
    --algorithms grpo groupedppo \
    --dataset 690-Project-Dataset-final.csv \
    --num_iters 300 \
    --batch_size 64 \
    --output_dir results
```

**Note**: This will:
- Train all specified algorithms on the tradeoff dataset
- Generate utility-privacy tradeoff graphs showing how each directive affects utility/privacy for each algorithm
- Create comparison table with metrics for all directives (strictly, balanced, accurately)

**Parameters**:
- `--algorithms`: Space-separated list of algorithms (grpo, groupedppo, vanillarl)
- `--dataset`: Path to dataset CSV file
- `--num_iters`: Training iterations per algorithm (default: 200)
- `--batch_size`: Batch size for training (default: 64)
- `--log_every`: Log every N iterations (default: 10)
- `--output_dir`: Output directory for results (default: comparison_results)

**Outputs**:
- `results/training_curves.png`: Learning curves for all algorithms
- `results/utility_privacy.png`: Utility-privacy tradeoff by directive for each algorithm (shows 3 points per algorithm: strictly, balanced, accurately)
- `results/bank_directive_tradeoff.png`: Bank domain utility vs privacy across directives
- `results/performance.png`: Performance comparison bar charts
- `results/comparison_table.csv`: Numerical comparison table with metrics for all directives:
  - `Restaurant Utility (strictly)`, `Restaurant Privacy (strictly)`
  - `Restaurant Utility (balanced)`, `Restaurant Privacy (balanced)`
  - `Restaurant Utility (accurately)`, `Restaurant Privacy (accurately)`
  - Same columns for Bank domain
- `results/results.json`: Complete results in JSON format

---

##  5. File Structure and Outputs

### 5.1 Training Outputs

```
models/
├── grpo_model.pt              # Trained GRPO model
├── grpo_history.json          # Training history
├── groupedppo_model.pt        # Trained GroupedPPO model
├── groupedppo_history.json    # Training history
├── vanillarl_model.pt         # Trained VanillaRL model
└── vanillarl_history.json     # Training history
```

### 5.2 Evaluation Outputs

```
evaluation_results.json        # Full evaluation metrics
```

### 5.3 Comparison Outputs

```
results/
├── training_curves.png        # Learning curves for all algorithms
├── utility_privacy.png        # Utility-privacy tradeoff
├── performance.png            # Performance bar charts
└── comparison_table.csv       # Numerical comparison
```

---
## 6. Integration Pipeline: End-to-End PII Minimization

The integration pipeline takes a third-party prompt and user data, automatically detects the domain, gets allowed PII types from GRPO, and returns minimized data with only allowed PII.

### 6.1 Basic Usage

**Restaurant Domain Example**:
```bash
cd final_project
python pipeline/integration_pipeline.py \
  --prompt "I need to book a table" \
  --data "Hi, my name is John Smith. My email is john@example.com and you can reach me at 555-1234. My SSN is 123-45-6789."
```

**Bank Domain Example**:
```bash
python pipeline/integration_pipeline.py \
  --prompt "I need to check my account balance" \
  --data "Name: John, Email: john@example.com, Phone: 555-1234, SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111" \
  --algorithm groupedppo
```

### 6.2 With Custom Directives

```bash
# Strictly directive (higher privacy)
python pipeline/integration_pipeline.py \
  --prompt "Book a table" \
  --data "Email: user@example.com, Phone: 555-1234, SSN: 123-45-6789" \
  --directive strictly

# Accurately directive (higher utility)
python pipeline/integration_pipeline.py \
  --prompt "Check account balance" \
  --data "Email: user@example.com, SSN: 123-45-6789" \
  --directive accurately
```

### 6.3 JSON Output with Full Details

```bash
python pipeline/integration_pipeline.py \
  --prompt "Reserve a table" \
  --data "Name: Jane, Email: jane@example.com, Phone: 555-5678" \
  --json --full-details
```

### 6.4 Integration Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--prompt` | Third-party prompt/query | Required |
| `--data` | User data containing PII (raw text) | Required |
| `--algorithm` | RL algorithm (grpo/groupedppo/vanillarl) | `grpo` |
| `--directive` | Privacy directive (strictly/balanced/accurately) | `balanced` |
| `--classifier-model` | Path to context classifier model | `MLP/context_agent_mlp.pth` |
| `--grpo-model` | Path to GRPO model | `models/{algorithm}_model.pt` |
| `--json` | Output as JSON | `False` |
| `--full-details` | Return detailed breakdown | `False` |

### 6.5 Output Format

The pipeline returns:
- **Domain**: Detected domain (restaurant/bank) with confidence
- **Allowed PII Types**: List of PII types allowed for this domain
- **Minimized Data (Redacted Text)**: Original text with disallowed PII replaced with `[REDACTED]`
- **Minimized Data (Structured)**: Structured format with PII types and values

**Example Output**:
```
============================================================
INTEGRATION PIPELINE RESULTS
============================================================

Third-party Prompt: I need to book a table
Detected Domain: RESTAURANT (confidence: 99.17%)

Allowed PII Types: PHONE, EMAIL

Minimized Data (Redacted Text):
  Hi, my name is [REDACTED]. My email is john@example.com and you can reach me at 555-1234. My SSN is [REDACTED].

Minimized Data (Structured):
  PHONE: 555-1234, EMAIL: john@example.com
============================================================
```

### 6.6 Using as Python Function

```python
from pipeline.integration_pipeline import minimize_data

result = minimize_data(
    third_party_prompt="I need to book a table",
    user_data="Hi, my name is John Smith. My email is john@example.com and you can reach me at 555-1234. My SSN is 123-45-6789.",
    algorithm="grpo",
    directive="balanced"
)

print(result['minimized_data'])  # Redacted text
print(result['minimized_data_structured'])  # Structured format
```

---

## 7. Comparison: Baseline LLM Minimizer vs RL Integration Pipeline

Compare the original AirGap LLM-based minimizer with the RL-based integration pipeline on utility, privacy, and inference speed.

### 7.1 Basic Comparison

**Restaurant Domain**:
```bash
cd final_project
python pipeline/compare_baseline_vs_rl.py \
  --num-samples 5 \
  --domain restaurant
```

**Bank Domain with GroupedPPO**:
```bash
python pipeline/compare_baseline_vs_rl.py \
  --num-samples 10 \
  --domain bank \
  --rl-algorithm groupedppo \
  --output bank_comparison.csv
```

### 7.2 Using Custom Test Cases

Create a JSON file `test_cases.json`:
```json
[
  {
    "prompt": "I need to book a table",
    "user_data": "Hi, my name is John Smith. My email is john@example.com and you can reach me at 555-1234. My SSN is 123-45-6789.",
    "allowed_pii": ["PHONE", "EMAIL"],
    "domain": "restaurant"
  },
  {
    "prompt": "Check my account balance",
    "user_data": "Name: Jane Doe, Email: jane@example.com, Phone: 555-5678, SSN: 987-65-4321",
    "allowed_pii": ["PHONE", "EMAIL", "SSN"],
    "domain": "bank"
  }
]
```

Then run:
```bash
python pipeline/compare_baseline_vs_rl.py --test-cases test_cases.json
```

### 7.3 Comparison Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--test-cases` | JSON file with test cases | Auto-generate |
| `--num-samples` | Number of test samples to generate | `5` |
| `--domain` | Domain (restaurant/bank) | `restaurant` |
| `--baseline-model` | Baseline model name | `Qwen/Qwen2.5-7B-Instruct` |
| `--rl-algorithm` | RL algorithm (grpo/groupedppo/vanillarl) | `grpo` |
| `--rl-directive` | Privacy directive (strictly/balanced/accurately) | `balanced` |
| `--output` | Output CSV file | `comparison_results.csv` |

### 7.4 Comparison Metrics

The script compares:

1. **Utility**: % of allowed PII correctly shared (higher is better)
2. **Privacy**: % of disallowed PII correctly NOT shared (higher is better)
3. **Quickness**: Inference time in seconds (lower is better)

### 7.5 Output

**Console Output**: Real-time comparison for each test case and overall summary:
```
================================================================================
OVERALL SUMMARY
================================================================================

Baseline LLM Minimizer:
  Average Utility: 95.0%
  Average Privacy: 98.0%
  Average Time: 2.123s
  Total Time: 10.615s

RL Integration Pipeline (grpo):
  Average Utility: 100.0%
  Average Privacy: 100.0%
  Average Time: 0.015s
  Total Time: 0.075s

Comparison:
  Utility Difference: +5.0% (better)
  Privacy Difference: +2.0% (better)
  Average Speedup: 141.5x faster
  Total Time Saved: 10.540s
```

**CSV Output**: Detailed results with all metrics for each test case:
- `baseline_utility`, `baseline_privacy`, `baseline_time`
- `rl_utility`, `rl_privacy`, `rl_time`
- `utility_diff`, `privacy_diff`, `time_speedup`

### 7.6 Requirements

**Baseline LLM Minimizer**:
- **Apple Silicon / CPU (recommended)**: install MLX baseline
  ```bash
  conda activate overthink
  pip install mlx mlx-lm
  ```
- **GPU fallback**: CUDA-capable GPU + `bitsandbytes`
- Large language model (default MLX model: `mlx-community/Qwen2.5-7B-Instruct-4bit`)
- `transformers` library

**RL Integration Pipeline**:
- Trained RL model (`models/{algorithm}_model.pt`)
- Context classifier (`MLP/context_agent_mlp.pth`)
- `spacy` and `en_core_web_sm` model

**Note**:
- The comparison script automatically tries the MLX baseline first (works on Apple Silicon without a discrete GPU).  
- If `mlx` is missing you will see “Baseline skipped – not available”; install it with the command above.  
- Use `--use-gpu` to force the GPU baseline if you have CUDA and `bitsandbytes` installed.

---

## 8. Endpoint for the RL model

```bash
# Command line - plain output
python scripts/endpoint.py --algorithm grpo --directive balanced --domain bank

# Command line - JSON output
python scripts/endpoint.py --algorithm grpo --directive strictly --domain bank --json

# As a Python function
from scripts.endpoint import get_regex
result = get_regex("grpo", "balanced", "bank")
# Returns: ['PHONE', 'EMAIL', 'DATE/DOB', 'SSN', 'CREDIT_CARD']
```

Features:
- Takes algorithm, directive, and domain as input
- Returns regex as a list of PII types
- Supports JSON output with --json flag
- Can be imported and used as a function

---

For more details, see:
- `README.md`: Project overview
- `ALGORITHM_EXPLANATION.md`: Algorithm details
- `pipeline/COMPARISON_README.md`: Detailed comparison guide

