# Complete Flow Diagram: From Dataset to Trained Model

##  End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET                                      │
│  File: 690-Project-Dataset-final.csv (15,805 rows)              │
│  Row: {ground_truth, allowed_restaurant, allowed_bank}          │
│  Example:                                                       │
│    ground_truth = [NAME, PHONE, EMAIL, SSN]                     │
│    allowed_restaurant = [PHONE, EMAIL]                          │
│    allowed_bank = [EMAIL, PHONE, SSN]                           │
│                                                                 │
│  Frequencies (optimized for utility tradeoff):                  │
│    EMAIL: 99.8%, PHONE: 64.6%                                   │
│    DATE/DOB: 48.4%, SSN: 41.9%, CREDIT_CARD: 41.9%              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STATE CONSTRUCTION                           │
│  present_mask = [1,1,1,0,0,0,0,1,0,0,0]  (11 dim)               │
│  scenario_one_hot = [1,0] or [0,1]  (2 dim)                     │
│  state = concat(present_mask, scenario_one_hot)  (13 dim)       │
│                                                                 │
│    allowed_mask is NOT in state - model must learn!             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    POLICY NETWORK                               │
│                                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │  Shared Encoder: MLP(13 → 64 → 64)       │                   │
│  └──────────────────────────────────────────┘                   │
│                    ↓                                            │
│  ┌──────────────────────────────────────────┐                   │
│  │  Algorithm-Specific Heads                │                   │
│  │                                          │                   │
│  │  GRPO:                                   │                   │
│  │    Policy: Linear(64 → 11)               │                   │
│  │    Value: Linear(64 → 1)                 │                   │
│  │                                          │                   │
│  │  GroupedPPO/VanillaRL:                   │                   │
│  │    Policy: Linear(64 → 3) per group      │                   │
│  │    Value: Linear(64 → 1) per group       │                   │
│  │    (VanillaRL: no value head)            │                   │
│  └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ACTION SELECTION                             │
│                                                                 │
│  GRPO:                                                          │
│    logits[11] → sigmoid → probs[11]                             │
│    Sample: Bernoulli(probs) → actions[11] = [0,1,1,0,...]       │
│                                                                 │
│  GroupedPPO/VanillaRL:                                          │
│    For each group:                                              │
│      logits[3] → softmax → probs[3]                             │
│      Sample: Categorical(probs) → action ∈ {0,1,2}              │
│    Result: {group: action} = {"contact": 1, "financial": 2}     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ACTION APPLICATION                           │
│                                                                 │
│  GRPO:                                                          │
│    actions directly indicate which PII to share                 │
│    shared = [PII_i where actions[i] == 1]                       │
│                                                                 │
│  GroupedPPO/VanillaRL:                                          │
│    For each group:                                              │
│      if action == 0: share = []                                 │
│      if action == 1: share = all present in group               │
│      if action == 2: share = allowed subset in group            │
│    Combine: shared = union of all group shares                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    REWARD COMPUTATION                           │
│                                                                 │
│  Compare shared vs allowed_mask:                                │
│    shared_allowed = shared ∩ allowed                            │
│    shared_disallowed = shared ∩ (present - allowed)             │
│                                                                 │
│  Utility = |shared_allowed| / |allowed|                         │
│  Privacy = 1 - |shared_disallowed| / |disallowed|               │
│                                                                 │
│  Reward = α·utility + β·privacy - λ·complexity                  │
│                                                                 │
│  (For GRPO: compute per group, then average)                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    POLICY UPDATE                                │
│                                                                 │
│  GRPO:                                                          │
│    advantages = rewards - value_baseline                        │
│    ratio = exp(new_log_prob - old_log_prob)                     │
│    policy_loss = -mean(ratio * advantages)                      │
│    value_loss = MSE(new_value, rewards)                         │
│    kl_penalty = KL(new_probs || old_probs)                      │
│    loss = policy_loss + value_loss + kl_penalty                 │
│                                                                 │
│  GroupedPPO:                                                    │
│    Same as GRPO but:                                            │
│    - Per-group advantages and values                            │
│    - PPO clipping: min(surr1, clip(surr2))                      │
│    - Entropy bonus for exploration                              │
│                                                                 │
│  VanillaRL:                                                     │
│    advantages = normalize(rewards)                              │
│    loss = -mean(log_prob * advantages)  # Simple REINFORCE      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERGENCE CHECK                            │
│                                                                 │
│  Evaluate on validation set                                     │
│  If reward improved > threshold:                                │
│    → Continue training                                          │
│  Else if no improvement for 'patience' iterations:              │
│    → STOP (converged)                                           │
│  Else if reached max_iters:                                     │
│    → STOP (max iterations)                                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINED MODEL                                │
│                                                                 │
│  Model has learned:                                             │
│    - Which PII to share for restaurant domain                   |
│    - Which PII to share for bank domain                         │
│    - Domain-specific patterns (generalized regex)               │
│                                                                 │
│  Can now make decisions on new conversations!                   │
└─────────────────────────────────────────────────────────────────┘
```

##  State-Action-Reward Cycle

```
┌─────────┐
│  State  │  s = [present_mask, scenario]
│   (s)   │
└────┬────┘
     │
     │ Policy π(a|s)
     ↓
┌─────────┐
│ Action  │  GRPO: a ∈ {0,1}^11
│   (a)   │  GroupedPPO/VanillaRL: a ∈ {0,1,2} per group
└────┬────┘
     │
     │ Apply action
     ↓
┌─────────┐
│ Reward  │  r = α·utility + β·privacy - complexity
│   (r)   │
└────┬────┘
     │
     │ Update policy
     ↓
┌─────────┐
│  New    │  θ_new = θ_old + α·∇J(θ)
│ Policy  │
└─────────┘
```

##  Learning Objective

All algorithms maximize:
```
J(θ) = E[Σ r(s,a)] 
     = E[Σ (α·utility + β·privacy - complexity)]
```

Where:
- **θ**: Policy parameters (neural network weights)
- **E**: Expectation over state-action distribution
- **r(s,a)**: Reward for taking action a in state s

The model learns to:
1. **Recognize domain** from scenario one-hot
2. **Infer patterns** from rewards (never sees allowed_mask!)
3. **Balance utility and privacy** based on domain weights
4. **Generalize** to new conversations

