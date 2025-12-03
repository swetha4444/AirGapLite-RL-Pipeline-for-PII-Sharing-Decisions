"""
Training utilities for the GRPO Rule Agent.

This version implements a grouped PPO / GRPO-style update:
- per-group rewards (identity/contact/financial/network)
- per-group value heads
- clipped policy objective
- entropy bonus
- optional KL regularization
"""

from dataclasses import dataclass
from typing import List, Dict
import random

import pandas as pd
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.config import SCENARIOS, GROUP2TYPEIDX, SCENARIO_NAME2ID, SCENARIO_WEIGHTS, LAMBDA_COMPLEXITY, NUM_PII
from common.mdp import fields_to_mask, build_state
from .grpo_policy import RulePolicy


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class DatasetRow:
    present_mask: List[int]
    allowed_mask_restaurant: List[int]
    allowed_mask_bank: List[int]


def parse_list_str(s) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def load_dataset_from_excel(path: str) -> List[DatasetRow]:
    """
    Load the Excel dataset (sheet 'dataset') and produce masks for:
    - ground_truth       -> present_mask
    - allowed_restaurant -> allowed_mask_restaurant
    - allowed_bank       -> allowed_mask_bank

    We do a case-insensitive lookup of column names. For "bank", we fall back
    to "allowed_bank" if present, else we treat all PII types as allowed.
    """
    from common.config import PII_TYPES

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="dataset")
    rows: List[DatasetRow] = []

    # Case-insensitive column lookup
    col_names = {c.lower(): c for c in df.columns}

    gt_col = col_names.get("ground_truth")
    rest_col = col_names.get("allowed_restaurant")
    bank_col = col_names.get("allowed_bank")

    if gt_col is None:
        raise ValueError("Expected a 'ground_truth' column in the dataset sheet.")
    if rest_col is None:
        raise ValueError("Expected an 'allowed_restaurant' column in the dataset sheet.")

    for _, row in df.iterrows():
        gt = parse_list_str(row.get(gt_col, ""))

        # Start from what the dataset says
        allowed_rest = parse_list_str(row.get(rest_col, ""))

        # Allow explicit bank column if present, else default to 'all PII allowed'
        if bank_col is not None:
            allowed_bank = parse_list_str(row.get(bank_col, ""))
        else:
            allowed_bank = list(PII_TYPES)

        # ---- Override policy goals for training if needed ----
        # Example: ensure NAME is allowed in restaurant scenario
        if "NAME" not in allowed_rest and "NAME" in gt:
            allowed_rest.append("NAME")

        # Bank: treat *all* PII types as potentially allowed if no explicit col
        if bank_col is None:
            allowed_bank = list(PII_TYPES)

        present_mask = fields_to_mask(gt)
        allowed_mask_rest = fields_to_mask(allowed_rest)
        allowed_mask_bank = fields_to_mask(allowed_bank)

        rows.append(
            DatasetRow(
                present_mask=present_mask,
                allowed_mask_restaurant=allowed_mask_rest,
                allowed_mask_bank=allowed_mask_bank,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Transition buffer for GRPO
# ---------------------------------------------------------------------------

def compute_reward(
    present_mask: List[int],
    allowed_mask: List[int],
    action_mask: List[int],
    scenario_name: str,
) -> float:
    """
    Compute reward based on PII grouping (same as GRPO).
    
    The reward encourages the model to learn domain-specific patterns:
    - For each PII group, check if the sharing decision matches the allowed pattern
    - Reward is based on group-level accuracy
    - The model should learn which groups should be shared for each domain
    """
    # Indices actually present in this conversation
    present = [i for i, m in enumerate(present_mask) if m == 1]
    if not present:
        return 0.0

    # Compute reward per group
    group_rewards = []
    total_groups = 0
    
    for group_name, type_indices in GROUP2TYPEIDX.items():
        # Get PII types in this group that are present
        present_in_group = [i for i in type_indices if present_mask[i] == 1]
        if not present_in_group:
            continue  # Skip groups with no present fields
        
        total_groups += 1
        
        # Check sharing decisions for this group
        shared_in_group = [i for i in present_in_group if action_mask[i] == 1]
        allowed_in_group = [i for i in present_in_group if allowed_mask[i] == 1]
        
        # Group-level reward: how well does the sharing match the allowed pattern?
        if allowed_in_group:
            # If some fields in this group are allowed, reward sharing them
            shared_allowed = [i for i in shared_in_group if i in allowed_in_group]
            group_utility = len(shared_allowed) / len(allowed_in_group)
        else:
            # If no fields are allowed, reward not sharing
            group_utility = 1.0 if len(shared_in_group) == 0 else 0.0
        
        # Penalty for sharing disallowed fields in this group
        disallowed_in_group = [i for i in present_in_group if allowed_mask[i] == 0]
        if disallowed_in_group:
            shared_disallowed = [i for i in shared_in_group if i in disallowed_in_group]
            group_privacy = 1.0 - (len(shared_disallowed) / len(disallowed_in_group))
        else:
            group_privacy = 1.0
        
        # Group-level reward
        weights = SCENARIO_WEIGHTS[scenario_name]
        alpha = weights["alpha"]
        beta = weights["beta"]
        group_reward = alpha * group_utility + beta * group_privacy
        group_rewards.append(group_reward)
    
    if total_groups == 0:
        return 0.0
    
    # Average reward across groups
    avg_group_reward = sum(group_rewards) / total_groups
    
    # Complexity penalty: sharing more fields should cost something
    frac_shared = sum(action_mask) / max(1, sum(present_mask))
    complexity_penalty = LAMBDA_COMPLEXITY * frac_shared
    
    reward = avg_group_reward - complexity_penalty
    return float(reward)


class Transition:
    """
    Container to hold transitions for training.

    Each element corresponds to one (state, action) decision:
        - state_vec: List[float]
        - action_mask: List[int]  (binary per-PII actions)
        - reward: float
        - old_log_prob: float
        - old_value: float      (V_old(s))
        - old_probs: List[float] (full old action distribution for KL)
        - scenario_name: str
    """

    def __init__(self):
        self.states: List[List[float]] = []
        self.action_masks: List[List[int]] = []
        self.rewards: List[float] = []
        self.old_log_probs: List[float] = []
        self.old_values: List[float] = []
        self.old_probs: List[List[float]] = []
        self.scenario_names: List[str] = []

    def add(
        self,
        state: List[float],
        action_mask: List[int],
        reward: float,
        old_log_prob: float,
        old_value: float,
        old_probs: List[float],
        scenario_name: str,
    ):
        self.states.append(state)
        self.action_masks.append(action_mask)
        self.rewards.append(reward)
        self.old_log_probs.append(old_log_prob)
        self.old_values.append(old_value)
        self.old_probs.append(old_probs)
        self.scenario_names.append(scenario_name)

    def __len__(self) -> int:
        return len(self.states)


# ---------------------------------------------------------------------------
# GRPO rollout: collect per-group transitions
# ---------------------------------------------------------------------------

def rollout_batch_grpo(
    policy: RulePolicy,
    dataset_rows: List[DatasetRow],
    batch_size: int = 64,
) -> Transition:
    """
    Generate a batch of transitions using the current policy.
    
    Key points:
    - Actions are binary (0=don't share, 1=share) for each PII type
    - State only contains present_mask and scenario (no allowed_mask)
    - Reward is computed based on PII grouping to encourage learning domain patterns
    - Uses PPO-style updates (clipping, etc.)
    """
    trans = Transition()
    if not dataset_rows:
        return trans

    device = next(policy.parameters()).device

    for _ in range(batch_size):
        row = random.choice(dataset_rows)

        # Randomly choose which scenario to simulate for this row.
        scenario_name = random.choice(["restaurant", "bank"])
        scenario_id = SCENARIO_NAME2ID[scenario_name]

        present_mask = row.present_mask
        # allowed_mask is used ONLY for reward computation (supervision signal)
        # The model never sees it in the state - it must learn the pattern
        allowed_mask = (
            row.allowed_mask_restaurant if scenario_name == "restaurant" else row.allowed_mask_bank
        )

        # State: only present_mask + scenario (no allowed_mask)
        state_tensor = build_state(present_mask, scenario_id).to(device)
        logits, value = policy(state_tensor)
        probs = torch.sigmoid(logits)[0]  # [NUM_PII]

        # Sample binary actions (0 or 1) for each PII
        dist = torch.distributions.Bernoulli(probs=probs)
        actions = dist.sample()  # [NUM_PII]
        log_prob = dist.log_prob(actions).sum().item()

        action_mask = actions.long().tolist()
        # Reward computed based on PII grouping
        reward = compute_reward(present_mask, allowed_mask, action_mask, scenario_name)

        trans.add(
            state=state_tensor.detach().cpu().tolist(),
            action_mask=action_mask,
            reward=float(reward),
            old_log_prob=log_prob,
            old_value=float(value.item()),
            old_probs=probs.detach().cpu().tolist(),
            scenario_name=scenario_name,
        )
    return trans


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def _compute_returns_and_advantages(trans: Transition):
    """
    For this 1-step MDP, the return is equal to the immediate reward.
    Advantage uses a simple baseline: A = R - V_old.
    """
    if len(trans) == 0:
        return None, None

    rewards = torch.tensor(trans.rewards, dtype=torch.float32)
    values_old = torch.tensor(trans.old_values, dtype=torch.float32)
    returns = rewards  # 1-step
    advantages = returns - values_old

    # Optional normalization for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


# ---------------------------------------------------------------------------
# GRPO / PPO-style update
# ---------------------------------------------------------------------------

def grpo_update(
    policy: RulePolicy,
    optimizer,
    transitions: Transition,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    kl_coef: float = 0.1,
):
    """
    PPO-style update with per-PII binary actions and group-based rewards.

    Uses PPO clipping for stable learning:
        - compute ratio = π_new / π_old
        - clipped surrogate objective
        - value loss
        - entropy bonus
        - KL penalty between old and new Bernoulli distributions
    """
    if len(transitions) == 0:
        return

    device = next(policy.parameters()).device

    returns, advantages = _compute_returns_and_advantages(transitions)
    returns = returns.to(device)
    advantages = advantages.to(device)

    states = torch.tensor(transitions.states, dtype=torch.float32).to(device)
    action_masks = torch.tensor(transitions.action_masks, dtype=torch.float32).to(device)  # [N, NUM_PII]
    old_log_probs = torch.tensor(transitions.old_log_probs, dtype=torch.float32).to(device)
    old_values = torch.tensor(transitions.old_values, dtype=torch.float32).to(device)
    old_probs = torch.tensor(transitions.old_probs, dtype=torch.float32).to(device)  # [N, NUM_PII]

    for _ in range(ppo_epochs):
        optimizer.zero_grad()

        # Forward pass
        logits, values = policy(states)  # logits: [N, NUM_PII], values: [N]
        probs = torch.sigmoid(logits)  # [N, NUM_PII]

        # Bernoulli distribution for per-PII actions
        dist = torch.distributions.Bernoulli(probs=probs)
        log_probs = dist.log_prob(action_masks).sum(dim=1)  # [N]
        entropy = dist.entropy().sum(dim=1).mean()  # scalar

            # PPO ratio
        ratio = torch.exp(log_probs - old_log_probs)  # [N]

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
        value_loss = F.mse_loss(values, returns)

        # KL between old and new Bernoulli distributions
        eps = 1e-8
        kl = (
            old_probs * torch.log((old_probs + eps) / (probs + eps))
            + (1.0 - old_probs) * torch.log((1.0 - old_probs + eps) / (1.0 - probs + eps))
        )
        kl_mean = kl.sum(dim=1).mean()

        # Total loss
        loss = policy_loss + value_coef * value_loss + kl_coef * kl_mean - entropy_coef * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers
# ---------------------------------------------------------------------------

def rollout_batch(policy: RulePolicy, dataset_rows: List[DatasetRow], batch_size: int = 64) -> Transition:
    """
    Backwards-compatible name: now uses GRPO-style rollout.
    """
    return rollout_batch_grpo(policy, dataset_rows, batch_size=batch_size)


def policy_gradient_update(
    policy: RulePolicy,
    optimizer,
    transitions: Transition,
    epochs: int = 3,
):
    """
    Backwards-compatible name: now wraps the GRPO / PPO-style update.

    `epochs` is mapped to `ppo_epochs`.
    """
    return grpo_update(policy, optimizer, transitions, ppo_epochs=epochs)


# ---------------------------------------------------------------------------
# Evaluation: average reward under greedy policy
# ---------------------------------------------------------------------------

def evaluate_average_reward(
    policy: RulePolicy,
    dataset_rows: List[DatasetRow],
    num_samples: int = 200,
) -> float:
    """
    Estimate average reward over random samples from the dataset.
    
    Uses deterministic actions (threshold at 0.5) for evaluation.
    Reward is computed based on PII grouping.
    """
    if not dataset_rows:
        return 0.0

    device = next(policy.parameters()).device

    total_reward = 0.0
    count = 0

    for _ in range(num_samples):
        row = random.choice(dataset_rows)
        scenario_name = random.choice(["restaurant", "bank"])
        scenario_id = SCENARIO_NAME2ID[scenario_name]

        present_mask = row.present_mask
        allowed_mask = (
            row.allowed_mask_restaurant if scenario_name == "restaurant" else row.allowed_mask_bank
        )

        state_tensor = build_state(present_mask, scenario_id).to(device)
        logits, _ = policy(state_tensor)
        probs = torch.sigmoid(logits)[0]  # [NUM_PII]
        actions = (probs >= 0.5).long().tolist()  # Deterministic threshold

        reward = compute_reward(present_mask, allowed_mask, actions, scenario_name)
        total_reward += reward
        count += 1

    if count == 0:
        return 0.0
    return float(total_reward / count)
