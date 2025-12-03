"""
GRPO-style training utilities for the per-PII RulePolicy.

Key choices:
    - Action space: Binary {0,1} per PII type (0=don't share, 1=share).
      Each PII gets an independent binary decision.
    - State: [present_mask (length NUM_PII), scenario_one_hot (length NUM_SCENARIOS)].
      NO allowed_mask in state - model must learn domain-specific patterns.
    - Reward: Based on PII grouping to encourage learning generalized regex patterns:
        * Computed at group level (identity, contact, financial, network, org, demographic)
        * For each group: reward sharing allowed fields, penalize sharing disallowed fields
        * Reward = alpha * group_utility + beta * group_privacy - complexity_penalty
        * The model learns which groups should be shared for each domain (restaurant vs bank)
    - GRPO/PPO-style update:
        * Uses old_log_prob and old_probs snapshot for KL regularisation.
        * The model generalizes to learn domain-specific sharing patterns.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import random

import pandas as pd
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.config import (
    PII_TYPES,
    TYPE2IDX,
    NUM_PII,
    SCENARIO_NAME2ID,
    SCENARIO_WEIGHTS,
    LAMBDA_COMPLEXITY,
    GROUP2TYPEIDX,
)
from common.mdp import build_state


# ---------------------------------------------------------------------------
# Dataset row representation
# ---------------------------------------------------------------------------


@dataclass
class DatasetRow:
    # Which PII types appear in the conversation at all
    present_mask: List[int]
    # Which PII of those present are considered allowed for each scenario
    # NOTE: These are used ONLY for reward computation during training.
    # The model never sees these in its state - it must learn domain-specific patterns.
    allowed_mask_restaurant: List[int]
    allowed_mask_bank: List[int]


def parse_list_str(s: str) -> List[str]:
    """Parse strings like "[PHONE, EMAIL]" into ["PHONE", "EMAIL"]."""
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s or s == "[]":
        return []
    s = s.strip("[]")
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def fields_to_mask(fields: List[str]) -> List[int]:
    mask = [0] * NUM_PII
    for f in fields:
        if f in TYPE2IDX:
            mask[TYPE2IDX[f]] = 1
    return mask


def load_dataset_from_excel(path: str) -> List[DatasetRow]:
    """Load your 690-Project-Dataset (.csv or .xlsx).

    Columns expected:
        - ground_truth       : list-like, e.g. "[NAME, PHONE, EMAIL]"
        - allowed_restaurant : list-like (used only for reward computation, not in state)
        - allowed_bank       : list-like (used only for reward computation, not in state)
    
    The model learns domain-specific patterns (generalized regex) for each domain.
    Actions are binary (0=don't share, 1=share) for each PII type.
    Reward is computed based on PII grouping.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        # Fallback for .xlsx or .xls; no sheet-name assumption needed.
        df = pd.read_excel(path)

    rows: List[DatasetRow] = []

    for _, row in df.iterrows():
        gt_fields = parse_list_str(str(row.get("ground_truth", "")))
        rest_fields = parse_list_str(str(row.get("allowed_restaurant", "")))
        bank_fields = parse_list_str(str(row.get("allowed_bank", "")))

        present_mask = fields_to_mask(gt_fields)
        allowed_restaurant = fields_to_mask(rest_fields)
        allowed_bank = fields_to_mask(bank_fields)

        rows.append(
            DatasetRow(
                present_mask=present_mask,
                allowed_mask_restaurant=allowed_restaurant,
                allowed_mask_bank=allowed_bank,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Reward function based on PII grouping.
# The model learns domain-specific patterns (generalized regex) for each domain.
# Reward is computed at the group level to encourage consistent group-level decisions.
# ---------------------------------------------------------------------------


def compute_reward(
    present_mask: List[int],
    allowed_mask: List[int],
    action_mask: List[int],
    scenario_name: str,
) -> float:
    """
    Compute reward based on PII grouping.
    
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


# ---------------------------------------------------------------------------
# Rollout buffer for GRPO / PPO-style updates
# ---------------------------------------------------------------------------


@dataclass
class TransitionBatch:
    states: List[List[float]] = field(default_factory=list)
    actions: List[List[int]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    old_log_probs: List[float] = field(default_factory=list)
    old_values: List[float] = field(default_factory=list)
    old_probs: List[List[float]] = field(default_factory=list)
    scenario_names: List[str] = field(default_factory=list)

    def add(
        self,
        state_vec: List[float],
        action_mask: List[int],
        reward: float,
        log_prob: float,
        value: float,
        probs: List[float],
        scenario_name: str,
    ):
        self.states.append(state_vec)
        self.actions.append(action_mask)
        self.rewards.append(reward)
        self.old_log_probs.append(log_prob)
        self.old_values.append(value)
        self.old_probs.append(probs)
        self.scenario_names.append(scenario_name)

    def to_tensors(self, device) -> Tuple[torch.Tensor, ...]:
        states = torch.tensor(self.states, dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(self.old_log_probs, dtype=torch.float32, device=device)
        old_values = torch.tensor(self.old_values, dtype=torch.float32, device=device)
        old_probs = torch.tensor(self.old_probs, dtype=torch.float32, device=device)
        return states, actions, rewards, old_log_probs, old_values, old_probs


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------


def bernoulli_log_prob(probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Log p(a | p) for multi-PII Bernoulli where
    probs/actions shape = [B, NUM_PII].
    """
    eps = 1e-8
    log_p = actions * torch.log(probs + eps) + (1.0 - actions) * torch.log(1.0 - probs + eps)
    return log_p.sum(dim=1)


def rollout_batch(
    policy,
    dataset_rows: List[DatasetRow],
    batch_size: int = 64,
) -> TransitionBatch:
    """
    Generate a batch of transitions using the current policy.
    
    Key points:
    - Actions are binary (0=don't share, 1=share) for each PII type
    - State only contains present_mask and scenario (no allowed_mask)
    - Reward is computed based on PII grouping to encourage learning domain patterns
    - The model learns generalized regex patterns for each domain
    """
    device = next(policy.parameters()).device
    batch = TransitionBatch()

    if not dataset_rows:
        return batch

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

        batch.add(
            state_vec=state_tensor.cpu().tolist(),
            action_mask=action_mask,
            reward=reward,
            log_prob=log_prob,
            value=value.item(),
            probs=probs.detach().cpu().tolist(),
            scenario_name=scenario_name,
        )

    return batch


# ---------------------------------------------------------------------------
# Policy gradient update (PPO/GRPO style with KL penalty)
# ---------------------------------------------------------------------------


def policy_gradient_update(
    policy,
    optimizer,
    batch: TransitionBatch,
    epochs: int = 4,
    kl_coef: float = 0.1,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
):
    """Perform several epochs of policy update on a rollout batch."""
    if not batch.states:
        return

    device = next(policy.parameters()).device
    states, actions, rewards, old_log_probs, old_values, old_probs = batch.to_tensors(device)

    with torch.no_grad():
        advantages = rewards - old_values
        # Normalise advantages for stability.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        logits, values = policy(states)
        probs = torch.sigmoid(logits)

        dist = torch.distributions.Bernoulli(probs=probs)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()

        ratio = torch.exp(log_probs - old_log_probs)

        # Policy loss (no clipping -> GRPO-style; you can add clipping if you like).
        surr = ratio * advantages
        policy_loss = -surr.mean()

        # Value loss against *actual* returns (one-step here).
        value_loss = F.mse_loss(values, rewards)

        # KL(new || old) between Bernoulli distributions
        eps = 1e-8
        kl = (
            old_probs * torch.log((old_probs + eps) / (probs + eps))
            + (1.0 - old_probs) * torch.log((1.0 - old_probs + eps) / (1.0 - probs + eps))
        )
        kl_mean = kl.sum(dim=1).mean()

        loss = policy_loss + value_coef * value_loss + kl_coef * kl_mean - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate_average_reward(
    policy,
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
    policy.eval()

    total_reward = 0.0
    n = 0

    with torch.no_grad():
        for _ in range(num_samples):
            row = random.choice(dataset_rows)
            scenario_name = random.choice(["restaurant", "bank"])
            scenario_id = SCENARIO_NAME2ID[scenario_name]

            present_mask = row.present_mask
            # allowed_mask used only for reward computation
            allowed_mask = (
                row.allowed_mask_restaurant
                if scenario_name == "restaurant"
                else row.allowed_mask_bank
            )

            # State: only present_mask + scenario
            state = build_state(present_mask, scenario_id).to(device)
            logits, _ = policy(state)
            probs = torch.sigmoid(logits)[0]
            # Deterministic: threshold at 0.5
            actions = (probs >= 0.5).long().tolist()

            r = compute_reward(present_mask, allowed_mask, actions, scenario_name)
            total_reward += r
            n += 1

    policy.train()
    return total_reward / max(1, n)
