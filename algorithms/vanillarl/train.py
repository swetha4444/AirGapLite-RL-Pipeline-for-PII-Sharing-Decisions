"""
Training utilities for the GRPO Rule Agent.
"""

from dataclasses import dataclass
from typing import List
import random

import pandas as pd
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.config import SCENARIOS, GROUP2TYPEIDX, SCENARIO_NAME2ID, SCENARIO_WEIGHTS, LAMBDA_COMPLEXITY
from common.mdp import fields_to_mask, build_state
from .policy import RulePolicy


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
    Load dataset from CSV or Excel automatically.
    Columns required:
    - ground_truth
    - allowed_restaurant
    - allowed_bank
    """

    # Auto-detect CSV vs Excel
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="dataset")

    rows: List[DatasetRow] = []

    # Normalize column names
    col_names = {c.lower(): c for c in df.columns}

    gt_col = col_names.get("ground_truth")
    rest_col = col_names.get("allowed_restaurant")
    bank_col = col_names.get("allowed_bank")

    if gt_col is None or rest_col is None or bank_col is None:
        raise ValueError("Missing required columns: ground_truth, allowed_restaurant, allowed_bank")

    for _, row in df.iterrows():
        gt = parse_list_str(row.get(gt_col, ""))
        allowed_rest = parse_list_str(row.get(rest_col, ""))
        allowed_bank = parse_list_str(row.get(bank_col, ""))

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
    Simple container to hold transitions for training.
    Each element is (state_vec, action_mask, log_prob, reward, scenario_name).
    """

    def __init__(self):
        self.states = []
        self.action_masks = []
        self.log_probs = []
        self.rewards = []
        self.scenario_names = []

    def add(self, state, action_mask, log_prob, reward, scenario_name):
        self.states.append(state)
        self.action_masks.append(action_mask)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.scenario_names.append(scenario_name)

    def __len__(self):
        return len(self.states)


def rollout_batch(policy: RulePolicy, dataset_rows: List[DatasetRow], batch_size: int = 64) -> Transition:
    """
    Generate a batch of transitions using the current policy.
    
    Key points:
    - Actions are binary (0=don't share, 1=share) for each PII type
    - State only contains present_mask and scenario (no allowed_mask)
    - Reward is computed based on PII grouping to encourage learning domain patterns
    - Uses simple REINFORCE updates (no value function)
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
        logits = policy(state_tensor)
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
            log_prob=log_prob,
            reward=reward,
            scenario_name=scenario_name,
        )
    return trans


def policy_gradient_update(policy: RulePolicy, optimizer, transitions: Transition, epochs: int = 3):
    """
    Simple REINFORCE update with per-PII binary actions.
    
    Loss = -mean(log_prob * advantage)
    where advantage = reward - baseline (normalized)
    """
    if len(transitions) == 0:
        return

    device = next(policy.parameters()).device
    rewards = torch.tensor(transitions.rewards, dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    for _ in range(epochs):
        optimizer.zero_grad()
        
        states = torch.tensor(transitions.states, dtype=torch.float32).to(device)
        action_masks = torch.tensor(transitions.action_masks, dtype=torch.float32).to(device)
        advantages_t = advantages.to(device)

        # Forward pass
        logits = policy(states)  # [N, NUM_PII]
        probs = torch.sigmoid(logits)  # [N, NUM_PII]

        # Bernoulli distribution for per-PII actions
        dist = torch.distributions.Bernoulli(probs=probs)
        log_probs = dist.log_prob(action_masks).sum(dim=1)  # [N]

        # REINFORCE loss
        loss = -(log_probs * advantages_t).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()


def evaluate_average_reward(policy: RulePolicy, dataset_rows: List[DatasetRow], num_samples: int = 200) -> float:
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
        logits = policy(state_tensor)
        probs = torch.sigmoid(logits)[0]  # [NUM_PII]
        actions = (probs >= 0.5).long().tolist()  # Deterministic threshold

        reward = compute_reward(present_mask, allowed_mask, actions, scenario_name)
        total_reward += reward
        count += 1

    if count == 0:
        return 0.0
    return float(total_reward / count)
