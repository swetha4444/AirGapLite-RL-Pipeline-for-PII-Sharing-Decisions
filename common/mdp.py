"""
MDP helpers and manual decision interface for the GRPO Rule Agent.

IMPORTANT:
    - The policy chooses a binary action (0 = don't share, 1 = share) for every individual PII type.
    - Actions are independent per PII - no allowed masks in the state.
    - The model learns domain-specific patterns (generalized regex) for each domain.
    - Reward is computed based on PII grouping to encourage learning consistent patterns.
    - Grouping into identity/contact/financial/network/org/demographic is used only
      when printing decisions for humans and for reward computation.
"""

from dataclasses import dataclass
from typing import Dict, List
import math

import torch

from .config import (
    PII_TYPES,
    TYPE2IDX,
    NUM_PII,
    GROUP2TYPEIDX,
    SCENARIO_NAME2ID,
    NUM_SCENARIOS,
    SCENARIO_WEIGHTS,
    LAMBDA_COMPLEXITY,
)


@dataclass
class ManualInput:
    """
    What you pass from `manual_demo_before_after.py` for a single scenario.

    - present_fields: which PII types are present in the user's utterance
    - scenario_name:  "restaurant" or "bank"
    - allowed_fields_*: these are NOT used by the model - kept for backwards compatibility.
      The model learns domain-specific patterns and makes binary decisions per PII.
    """

    present_fields: List[str]
    scenario_name: str
    allowed_fields_restaurant: List[str]
    allowed_fields_bank: List[str]


@dataclass
class DecisionOutput:
    scenario_name: str
    # For backwards compatibility with your existing demo script, we still
    # expose group-level actions (0=share none, 1=share all present,
    # 2=share subset), even though internally the policy acts per-PII.
    actions_by_group: Dict[str, int]
    shared_fields_by_group: Dict[str, List[str]]
    present_fields: List[str]


def fields_to_mask(fields: List[str]) -> List[int]:
    mask = [0] * NUM_PII
    for f in fields:
        if f in TYPE2IDX:
            mask[TYPE2IDX[f]] = 1
    return mask


def mask_to_fields(mask: List[int]) -> List[str]:
    return [PII_TYPES[i] for i, v in enumerate(mask) if v == 1]


def build_state(present_mask: List[int], scenario_id: int) -> torch.Tensor:
    """Concatenate PII-present mask with scenario one-hot.

    State dimension = NUM_PII + NUM_SCENARIOS.
    """
    scenario_oh = [0] * NUM_SCENARIOS
    scenario_oh[scenario_id] = 1
    state_vec = present_mask + scenario_oh
    return torch.tensor(state_vec, dtype=torch.float32)


def decide_sharing_for_manual_input(
    policy,
    manual: ManualInput,
    deterministic: bool = True,
    threshold: float = 0.5,
    directive: str = "balanced",
) -> DecisionOutput:
    """
    Run the learned policy on a hand-crafted scenario.

    The agent:
      - Sees: which PII types are present + the scenario (restaurant/bank)
      - Outputs: a binary decision per PII (share / do not share)
    
    Args:
        policy: Trained policy model
        manual: ManualInput with scenario and present fields
        deterministic: If True, use threshold; if False, sample
        threshold: Probability threshold for sharing (0.0 to 1.0)
        directive: "strictly" (high threshold, more privacy),
                  "accurately" (low threshold, more utility),
                  "balanced" (default threshold)
    """

    present_mask = fields_to_mask(manual.present_fields)
    scenario_id = SCENARIO_NAME2ID[manual.scenario_name]

    # Build state and get per-PII 0/1 actions from the policy.
    state = build_state(present_mask, scenario_id)
    action_mask = policy.act(state, deterministic=deterministic, threshold=threshold, directive=directive)  # List[int], len = NUM_PII

    # Convert per-PII actions into group-level actions for pretty-printing.
    actions_by_group: Dict[str, int] = {}
    shared_fields_by_group: Dict[str, List[str]] = {}

    for group_name, type_indices in GROUP2TYPEIDX.items():
        # PII present in this group for this conversation
        present_indices = [i for i in type_indices if present_mask[i] == 1]
        if not present_indices:
            # No signal in this group for this example.
            actions_by_group[group_name] = 0
            shared_fields_by_group[group_name] = []
            continue

        shared_indices = [i for i in present_indices if action_mask[i] == 1]
        shared_fields = [PII_TYPES[i] for i in shared_indices]

        if not shared_indices:
            action_code = 0
        elif len(shared_indices) == len(present_indices):
            action_code = 1
        else:
            action_code = 2

        actions_by_group[group_name] = action_code
        shared_fields_by_group[group_name] = shared_fields

    return DecisionOutput(
        scenario_name=manual.scenario_name,
        actions_by_group=actions_by_group,
        shared_fields_by_group=shared_fields_by_group,
        present_fields=manual.present_fields,
    )


# ---------------------------------------------------------------------------
# Helper functions for group-based approaches (GroupedPPO, VanillaRL)
# ---------------------------------------------------------------------------

def apply_group_action(
    type_indices: List[int],
    present_mask: List[int],
    action: int,
    pii_logits: List[float] = None,
    threshold: float = 0.5,
) -> List[int]:
    """
    Apply a group-level action to determine which PII indices are shared.
    
    Args:
        type_indices: List of PII type indices in this group
        present_mask: Which PII types are present (1 if present, 0 otherwise)
        action: Group action (0=none, 1=share all present, 2=share learned subset)
        pii_logits: Per-PII logits for this group (required for action 2)
        threshold: Threshold for action 2 (default 0.5)
    
    Returns:
        List of PII indices that should be shared
    """
    present_in_group = [i for i in type_indices if present_mask[i] == 1]
    
    if action == 0:
        # Share none
        return []
    elif action == 1:
        # Share all present in this group
        return present_in_group
    elif action == 2:
        # Share learned subset based on per-PII logits
        if pii_logits is None:
            # Fallback: share none if no logits provided
            return []
        # Map logits to present PII indices
        shared = []
        for idx, pii_idx in enumerate(type_indices):
            if present_mask[pii_idx] == 1:  # Only consider present PII
                if idx < len(pii_logits):
                    prob = 1.0 / (1.0 + math.exp(-pii_logits[idx]))  # sigmoid
                    if prob >= threshold:
                        shared.append(pii_idx)
        return shared
    else:
        return []


def compute_group_reward(
    group_name: str,
    scenario_name: str,
    group_type_indices: List[int],
    present_mask: List[int],
    allowed_mask: List[int],
    shared_indices: List[int],
    action: int,
) -> float:
    """
    Compute reward for a group-level action.
    
    Args:
        group_name: Name of the PII group
        scenario_name: "restaurant" or "bank"
        group_type_indices: PII type indices in this group
        present_mask: Which PII types are present
        allowed_mask: Which PII types are allowed
        shared_indices: Which PII indices were actually shared
        action: The action taken (0/1/2)
    
    Returns:
        Scalar reward for this group action
    """
    present_in_group = [i for i in group_type_indices if present_mask[i] == 1]
    if not present_in_group:
        return 0.0
    
    allowed_in_group = [i for i in present_in_group if allowed_mask[i] == 1]
    disallowed_in_group = [i for i in present_in_group if allowed_mask[i] == 0]
    
    shared_allowed = [i for i in shared_indices if i in allowed_in_group]
    shared_disallowed = [i for i in shared_indices if i in disallowed_in_group]
    
    # Utility: fraction of allowed PII that was shared
    if allowed_in_group:
        utility = len(shared_allowed) / len(allowed_in_group)
    else:
        utility = 1.0 if len(shared_indices) == 0 else 0.0
    
    # Privacy: fraction of disallowed PII that was NOT shared
    if disallowed_in_group:
        privacy = 1.0 - (len(shared_disallowed) / len(disallowed_in_group))
    else:
        privacy = 1.0
    
    # Get scenario weights
    weights = SCENARIO_WEIGHTS[scenario_name]
    alpha = weights["alpha"]
    beta = weights["beta"]
    
    # Group reward
    group_reward = alpha * utility + beta * privacy
    
    # Complexity penalty (small penalty for sharing more)
    if present_in_group:
        frac_shared = len(shared_indices) / len(present_in_group)
        complexity_penalty = LAMBDA_COMPLEXITY * frac_shared
        group_reward -= complexity_penalty
    
    return float(group_reward)
