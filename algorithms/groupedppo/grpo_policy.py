"""
Policy network for GroupedPPO: per-PII binary actions with group-based rewards.

Same as GRPO: binary (0/1) for each PII type, but uses PPO-style updates.
Rewards are computed by group to encourage learning group patterns.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.config import NUM_PII, NUM_SCENARIOS


class RulePolicy(nn.Module):
    """
    MLP policy with per-PII binary actions (same as GRPO):
      - Input: state = [PII-present mask, scenario one-hot]
      - Output: Bernoulli logits for each PII (length = NUM_PII)
      - Value head: V(s) for advantage estimation
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        state_dim = NUM_PII + NUM_SCENARIOS

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, NUM_PII)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: [B, state_dim] or [state_dim]

        Returns:
            logits: [B, NUM_PII]  (Bernoulli logits per PII)
            value:  [B]           (scalar value per state)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = True, threshold: float = 0.5, directive: str = "balanced") -> list:
        """
        Sample or threshold actions for a *single* state.

        Args:
            state: State tensor or list
            deterministic: If True, use threshold; if False, sample from distribution
            threshold: Probability threshold for sharing (0.0 to 1.0)
            directive: "strictly" (high threshold, more privacy), 
                      "accurately" (low threshold, more utility),
                      "balanced" (default 0.5 threshold)
        
        Returns:
            List[int] of length NUM_PII with 0/1 decisions.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        logits, _ = self.forward(state)
        probs = torch.sigmoid(logits)[0]  # [NUM_PII]

        # Adjust threshold based on directive
        if directive == "strictly":
            # Share only highly probable PII (more privacy, less utility)
            threshold = max(threshold, 0.7)  # At least 0.7
        elif directive == "accurately":
            # Share PII even with minimal probability (less privacy, more utility)
            threshold = min(threshold, 0.3)  # At most 0.3
        # "balanced" uses the provided threshold (default 0.5)

        if deterministic:
            actions = (probs >= threshold).long()
        else:
            dist = torch.distributions.Bernoulli(probs=probs)
            actions = dist.sample().long()

        return actions.tolist()
