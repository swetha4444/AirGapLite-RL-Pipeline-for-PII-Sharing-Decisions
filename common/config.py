"""
Configuration and constants for the GRPO Rule Agent MDP (per-PII actions).

This module only defines static metadata and hyperparameters.
The policy never sees any ground-truth "allowed" masks here – those are
used only inside the training loop as reward signals.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# PII universe
# ---------------------------------------------------------------------------

# 11 PII types (fixed order, must match dataset strings exactly)
PII_TYPES: List[str] = [
    "NAME",
    "PHONE",
    "EMAIL",
    "DATE/DOB",
    "company",
    "location",
    "IP",
    "SSN",
    "CREDIT_CARD",
    "age",
    "sex",
]

TYPE2IDX: Dict[str, int] = {t: i for i, t in enumerate(PII_TYPES)}
NUM_PII: int = len(PII_TYPES)

# ---------------------------------------------------------------------------
# Grouping (used only for pretty-printing / analysis)
# ---------------------------------------------------------------------------

# Logical groupings of PII types – the agent acts per-PII, but we still
# use these groups when printing decisions so the output is easier to read.
# DATE/DOB is added to identity group as it's often used for identity verification.
GROUPS: Dict[str, List[str]] = {
    "identity": ["NAME", "DATE/DOB"],
    "contact": ["PHONE", "EMAIL"],
    "financial": ["SSN", "CREDIT_CARD"],
    "network": ["IP"],
    "org": ["company", "location"],
    "demographic": ["age", "sex"],
}

GROUP2TYPEIDX: Dict[str, List[int]] = {
    g: [TYPE2IDX[t] for t in types if t in TYPE2IDX]
    for g, types in GROUPS.items()
}

# ---------------------------------------------------------------------------
# Scenarios / domains (restaurant vs bank)
# ---------------------------------------------------------------------------

SCENARIOS = {
    0: "restaurant",
    1: "bank",
}
SCENARIO_NAME2ID: Dict[str, int] = {v: k for k, v in SCENARIOS.items()}
NUM_SCENARIOS: int = len(SCENARIOS)

# Per-scenario trade-off between utility and privacy.
#   reward = alpha * utility + beta * privacy - complexity_penalty
SCENARIO_WEIGHTS = {
    "restaurant": {"alpha": 0.6, "beta": 0.4},  # more privacy-leaning
    "bank": {"alpha": 0.7, "beta": 0.3},        # more utility-leaning
}

# ---------------------------------------------------------------------------
# Complexity regularisation (penalise sharing too many fields)
# ---------------------------------------------------------------------------

# Global coefficient for number-of-fields shared.
LAMBDA_COMPLEXITY: float = 0.05

# Number of actions for group-based approaches (0=none, 1=share all, 2=share subset)
NUM_ACTIONS: int = 3

__all__ = [
    "PII_TYPES",
    "TYPE2IDX",
    "NUM_PII",
    "GROUPS",
    "GROUP2TYPEIDX",
    "SCENARIOS",
    "SCENARIO_NAME2ID",
    "NUM_SCENARIOS",
    "SCENARIO_WEIGHTS",
    "LAMBDA_COMPLEXITY",
    "NUM_ACTIONS",
]
