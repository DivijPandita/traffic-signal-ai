"""
Neural network architectures shared by the RL agents.

DQN (this phase) uses QNetwork: state -> Q-value per action.
PPO (Phase 11) will use a separate policy/value network defined here
later, but we keep this file as the single place all agent
architectures live, so changes to e.g. hidden layer sizes are made
in one spot.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Simple feedforward network: state vector -> one Q-value per
    discrete action. Deliberately plain (no dueling/attention/etc.)
    since DQN architecture is not this project's novelty -- see
    Phase 9 rationale.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)