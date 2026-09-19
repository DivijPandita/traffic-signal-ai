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

class ActorCriticNetwork(nn.Module):
    """
    Shared-trunk actor-critic network for PPO.

    The "actor" head outputs action logits (turned into a probability
    distribution via softmax during action selection). The "critic"
    head outputs a single scalar: the estimated value of the current
    state. Sharing the trunk (early layers) is a common efficiency
    choice -- both heads benefit from the same learned traffic-state
    features, and it keeps the network small relative to having two
    fully separate networks.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_logits, state_value