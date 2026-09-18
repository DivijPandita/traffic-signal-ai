"""
DQN agent: epsilon-greedy action selection, experience replay buffer,
and the target-network training update.

This agent has NO knowledge of SUMO, TraCI, or traffic-specific
concepts -- it only knows "state vector in, action index out, learn
from (state, action, reward, next_state, done) tuples." This
separation is what lets the exact same agent code be reused unchanged
in Phase 13/14 when we swap in the multi-objective / context-aware
reward -- the reward is a number the environment computes, the agent
just consumes it.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import QNetwork


class ReplayBuffer:
    """
    Fixed-size buffer of past experiences. Sampling uniformly at
    random from this buffer, rather than learning from consecutive
    steps directly, breaks the strong temporal correlation between
    consecutive traffic states (e.g. queue length at second 100 and
    101 are nearly identical) that would otherwise destabilize
    training.
    """

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20000,
        target_update_freq: int = 500,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self._epsilon_step_count = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # The "live" network we act with and train directly.
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        # The target network: a periodically-synced copy used only to
        # compute stable training targets, never used for exploration.
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self._train_step_count = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Epsilon-greedy action selection. With probability epsilon,
        picks a uniformly random action (exploration); otherwise picks
        the action with the highest predicted Q-value (exploitation).
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def _decay_epsilon(self):
        self._epsilon_step_count += 1
        fraction = min(1.0, self._epsilon_step_count / self.epsilon_decay_steps)
        self.epsilon = self.epsilon + fraction * (self.epsilon_end - 1.0) * (1.0)
        # Linear decay from epsilon_start to epsilon_end over decay_steps.
        self.epsilon = max(
            self.epsilon_end,
            1.0 - fraction * (1.0 - self.epsilon_end),
        )

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        One gradient update using a random batch from the replay
        buffer. Returns the loss value (for logging), or None if the
        buffer doesn't have enough experiences yet.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        # Current Q-value estimates for the actions actually taken.
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: reward + gamma * max_a' Q_target(next_state, a'),
        # zeroed out for terminal states (done=1) since there is no
        # future reward beyond an episode's end.
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents occasional large TD-error spikes
        # (e.g. from a sudden queue-length jump) from causing a
        # destabilizing update.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._train_step_count += 1
        if self._train_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self._decay_epsilon()

        return loss.item()

    def save(self, path: str):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)