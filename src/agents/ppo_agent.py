"""
PPO agent: on-policy rollout collection, advantage estimation (GAE),
and the clipped surrogate objective update.

Like DQNAgent, this has no knowledge of SUMO/TraCI -- it only knows
state vectors, action indices, and rewards. This is what lets Phase
13/14 swap the reward function without touching this file at all.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.networks import ActorCriticNetwork

class RunningNormalizer:
    """
    Tracks a running mean and standard deviation (Welford's online
    algorithm) and uses it to normalize rewards on the fly. This
    prevents occasional catastrophic rollouts (e.g. a policy that
    briefly causes gridlock) from producing enormous raw reward
    magnitudes that blow up the value function's MSE loss and
    destabilize the whole policy update -- exactly the failure mode
    causing the loss/avg_wait explosion seen in early PPO training.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: float) -> float:
        self.update(x)
        std = (self.var ** 0.5) + 1e-8
        return x / std  # scale only, don't subtract mean -- preserves reward sign/ordering

class RolloutBuffer:
    """
    Stores exactly one batch of on-policy experience (collected under
    the CURRENT policy). Unlike DQN's replay buffer, this is cleared
    after every training update -- PPO cannot reuse stale experience
    collected under an old policy version.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.25,
        entropy_coef: float = 0.02,
        num_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.buffer = RolloutBuffer()
        self.reward_normalizer = RunningNormalizer()

    def select_action(self, state: np.ndarray):
        """
        Samples an action from the current policy's distribution
        (not argmax -- PPO's exploration comes from sampling, not
        epsilon-greedy). Returns the action, its log-probability
        (needed for the clipped objective later), and the critic's
        value estimate for this state (needed for advantage
        computation).
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Deterministic action selection for EVALUATION (no sampling), analogous to DQN's explore=False."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.network(state_t)
            action = torch.argmax(logits, dim=1)
        return int(action.item())

    def store_transition(self, state, action, reward, log_prob, value, done):
        normalized_reward = self.reward_normalizer.normalize(reward)
        self.buffer.add(state, action, normalized_reward, log_prob, value, done)

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        Generalized Advantage Estimation: a smoothed, lower-variance
        version of "advantage" than a raw one-step TD estimate.
        gae_lambda controls the bias-variance tradeoff -- lambda=1 is
        pure Monte Carlo (high variance, low bias), lambda=0 is pure
        one-step TD (low variance, higher bias); 0.95 is a common,
        well-tested middle ground.
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        values_ext = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, last_state: np.ndarray, last_done: bool):
        """
        Runs num_epochs passes over the collected rollout buffer,
        performing the clipped-surrogate-objective PPO update. Called
        once a full rollout has been collected (see train_ppo.py).
        """
        if last_done:
            last_value = 0.0
        else:
            state_t = torch.as_tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, last_value_t = self.network(state_t)
            last_value = float(last_value_t.item())

        advantages, returns = self._compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.dones, last_value
        )
        # Normalizing advantages stabilizes training -- without this,
        # reward scale differences between scenarios (e.g. moderate vs
        # heavy having very different waiting-time magnitudes) would
        # require re-tuning the learning rate per scenario.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.as_tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = len(self.buffer)
        indices = np.arange(n)

        total_loss_log = 0.0
        num_updates = 0

        for _ in range(self.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                batch_idx = indices[start:start + self.minibatch_size]
                if len(batch_idx) < 2:
                    continue

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                logits, values = self.network(b_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # The clipped surrogate objective: PPO's core trick.
                # ratio = how much more/less likely the new policy makes
                # this action vs the old policy that actually collected it.
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values.squeeze(-1), b_returns)

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_log += loss.item()
                num_updates += 1

        self.buffer.clear()
        return total_loss_log / max(1, num_updates)

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "reward_normalizer_mean": self.reward_normalizer.mean,
            "reward_normalizer_var": self.reward_normalizer.var,
            "reward_normalizer_count": self.reward_normalizer.count,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        if "reward_normalizer_mean" in checkpoint:
            self.reward_normalizer.mean = checkpoint["reward_normalizer_mean"]
            self.reward_normalizer.var = checkpoint["reward_normalizer_var"]
            self.reward_normalizer.count = checkpoint["reward_normalizer_count"]
