"""
Solver Agent — Navigates the heist environment to reach the vault.
Uses PPO with CNN+LSTM for temporal reasoning about camera rotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from ..networks import SolverNetwork
from ..utils import DEVICE


class SolverAgent:
    """
    The Solver (Spy) agent that navigates security layouts.
    
    Uses Proximal Policy Optimization (PPO) with:
    - CNN for spatial awareness (walls, cameras, guards)
    - LSTM for temporal reasoning (camera rotation cycles)
    - Actor-Critic architecture for stable training
    """
    
    def __init__(self, grid_rows: int = 20, grid_cols: int = 20,
                 num_actions: int = 5, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coeff: float = 0.05,
                 value_coeff: float = 0.5, max_grad_norm: float = 0.5,
                 ppo_epochs: int = 3, batch_size: int = 64):
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Network
        self.network = SolverNetwork(
            grid_rows=grid_rows, grid_cols=grid_cols,
            num_actions=num_actions
        ).to(DEVICE)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # LSTM hidden state (reset per episode)
        self.hidden = None
        
        # Experience buffer (per-episode rollouts)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # Training stats
        self.episode_count = 0
        self.total_reward = 0.0
        self.recent_rewards = deque(maxlen=100)
    
    def reset(self):
        """Reset LSTM hidden state for a new episode."""
        self.hidden = None
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action based on current state.
        
        Args:
            state: (3, grid_rows, grid_cols) state tensor from environment
        
        Returns:
            action: integer action (0-4)
        """
        self.network.eval()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action, log_prob, value, self.hidden = \
                self.network.get_action(state_tensor, self.hidden)
        
        # Store experience
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        return action.item()
    
    def store_transition(self, reward: float, done: bool):
        """Store reward and done flag for the last action."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def end_episode(self, final_reward: float):
        """Mark the end of an episode."""
        self.total_reward += final_reward
        self.recent_rewards.append(final_reward)
        self.episode_count += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update the Solver's policy using PPO with GAE.
        
        Returns:
            Dict with training metrics
        """
        if len(self.states) == 0:
            return {"solver_loss": 0.0}
        
        self.network.train()
        
        # Synchronize buffer lengths (they can desync by 1 in edge cases)
        min_len = min(len(self.states), len(self.actions), 
                      len(self.log_probs), len(self.values),
                      len(self.rewards), len(self.dones))
        
        if min_len == 0:
            self._clear_buffers()
            return {"solver_loss": 0.0}
        
        # Convert to tensors (truncated to min_len)
        states = torch.FloatTensor(np.array(self.states[:min_len])).to(DEVICE)
        actions = torch.LongTensor(self.actions[:min_len]).to(DEVICE)
        old_log_probs = torch.FloatTensor(self.log_probs[:min_len]).to(DEVICE)
        values = torch.FloatTensor(self.values[:min_len]).to(DEVICE)
        rewards = torch.FloatTensor(self.rewards[:min_len]).to(DEVICE)
        dones = torch.FloatTensor(self.dones[:min_len]).to(DEVICE)
        
        # Compute GAE advantages
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update (multiple epochs)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        dataset_size = len(states)
        
        for _ in range(self.ppo_epochs):
            # Mini-batch training
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Forward pass (no LSTM for batch training — use feedforward)
                logits, new_values, _ = self.network(batch_states)
                
                # New log probs
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                    1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # Total loss
                loss = (policy_loss 
                        + self.value_coeff * value_loss 
                        - self.entropy_coeff * entropy)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 
                                         self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        metrics = {
            "solver_policy_loss": total_policy_loss / max(num_updates, 1),
            "solver_value_loss": total_value_loss / max(num_updates, 1),
            "solver_entropy": total_entropy / max(num_updates, 1),
            "solver_avg_reward": np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            "solver_episodes": self.episode_count,
        }
        
        # Clear buffers
        self._clear_buffers()
        
        return metrics
    
    def _clear_buffers(self):
        """Clear all experience buffers."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        return advantages
    
    def save(self, path: str):
        """Save the Solver's model."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
        }, path)
    
    def load(self, path: str):
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.episode_count = checkpoint.get("episode_count", 0)
