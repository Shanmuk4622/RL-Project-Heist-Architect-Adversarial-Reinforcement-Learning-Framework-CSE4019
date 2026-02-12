"""
Architect Agent — Designs security layouts to detect the Solver.
Uses PPO with an encoder-decoder CNN to generate level layouts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional

from ..networks import ArchitectNetwork
from ..utils import DEVICE, TileType


class ArchitectAgent:
    """
    The Architect agent that designs security layouts.
    
    Uses Proximal Policy Optimization (PPO) to learn placement strategies
    that maximize Solver detection while maintaining level solvability.
    """
    
    def __init__(self, grid_rows: int = 20, grid_cols: int = 20,
                 budget: int = 15, lr: float = 3e-4,
                 gamma: float = 0.99, clip_epsilon: float = 0.2,
                 entropy_coeff: float = 0.01, value_coeff: float = 0.5):
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.budget = budget
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        
        # Network
        self.network = ArchitectNetwork(
            grid_rows=grid_rows, grid_cols=grid_cols
        ).to(DEVICE)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Experience buffer
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        # Training stats
        self.episode_count = 0
        self.total_reward = 0.0
    
    def generate_layout(self, temperature: float = 1.0
                        ) -> Tuple[List, List, List]:
        """
        Generate a security layout for the current episode.
        
        Args:
            temperature: Exploration temperature (higher = more random)
        
        Returns:
            (walls, cameras, guards) — placement specifications
        """
        self.network.eval()
        
        # Create initial grid state (just start and vault markers)
        grid_state = np.zeros((1, 1, self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Mark start (2) and vault (3) normalized
        grid_state[0, 0, 1, 1] = TileType.START / 5.0
        grid_state[0, 0, self.grid_rows - 2, self.grid_cols - 2] = TileType.VAULT / 5.0
        
        state_tensor = torch.FloatTensor(grid_state).to(DEVICE)
        
        with torch.no_grad():
            walls, cameras, guards, log_prob, value = \
                self.network.generate_layout(state_tensor, self.budget, temperature)
        
        # Store for training
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return walls, cameras, guards
    
    def store_reward(self, reward: float):
        """Store the reward received after solver evaluation."""
        self.rewards.append(reward)
        self.total_reward += reward
        self.episode_count += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update the Architect's policy using PPO.
        
        Returns:
            Dict with training metrics
        """
        if len(self.rewards) == 0:
            return {"architect_loss": 0.0}
        
        self.network.train()
        
        # Convert to tensors
        rewards = torch.FloatTensor(self.rewards).to(DEVICE)
        old_log_probs = torch.stack(self.log_probs).to(DEVICE).detach()
        old_values = torch.stack([v.squeeze() for v in self.values]).to(DEVICE).detach()
        
        # Normalize rewards
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute advantages
        advantages = rewards - old_values
        
        # PPO update
        # Re-evaluate (simplified: use stored log_probs since layout is fixed)
        grid_state = np.zeros((1, 1, self.grid_rows, self.grid_cols), dtype=np.float32)
        grid_state[0, 0, 1, 1] = TileType.START / 5.0
        grid_state[0, 0, self.grid_rows - 2, self.grid_cols - 2] = TileType.VAULT / 5.0
        state_tensor = torch.FloatTensor(grid_state).to(DEVICE)
        
        _, new_values, _ = self.network(state_tensor)
        new_value = new_values.squeeze()
        
        # Value loss (new_value is scalar from single state, so compare against mean reward)
        if len(rewards) > 0:
            target_value = rewards.mean()
            value_loss = F.mse_loss(new_value, target_value)
        else:
            value_loss = torch.tensor(0.0, device=DEVICE)
        
        # Policy loss (simplified PPO)
        policy_loss = -(old_log_probs * advantages.detach()).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coeff * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        metrics = {
            "architect_policy_loss": policy_loss.item(),
            "architect_value_loss": value_loss.item() if isinstance(value_loss, torch.Tensor) else 0.0,
            "architect_total_loss": total_loss.item(),
            "architect_avg_reward": self.total_reward / max(self.episode_count, 1),
        }
        
        # Clear buffers
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        
        return metrics
    
    def save(self, path: str):
        """Save the Architect's model."""
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


# Need F for the loss function
import torch.nn.functional as F
