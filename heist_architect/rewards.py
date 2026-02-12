"""
Reward Calculator for the Heist Architect framework.
Implements the zero-sum adversarial reward system.
"""

from typing import Dict, Tuple
from .environment import HeistEnvironment


class RewardCalculator:
    """
    Calculates rewards for both the Architect and Solver agents.
    
    Reward Structure (Zero-Sum Adversarial):
    
    Architect:
        +1.0  — Solver detected by camera/guard
        -1.0  — Level is unsolvable (no path from start to vault)
         0.0  — Solver reaches the vault (Architect failed)
        +0.2  — Bonus for longer solver paths (level is challenging)
    
    Solver:
        +10.0 — Successfully reaching the vault
        -1.0  — Detected by a camera or guard
        -0.01 — Small step penalty (encourages efficiency)
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Architect rewards
        self.architect_detect = self.config.get("architect_detect", 1.0)
        self.architect_invalid = self.config.get("architect_invalid", -1.0)
        self.architect_vault_fail = self.config.get("architect_vault_fail", -0.5)
        self.architect_difficulty_bonus = self.config.get("architect_difficulty_bonus", 0.2)
        
        # Solver rewards
        self.solver_vault = self.config.get("solver_vault", 10.0)
        self.solver_detected = self.config.get("solver_detected", -1.0)
        self.solver_step = self.config.get("solver_step", -0.01)
        self.solver_timeout = self.config.get("solver_timeout", -0.5)
    
    def calculate_architect_reward(self, env: HeistEnvironment,
                                    solve_rate: float = 0.0) -> float:
        """
        Calculate the Architect's reward after a batch of Solver episodes.
        
        Args:
            env: The HeistEnvironment after level setup
            solve_rate: Fraction of solver episodes that succeeded (0-1)
        
        Returns:
            float: Total reward for the Architect
        """
        reward = 0.0
        
        # Penalty for unsolvable level
        if not env.is_level_valid():
            return self.architect_invalid
        
        # Reward based on detection rate (inverse of solve rate)
        detection_rate = 1.0 - solve_rate
        reward += detection_rate * self.architect_detect
        
        # Penalty if solver reaches vault too easily
        if solve_rate > 0.8:
            reward += self.architect_vault_fail
        
        # Bonus for creating challenging levels (moderate solve rate)
        if 0.2 <= solve_rate <= 0.6:
            reward += self.architect_difficulty_bonus
        
        return reward
    
    def calculate_solver_episode_reward(self, env: HeistEnvironment) -> float:
        """
        Calculate the Solver's total reward from a single episode.
        
        Args:
            env: The HeistEnvironment after episode completion
        
        Returns:
            float: Total episode reward
        """
        reward = 0.0
        
        if env.vault_reached:
            reward += self.solver_vault
        
        if env.solver_detected:
            reward += self.solver_detected
        
        # Step penalty is already applied during env.step()
        # Add timeout penalty
        if env.tick >= env.config.max_steps and not env.vault_reached:
            reward += self.solver_timeout
        
        return reward
    
    def get_reward_summary(self) -> Dict[str, float]:
        """Return a summary of all reward values for logging."""
        return {
            "architect_detect": self.architect_detect,
            "architect_invalid": self.architect_invalid,
            "architect_vault_fail": self.architect_vault_fail,
            "architect_difficulty_bonus": self.architect_difficulty_bonus,
            "solver_vault": self.solver_vault,
            "solver_detected": self.solver_detected,
            "solver_step": self.solver_step,
            "solver_timeout": self.solver_timeout,
        }
