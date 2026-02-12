"""
Adversarial Training Loop — GAN-style co-evolution of Architect and Solver.

The training loop:
1. WARMUP: Solver learns basic navigation on empty/simple layouts
2. Architect generates a security layout
3. Solver attempts the level N times
4. Both agents update their policies
5. Difficulty scales via curriculum (increasing Architect budget)
"""

import torch
import numpy as np
import json
import os
import time
from typing import Dict, Optional
from collections import deque

from .environment import HeistEnvironment, EnvironmentConfig
from .agents.architect import ArchitectAgent
from .agents.solver import SolverAgent
from .rewards import RewardCalculator
from .utils import DEVICE


class TrainingMetrics:
    """Tracks and logs training metrics."""
    
    def __init__(self):
        self.history = {
            "episode": [],
            "solve_rate": [],
            "detection_rate": [],
            "timeout_rate": [],
            "architect_reward": [],
            "solver_reward": [],
            "architect_loss": [],
            "solver_loss": [],
            "avg_steps": [],
            "budget": [],
            "phase": [],
        }
        self.recent_solve_rates = deque(maxlen=50)
    
    def log(self, episode: int, metrics: Dict):
        for key in self.history:
            if key in metrics:
                self.history[key].append(metrics[key])
        self.history["episode"].append(episode)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_summary(self, last_n: int = 10) -> str:
        lines = []
        for key in ["solve_rate", "detection_rate", "architect_reward", "solver_reward"]:
            vals = self.history.get(key, [])
            if vals:
                recent = vals[-last_n:]
                lines.append(f"  {key}: {np.mean(recent):.3f}")
        return "\n".join(lines)


class AdversarialTrainer:
    """
    GAN-style adversarial training loop for the Heist Architect framework.
    
    Training phases (curriculum):
        Phase 0 (Warmup):     No security — solver learns to navigate to vault
        Phase 1 (Ep 1-80):    Budget=5,  walls only (no cameras/guards yet)
        Phase 2 (Ep 81-200):  Budget=8,  walls + cameras (slow rotation)
        Phase 3 (Ep 201-400): Budget=15, full complexity
        Phase 4 (Ep 400+):    Budget=22, expert difficulty
    """
    
    # Curriculum schedule: (episode_threshold, budget, allow_cameras, allow_guards, description)
    CURRICULUM = [
        (0,   5,  False, False, "Walls Only"),
        (80,  8,  True,  False, "Walls + Cameras"),
        (200, 15, True,  True,  "Full Security"),
        (400, 22, True,  True,  "Expert"),
    ]
    
    WARMUP_EPISODES = 30  # Pure navigation warmup
    
    def __init__(self, config: Optional[EnvironmentConfig] = None,
                 solver_episodes_per_layout: int = 20,
                 total_episodes: int = 500,
                 save_dir: str = "checkpoints",
                 log_dir: str = "logs",
                 architect_lr: float = 3e-4,
                 solver_lr: float = 1e-3):
        
        self.config = config or EnvironmentConfig()
        self.solver_episodes = solver_episodes_per_layout
        self.total_episodes = total_episodes
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Initialize environment
        self.env = HeistEnvironment(self.config)
        
        # Initialize agents
        self.architect = ArchitectAgent(
            grid_rows=self.config.grid_rows,
            grid_cols=self.config.grid_cols,
            budget=self.config.architect_budget,
            lr=architect_lr
        )
        
        self.solver = SolverAgent(
            grid_rows=self.config.grid_rows,
            grid_cols=self.config.grid_cols,
            lr=solver_lr
        )
        
        # Reward calculator
        self.reward_calc = RewardCalculator()
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # State for visualization server
        self.current_state = None
        self.training_active = False
    
    def get_curriculum_phase(self, episode: int):
        """Determine budget and allowed asset types based on curriculum schedule."""
        phase = self.CURRICULUM[0]
        for threshold, budget, cams, guards, desc in self.CURRICULUM:
            if episode >= threshold:
                phase = (threshold, budget, cams, guards, desc)
        return phase
    
    def _run_warmup(self):
        """
        Warmup phase: Train the Solver on empty grids (no security).
        This teaches the Solver basic navigation toward the vault.
        """
        print(f"\n{'='*60}")
        print(f"  WARMUP PHASE — Teaching Solver basic navigation")
        print(f"  {self.WARMUP_EPISODES} episodes on empty grids")
        print(f"{'='*60}\n")
        
        # Set empty layout (no security)
        self.env.set_layout([], [], [])
        
        for ep in range(1, self.WARMUP_EPISODES + 1):
            # Run multiple attempts per episode
            solve_count = 0
            total_reward = 0.0
            total_steps = 0
            
            for attempt in range(self.solver_episodes):
                obs = self.env.reset()
                self.solver.reset()
                state = self.env.get_state_tensor()
                ep_reward = 0.0
                
                for step in range(self.config.max_steps):
                    action = self.solver.select_action(state)
                    obs, reward, done, info = self.env.step(action)
                    self.solver.store_transition(reward, done)
                    ep_reward += reward
                    state = self.env.get_state_tensor()
                    
                    if done:
                        break
                
                if info.get("status") == "vault_reached":
                    solve_count += 1
                total_reward += ep_reward
                total_steps += self.env.tick
                self.solver.end_episode(ep_reward)
            
            # Update solver
            solver_metrics = self.solver.update()
            
            solve_rate = solve_count / self.solver_episodes
            avg_reward = total_reward / self.solver_episodes
            
            if ep % 5 == 0 or ep == 1:
                print(f"  [Warmup {ep:3d}/{self.WARMUP_EPISODES}] "
                      f"Solve: {solve_rate:.2f} | "
                      f"Reward: {avg_reward:+.2f} | "
                      f"Steps: {total_steps/self.solver_episodes:.0f}")
        
        print(f"\n  Warmup complete! Final solve rate: {solve_rate:.2f}\n")
    
    def train(self, callback=None):
        """
        Run the full adversarial training loop.
        
        Args:
            callback: Optional function called each episode with 
                      (episode, metrics, env_state)
        """
        self.training_active = True
        print(f"\n{'='*60}")
        print(f"  Heist Architect — Adversarial Training")
        print(f"  Device: {DEVICE}")
        print(f"  Grid: {self.config.grid_rows}x{self.config.grid_cols}")
        print(f"  Total Episodes: {self.total_episodes}")
        print(f"  Solver Attempts per Layout: {self.solver_episodes}")
        print(f"{'='*60}\n")
        
        # Phase 0: Warmup — solver learns basic navigation
        self._run_warmup()
        
        start_time = time.time()
        last_phase_desc = ""
        
        for episode in range(1, self.total_episodes + 1):
            # Update curriculum
            _, budget, allow_cameras, allow_guards, phase_desc = \
                self.get_curriculum_phase(episode)
            self.architect.budget = budget
            self.env.budget.scale_budget(budget)
            
            # Announce phase changes
            if phase_desc != last_phase_desc:
                print(f"\n  >>> Phase: {phase_desc} (budget={budget}) <<<\n")
                last_phase_desc = phase_desc
            
            # Temperature for exploration (anneal over time)
            temperature = max(0.5, 2.0 - episode / self.total_episodes * 1.5)
            
            # -------------------------------------------------------
            # Step 1: Architect generates a layout
            # -------------------------------------------------------
            walls, cameras, guards = self.architect.generate_layout(temperature)
            
            # Filter by curriculum phase
            if not allow_cameras:
                cameras = []
            if not allow_guards:
                guards = []
            
            # Apply layout to environment
            is_valid = self.env.set_layout(walls, cameras, guards)
            
            if not is_valid:
                # Penalize invalid layout, skip solver phase
                self.architect.store_reward(self.reward_calc.architect_invalid)
                
                ep_metrics = {
                    "solve_rate": 0.0,
                    "detection_rate": 0.0,
                    "timeout_rate": 1.0,
                    "architect_reward": self.reward_calc.architect_invalid,
                    "solver_reward": 0.0,
                    "avg_steps": 0,
                    "budget": budget,
                    "phase": phase_desc,
                }
                self.metrics.log(episode, ep_metrics)
                
                if episode % 10 == 0:
                    self._print_progress(episode, ep_metrics, start_time)
                
                continue
            
            # -------------------------------------------------------
            # Step 2: Solver attempts the level multiple times
            # -------------------------------------------------------
            solve_count = 0
            detect_count = 0
            timeout_count = 0
            total_steps = 0
            total_solver_reward = 0.0
            
            for attempt in range(self.solver_episodes):
                obs = self.env.reset()
                self.solver.reset()
                
                episode_reward = 0.0
                state = self.env.get_state_tensor()
                
                for step in range(self.config.max_steps):
                    action = self.solver.select_action(state)
                    obs, reward, done, info = self.env.step(action)
                    
                    self.solver.store_transition(reward, done)
                    episode_reward += reward
                    
                    state = self.env.get_state_tensor()
                    
                    if done:
                        break
                
                # Track outcomes
                if info.get("status") == "vault_reached":
                    solve_count += 1
                elif info.get("status") == "detected":
                    detect_count += 1
                else:
                    timeout_count += 1
                
                total_steps += self.env.tick
                total_solver_reward += episode_reward
                self.solver.end_episode(episode_reward)
            
            # -------------------------------------------------------
            # Step 3: Calculate rewards and update both agents
            # -------------------------------------------------------
            solve_rate = solve_count / self.solver_episodes
            detection_rate = detect_count / self.solver_episodes
            
            # Architect reward
            arch_reward = self.reward_calc.calculate_architect_reward(
                self.env, solve_rate
            )
            self.architect.store_reward(arch_reward)
            
            # Update both agents
            arch_metrics = self.architect.update()
            solver_metrics = self.solver.update()
            
            # Store current state for visualization
            self.current_state = self.env.get_environment_state()
            
            # -------------------------------------------------------
            # Step 4: Log metrics
            # -------------------------------------------------------
            ep_metrics = {
                "solve_rate": solve_rate,
                "detection_rate": detection_rate,
                "timeout_rate": timeout_count / self.solver_episodes,
                "architect_reward": arch_reward,
                "solver_reward": total_solver_reward / self.solver_episodes,
                "architect_loss": arch_metrics.get("architect_total_loss", 0),
                "solver_loss": solver_metrics.get("solver_policy_loss", 0),
                "avg_steps": total_steps / self.solver_episodes,
                "budget": budget,
                "phase": phase_desc,
            }
            self.metrics.log(episode, ep_metrics)
            self.metrics.recent_solve_rates.append(solve_rate)
            
            # Callback for visualization
            if callback:
                callback(episode, ep_metrics, self.current_state)
            
            # Print progress
            if episode % 10 == 0:
                self._print_progress(episode, ep_metrics, start_time)
            
            # Save checkpoints
            if episode % 50 == 0:
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint(self.total_episodes)
        self.metrics.save(os.path.join(self.log_dir, "training_metrics.json"))
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Total Time: {elapsed/60:.1f} minutes")
        print(f"  Final Metrics:\n{self.metrics.get_summary()}")
        print(f"{'='*60}\n")
        
        self.training_active = False
    
    def _print_progress(self, episode: int, metrics: Dict, start_time: float):
        """Print a progress line."""
        elapsed = time.time() - start_time
        eps_per_sec = episode / max(elapsed, 1)
        
        print(
            f"[Ep {episode:4d}/{self.total_episodes}] "
            f"Solve: {metrics['solve_rate']:.2f} | "
            f"Detect: {metrics['detection_rate']:.2f} | "
            f"ArchR: {metrics['architect_reward']:+.2f} | "
            f"SolvR: {metrics['solver_reward']:+.2f} | "
            f"Steps: {metrics['avg_steps']:.0f} | "
            f"Budget: {metrics['budget']} | "
            f"Phase: {metrics.get('phase', '?')} | "
            f"{eps_per_sec:.1f} ep/s"
        )
    
    def _save_checkpoint(self, episode: int):
        """Save models and metrics."""
        self.architect.save(
            os.path.join(self.save_dir, f"architect_ep{episode}.pt")
        )
        self.solver.save(
            os.path.join(self.save_dir, f"solver_ep{episode}.pt")
        )
        self.metrics.save(
            os.path.join(self.log_dir, "training_metrics.json")
        )
    
    def demo_episode(self) -> Dict:
        """
        Run a single demo episode for visualization.
        Returns the full environment state at each step.
        """
        # Generate a layout
        walls, cameras, guards = self.architect.generate_layout(temperature=0.5)
        self.env.set_layout(walls, cameras, guards)
        
        obs = self.env.reset()
        self.solver.reset()
        
        frames = []
        state = self.env.get_state_tensor()
        
        for step in range(self.config.max_steps):
            frames.append(self.env.get_environment_state())
            
            action = self.solver.select_action(state)
            obs, reward, done, info = self.env.step(action)
            state = self.env.get_state_tensor()
            
            if done:
                frames.append(self.env.get_environment_state())
                break
        
        return {
            "frames": frames,
            "outcome": info.get("status", "unknown"),
            "total_steps": self.env.tick,
        }
