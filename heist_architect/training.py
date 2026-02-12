"""
Adversarial Training Loop — GAN-style co-evolution of Architect and Solver.

The training loop:
1. WARMUP: Solver learns basic navigation on empty/simple layouts
2. Architect generates a security layout
3. Solver attempts the level N times
4. Both agents update their policies
5. Difficulty scales via curriculum (increasing Architect budget)

Features:
- Resume from checkpoint (loads latest saved weights)
- Interactive episodes (manual budget, freeze agents, etc.)
- Game log (persistent record of every episode)
"""

import torch
import numpy as np
import json
import os
import time
import glob
import re
from typing import Dict, Optional, List
from collections import deque
from datetime import datetime

from .environment import HeistEnvironment, EnvironmentConfig
from .agents.architect import ArchitectAgent
from .agents.solver import SolverAgent
from .rewards import RewardCalculator
from .utils import DEVICE


class GameLogEntry:
    """A single episode record in the game log."""
    
    def __init__(self, episode: int, phase: str, budget: int,
                 walls: int, cameras: int, guards: int,
                 solve_rate: float, detection_rate: float, timeout_rate: float,
                 architect_reward: float, solver_reward: float,
                 avg_steps: float, level_valid: bool,
                 is_interactive: bool = False,
                 freeze_architect: bool = False, freeze_solver: bool = False,
                 temperature: float = 1.0, timestamp: str = ""):
        self.data = {
            "episode": episode,
            "phase": phase,
            "budget": budget,
            "walls": walls,
            "cameras": cameras,
            "guards": guards,
            "solve_rate": round(solve_rate, 3),
            "detection_rate": round(detection_rate, 3),
            "timeout_rate": round(timeout_rate, 3),
            "architect_reward": round(architect_reward, 3),
            "solver_reward": round(solver_reward, 3),
            "avg_steps": round(avg_steps, 1),
            "level_valid": level_valid,
            "is_interactive": is_interactive,
            "freeze_architect": freeze_architect,
            "freeze_solver": freeze_solver,
            "temperature": round(temperature, 2),
            "timestamp": timestamp or datetime.now().strftime("%H:%M:%S"),
        }
    
    def to_dict(self):
        return self.data


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
    
    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.history = json.load(f)

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
        
        # Game log — persistent record of ALL episodes
        self.game_log: List[GameLogEntry] = []
        
        # Track total episodes across sessions (for resume)
        self.global_episode = 0
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # State for visualization server
        self.current_state = None
        self.training_active = False
    
    # ==================================================================
    # Resume from Checkpoint
    # ==================================================================
    
    def find_latest_checkpoint(self) -> Optional[int]:
        """Find the latest checkpoint episode number in save_dir."""
        pattern = os.path.join(self.save_dir, "architect_ep*.pt")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        episodes = []
        for f in files:
            match = re.search(r'architect_ep(\d+)\.pt', f)
            if match:
                episodes.append(int(match.group(1)))
        
        return max(episodes) if episodes else None
    
    def list_checkpoints(self) -> List[int]:
        """Return a sorted list of available checkpoint episode numbers."""
        checkpoints = []
        # Look for solver checkpoints as the source of truth
        files = glob.glob(os.path.join(self.save_dir, "solver_ep*.pt"))
        for f in files:
            match = re.search(r"solver_ep(\d+).pt", f)
            if match:
                checkpoints.append(int(match.group(1)))
        return sorted(checkpoints)

    def load_checkpoint(self, episode: int) -> bool:
        """Load a specific checkpoint episode. Returns True if successful."""
        arch_path = os.path.join(self.save_dir, f"architect_ep{episode}.pt")
        solver_path = os.path.join(self.save_dir, f"solver_ep{episode}.pt")
        
        if not (os.path.exists(arch_path) and os.path.exists(solver_path)):
            print(f"Checkpoint not found for episode {episode}")
            return False
            
        print(f"Loading checkpoint from episode {episode}...")
        self.architect.load(arch_path)
        self.solver.load(solver_path)
        
        # Try to load metrics if available, but not critical for simulation
        metrics_path = os.path.join(self.log_dir, "training_metrics.json")
        if os.path.exists(metrics_path):
            self.metrics.load(metrics_path)
            
        # Try to load game log
        log_path = os.path.join(self.log_dir, "game_log.json")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                self.game_log = [GameLogEntry(**entry) for entry in json.load(f)]
                
        self.global_episode = episode
        return True

    def resume_from_checkpoint(self) -> int:
        """
        Load the latest checkpoint and return the episode number.
        Returns 0 if no checkpoint found.
        """
        latest_ep = self.find_latest_checkpoint()
        if latest_ep == 0 or latest_ep is None:
            print("  No checkpoints found. Starting from scratch.")
            return 0
        
        if self.load_checkpoint(latest_ep):
            print(f"  Resuming from episode {latest_ep}")
            return latest_ep
        return 0
    
    # ==================================================================
    # Curriculum
    # ==================================================================
    
    def get_curriculum_phase(self, episode: int):
        """Determine budget and allowed asset types based on curriculum schedule."""
        phase = self.CURRICULUM[0]
        for threshold, budget, cams, guards, desc in self.CURRICULUM:
            if episode >= threshold:
                phase = (threshold, budget, cams, guards, desc)
        return phase
    
    # ==================================================================
    # Warmup
    # ==================================================================
    
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
    
    # ==================================================================
    # Main Training Loop
    # ==================================================================
    
    def train(self, callback=None, resume: bool = False):
        """
        Run the full adversarial training loop.
        
        Args:
            callback: Optional function called each episode with 
                      (episode, metrics, env_state)
            resume: If True, load latest checkpoint before training
        """
        self.training_active = True
        
        start_episode = 0
        if resume:
            start_episode = self.resume_from_checkpoint()
        
        print(f"\n{'='*60}")
        print(f"  Heist Architect — Adversarial Training")
        print(f"  Device: {DEVICE}")
        print(f"  Grid: {self.config.grid_rows}x{self.config.grid_cols}")
        print(f"  Episodes: {start_episode + 1} → {start_episode + self.total_episodes}")
        print(f"  Solver Attempts per Layout: {self.solver_episodes}")
        if resume and start_episode > 0:
            print(f"  RESUMED from episode {start_episode}")
        print(f"{'='*60}\n")
        
        # Phase 0: Warmup — solver learns basic navigation (skip if resuming)
        if start_episode == 0:
            self._run_warmup()
        
        start_time = time.time()
        last_phase_desc = ""
        
        for ep_idx in range(1, self.total_episodes + 1):
            episode = start_episode + ep_idx
            self.global_episode = episode
            
            # Run one training episode
            ep_metrics, log_entry = self._run_one_episode(
                episode=episode,
                is_interactive=False,
            )
            
            # Log
            self.metrics.log(episode, ep_metrics)
            self.metrics.recent_solve_rates.append(ep_metrics["solve_rate"])
            self.game_log.append(log_entry)
            
            # Callback for visualization
            if callback:
                callback(episode, ep_metrics, self.current_state)
            
            # Announce phase changes
            phase_desc = ep_metrics.get("phase", "?")
            if phase_desc != last_phase_desc:
                print(f"\n  >>> Phase: {phase_desc} (budget={ep_metrics['budget']}) <<<\n")
                last_phase_desc = phase_desc
            
            # Print progress
            if ep_idx % 10 == 0:
                self._print_progress(episode, ep_idx, ep_metrics, start_time)
            
            # Save checkpoints
            if ep_idx % 50 == 0:
                self._save_checkpoint(episode)
        
        # Final save
        final_ep = start_episode + self.total_episodes
        self._save_checkpoint(final_ep)
        self._save_game_log()
        self.metrics.save(os.path.join(self.log_dir, "training_metrics.json"))
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Total Time: {elapsed/60:.1f} minutes")
        print(f"  Episodes Trained: {self.total_episodes}")
        print(f"  Global Episode: {final_ep}")
        print(f"  Final Metrics:\n{self.metrics.get_summary()}")
        print(f"{'='*60}\n")
        
        self.training_active = False
    
    def _run_one_episode(self, episode: int, is_interactive: bool = False,
                          budget_override: int = None,
                          freeze_architect: bool = False,
                          freeze_solver: bool = False,
                          temperature_override: float = None,
                          solver_attempts_override: int = None,
                          allow_cameras_override: bool = None,
                          allow_guards_override: bool = None):
        """
        Run a single training episode. Used by both train() and interactive mode.
        
        Returns:
            (ep_metrics dict, GameLogEntry)
        """
        # Determine curriculum phase
        _, budget, allow_cameras, allow_guards, phase_desc = \
            self.get_curriculum_phase(episode)
        
        # Apply overrides for interactive mode
        if budget_override is not None:
            budget = budget_override
        if allow_cameras_override is not None:
            allow_cameras = allow_cameras_override
        if allow_guards_override is not None:
            allow_guards = allow_guards_override
        
        self.architect.budget = budget
        self.env.budget.scale_budget(budget)
        
        # Temperature
        if temperature_override is not None:
            temperature = temperature_override
        else:
            temperature = max(0.5, 2.0 - episode / max(self.total_episodes, 1) * 1.5)
        
        solver_attempts = solver_attempts_override or self.solver_episodes
        
        if is_interactive:
            phase_desc = f"Interactive (budget={budget})"
        
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
        
        num_walls = len(walls)
        num_cameras = len(cameras)
        num_guards = len(guards)
        
        if not is_valid:
            # Penalize invalid layout, skip solver phase
            if not freeze_architect:
                self.architect.store_reward(self.reward_calc.architect_invalid)
                self.architect.update()
            
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
            
            log_entry = GameLogEntry(
                episode=episode, phase=phase_desc, budget=budget,
                walls=num_walls, cameras=num_cameras, guards=num_guards,
                solve_rate=0, detection_rate=0, timeout_rate=1,
                architect_reward=self.reward_calc.architect_invalid,
                solver_reward=0, avg_steps=0, level_valid=False,
                is_interactive=is_interactive,
                freeze_architect=freeze_architect, freeze_solver=freeze_solver,
                temperature=temperature,
            )
            
            return ep_metrics, log_entry
        
        # -------------------------------------------------------
        # Step 2: Solver attempts the level multiple times
        # -------------------------------------------------------
        solve_count = 0
        detect_count = 0
        timeout_count = 0
        total_steps = 0
        total_solver_reward = 0.0
        
        for attempt in range(solver_attempts):
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
        # Step 3: Calculate rewards and update agents
        # -------------------------------------------------------
        solve_rate = solve_count / solver_attempts
        detection_rate = detect_count / solver_attempts
        
        # Architect reward
        arch_reward = self.reward_calc.calculate_architect_reward(
            self.env, solve_rate
        )
        
        if not freeze_architect:
            self.architect.store_reward(arch_reward)
            self.architect.update()
        
        if not freeze_solver:
            self.solver.update()
        else:
            # Clear solver buffers without updating
            self.solver._clear_buffers()
        
        # Store current state for visualization
        self.current_state = self.env.get_environment_state()
        
        # -------------------------------------------------------
        # Step 4: Build metrics and log entry
        # -------------------------------------------------------
        ep_metrics = {
            "solve_rate": solve_rate,
            "detection_rate": detection_rate,
            "timeout_rate": timeout_count / solver_attempts,
            "architect_reward": arch_reward,
            "solver_reward": total_solver_reward / solver_attempts,
            "architect_loss": 0,
            "solver_loss": 0,
            "avg_steps": total_steps / solver_attempts,
            "budget": budget,
            "phase": phase_desc,
        }
        
        log_entry = GameLogEntry(
            episode=episode, phase=phase_desc, budget=budget,
            walls=num_walls, cameras=num_cameras, guards=num_guards,
            solve_rate=solve_rate, detection_rate=detection_rate,
            timeout_rate=timeout_count / solver_attempts,
            architect_reward=arch_reward,
            solver_reward=total_solver_reward / solver_attempts,
            avg_steps=total_steps / solver_attempts,
            level_valid=True,
            is_interactive=is_interactive,
            freeze_architect=freeze_architect, freeze_solver=freeze_solver,
            temperature=temperature,
        )
        
        return ep_metrics, log_entry
    
    # ==================================================================
    # Interactive Episode Mode
    # ==================================================================
    
    def run_interactive_episodes(self, num_episodes: int = 1,
                                  budget: int = 15,
                                  freeze_architect: bool = False,
                                  freeze_solver: bool = False,
                                  temperature: float = 1.0,
                                  solver_attempts: int = 20,
                                  allow_cameras: bool = True,
                                  allow_guards: bool = True,
                                  callback=None) -> List[Dict]:
        """
        Run interactive episodes with custom parameters.
        Both agents use their current learned weights.
        
        Args:
            num_episodes: How many episodes to run
            budget: Architect's budget for these episodes
            freeze_architect: If True, Architect doesn't learn
            freeze_solver: If True, Solver doesn't learn
            temperature: Architect exploration temperature
            solver_attempts: Solver attempts per layout
            allow_cameras: Allow cameras in layout
            allow_guards: Allow guards in layout
            callback: Called per episode with (episode, metrics, env_state)
        
        Returns:
            List of metrics dicts for each episode
        """
        results = []
        
        for i in range(num_episodes):
            self.global_episode += 1
            episode = self.global_episode
            
            ep_metrics, log_entry = self._run_one_episode(
                episode=episode,
                is_interactive=True,
                budget_override=budget,
                freeze_architect=freeze_architect,
                freeze_solver=freeze_solver,
                temperature_override=temperature,
                solver_attempts_override=solver_attempts,
                allow_cameras_override=allow_cameras,
                allow_guards_override=allow_guards,
            )
            
            self.metrics.log(episode, ep_metrics)
            self.game_log.append(log_entry)
            results.append(ep_metrics)
            
            if callback:
                callback(episode, ep_metrics, self.current_state)
        
        # Auto-save after interactive block
        self._save_checkpoint(self.global_episode)
        self._save_game_log()
        self.metrics.save(os.path.join(self.log_dir, "training_metrics.json"))
        
        return results
    
    # ==================================================================
    # Game Log
    # ==================================================================
    
    def get_game_log(self) -> List[Dict]:
        """Return the full game log as a list of dicts."""
        return [entry.to_dict() for entry in self.game_log]
    
    def _save_game_log(self):
        """Save game log to disk."""
        log_path = os.path.join(self.log_dir, "game_log.json")
        with open(log_path, 'w') as f:
            json.dump([e.to_dict() for e in self.game_log], f, indent=2)
    
    # ==================================================================
    # Utilities
    # ==================================================================
    
    def _print_progress(self, episode: int, ep_idx: int, metrics: Dict, start_time: float):
        """Print a progress line."""
        elapsed = time.time() - start_time
        eps_per_sec = ep_idx / max(elapsed, 1)
        
        print(
            f"[Ep {episode:4d}] "
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
        self._save_game_log()
    
    def simulate_episode(self, budget: int = 15, solver_attempts: int = 1) -> Dict:
        """
        Run a simulation with specific parameters and return frames for playback.
        Runs multiple attempts and returns the best one.
        """
        # Set architect budget temporarily
        original_budget = self.architect.budget
        self.architect.budget = budget
        
        # Generate layout
        walls, cameras, guards = self.architect.generate_layout(temperature=0.5)
        self.env.set_layout(walls, cameras, guards)
        
        # Restore budget
        self.architect.budget = original_budget
        
        best_outcome = "timeout"
        best_frames = []
        max_reward = -float('inf')
        
        # Run attempts
        for i in range(solver_attempts):
            obs = self.env.reset()
            self.solver.reset()
            frames = []
            episode_reward = 0.0
            state = self.env.get_state_tensor()
            outcome = "timeout"
            
            for step in range(self.config.max_steps):
                # Store frame only for the potential best episode
                frames.append(self.env.get_environment_state())
                
                action = self.solver.select_action(state)
                obs, reward, done, info = self.env.step(action)
                state = self.env.get_state_tensor()
                episode_reward += reward
                
                if done:
                    frames.append(self.env.get_environment_state())
                    outcome = info.get("status", "timeout")
                    break
            
            # Clean up buffers immediately
            self.solver._clear_buffers()
            
            # Determine if this attempt is better
            # Priority: Vault Reached > Not Detected > Higher Reward
            is_better = False
            
            if i == 0:
                is_better = True
            else:
                if outcome == "vault_reached":
                    if best_outcome != "vault_reached":
                        is_better = True
                    elif episode_reward > max_reward:
                        is_better = True
                elif outcome == "detected":
                    if best_outcome == "timeout":
                        is_better = True
                    elif best_outcome == "detected" and episode_reward > max_reward:
                        is_better = True
                elif outcome == "timeout":
                    if best_outcome == "timeout" and episode_reward > max_reward:
                        is_better = True
            
            if is_better:
                best_outcome = outcome
                max_reward = episode_reward
                best_frames = frames
        
        return {
            "frames": best_frames,
            "outcome": best_outcome,
            "total_steps": len(best_frames) - 1,
            "reward": max_reward
        }
