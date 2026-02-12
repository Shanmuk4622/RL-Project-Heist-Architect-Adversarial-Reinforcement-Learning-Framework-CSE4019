"""
Heist Environment — The 2D Grid Battleground.

A time-stepped simulation where the Architect places security assets
and the Solver navigates to the vault while avoiding detection.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

from .utils import TileType, bfs_path_exists, create_empty_grid, grid_to_text, manhattan_distance
from .components.security import Wall, Camera, Guard
from .components.visibility import DynamicVisibilityMap
from .components.budget import BudgetManager, BUDGET_COSTS


@dataclass
class EnvironmentConfig:
    """Configuration for the Heist Environment."""
    grid_rows: int = 20
    grid_cols: int = 20
    max_steps: int = 200  # max ticks per episode
    start_pos: Tuple[int, int] = (1, 1)
    vault_pos: Tuple[int, int] = None  # auto-set if None
    architect_budget: int = 15
    # Reward values
    reward_vault: float = 10.0
    reward_detection: float = -1.0
    reward_step: float = -0.01  # small penalty per step to encourage efficiency
    reward_architect_detect: float = 1.0
    reward_architect_invalid: float = -1.0

    def __post_init__(self):
        """Auto-set vault_pos if not provided."""
        if self.vault_pos is None:
            self.vault_pos = (self.grid_rows - 2, self.grid_cols - 2)


class HeistEnvironment:
    """
    The 2D grid-based Heist environment.
    
    This environment manages the game loop between the Architect and Solver:
    1. Architect places security assets (walls, cameras, guards)
    2. Level is validated (path must exist from start to vault)
    3. Solver navigates the level while cameras rotate and guards patrol
    4. Episode ends when: vault reached, solver detected, or max steps exceeded
    """
    
    # Action mapping for the Solver
    ACTIONS = {
        0: (0, 0),    # WAIT
        1: (-1, 0),   # UP
        2: (1, 0),    # DOWN
        3: (0, -1),   # LEFT
        4: (0, 1),    # RIGHT
    }
    ACTION_NAMES = {0: "WAIT", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    NUM_SOLVER_ACTIONS = 5
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        
        # Grid state
        self.grid = create_empty_grid(self.config.grid_rows, self.config.grid_cols)
        self.grid[self.config.start_pos[0], self.config.start_pos[1]] = TileType.START
        self.grid[self.config.vault_pos[0], self.config.vault_pos[1]] = TileType.VAULT
        
        # Security components
        self.walls: List[Wall] = []
        self.cameras: List[Camera] = []
        self.guards: List[Guard] = []
        
        # Visibility
        self.visibility_map = DynamicVisibilityMap(
            self.config.grid_rows, self.config.grid_cols
        )
        
        # Budget
        self.budget = BudgetManager(total_budget=self.config.architect_budget)
        
        # Solver state
        self.solver_pos = self.config.start_pos
        self.tick = 0
        self.done = False
        self.solver_detected = False
        self.vault_reached = False
        
        # Distance tracking for reward shaping
        self._prev_dist = manhattan_distance(self.config.start_pos, self.config.vault_pos)
        self._initial_dist = self._prev_dist
        
        # Episode history (for visualization)
        self.solver_path: List[Tuple[int, int]] = [self.solver_pos]
        self.detection_events: List[Dict] = []
    
    # =======================================================================
    # Architect Phase: Level Generation
    # =======================================================================
    
    def set_layout(self, walls: List[Tuple[int, int]],
                   cameras: List[Dict], guards: List[Dict]) -> bool:
        """
        Set the security layout for this episode (Architect's action).
        
        Args:
            walls: List of (row, col) positions for walls
            cameras: List of dicts with keys: row, col, fov_angle, heading, rotation_speed, vision_range
            guards: List of dicts with keys: patrol_path, speed, vision_range, fov_angle
        
        Returns:
            True if the layout is valid (path exists from start to vault)
        """
        self._reset_layout()
        
        # Place walls
        for r, c in walls:
            if self._is_valid_placement(r, c) and self.budget.purchase("wall"):
                self.grid[r, c] = TileType.WALL
                self.walls.append(Wall(r, c))
        
        # Place cameras
        for cam_def in cameras:
            r, c = cam_def["row"], cam_def["col"]
            if self._is_valid_placement(r, c) and self.budget.purchase("camera"):
                cam = Camera(
                    row=r, col=c,
                    fov_angle=cam_def.get("fov_angle", 60.0),
                    heading=cam_def.get("heading", 0.0),
                    rotation_speed=cam_def.get("rotation_speed", 15.0),
                    vision_range=cam_def.get("vision_range", 6)
                )
                self.grid[r, c] = TileType.CAMERA
                self.cameras.append(cam)
        
        # Place guards
        for guard_def in guards:
            path = guard_def["patrol_path"]
            if path and self.budget.purchase("guard"):
                guard = Guard(
                    patrol_path=path,
                    speed=guard_def.get("speed", 1),
                    vision_range=guard_def.get("vision_range", 4),
                    fov_angle=guard_def.get("fov_angle", 90.0)
                )
                # Mark guard's starting position
                self.grid[guard.row, guard.col] = TileType.GUARD
                self.guards.append(guard)
        
        # Validate layout
        return self.is_level_valid()
    
    def is_level_valid(self) -> bool:
        """Check if a path exists from start to vault."""
        return bfs_path_exists(
            self.grid, self.config.start_pos, self.config.vault_pos
        )
    
    def _is_valid_placement(self, row: int, col: int) -> bool:
        """Check if a tile can have an asset placed on it."""
        if row <= 0 or row >= self.config.grid_rows - 1:
            return False
        if col <= 0 or col >= self.config.grid_cols - 1:
            return False
        tile = self.grid[row, col]
        return tile == TileType.EMPTY
    
    def _reset_layout(self):
        """Clear the current layout (keep border walls, start, vault)."""
        self.grid = create_empty_grid(self.config.grid_rows, self.config.grid_cols)
        self.grid[self.config.start_pos[0], self.config.start_pos[1]] = TileType.START
        self.grid[self.config.vault_pos[0], self.config.vault_pos[1]] = TileType.VAULT
        self.walls.clear()
        self.cameras.clear()
        self.guards.clear()
        self.budget.reset()
    
    # =======================================================================
    # Solver Phase: Navigation
    # =======================================================================
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the Solver's position and tick counter for a new attempt.
        The layout remains the same (Architect doesn't regenerate).
        
        Returns:
            Initial observation dict for the Solver
        """
        self.solver_pos = self.config.start_pos
        self.tick = 0
        self.done = False
        self.solver_detected = False
        self.vault_reached = False
        self.solver_path = [self.solver_pos]
        self.detection_events = []
        self.visibility_map.reset()
        
        # Reset distance tracking
        self._prev_dist = manhattan_distance(self.solver_pos, self.config.vault_pos)
        self._initial_dist = self._prev_dist
        
        # Reset camera headings and guard positions
        for cam in self.cameras:
            cam.heading = cam.heading  # keep initial heading
        for guard in self.guards:
            guard.current_idx = 0
        
        # Compute initial visibility
        wall_mask = (self.grid == TileType.WALL)
        self.visibility_map.update(self.cameras, self.guards, wall_mask)
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one time step.
        
        1. Solver moves (or waits)
        2. Cameras rotate, guards patrol
        3. Visibility map updates
        4. Check for detection / vault reached
        5. Apply reward shaping (distance-based guidance)
        
        Args:
            action: 0=WAIT, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        
        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            return self._get_observation(), 0.0, True, {"status": "already_done"}
        
        reward = self.config.reward_step  # small step penalty
        info = {"status": "running", "tick": self.tick}
        
        # 1. Move the Solver
        dr, dc = self.ACTIONS[action]
        new_r = self.solver_pos[0] + dr
        new_c = self.solver_pos[1] + dc
        
        if (0 <= new_r < self.config.grid_rows and
            0 <= new_c < self.config.grid_cols and
            self.grid[new_r, new_c] != TileType.WALL):
            self.solver_pos = (new_r, new_c)
        
        self.solver_path.append(self.solver_pos)
        
        # 2. Update cameras and guards
        for cam in self.cameras:
            cam.update()
        for guard in self.guards:
            guard.update()
        
        # 3. Update visibility map
        wall_mask = (self.grid == TileType.WALL)
        self.visibility_map.update(self.cameras, self.guards, wall_mask)
        
        # 4. Distance-based reward shaping (CRITICAL for learning)
        curr_dist = manhattan_distance(self.solver_pos, self.config.vault_pos)
        # Reward for getting closer to vault, penalty for moving away
        dist_reward = (self._prev_dist - curr_dist) * 0.1
        reward += dist_reward
        self._prev_dist = curr_dist
        
        # Proximity bonus — increasing reward as solver gets very close
        if curr_dist <= 3 and self._initial_dist > 3:
            reward += 0.05 * (3 - curr_dist)
        
        # 5. Check conditions
        # Detection check
        if self.visibility_map.is_visible(self.solver_pos[0], self.solver_pos[1]):
            self.solver_detected = True
            reward += self.config.reward_detection
            self.detection_events.append({
                "tick": self.tick,
                "position": self.solver_pos
            })
            self.done = True
            info["status"] = "detected"
        
        # Vault check
        if self.solver_pos == self.config.vault_pos:
            self.vault_reached = True
            reward += self.config.reward_vault
            self.done = True
            info["status"] = "vault_reached"
        
        # Timeout
        self.tick += 1
        if self.tick >= self.config.max_steps:
            self.done = True
            info["status"] = "timeout"
            # Give partial reward based on how close solver got
            closest_fraction = max(0, 1.0 - curr_dist / max(self._initial_dist, 1))
            reward += closest_fraction * 2.0  # up to +2 for getting close
        
        return self._get_observation(), reward, self.done, info
    
    # =======================================================================
    # Observations
    # =======================================================================
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Build the Solver's observation.
        
        Returns dict with:
            - occupancy_grid: 2D array of tile types (normalized)
            - visibility_map: 2D binary array of current surveillance
            - solver_position: (row, col) normalized
            - vault_direction: relative direction to vault (dr, dc) normalized
            - time_feature: normalized tick counter
        """
        rows, cols = self.config.grid_rows, self.config.grid_cols
        
        # Occupancy grid (normalized to 0-1)
        occupancy = self.grid.astype(np.float32) / max(TileType.GUARD, 1)
        
        # Current visibility
        visibility = self.visibility_map.visibility.copy()
        
        # Solver position (normalized)
        solver_norm = np.array([
            self.solver_pos[0] / rows,
            self.solver_pos[1] / cols
        ], dtype=np.float32)
        
        # Direction to vault (normalized)
        vault_dir = np.array([
            (self.config.vault_pos[0] - self.solver_pos[0]) / rows,
            (self.config.vault_pos[1] - self.solver_pos[1]) / cols
        ], dtype=np.float32)
        
        # Time feature
        time_feat = np.array([self.tick / self.config.max_steps], dtype=np.float32)
        
        return {
            "occupancy_grid": occupancy,
            "visibility_map": visibility,
            "solver_position": solver_norm,
            "vault_direction": vault_dir,
            "time_feature": time_feat,
        }
    
    def get_state_tensor(self) -> np.ndarray:
        """
        Flatten the observation into a single tensor for neural network input.
        Shape: (channels, rows, cols) — 3 spatial channels + embedded features
        """
        obs = self._get_observation()
        rows, cols = self.config.grid_rows, self.config.grid_cols
        
        # Create a position/direction channel with gradient toward vault
        pos_channel = np.zeros((rows, cols), dtype=np.float32)
        pos_channel[self.solver_pos[0], self.solver_pos[1]] = 1.0
        pos_channel[self.config.vault_pos[0], self.config.vault_pos[1]] = -1.0
        
        # Add distance gradient to help the network learn direction
        for r in range(rows):
            for c in range(cols):
                d = manhattan_distance((r, c), self.config.vault_pos)
                max_d = rows + cols
                pos_channel[r, c] += -0.3 * (d / max_d)  # gentle gradient toward vault
        
        # Stack all: 3 channels total
        state = np.stack([
            obs["occupancy_grid"],
            obs["visibility_map"],
            pos_channel
        ], axis=0)
        
        return state
    
    # =======================================================================
    # Info & Rendering
    # =======================================================================
    
    def get_architect_reward(self) -> float:
        """Calculate the Architect's reward for this episode."""
        if not self.is_level_valid():
            return self.config.reward_architect_invalid
        if self.solver_detected:
            return self.config.reward_architect_detect
        return 0.0
    
    def get_environment_state(self) -> Dict[str, Any]:
        """Get the full environment state for visualization."""
        return {
            "grid": self.grid.tolist(),
            "visibility": self.visibility_map.visibility.tolist(),
            "solver_pos": self.solver_pos,
            "solver_path": self.solver_path,
            "vault_pos": self.config.vault_pos,
            "start_pos": self.config.start_pos,
            "tick": self.tick,
            "done": self.done,
            "cameras": [
                {
                    "row": c.row, "col": c.col,
                    "heading": c.heading, "fov_angle": c.fov_angle,
                    "vision_range": c.vision_range
                }
                for c in self.cameras
            ],
            "guards": [
                {
                    "row": g.row, "col": g.col,
                    "heading": g.heading,
                    "patrol_path": g.patrol_path,
                    "current_idx": g.current_idx
                }
                for g in self.guards
            ],
            "detection_events": self.detection_events,
        }
    
    def render_text(self) -> str:
        """Render the grid as ASCII text."""
        return grid_to_text(self.grid, self.solver_pos)
    
    def __repr__(self):
        return (f"HeistEnvironment(grid={self.config.grid_rows}x{self.config.grid_cols}, "
                f"cameras={len(self.cameras)}, guards={len(self.guards)}, "
                f"walls={len(self.walls)}, tick={self.tick})")
