"""
Security Components for the Heist Architect framework.
Defines Walls, Rotating Cameras with vision cones, and Patrol Guards.
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Wall
# ---------------------------------------------------------------------------

@dataclass
class Wall:
    """A static wall that blocks movement and line-of-sight."""
    row: int
    col: int
    
    def __repr__(self):
        return f"Wall({self.row}, {self.col})"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """
    A rotating camera with a vision cone.
    
    Attributes:
        row, col: Fixed position on the grid
        fov_angle: Field of view in degrees (e.g., 60°)
        heading: Current direction the camera faces (degrees, 0=right, 90=up)
        rotation_speed: Degrees rotated per tick (positive=CCW, negative=CW)
        vision_range: How far the camera can see (in tiles)
    """
    row: int
    col: int
    fov_angle: float = 60.0
    heading: float = 0.0
    rotation_speed: float = 15.0  # degrees per tick
    vision_range: int = 6
    
    def update(self, tick: int = 1):
        """Rotate the camera by one tick."""
        self.heading = (self.heading + self.rotation_speed * tick) % 360.0
    
    def get_vision_cone_tiles(self, grid_rows: int, grid_cols: int, walls: np.ndarray) -> List[Tuple[int, int]]:
        """
        Calculate which tiles are visible by this camera using raycasting.
        
        Args:
            grid_rows, grid_cols: Dimensions of the grid
            walls: 2D boolean array where True = wall present
        Returns:
            List of (row, col) positions visible by this camera
        """
        visible = []
        half_fov = self.fov_angle / 2.0
        
        # Cast rays across the FOV
        num_rays = max(int(self.fov_angle * 2), 30)  # more rays for wider FOV
        
        for i in range(num_rays + 1):
            angle_deg = self.heading - half_fov + (self.fov_angle * i / num_rays)
            angle_rad = math.radians(angle_deg)
            
            # Ray direction (note: row increases downward)
            dx = math.cos(angle_rad)
            dy = -math.sin(angle_rad)  # negative because row increases downward
            
            # March along the ray
            for step in range(1, self.vision_range + 1):
                # Use finer steps for accuracy
                for sub in np.linspace(0, 1, 3):
                    dist = step - 1 + sub * 1.0
                    if dist == 0:
                        continue
                    fx = self.col + dx * dist
                    fy = self.row + dy * dist
                    
                    c = int(round(fx))
                    r = int(round(fy))
                    
                    if 0 <= r < grid_rows and 0 <= c < grid_cols:
                        if walls[r, c]:
                            break  # wall blocks vision
                        if (r, c) != (self.row, self.col) and (r, c) not in visible:
                            visible.append((r, c))
                    else:
                        break  # out of bounds
                else:
                    continue
                break  # wall was hit, stop this ray
        
        return visible
    
    def __repr__(self):
        return (f"Camera(pos=({self.row},{self.col}), "
                f"heading={self.heading:.0f}°, fov={self.fov_angle:.0f}°, "
                f"speed={self.rotation_speed:.0f}°/tick, range={self.vision_range})")


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

@dataclass
class Guard:
    """
    A patrol guard that moves along a predefined path.
    
    Attributes:
        patrol_path: List of (row, col) waypoints
        speed: How many waypoints to advance per tick
        current_idx: Current position index in patrol_path
        vision_range: Detection radius around the guard
        fov_angle: Field of view in degrees (guards have wider FOV than cameras)
        heading: Current facing direction (derived from movement)
    """
    patrol_path: List[Tuple[int, int]] = field(default_factory=list)
    speed: int = 1
    current_idx: int = 0
    vision_range: int = 4
    fov_angle: float = 90.0
    heading: float = 0.0
    
    @property
    def row(self) -> int:
        return self.patrol_path[self.current_idx][0] if self.patrol_path else 0
    
    @property
    def col(self) -> int:
        return self.patrol_path[self.current_idx][1] if self.patrol_path else 0
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.row, self.col)
    
    def update(self, tick: int = 1):
        """Move the guard along its patrol path."""
        if not self.patrol_path or len(self.patrol_path) < 2:
            return
        
        old_idx = self.current_idx
        self.current_idx = (self.current_idx + self.speed * tick) % len(self.patrol_path)
        
        # Update heading based on movement direction
        old_pos = self.patrol_path[old_idx]
        new_pos = self.patrol_path[self.current_idx]
        dr = new_pos[0] - old_pos[0]
        dc = new_pos[1] - old_pos[1]
        if dr != 0 or dc != 0:
            self.heading = math.degrees(math.atan2(-dr, dc)) % 360.0
    
    def get_visible_tiles(self, grid_rows: int, grid_cols: int, walls: np.ndarray) -> List[Tuple[int, int]]:
        """
        Calculate visible tiles around the guard using raycasting.
        Guards use a directional cone based on their heading.
        """
        visible = []
        half_fov = self.fov_angle / 2.0
        num_rays = max(int(self.fov_angle * 2), 30)
        
        for i in range(num_rays + 1):
            angle_deg = self.heading - half_fov + (self.fov_angle * i / num_rays)
            angle_rad = math.radians(angle_deg)
            
            dx = math.cos(angle_rad)
            dy = -math.sin(angle_rad)
            
            for step in range(1, self.vision_range + 1):
                fx = self.col + dx * step
                fy = self.row + dy * step
                
                c = int(round(fx))
                r = int(round(fy))
                
                if 0 <= r < grid_rows and 0 <= c < grid_cols:
                    if walls[r, c]:
                        break
                    if (r, c) != (self.row, self.col) and (r, c) not in visible:
                        visible.append((r, c))
                else:
                    break
        
        return visible
    
    def __repr__(self):
        return (f"Guard(pos=({self.row},{self.col}), "
                f"path_len={len(self.patrol_path)}, "
                f"heading={self.heading:.0f}°, range={self.vision_range})")
