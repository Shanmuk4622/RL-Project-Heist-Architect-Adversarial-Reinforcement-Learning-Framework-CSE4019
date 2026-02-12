"""
Dynamic Visibility Map for the Heist Architect framework.
Computes a time-dependent surveillance map from all cameras and guards.
"""

import numpy as np
from typing import List, Tuple
from .security import Camera, Guard


class DynamicVisibilityMap:
    """
    Computes and maintains a binary visibility grid that updates every tick.
    
    The visibility map shows which tiles are currently under surveillance
    by any camera or guard. Walls block line-of-sight.
    
    Attributes:
        rows, cols: Grid dimensions
        visibility: 2D array where 1 = under surveillance, 0 = safe
        heat_map: Accumulated visibility over time (for analysis)
    """
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.visibility = np.zeros((rows, cols), dtype=np.float32)
        self.heat_map = np.zeros((rows, cols), dtype=np.float32)
        self._total_updates = 0
    
    def update(self, cameras: List[Camera], guards: List[Guard],
               walls: np.ndarray) -> np.ndarray:
        """
        Recalculate the visibility map based on current camera headings
        and guard positions.
        
        Args:
            cameras: List of Camera objects with current headings
            guards: List of Guard objects with current positions
            walls: 2D boolean array where True = wall
        
        Returns:
            2D float array: 1.0 = visible, 0.0 = safe
        """
        self.visibility = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # Add camera vision cones
        for cam in cameras:
            tiles = cam.get_vision_cone_tiles(self.rows, self.cols, walls)
            for r, c in tiles:
                self.visibility[r, c] = 1.0
        
        # Add guard vision zones
        for guard in guards:
            tiles = guard.get_visible_tiles(self.rows, self.cols, walls)
            for r, c in tiles:
                self.visibility[r, c] = 1.0
            # Guard's own tile is also dangerous
            self.visibility[guard.row, guard.col] = 1.0
        
        # Update heat map (running average)
        self._total_updates += 1
        self.heat_map += self.visibility
        
        return self.visibility
    
    def is_visible(self, row: int, col: int) -> bool:
        """Check if a specific tile is currently under surveillance."""
        return self.visibility[row, col] > 0.5
    
    def get_safe_tiles(self) -> List[Tuple[int, int]]:
        """Return list of all tiles NOT currently under surveillance."""
        safe = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.visibility[r, c] < 0.5:
                    safe.append((r, c))
        return safe
    
    def get_normalized_heat_map(self) -> np.ndarray:
        """Return heat map normalized to [0, 1] for analysis."""
        if self._total_updates == 0:
            return self.heat_map.copy()
        return self.heat_map / self._total_updates
    
    def reset(self):
        """Reset visibility and heat map."""
        self.visibility = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.heat_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._total_updates = 0
