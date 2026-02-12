"""
Utility functions for the Heist Architect framework.
Includes pathfinding (BFS), grid helpers, and device management.
"""

import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Device Management
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Get the best available device (CUDA GPU preferred)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device

DEVICE = get_device()

# ---------------------------------------------------------------------------
# Tile Types
# ---------------------------------------------------------------------------

class TileType:
    EMPTY   = 0
    WALL    = 1
    START   = 2
    VAULT   = 3
    CAMERA  = 4
    GUARD   = 5

TILE_NAMES = {
    TileType.EMPTY:  "Empty",
    TileType.WALL:   "Wall",
    TileType.START:  "Start",
    TileType.VAULT:  "Vault",
    TileType.CAMERA: "Camera",
    TileType.GUARD:  "Guard",
}

# ---------------------------------------------------------------------------
# Pathfinding (BFS)
# ---------------------------------------------------------------------------

def bfs_path_exists(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """
    Check if a path exists between start and goal on the grid using BFS.
    Walls (TileType.WALL) block movement. All other tiles are walkable.
    
    Args:
        grid: 2D numpy array of tile types
        start: (row, col) start position
        goal: (row, col) goal position
    Returns:
        True if a walkable path exists from start to goal
    """
    rows, cols = grid.shape
    if start == goal:
        return True
    
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr, nc] != TileType.WALL:
                    if (nr, nc) == goal:
                        return True
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    return False


def bfs_shortest_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Find the shortest path from start to goal using BFS.
    Returns list of (row, col) positions, or None if no path exists.
    """
    rows, cols = grid.shape
    if start == goal:
        return [start]
    
    visited = {start: None}
    queue = deque([start])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr, nc] != TileType.WALL:
                    visited[(nr, nc)] = (r, c)
                    if (nr, nc) == goal:
                        # Reconstruct path
                        path = []
                        pos = (nr, nc)
                        while pos is not None:
                            path.append(pos)
                            pos = visited[pos]
                        return list(reversed(path))
                    queue.append((nr, nc))
    
    return None


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Grid Helpers
# ---------------------------------------------------------------------------

def create_empty_grid(rows: int, cols: int) -> np.ndarray:
    """Create an empty grid with border walls."""
    grid = np.full((rows, cols), TileType.EMPTY, dtype=np.int32)
    # Add border walls
    grid[0, :] = TileType.WALL
    grid[-1, :] = TileType.WALL
    grid[:, 0] = TileType.WALL
    grid[:, -1] = TileType.WALL
    return grid


def grid_to_text(grid: np.ndarray, solver_pos: Optional[Tuple[int, int]] = None) -> str:
    """
    Convert a grid to a text representation for debugging.
    Symbols: # wall, S start, V vault, C camera, G guard, . empty, @ solver
    """
    symbols = {
        TileType.EMPTY:  '.',
        TileType.WALL:   '#',
        TileType.START:  'S',
        TileType.VAULT:  'V',
        TileType.CAMERA: 'C',
        TileType.GUARD:  'G',
    }
    rows, cols = grid.shape
    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            if solver_pos and (r, c) == solver_pos:
                row_chars.append('@')
            else:
                row_chars.append(symbols.get(grid[r, c], '?'))
        lines.append(''.join(row_chars))
    return '\n'.join(lines)
