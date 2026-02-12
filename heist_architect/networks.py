"""
Neural Network Architectures for the Heist Architect framework.
Includes policy and value networks for both Architect and Solver agents.
Uses PyTorch with GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SolverNetwork(nn.Module):
    """
    Policy + Value network for the Solver agent.
    
    Architecture:
        - CNN backbone for spatial features (occupancy + visibility + position)
        - LSTM for temporal reasoning (remembering camera rotation patterns)
        - Dual heads: policy (action probabilities) + value (state value)
    
    Input: (batch, 3, grid_rows, grid_cols) — 3-channel spatial state
    Output: (action_logits, state_value)
    """
    
    def __init__(self, grid_rows: int = 20, grid_cols: int = 20, 
                 num_actions: int = 5, hidden_dim: int = 256,
                 lstm_hidden: int = 128):
        super().__init__()
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        
        # CNN backbone for spatial features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate CNN output size
        cnn_out_size = 64 * 4 * 4  # 1024
        
        # FC layer after CNN
        self.fc_spatial = nn.Linear(cnn_out_size, hidden_dim)
        
        # LSTM for temporal reasoning
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, 
                hidden: Tuple[torch.Tensor, torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            state: (batch, 3, rows, cols) spatial state tensor
            hidden: Optional LSTM hidden state (h, c)
        
        Returns:
            action_logits: (batch, num_actions)
            state_value: (batch, 1) 
            new_hidden: Updated LSTM state
        """
        batch_size = state.size(0)
        
        # CNN backbone
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        
        # Spatial features
        spatial = F.relu(self.fc_spatial(x))
        
        # LSTM (add sequence dimension)
        spatial = spatial.unsqueeze(1)  # (batch, 1, hidden)
        
        if hidden is None:
            hidden = self._init_hidden(batch_size, state.device)
        
        lstm_out, new_hidden = self.lstm(spatial, hidden)
        lstm_out = lstm_out.squeeze(1)  # (batch, lstm_hidden)
        
        # Dual heads
        action_logits = self.policy_head(lstm_out)
        state_value = self.value_head(lstm_out)
        
        return action_logits, state_value, new_hidden
    
    def _init_hidden(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        )
    
    def get_action(self, state: torch.Tensor, hidden=None):
        """Sample an action from the policy (for inference)."""
        logits, value, new_hidden = self.forward(state, hidden)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, new_hidden


class ArchitectNetwork(nn.Module):
    """
    Policy + Value network for the Architect agent.
    
    The Architect outputs a full level layout given a budget constraint.
    It generates placement probabilities for each cell and asset type.
    
    Architecture:
        - Encoder: Process current grid state
        - Decoder: Generate placement probabilities per cell
        - Budget-aware: Masked by remaining budget
    
    Input: (batch, 1, grid_rows, grid_cols) — empty grid with start/vault
    Output: placement_probs for each cell × asset_type (wall, camera, guard)
    """
    
    def __init__(self, grid_rows: int = 20, grid_cols: int = 20,
                 num_asset_types: int = 3, hidden_dim: int = 256):
        super().__init__()
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_asset_types = num_asset_types  # wall, camera, guard
        
        # Encoder: process grid state
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_global = nn.Linear(64 * 4 * 4, hidden_dim)
        
        # Decoder: generate placement probabilities
        # Output: (batch, num_asset_types + 1, rows, cols)
        # Channel per asset type + 1 for "no placement"
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_asset_types + 1, kernel_size=1),  # per-cell logits
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Camera parameter heads (for cells predicted as cameras)
        self.camera_fov_head = nn.Linear(hidden_dim, 1)      # FOV angle
        self.camera_speed_head = nn.Linear(hidden_dim, 1)     # Rotation speed
        self.camera_heading_head = nn.Linear(hidden_dim, 1)   # Initial heading
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, grid_state: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass.
        
        Args:
            grid_state: (batch, 1, rows, cols) grid with start/vault marked
        
        Returns:
            placement_logits: (batch, num_asset_types+1, rows, cols) 
            state_value: (batch, 1)
            camera_params: dict with FOV, speed, heading predictions
        """
        # Encode
        features = self.encoder(grid_state)
        
        # Global features for value and camera params
        global_feat = self.global_pool(features)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.relu(self.fc_global(global_feat))
        
        # Decode placement logits
        placement_logits = self.decoder(features)
        
        # Value
        state_value = self.value_head(global_feat)
        
        # Camera parameters (clamped to reasonable ranges)
        camera_params = {
            "fov": torch.sigmoid(self.camera_fov_head(global_feat)) * 90 + 30,   # [30, 120] degrees
            "speed": torch.sigmoid(self.camera_speed_head(global_feat)) * 30 + 5, # [5, 35] deg/tick
            "heading": torch.sigmoid(self.camera_heading_head(global_feat)) * 360, # [0, 360] degrees
        }
        
        return placement_logits, state_value, camera_params
    
    def generate_layout(self, grid_state: torch.Tensor, 
                        budget: int, temperature: float = 1.0):
        """
        Generate a complete level layout.
        
        Args:
            grid_state: (1, 1, rows, cols) initial grid
            budget: Available budget points
            temperature: Sampling temperature (higher = more exploration)
        
        Returns:
            walls, cameras, guards: Lists of placements
            log_probs: Log probabilities for training
            value: State value estimate
        """
        placement_logits, value, cam_params = self.forward(grid_state)
        
        # Apply temperature
        placement_logits = placement_logits / temperature
        
        # Softmax over asset types per cell
        # Shape: (1, num_asset_types+1, rows, cols)
        probs = F.softmax(placement_logits, dim=1)
        
        # Sample placements
        b, c, h, w = probs.shape
        probs_flat = probs.view(b, c, -1).permute(0, 2, 1)  # (1, H*W, C)
        
        dist = torch.distributions.Categorical(probs_flat)
        sampled = dist.sample()  # (1, H*W)
        log_probs = dist.log_prob(sampled)  # (1, H*W)
        
        # Decode placements
        asset_map = sampled.view(h, w).cpu().numpy()
        
        walls = []
        cameras = []
        guards = []
        remaining = budget
        
        from .components.budget import BUDGET_COSTS
        
        for r in range(1, h - 1):  # skip borders
            for c_idx in range(1, w - 1):
                asset_type = asset_map[r, c_idx]
                
                if asset_type == 0:  # no placement
                    continue
                elif asset_type == 1 and remaining >= BUDGET_COSTS["wall"]:
                    walls.append((r, c_idx))
                    remaining -= BUDGET_COSTS["wall"]
                elif asset_type == 2 and remaining >= BUDGET_COSTS["camera"]:
                    fov = cam_params["fov"].item()
                    speed = cam_params["speed"].item()
                    heading = cam_params["heading"].item()
                    cameras.append({
                        "row": r, "col": c_idx,
                        "fov_angle": fov,
                        "rotation_speed": speed,
                        "heading": heading,
                        "vision_range": 6
                    })
                    remaining -= BUDGET_COSTS["camera"]
                elif asset_type == 3 and remaining >= BUDGET_COSTS["guard"]:
                    # Generate a simple patrol path around the guard position
                    patrol = self._generate_patrol(r, c_idx, h, w)
                    guards.append({
                        "patrol_path": patrol,
                        "speed": 1,
                        "vision_range": 4,
                        "fov_angle": 90.0
                    })
                    remaining -= BUDGET_COSTS["guard"]
                
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
        
        total_log_prob = log_probs.sum()
        
        return walls, cameras, guards, total_log_prob, value
    
    def _generate_patrol(self, row: int, col: int, 
                         grid_h: int, grid_w: int) -> list:
        """Generate a simple patrol path around a position."""
        path = []
        # Create a rectangular patrol
        offsets = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), 
                   (2, 1), (2, 0), (1, 0)]
        for dr, dc in offsets:
            r = max(1, min(grid_h - 2, row + dr - 1))
            c = max(1, min(grid_w - 2, col + dc - 1))
            path.append((r, c))
        return path
