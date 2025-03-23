import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.num_frequencies):
            outputs.append(torch.sin(2.0**i * x))
            outputs.append(torch.cos(2.0**i * x))
        return torch.cat(outputs, dim=-1)

class NeRF(nn.Module):
    def __init__(
        self,
        pos_enc_frequencies: int = 10,
        dir_enc_frequencies: int = 4,
        hidden_size: int = 256,
        num_layers: int = 8,
        use_view_dirs: bool = True
    ):
        super().__init__()
        self.use_view_dirs = use_view_dirs
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(pos_enc_frequencies)
        self.dir_encoder = PositionalEncoding(dir_enc_frequencies)
        
        # Calculate dimensions
        pos_enc_dims = pos_enc_frequencies * 6  # 3D position * 2 (sin/cos)
        dir_enc_dims = dir_enc_frequencies * 6  # 3D direction * 2 (sin/cos)
        
        # Main MLP layers
        self.layers = nn.ModuleList([
            nn.Linear(pos_enc_dims, hidden_size),
            *[nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)]
        ])
        
        # Density prediction
        self.density_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.ReLU()
        )
        
        # Feature vector for color prediction
        self.feature_layer = nn.Linear(hidden_size, hidden_size)
        
        # Color prediction
        if use_view_dirs:
            color_input_dim = hidden_size + dir_enc_dims
        else:
            color_input_dim = hidden_size
            
        self.color_layer = nn.Sequential(
            nn.Linear(color_input_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 3),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode positions and directions
        pos_enc = self.pos_encoder(positions)
        if self.use_view_dirs and directions is not None:
            dir_enc = self.dir_encoder(directions)
        
        # Process through main network
        x = pos_enc
        for layer in self.layers:
            x = torch.relu(layer(x))
        
        # Predict density
        density = self.density_layer(x)
        
        # Get feature vector for color prediction
        features = self.feature_layer(x)
        
        # Predict color
        if self.use_view_dirs and directions is not None:
            color_input = torch.cat([features, dir_enc], dim=-1)
        else:
            color_input = features
            
        color = self.color_layer(color_input)
        
        return density, color

class NeRFTrainer:
    def __init__(
        self,
        model: NeRF,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
    def train_step(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        target_colors: torch.Tensor,
        target_densities: torch.Tensor
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        positions = positions.to(self.device)
        directions = directions.to(self.device)
        target_colors = target_colors.to(self.device)
        target_densities = target_densities.to(self.device)
        
        # Forward pass
        pred_densities, pred_colors = self.model(positions, directions)
        
        # Compute losses
        color_loss = torch.mean((pred_colors - target_colors) ** 2)
        density_loss = torch.mean((pred_densities - target_densities) ** 2)
        total_loss = color_loss + 0.1 * density_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "color_loss": color_loss.item(),
            "density_loss": density_loss.item()
        }
        
    def render_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        num_samples: int = 64,
        near: float = 0.0,
        far: float = 1.0
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            # Sample points along rays
            t = torch.linspace(near, far, num_samples).to(self.device)
            t = t.view(1, -1, 1).expand(ray_origins.shape[0], -1, 3)
            
            # Get 3D points
            points = ray_origins.unsqueeze(1) + t * ray_directions.unsqueeze(1)
            points = points.view(-1, 3)
            
            # Get predictions
            densities, colors = self.model(points, ray_directions.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 3))
            
            # Reshape predictions
            densities = densities.view(-1, num_samples)
            colors = colors.view(-1, num_samples, 3)
            
            # Compute weights for volume rendering
            weights = torch.exp(-torch.cumsum(densities, dim=-1))
            weights = weights * (1 - torch.exp(-densities))
            
            # Render final color
            rendered_colors = torch.sum(weights.unsqueeze(-1) * colors, dim=1)
            
            return rendered_colors 