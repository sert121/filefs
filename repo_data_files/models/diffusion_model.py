import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

class DiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_classes: Optional[int] = None
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        
        # Define noise schedule
        self.register_buffer(
            "betas",
            torch.linspace(beta_start, beta_end, num_timesteps)
        )
        self.register_buffer(
            "alphas",
            1 - self.betas
        )
        self.register_buffer(
            "alphas_cumprod",
            torch.cumprod(self.alphas, dim=0)
        )
        
        # U-Net backbone
        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            *[ResBlock(hidden_channels) for _ in range(num_res_blocks)],
            nn.Conv2d(hidden_channels, hidden_channels*2, 4, stride=2, padding=1),
            *[ResBlock(hidden_channels*2) for _ in range(num_res_blocks)],
            nn.Conv2d(hidden_channels*2, hidden_channels*4, 4, stride=2, padding=1),
            *[ResBlock(hidden_channels*4) for _ in range(num_res_blocks)],
            nn.Conv2d(hidden_channels*4, hidden_channels*8, 4, stride=2, padding=1),
            *[ResBlock(hidden_channels*8) for _ in range(num_res_blocks)]
        ])
        
        # Middle block
        self.middle = nn.Sequential(
            ResBlock(hidden_channels*8),
            ResBlock(hidden_channels*8)
        )
        
        # Decoder
        self.decoder = nn.ModuleList([
            *[ResBlock(hidden_channels*8) for _ in range(num_res_blocks)],
            nn.ConvTranspose2d(hidden_channels*8, hidden_channels*4, 4, stride=2, padding=1),
            *[ResBlock(hidden_channels*4) for _ in range(num_res_blocks)],
            nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, 4, stride=2, padding=1),
            *[ResBlock(hidden_channels*2) for _ in range(num_res_blocks)],
            nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 4, stride=2, padding=1),
            *[ResBlock(hidden_channels) for _ in range(num_res_blocks)],
            nn.Conv2d(hidden_channels, in_channels, 1)
        ])
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels*4),
            nn.SiLU(),
            nn.Linear(hidden_channels*4, hidden_channels*4)
        )
        
        # Class embedding (if conditional)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, hidden_channels*4)
            
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Class embedding (if conditional)
        if y is not None:
            c_emb = self.class_embed(y)
            emb = t_emb + c_emb
        else:
            emb = t_emb
            
        # Encoder
        h = x
        skip_connections = []
        for layer in self.encoder:
            h = layer(h, emb)
            skip_connections.append(h)
            
        # Middle block
        h = self.middle(h, emb)
        
        # Decoder
        for layer in self.decoder:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
                if len(skip_connections) > 0:
                    h = h + skip_connections.pop()
                    
        return h
        
    def sample(
        self,
        batch_size: int,
        image_size: int,
        num_inference_steps: int = 50,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample from the model."""
        self.eval()
        
        with torch.no_grad():
            # Initialize noise
            x = torch.randn(batch_size, 3, image_size, image_size)
            x = x.to(next(self.parameters()).device)
            
            # Time steps for sampling
            timesteps = torch.linspace(
                self.num_timesteps - 1,
                0,
                num_inference_steps,
                dtype=torch.long
            )
            
            # Sampling loop
            for t in timesteps:
                t_batch = torch.full((batch_size,), t, device=x.device)
                
                # Predict noise
                noise_pred = self(x, t_batch, y)
                
                # Update sample
                alpha_t = self.alphas[t]
                alpha_t_bar = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                    
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - (1 - alpha_t) / torch.sqrt(1 - alpha_t_bar) * noise_pred
                ) + torch.sqrt(beta_t) * noise
                
            return x
            
    def training_step(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform a training step."""
        self.train()
        
        batch_size = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,))
        
        # Add noise to input
        noise = torch.randn_like(x)
        alpha_t_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        noisy_x = torch.sqrt(alpha_t_bar) * x + torch.sqrt(1 - alpha_t_bar) * noise
        
        # Predict noise
        noise_pred = self(noisy_x, t, y)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss}

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Linear(channels*4, channels)
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.silu(h)
        
        # Time embedding
        emb = self.time_mlp(emb)
        emb = emb.unsqueeze(-1).unsqueeze(-1)
        h = h + emb
        
        h = self.conv2(h)
        h = F.silu(h)
        
        return h + x

class DiffusionTrainer:
    def __init__(
        self,
        model: DiffusionModel,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
    def train_step(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
            
        # Training step
        losses = self.model.training_step(x, y)
        
        # Backward pass
        losses["loss"].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
        
    def sample(
        self,
        batch_size: int,
        image_size: int,
        num_inference_steps: int = 50,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate samples from the model."""
        if y is not None:
            y = y.to(self.device)
            
        return self.model.sample(
            batch_size,
            image_size,
            num_inference_steps,
            y
        ) 