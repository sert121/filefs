import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple

class ClinicalTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,  # HR, BP, SpO2
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 1440,  # 24 hours at 1-min intervals
        num_classes: int = 2  # binary classification
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.register_buffer(
            "pos_encoding",
            self._create_pos_encoding(max_seq_length, hidden_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output heads
        self.mortality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes),
            nn.Sigmoid()
        )
        
        self.vitals_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, input_dim)
        )
        
        # Attention mask for causal prediction
        self.register_buffer(
            "causal_mask",
            self._create_causal_mask(max_seq_length)
        )
        
    def _create_pos_encoding(
        self,
        max_seq_length: int,
        hidden_dim: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pos_encoding = torch.zeros(max_seq_length, hidden_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
        
    def _create_causal_mask(self, max_seq_length: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1)
        return mask.bool()
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:x.size(1)]
        
        # Create attention mask
        if mask is not None:
            # Combine padding mask with causal mask
            attn_mask = mask.unsqueeze(1) | self.causal_mask[:x.size(1), :x.size(1)]
        else:
            attn_mask = self.causal_mask[:x.size(1), :x.size(1)]
            
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Get sequence representation (use last token)
        seq_repr = x[:, -1]
        
        # Generate predictions
        mortality_pred = self.mortality_head(seq_repr)
        vitals_pred = self.vitals_prediction_head(x)
        
        if self.training and targets is not None:
            # Compute losses
            losses = self.compute_losses(
                mortality_pred,
                vitals_pred,
                targets
            )
            return losses
            
        return {
            "mortality": mortality_pred,
            "vitals_forecast": vitals_pred
        }
        
    def compute_losses(
        self,
        mortality_pred: torch.Tensor,
        vitals_pred: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        # Mortality prediction loss
        mortality_loss = nn.BCELoss()(
            mortality_pred,
            targets["mortality"].float()
        )
        
        # Vitals prediction loss
        vitals_loss = nn.MSELoss()(
            vitals_pred,
            targets["vitals"]
        )
        
        # Combine losses
        total_loss = mortality_loss + 0.5 * vitals_loss
        
        return {
            "total_loss": total_loss,
            "mortality_loss": mortality_loss,
            "vitals_loss": vitals_loss
        }

class ClinicalDataLoader:
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        sequence_length: int = 1440,
        num_workers: int = 4
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and preprocess clinical data."""
        # Load raw data
        data = torch.load(self.data_path)
        
        # Extract features and labels
        vitals = data["vitals"]
        mortality = data["mortality"]
        timestamps = data["timestamps"]
        
        # Create sequences
        sequences = []
        labels = []
        masks = []
        
        for i in range(len(vitals) - self.sequence_length):
            seq = vitals[i:i + self.sequence_length]
            label = mortality[i + self.sequence_length]
            mask = torch.ones(self.sequence_length)
            
            sequences.append(seq)
            labels.append(label)
            masks.append(mask)
            
        return (
            torch.stack(sequences),
            torch.stack(labels),
            torch.stack(masks)
        )
        
    def create_dataloader(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader."""
        dataset = torch.utils.data.TensorDataset(
            sequences,
            labels,
            masks
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        ) 