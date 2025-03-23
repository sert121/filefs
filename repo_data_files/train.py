import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import wandb
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional

from models.nerf_model import NeRF, NeRFTrainer
from models.license_plate_detector import LicensePlateDetector
from models.clinical_forecasting import ClinicalTimeSeriesTransformer, ClinicalDataLoader
from models.legal_bert import LegalBertForSequenceClassification, LegalBertTokenizer, LegalBertTrainer
from models.diffusion_model import DiffusionModel, DiffusionTrainer

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_dir = config["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )

def setup_wandb(config: Dict[str, Any]) -> None:
    """Setup Weights & Biases logging."""
    if config["logging"]["wandb"]["enabled"]:
        wandb.init(
            project=config["logging"]["wandb"]["project"],
            entity=config["logging"]["wandb"]["entity"],
            config=config
        )

def create_model(
    model_name: str,
    config: Dict[str, Any],
    device: str
) -> nn.Module:
    """Create model based on configuration."""
    if model_name == "nerf":
        model = NeRF(**config["nerf"]["model"])
    elif model_name == "license_plate":
        model = LicensePlateDetector(**config["license_plate"]["model"])
    elif model_name == "clinical":
        model = ClinicalTimeSeriesTransformer(**config["clinical"]["model"])
    elif model_name == "legal_bert":
        model = LegalBertForSequenceClassification(**config["legal_bert"]["model"])
    elif model_name == "diffusion":
        model = DiffusionModel(**config["diffusion"]["model"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return model.to(device)

def create_optimizer(
    model: nn.Module,
    model_name: str,
    config: Dict[str, Any]
) -> optim.Optimizer:
    """Create optimizer based on configuration."""
    if model_name == "nerf":
        lr = config["nerf"]["training"]["learning_rate"]
    elif model_name == "license_plate":
        lr = config["license_plate"]["training"]["learning_rate"]
    elif model_name == "clinical":
        lr = config["clinical"]["training"]["learning_rate"]
    elif model_name == "legal_bert":
        lr = config["legal_bert"]["training"]["learning_rate"]
    elif model_name == "diffusion":
        lr = config["diffusion"]["training"]["learning_rate"]
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config["global"]["weight_decay"]
    )

def create_trainer(
    model: nn.Module,
    model_name: str,
    config: Dict[str, Any],
    device: str
) -> Any:
    """Create trainer based on model type."""
    optimizer = create_optimizer(model, model_name, config)
    
    if model_name == "nerf":
        return NeRFTrainer(model, optimizer, device)
    elif model_name == "license_plate":
        return None  # Implement LicensePlateTrainer
    elif model_name == "clinical":
        return None  # Implement ClinicalTrainer
    elif model_name == "legal_bert":
        return LegalBertTrainer(model, None, optimizer, device)  # Add tokenizer
    elif model_name == "diffusion":
        return DiffusionTrainer(model, optimizer, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train(
    model_name: str,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None
) -> None:
    """Main training loop."""
    device = config["global"]["device"]
    
    # Create model and trainer
    model = create_model(model_name, config, device)
    trainer = create_trainer(model, model_name, config, device)
    
    # Setup mixed precision training
    scaler = GradScaler(enabled=config["global"]["mixed_precision"])
    
    # Training loop
    num_epochs = config[model_name]["training"]["num_epochs"]
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=config["global"]["mixed_precision"]):
                    losses = trainer.train_step(batch)
                    
                # Backward pass
                scaler.scale(losses["loss"]).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["global"]["max_grad_norm"]
                )
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update progress bar
                train_losses.append(losses["loss"])
                pbar.set_postfix({"loss": sum(train_losses) / len(train_losses)})
                
                # Log metrics
                if config["logging"]["wandb"]["enabled"]:
                    wandb.log(losses)
                    
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    losses = trainer.evaluate(batch)
                    val_losses.append(losses["loss"])
                    
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        config["checkpointing"]["save_dir"],
                        f"{model_name}_best.pt"
                    )
                )
                
            # Log validation metrics
            if config["logging"]["wandb"]["enabled"]:
                wandb.log({"val_loss": avg_val_loss})
                
        # Save checkpoint
        if (epoch + 1) % config["checkpointing"]["save_frequency"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                os.path.join(
                    config["checkpointing"]["save_dir"],
                    f"{model_name}_epoch_{epoch+1}.pt"
                )
            )

def main():
    # Load configuration
    with open("config/training_config.yml", "r") as f:
        config = yaml.safe_load(f)
        
    # Setup logging
    setup_logging(config)
    setup_wandb(config)
    
    # Set random seed
    torch.manual_seed(config["global"]["seed"])
    
    # Create data loaders
    # TODO: Implement data loading for each model
    
    # Train models
    models = ["nerf", "license_plate", "clinical", "legal_bert", "diffusion"]
    for model_name in models:
        logging.info(f"Training {model_name} model...")
        train(model_name, config, None, None)  # Add data loaders
        
    if config["logging"]["wandb"]["enabled"]:
        wandb.finish()

if __name__ == "__main__":
    main() 