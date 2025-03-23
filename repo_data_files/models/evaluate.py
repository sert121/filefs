import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)

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
            logging.FileHandler(os.path.join(log_dir, "evaluate.log")),
            logging.StreamHandler()
        ]
    )

def create_model(
    model_name: str,
    config: Dict[str, Any],
    device: str,
    checkpoint_path: str
) -> nn.Module:
    """Create and load model from checkpoint."""
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
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_classification(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str
) -> Dict[str, float]:
    """Evaluate classification model."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs[1]
            probs = torch.softmax(logits, dim=-1)
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted"
    )
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

def evaluate_detection(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate object detection model."""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            preds = outputs["boxes"]
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch["boxes"].cpu().numpy())
    
    # Calculate mAP
    # TODO: Implement mAP calculation
    
    return {
        "mAP": 0.0,  # Placeholder
        "mAP_50": 0.0,  # Placeholder
        "mAP_75": 0.0  # Placeholder
    }

def evaluate_regression(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str
) -> Dict[str, float]:
    """Evaluate regression model."""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            preds = outputs["predictions"]
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch["targets"].cpu().numpy())
    
    # Calculate metrics
    mse = np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }

def plot_metrics(
    metrics: Dict[str, float],
    model_name: str,
    output_dir: str
) -> None:
    """Plot evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix if available
    if "confusion_matrix" in metrics:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues"
        )
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
    
    # Plot ROC curve if available
    if "roc_auc" in metrics:
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"ROC curve (AUC = {metrics['roc_auc']:.2f})"
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
        plt.close()

def evaluate(
    model_name: str,
    config: Dict[str, Any],
    test_loader: torch.utils.data.DataLoader,
    checkpoint_path: str
) -> None:
    """Evaluate model on test set."""
    device = config["global"]["device"]
    
    # Create and load model
    model = create_model(model_name, config, device, checkpoint_path)
    
    # Evaluate based on model type
    if model_name in ["legal_bert", "clinical"]:
        metrics = evaluate_classification(model, test_loader, device)
    elif model_name == "license_plate":
        metrics = evaluate_detection(model, test_loader, device)
    elif model_name == "nerf":
        metrics = evaluate_regression(model, test_loader, device)
    elif model_name == "diffusion":
        # TODO: Implement diffusion model evaluation
        metrics = {}
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Plot metrics
    plot_metrics(
        metrics,
        model_name,
        os.path.join(config["logging"]["log_dir"], "plots")
    )
    
    # Log metrics
    logging.info(f"\n{model_name} Evaluation Results:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logging.info(f"{metric}: {value:.4f}")

def main():
    # Load configuration
    with open("config/training_config.yml", "r") as f:
        config = yaml.safe_load(f)
        
    # Setup logging
    setup_logging(config)
    
    # Set random seed
    torch.manual_seed(config["global"]["seed"])
    
    # Create test data loaders
    # TODO: Implement data loading for each model
    
    # Evaluate models
    models = ["nerf", "license_plate", "clinical", "legal_bert", "diffusion"]
    for model_name in models:
        logging.info(f"\nEvaluating {model_name} model...")
        evaluate(
            model_name,
            config,
            None,  # Add test loader
            os.path.join(
                config["checkpointing"]["save_dir"],
                f"{model_name}_best.pt"
            )
        )

if __name__ == "__main__":
    main() 