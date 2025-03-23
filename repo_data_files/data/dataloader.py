import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
import logging
from PIL import Image
import json

class NeRFDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Any] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load preprocessed data
        self.images = np.load(os.path.join(data_dir, f"nerf_images_{split}.npy"))
        self.poses = np.load(os.path.join(data_dir, f"nerf_poses_{split}.npy"))
        
    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[idx]).float()
        pose = torch.from_numpy(self.poses[idx]).float()
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "pose": pose
        }

class LicensePlateDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Any] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load preprocessed data
        self.data = torch.load(os.path.join(data_dir, f"license_plate_{split}.pt"))
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        image = torch.from_numpy(item["image"]).float()
        bbox = torch.tensor(item["bbox"], dtype=torch.float32)
        label = torch.tensor(item["class"], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "bbox": bbox,
            "label": label
        }

class ClinicalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Any] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load preprocessed data
        self.sequences = np.load(os.path.join(data_dir, f"clinical_{split}.npy"))
        self.labels = np.load(os.path.join(data_dir, f"clinical_{split}_labels.npy"))
        
    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = torch.from_numpy(self.sequences[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return {
            "sequence": sequence,
            "label": label
        }

class LegalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        tokenizer: Optional[Any] = None,
        max_length: int = 512
    ):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load preprocessed data
        data = torch.load(os.path.join(data_dir, f"legal_{split}.pt"))
        self.texts = data["texts"]
        self.labels = data["labels"]
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": label
        }

class DiffusionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Any] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load preprocessed data
        self.data = torch.load(os.path.join(data_dir, f"diffusion_{split}.pt"))
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        image = torch.from_numpy(item["image"]).float()
        label = torch.tensor(item["label"], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "label": label
        }

def create_dataloaders(
    data_dir: str,
    config: Dict[str, Any],
    model_name: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for a specific model."""
    
    # Select appropriate dataset class
    dataset_classes = {
        "nerf": NeRFDataset,
        "license_plate": LicensePlateDataset,
        "clinical": ClinicalDataset,
        "legal_bert": LegalDataset,
        "diffusion": DiffusionDataset
    }
    
    dataset_class = dataset_classes[model_name]
    
    # Create datasets
    train_dataset = dataset_class(
        data_dir=data_dir,
        split="train",
        **config[model_name]["data"]
    )
    
    val_dataset = dataset_class(
        data_dir=data_dir,
        split="val",
        **config[model_name]["data"]
    )
    
    test_dataset = dataset_class(
        data_dir=data_dir,
        split="test",
        **config[model_name]["data"]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["global"]["batch_size"],
        shuffle=True,
        num_workers=config["global"]["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["global"]["batch_size"],
        shuffle=False,
        num_workers=config["global"]["num_workers"],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["global"]["batch_size"],
        shuffle=False,
        num_workers=config["global"]["num_workers"],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 