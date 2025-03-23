import torch
import torchvision.transforms as T
from typing import Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class NeRFAugmentations:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.ElasticTransform(p=0.3),
            ToTensorV2()
        ])
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for albumentations
        image = image.numpy()
        
        # Apply augmentations
        augmented = self.transform(image=image)
        return augmented["image"]

class LicensePlateAugmentations:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1)
            ], p=0.3),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
        
    def __call__(self, image: torch.Tensor, bbox: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Convert to numpy for albumentations
        image = image.numpy()
        bbox = bbox.numpy()
        
        # Apply augmentations
        augmented = self.transform(
            image=image,
            bboxes=[bbox],
            labels=[0]  # dummy label
        )
        
        return {
            "image": augmented["image"],
            "bbox": torch.tensor(augmented["bboxes"][0])
        }

class ClinicalAugmentations:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        # Add Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(sequence) * 0.1
            sequence = sequence + noise
            
        # Random masking
        if random.random() < 0.2:
            mask = torch.rand_like(sequence) > 0.1
            sequence = sequence * mask
            
        return sequence

class LegalAugmentations:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def __call__(self, text: str) -> str:
        # Simple text augmentations
        if random.random() < 0.3:
            # Randomly remove some punctuation
            text = ''.join(c for c in text if not (c in '.,!?;:' and random.random() < 0.3))
            
        if random.random() < 0.2:
            # Randomly capitalize some words
            words = text.split()
            text = ' '.join(w.capitalize() if random.random() < 0.3 else w for w in words)
            
        return text

class DiffusionAugmentations:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1)
            ], p=0.3),
            A.ColorJitter(p=0.5),
            ToTensorV2()
        ])
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for albumentations
        image = image.numpy()
        
        # Apply augmentations
        augmented = self.transform(image=image)
        return augmented["image"]

def get_augmentations(
    model_name: str,
    config: Dict[str, Any]
) -> Any:
    """Get appropriate augmentations for a model."""
    
    augmentation_classes = {
        "nerf": NeRFAugmentations,
        "license_plate": LicensePlateAugmentations,
        "clinical": ClinicalAugmentations,
        "legal_bert": LegalAugmentations,
        "diffusion": DiffusionAugmentations
    }
    
    return augmentation_classes[model_name](config) 