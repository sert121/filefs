import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional

class LicensePlateDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,  # plate/no-plate
        backbone: str = "resnet50",
        pretrained: bool = True,
        anchor_sizes: List[Tuple[int, int]] = None
    ):
        super().__init__()
        
        # Default anchor sizes for license plates
        if anchor_sizes is None:
            anchor_sizes = [(32, 16), (64, 32), (128, 64), (256, 128)]
            
        self.num_classes = num_classes
        self.anchor_sizes = anchor_sizes
        self.num_anchors = len(anchor_sizes)
        
        # Load backbone with pretrained weights
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dims = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Feature Pyramid Network (FPN)
        self.fpn = FeaturePyramidNetwork(feature_dims)
        
        # Detection heads for each FPN level
        self.detection_heads = nn.ModuleList([
            DetectionHead(
                in_channels=256,
                num_classes=num_classes,
                num_anchors=1  # One anchor per level
            ) for _ in range(len(anchor_sizes))
        ])
        
        # Non-maximum suppression threshold
        self.nms_threshold = 0.5
        self.conf_threshold = 0.25
        
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        
        # Apply FPN
        fpn_features = self.fpn(features)
        
        # Get predictions from each detection head
        predictions = []
        for head, feature in zip(self.detection_heads, fpn_features):
            pred = head(feature)
            predictions.append(pred)
            
        # Combine predictions
        combined_pred = torch.cat(predictions, dim=1)
        
        if self.training and targets is not None:
            # Compute losses
            losses = self.compute_losses(combined_pred, targets)
            return losses
        else:
            # Post-process predictions
            boxes, scores, classes = self.post_process(combined_pred)
            return {
                "boxes": boxes,
                "scores": scores,
                "classes": classes
            }
            
    def compute_losses(
        self,
        predictions: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute detection losses."""
        # Split predictions into components
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4]
        pred_classes = predictions[..., 5:]
        
        # Get target components
        target_boxes = targets["boxes"]
        target_classes = targets["classes"]
        
        # Compute box regression loss
        box_loss = nn.SmoothL1Loss()(pred_boxes, target_boxes)
        
        # Compute confidence loss
        conf_loss = nn.BCEWithLogitsLoss()(pred_conf, targets["conf"])
        
        # Compute classification loss
        cls_loss = nn.CrossEntropyLoss()(pred_classes, target_classes)
        
        # Combine losses
        total_loss = box_loss + conf_loss + cls_loss
        
        return {
            "total_loss": total_loss,
            "box_loss": box_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss
        }
        
    def post_process(
        self,
        predictions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Post-process predictions with NMS."""
        # Split predictions
        boxes = predictions[..., :4]
        scores = predictions[..., 4]
        classes = predictions[..., 5:]
        
        # Apply confidence threshold
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        
        # Apply NMS
        keep = torchvision.ops.nms(boxes, scores, self.nms_threshold)
        
        return boxes[keep], scores[keep], classes[keep]

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels, 256, 1)
        self.smooth_conv = nn.Conv2d(256, 256, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Lateral connections
        c5 = self.lateral_conv(x)
        
        # Top-down pathway
        p5 = self.smooth_conv(c5)
        p4 = self._upsample_add(p5, c4)
        p3 = self._upsample_add(p4, c3)
        p2 = self._upsample_add(p3, c2)
        
        return [p2, p3, p4, p5]
        
    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, size=y.shape[-2:]) + y

class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        
        # Output layers
        self.box_layer = nn.Conv2d(256, num_anchors * 4, 1)
        self.conf_layer = nn.Conv2d(256, num_anchors, 1)
        self.cls_layer = nn.Conv2d(256, num_anchors * num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        
        # Get predictions
        boxes = self.box_layer(x)
        conf = self.conf_layer(x)
        classes = self.cls_layer(x)
        
        # Reshape predictions
        batch_size = x.shape[0]
        boxes = boxes.view(batch_size, -1, 4)
        conf = conf.view(batch_size, -1)
        classes = classes.view(batch_size, -1, self.num_classes)
        
        # Combine predictions
        return torch.cat([boxes, conf.unsqueeze(-1), classes], dim=-1) 