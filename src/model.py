"""
Model architecture for bird classification
Uses EfficientNet with custom classification head
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, efficientnet_b5, EfficientNet_B4_Weights, EfficientNet_B5_Weights

from config import NUM_CLASSES, DROPOUT_RATE, MODEL_NAME, PRETRAINED


class BirdClassifier(nn.Module):
    """
    EfficientNet-based classifier for bird species recognition
    Includes custom head with dropout and activation functions
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        model_name: str = MODEL_NAME,
        pretrained: bool = PRETRAINED,
        dropout_rate: float = DROPOUT_RATE
    ):
        """
        Args:
            num_classes: Number of output classes (25 bird species)
            model_name: Model architecture (efficientnet_b4 or efficientnet_b5)
            pretrained: Use ImageNet pre-trained weights
            dropout_rate: Dropout rate for regularization
        """
        super(BirdClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained EfficientNet
        if model_name == "efficientnet_b4":
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b4(weights=weights)
            in_features = 1792  # EfficientNet-B4 output features
            
        elif model_name == "efficientnet_b5":
            weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b5(weights=weights)
            in_features = 2048  # EfficientNet-B5 output features
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove the original classification head
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head for birds
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # Flatten already-pooled features
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),      # Hidden layer
            nn.BatchNorm1d(512),               # Batch normalization
            nn.GELU(),                         # GELU activation (better than ReLU)
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),               # Second hidden layer
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes)        # Output layer
        )
        
        print(f"[MODEL] Initialized {model_name} with {num_classes} classes")
        print(f"[MODEL] Pretrained: {pretrained}")
        print(f"[MODEL] Dropout rate: {dropout_rate}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
        Returns:
            Logits (batch_size, num_classes)
        """
        # Backbone extracts features
        features = self.backbone(x)  # Now returns flattened features
        
        # Classification head
        logits = self.classifier(features)
        
        return logits
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze backbone parameters
        Used for Stage 1 (freeze) and Stage 2 (unfreeze) training
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        print(f"[MODEL] Backbone {status} for training")
    
    def get_parameter_groups(self, lr_backbone: float = 1e-4, lr_head: float = 1e-3) -> list:
        """
        Get parameter groups for different learning rates
        Backbone gets lower LR, head gets higher LR
        """
        param_groups = [
            {"params": self.backbone.parameters(), "lr": lr_backbone, "weight_decay": 1e-4},
            {"params": self.classifier.parameters(), "lr": lr_head, "weight_decay": 1e-4},
        ]
        return param_groups
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_summary(self):
        """Print model architecture summary"""
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total trainable parameters: {self.num_parameters:,}")
        print(f"{'='*60}\n")
        
        # Print encoder summary
        print("Backbone (EfficientNet):")
        print(f"  Architecture: {self.model_name.upper()}")
        print(f"  Features: Extracted from ImageNet pre-trained model")
        
        print("\nClassification Head:")
        for name, module in self.classifier.named_children():
            print(f"  {name}: {module}")
        print()
