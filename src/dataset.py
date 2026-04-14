"""
Dataset and DataLoader utilities for bird classification
Includes data augmentation pipeline
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from config import (
    TRAIN_DIR, VALID_DIR, IMAGE_SIZE, TRAIN_BATCH_SIZE, 
    VALID_BATCH_SIZE, NUM_WORKERS, USE_MIXUP, USE_CUTMIX,
    MIXUP_ALPHA, CUTMIX_ALPHA, SEED
)


class BirdDataset(Dataset):
    """Custom dataset for bird species classification"""
    
    def __init__(self, root_dir: Path, transform=None, split: str = "train"):
        """
        Args:
            root_dir: Path to train or valid directory
            transform: Image transformations to apply
            split: 'train' or 'valid' (for logging)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Get class names from subdirectories
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name
                self.class_names.append(class_name)
                
                # Collect all images from this class
                for img_file in class_dir.glob("*.jpg"):
                    self.images.append(str(img_file))
                    self.labels.append(idx)
        
        print(f"[{split.upper()}] Loaded {len(self.images)} images from {len(self.class_names)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transforms():
    """Aggressive augmentation for training"""
    return transforms.Compose([
        # Resize and crop
        transforms.RandomResizedCrop(
            IMAGE_SIZE, 
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33)
        ),
        
        # Geometric transformations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        
        # Color augmentation
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        
        # Gaussian blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Random grayscale
        transforms.RandomGrayscale(p=0.1),
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_valid_transforms():
    """Minimal augmentation for validation/test"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup augmentation to a batch"""
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    
    index = torch.randperm(batch_size)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    
    # One-hot labels for mixup
    labels_a = labels
    labels_b = labels[index]
    
    return mixed_images, (labels_a, labels_b, lam)


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply CutMix augmentation to a batch"""
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    
    index = torch.randperm(batch_size)
    
    # Random box
    h, w = images.size(2), images.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
    
    labels_a = labels
    labels_b = labels[index]
    
    return images, (labels_a, labels_b, lam)


def get_data_loaders(
    train_dir: Path = TRAIN_DIR,
    valid_dir: Path = VALID_DIR,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    valid_batch_size: int = VALID_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED
) -> Tuple[DataLoader, DataLoader, BirdDataset]:
    """Create train and validation dataloaders"""
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create datasets
    train_dataset = BirdDataset(train_dir, transform=get_train_transforms(), split="train")
    valid_dataset = BirdDataset(valid_dir, transform=get_valid_transforms(), split="valid")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, train_dataset
