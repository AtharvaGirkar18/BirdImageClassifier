"""
Utility functions for bird classification project
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[SEED] Set to {seed}")


def load_checkpoint_for_inference(checkpoint_path: Path, device: str = "cuda"):
    """Load checkpoint for inference"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    from src.model import BirdClassifier
    
    model = BirdClassifier(
        num_classes=checkpoint["model_config"]["num_classes"],
        model_name=checkpoint["model_config"]["model_name"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"[LOAD] Loaded checkpoint from {checkpoint_path.name}")
    return model


def get_latest_checkpoint(checkpoint_dir: Path, stage: int = None) -> Path:
    """Get latest checkpoint from directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if stage is not None:
        # Get latest checkpoint for specific stage
        pattern = f"checkpoint_stage{stage}_*.pt"
        files = sorted(checkpoint_dir.glob(pattern))
    else:
        # Get latest checkpoint overall
        pattern = "checkpoint_*.pt"
        files = sorted(checkpoint_dir.glob(pattern))
    
    if not files:
        return None
    
    latest = files[-1]
    print(f"[CHECKPOINT] Found latest: {latest.name}")
    return latest


def create_submission_csv(predictions: np.ndarray, class_names: List[str], output_path: Path):
    """Create submission CSV with predictions"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pred_classes = [class_names[p] for p in predictions]
    
    with open(output_path, 'w') as f:
        f.write("image_id,species\n")
        for idx, species in enumerate(pred_classes):
            f.write(f"{idx},{species}\n")
    
    print(f"[SUBMISSION] Saved to {output_path}")


def save_config_to_json(config_dict: Dict, output_path: Path):
    """Save configuration to JSON for reproducibility"""
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"[CONFIG] Saved to {output_path}")


def count_parameters(model):
    """Count trainable and total parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[PARAMETERS]")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen:    {total - trainable:,}")
    
    return total, trainable


def get_device():
    """Get appropriate device"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] CUDA Version: {torch.version.cuda}")
    else:
        device = "cpu"
        print(f"[DEVICE] GPU not available, using CPU")
    
    return device


def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"\n[GPU INFO]")
        print(f"  Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        print()
    else:
        print("[GPU INFO] No GPU available")


class AverageMeter:
    """Track average and current value"""
    
    def __init__(self, name=''):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"
