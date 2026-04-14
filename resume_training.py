#!/usr/bin/env python3
"""
Resume training from checkpoint
Use this to pause and resume training anytime
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.utils import set_seed, print_gpu_info, get_device, get_latest_checkpoint
from src.model import BirdClassifier
from src.dataset import get_data_loaders
from src.train import TrainingManager, train_stage
from config import CHECKPOINT_DIR, DEVICE, STAGE1_EPOCHS, STAGE2_EPOCHS, NUM_CLASSES

if __name__ == "__main__":
    print("="*70)
    print("BIRD IMAGE CLASSIFIER - RESUME TRAINING")
    print("="*70 + "\n")
    
    device = get_device()
    print_gpu_info()
    set_seed(42)
    
    # Find latest checkpoint
    print("[RESUME] Searching for checkpoints...")
    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
    
    if latest_ckpt is None:
        print("[ERROR] No checkpoints found!")
        print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
        print("\nStart a new training session with: python run_training.py")
        sys.exit(1)
    
    print(f"[RESUME] Latest checkpoint: {latest_ckpt.name}\n")
    
    # Load data
    print("[SETUP] Loading data...")
    train_loader, valid_loader, train_dataset = get_data_loaders()
    
    # Build model
    print("[MODEL] Building architecture...")
    model = BirdClassifier(num_classes=NUM_CLASSES)
    
    # Create trainer
    trainer = TrainingManager(model, device=device)
    
    # Determine stage from checkpoint name
    if "stage1" in latest_ckpt.name:
        next_stage = 1
        next_epochs = STAGE1_EPOCHS
        print(f"[RESUME] Continuing Stage 1 training (epochs remaining: {STAGE1_EPOCHS})...")
    else:
        next_stage = 2
        next_epochs = STAGE2_EPOCHS
        print(f"[RESUME] Continuing Stage 2 fine-tuning (epochs remaining: {STAGE2_EPOCHS})...")
    
    # Resume training
    print(f"\n[RESUME] Starting from epoch {trainer.current_epoch}...\n")
    
    # Continue training
    train_stage(
        trainer, train_loader, valid_loader,
        stage=next_stage,
        epochs=next_epochs,
        resume_from=latest_ckpt
    )
    
    print(f"\n[COMPLETE] Best validation accuracy: {trainer.best_val_acc:.4f}")
