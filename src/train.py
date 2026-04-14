"""
Training script for bird classification with checkpoint save/resume functionality
Implements two-stage training: Stage 1 (feature extraction) and Stage 2 (fine-tuning)
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from config import (
    DEVICE, NUM_CLASSES, STAGE1_EPOCHS, STAGE1_LR, STAGE2_EPOCHS, STAGE2_LR,
    LABEL_SMOOTHING, CHECKPOINT_DIR, SAVE_INTERVAL, WARMUP_EPOCHS,
    COSINE_MIN_LR, EARLY_STOPPING_PATIENCE, LOG_INTERVAL, USE_AMP,
    SEED
)
from src.model import BirdClassifier
from src.dataset import get_data_loaders, mixup_batch, cutmix_batch


class TrainingManager:
    """Manages training, validation, checkpointing, and resuming"""
    
    def __init__(self, model: BirdClassifier, device: str = DEVICE):
        self.model = model.to(device)
        self.device = device
        
        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": []
        }
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # AMP scaler
        self.scaler = GradScaler() if USE_AMP else None
        
        print(f"[TRAINER] Initialized with device: {device}")
    
    def get_optimizer(self, stage: int = 1) -> optim.AdamW:
        """Create optimizer for training stage"""
        if stage == 1:
            # Stage 1: Only train classification head, freeze backbone
            self.model.freeze_backbone(freeze=True)
            lr = STAGE1_LR
            params = self.model.classifier.parameters()
            
        else:  # stage == 2
            # Stage 2: Fine-tune entire model
            self.model.freeze_backbone(freeze=False)
            lr = STAGE2_LR
            params = self.model.parameters()
        
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        print(f"[OPTIMIZER] Stage {stage}: LR={lr}, Training: {'Classifier head' if stage == 1 else 'Full model'}")
        return optimizer
    
    def get_scheduler(self, optimizer: optim.Optimizer, total_epochs: int):
        """Create learning rate scheduler with warmup"""
        def warmup_cosine_lr(epoch):
            if epoch < WARMUP_EPOCHS:
                # Linear warmup
                return float(epoch) / float(max(1, WARMUP_EPOCHS))
            else:
                # Cosine annealing
                progress = float(epoch - WARMUP_EPOCHS) / float(max(1, total_epochs - WARMUP_EPOCHS))
                return max(COSINE_MIN_LR, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int) -> Dict:
        """Train for one epoch with progress bar"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", unit=" batch", 
                    ncols=100, position=0, leave=True)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with AMP
            optimizer.zero_grad()
            
            if USE_AMP and self.scaler is not None:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar with current metrics
            avg_loss = running_loss / (batch_idx + 1)
            acc = correct / total
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})
        
        pbar.close()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc
        }
    
    @torch.no_grad()
    def validate(self, valid_loader: DataLoader) -> Dict:
        """Validate on validation set with progress bar"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(valid_loader, desc="Validation", unit=" batch", 
                    ncols=100, position=0, leave=True)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if USE_AMP and self.scaler is not None:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar with current metrics
            avg_loss = running_loss / (batch_idx + 1)
            acc = correct / total
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})
        
        pbar.close()
        
        val_loss = running_loss / len(valid_loader)
        val_acc = correct / total
        
        return {
            "loss": val_loss,
            "accuracy": val_acc
        }
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, stage: int) -> Path:
        """Save model checkpoint with full training state"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_stage{stage}_epoch{epoch:03d}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "epochs_no_improve": self.epochs_no_improve,
            "training_history": self.training_history,
            "model_config": {
                "num_classes": self.model.num_classes,
                "model_name": self.model.model_name
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"[CHECKPOINT] Saved: {checkpoint_path.name}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path, optimizer: optim.Optimizer) -> int:
        """Load model checkpoint and resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_acc = checkpoint["best_val_acc"]
        self.best_epoch = checkpoint["best_epoch"]
        self.epochs_no_improve = checkpoint["epochs_no_improve"]
        self.training_history = checkpoint["training_history"]
        
        print(f"[CHECKPOINT] Loaded: {checkpoint_path.name}")
        print(f"[CHECKPOINT] Resuming from epoch {self.current_epoch}")
        print(f"[CHECKPOINT] Best validation accuracy so far: {self.best_val_acc:.4f}")
        
        return self.current_epoch
    
    def save_best_model(self, stage: int):
        """Save the best model for this stage"""
        best_path = self.checkpoint_dir / f"best_model_stage{stage}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "num_classes": self.model.num_classes,
                "model_name": self.model.model_name
            }
        }, best_path)
        print(f"[MODEL] Best model saved: {best_path.name}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        print(f"[HISTORY] Saved to {history_path.name}")


def train_stage(
    trainer: TrainingManager,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    stage: int,
    epochs: int,
    resume_from: Optional[Path] = None
):
    """
    Train for a specific stage
    Stage 1: Freeze backbone, train classifier head
    Stage 2: Fine-tune entire model
    """
    print(f"\n{'='*70}", flush=True)
    print(f"STAGE {stage} TRAINING", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    print(f"[TRAIN_STAGE] Getting optimizer for stage {stage}...", flush=True)
    optimizer = trainer.get_optimizer(stage=stage)
    
    print(f"[TRAIN_STAGE] Getting scheduler...", flush=True)
    scheduler = trainer.get_scheduler(optimizer, total_epochs=epochs)
    
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if resume_from is not None:
        print(f"[TRAIN_STAGE] Loading checkpoint...", flush=True)
        start_epoch = trainer.load_checkpoint(resume_from, optimizer)
    
    print(f"[TRAIN_STAGE] Starting epoch loop from {start_epoch} to {epochs}...\n", flush=True)
    
    for epoch in range(start_epoch, epochs):
        print(f"\n[Epoch {epoch}/{epochs-1}]", flush=True)
        
        # Train
        print(f"  [TRAIN] Training epoch {epoch}...", flush=True)
        train_metrics = trainer.train_epoch(train_loader, optimizer, epoch)
        
        # Validate
        print(f"  [VALID] Validating epoch {epoch}...", flush=True)
        val_metrics = trainer.validate(valid_loader)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics with improved formatting
        print(f"\n  Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f}", flush=True)
        print(f"  Valid - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f}", flush=True)
        print(f"  LR: {current_lr:.2e}", flush=True)
        
        # Update history
        trainer.training_history["train_loss"].append(train_metrics["loss"])
        trainer.training_history["train_acc"].append(train_metrics["accuracy"])
        trainer.training_history["val_loss"].append(val_metrics["loss"])
        trainer.training_history["val_acc"].append(val_metrics["accuracy"])
        trainer.training_history["learning_rates"].append(current_lr)
        
        # Check if best
        if val_metrics["accuracy"] > trainer.best_val_acc:
            trainer.best_val_acc = val_metrics["accuracy"]
            trainer.best_epoch = epoch
            trainer.epochs_no_improve = 0
            print(f"  >>> NEW BEST! Val Acc: {trainer.best_val_acc:.4f} (Epoch {epoch})", flush=True)
        else:
            trainer.epochs_no_improve += 1
            print(f"  No improvement ({trainer.epochs_no_improve}/{EARLY_STOPPING_PATIENCE})", flush=True)
        
        # Save checkpoint and update history
        if (epoch + 1) % SAVE_INTERVAL == 0:
            trainer.save_checkpoint(epoch, optimizer, stage=stage)
            trainer.save_training_history()  # Update JSON after every checkpoint
        
        # Early stopping
        if trainer.epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n[EARLY STOPPING] No improvement for {EARLY_STOPPING_PATIENCE} epochs", flush=True)
            break
    
    # Save best model and history
    trainer.save_best_model(stage=stage)
    trainer.save_training_history()


def main():
    """Main training pipeline"""
    print("[MAIN] Setting random seeds...", flush=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    print("[MAIN] Loading data...", flush=True)
    train_loader, valid_loader, train_dataset = get_data_loaders()
    print(f"[MAIN] Data loaded: {len(train_loader)} train batches, {len(valid_loader)} valid batches", flush=True)
    
    print("\n[MAIN] Building architecture...", flush=True)
    model = BirdClassifier(num_classes=NUM_CLASSES)
    model.print_summary()
    
    print("[MAIN] Creating trainer...")
    trainer = TrainingManager(model, device=DEVICE)
    
    # STAGE 1: Feature extraction (freeze backbone)
    print(f"\n[MAIN] Starting STAGE 1 training for {STAGE1_EPOCHS} epochs")
    train_stage(
        trainer, train_loader, valid_loader,
        stage=1,
        epochs=STAGE1_EPOCHS,
        resume_from=None
    )
    
    # STAGE 2: Fine-tuning (unfreeze backbone)
    print(f"\n[MAIN] Starting STAGE 2 training for {STAGE2_EPOCHS} epochs")
    train_stage(
        trainer, train_loader, valid_loader,
        stage=2,
        epochs=STAGE2_EPOCHS,
        resume_from=None
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f} (Epoch {trainer.best_epoch})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
