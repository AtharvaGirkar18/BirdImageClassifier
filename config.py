"""
Configuration file for Bird Image Classification project
All hyperparameters and settings are defined here
"""

import os
from pathlib import Path

# Data paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "Birds_25"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

# Create checkpoint directory if it doesn't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset parameters
NUM_CLASSES = 25  # 25 bird species
IMAGE_SIZE = 224   # EfficientNet standard input size
TRAIN_BATCH_SIZE = 32  # Reduced for 4GB GPU (GTX 1650Ti) - was 64
VALID_BATCH_SIZE = 64  # Reduced for 4GB GPU - was 128
NUM_WORKERS = 0    # Set to 0 for Windows compatibility (avoids multiprocessing issues)

# Model parameters
MODEL_NAME = "efficientnet_b4"  # b4 works on 4GB with AMP + reduced batch size
PRETRAINED = True  # Use ImageNet pre-trained weights

# Training parameters - Stage 1 (Feature extraction)
STAGE1_EPOCHS = 8
STAGE1_LR = 1e-3
STAGE1_FREEZE_BACKBONE = True  # Freeze backbone, train only head

# Training parameters - Stage 2 (Fine-tuning)
STAGE2_EPOCHS = 15  # Reduced from 25
STAGE2_LR = 1e-4
STAGE2_FREEZE_BACKBONE = False  # Unfreeze and fine-tune everything

# Total epochs
TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS  # 23 epochs

# Optimizer parameters
OPTIMIZER = "adamw"
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3

# Loss function parameters
LABEL_SMOOTHING = 0.1  # Regularization

# Regularization
DROPOUT_RATE = 0.4
DROPOUT_CONNECT_RATE = 0.2

# Data augmentation
USE_MIXUP = True
MIXUP_ALPHA = 1.0
USE_CUTMIX = False
CUTMIX_ALPHA = 1.0
USE_AUGMENT = True

# Learning rate schedule
LR_SCHEDULER = "cosine"  # cosine annealing
COSINE_MIN_LR = 1e-5
NUM_CYCLES = 1

# Checkpoint & Resume
SAVE_BEST_ONLY = True
SAVE_INTERVAL = 1  # Save every N epochs
RESUME_FROM_CHECKPOINT = None  # Set to checkpoint path to resume training

# Device - dynamically select GPU if available
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[CONFIG] Using device: {DEVICE}", flush=True)
USE_AMP = False  # Disabled - causing NaN loss on GTX 1650 Ti

# Random seed for reproducibility
SEED = 42

# Early stopping (optional)
EARLY_STOPPING_PATIENCE = 5  # Reduced from 10
EARLY_STOPPING_MIN_DELTA = 0.001

# Testing parameters
TEST_JIT_COMPILE = False  # Use TorchScript
ENSEMBLE_MODELS = 3  # Number of models for ensemble

# Logging
LOG_INTERVAL = 10  # Log every N batches (reduced from 50 for more frequent updates)
VALIDATE_INTERVAL = 1  # Validate every N epochs
VERBOSE = True
