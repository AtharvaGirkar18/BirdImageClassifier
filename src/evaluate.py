"""
Evaluation script for bird classification
Includes metrics computation and visualization
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE, USE_AMP, CHECKPOINT_DIR
from src.model import BirdClassifier
from src.dataset import get_data_loaders


class Evaluator:
    """Evaluates model performance on validation/test sets"""
    
    def __init__(self, model: BirdClassifier, criterion: nn.Module, device: str = DEVICE):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, class_names: list = None) -> Dict:
        """
        Evaluate model on dataset
        Returns comprehensive metrics
        """
        all_logits = []
        all_labels = []
        running_loss = 0.0
        
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if USE_AMP:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            running_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
        
        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Get predictions
        predictions = torch.argmax(all_logits, dim=1)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, predictions, average='weighted', zero_division=0)
        
        avg_loss = running_loss / len(data_loader)
        
        results = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": predictions,
            "labels": all_labels,
            "logits": all_logits
        }
        
        # Print metrics
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Loss:      {avg_loss:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"{'='*60}\n")
        
        # Class-wise metrics
        if class_names:
            print("\nPER-CLASS METRICS:")
            print(classification_report(
                all_labels, predictions,
                target_names=class_names,
                digits=4
            ))
        
        return results
    
    @staticmethod
    def plot_confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, class_names: list, save_path: Path = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Bird Species Classification', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history: Dict, save_path: Path = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=4)
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=4)
        axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_learning_rate(learning_rates: list, save_path: Path = None):
        """Plot learning rate schedule"""
        plt.figure(figsize=(10, 4))
        plt.plot(learning_rates, marker='o', markersize=3, linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Learning rate plot saved to {save_path}")
        
        plt.show()


def evaluate_best_model(stage: int = 2):
    """Evaluate best model from a training stage"""
    print(f"[EVALUATE] Loading best model from stage {stage}...")
    
    best_model_path = CHECKPOINT_DIR / f"best_model_stage{stage}.pt"
    
    if not best_model_path.exists():
        print(f"Error: Model not found at {best_model_path}")
        return
    
    # Load model
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model = BirdClassifier(num_classes=checkpoint["model_config"]["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create evaluator
    evaluator = Evaluator(model, criterion, device=DEVICE)
    
    # Load data
    _, valid_loader, train_dataset = get_data_loaders()
    
    # Evaluate
    results = evaluator.evaluate(valid_loader, class_names=train_dataset.class_names)
    
    # Plot confusion matrix
    confusion_path = CHECKPOINT_DIR / f"confusion_matrix_stage{stage}.png"
    evaluator.plot_confusion_matrix(
        results['predictions'],
        results['labels'],
        class_names=train_dataset.class_names,
        save_path=confusion_path
    )
    
    return results


if __name__ == "__main__":
    # Example: Evaluate best model from stage 2
    evaluate_best_model(stage=2)
