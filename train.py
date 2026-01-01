"""
EEG Seizure Detection - Training Script
Ceribell Project

Features:
    - Weighted loss for class imbalance
    - Early stopping with patience
    - Learning rate scheduling
    - Model checkpointing
    - Comprehensive metrics (F1, AUROC, sensitivity, specificity)
    - TensorBoard logging
    - Gradient clipping for stability

Author: Ceribell Seizure Detector Project
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple
import json

from modules.model import create_seizure_detector
from modules.dataset import load_preprocessed_data, create_train_val_split, create_dataloaders


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class
    
    Returns:
        Dictionary of metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    return metrics


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model: nn.Module, train_loader, criterion, optimizer,
                device: torch.device, gradient_clip: float = 1.0) -> Tuple[float, Dict]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cpu/cuda)
        gradient_clip: Gradient clipping threshold
    
    Returns:
        avg_loss: Average training loss
        metrics: Training metrics
    """
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())  # Prob of seizure class
    
    avg_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    return avg_loss, metrics


def validate_epoch(model: nn.Module, val_loader, criterion,
                   device: torch.device) -> Tuple[float, Dict]:
    """
    Validate for one epoch.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device (cpu/cuda)
    
    Returns:
        avg_loss: Average validation loss
        metrics: Validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            logits, _ = model(data)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    return avg_loss, metrics


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.mode == 'min':
            improved = metric < (self.best_score - self.min_delta)
        else:
            improved = metric > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
                return True
        
        return False


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(model: nn.Module, train_loader, val_loader,
                class_weights: torch.Tensor, config: Dict,
                device: torch.device, save_dir: str = './checkpoints'):
    """
    Complete training loop with all bells and whistles.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Class weights for loss function
        config: Training configuration dictionary
        device: Device (cpu/cuda)
        save_dir: Directory to save checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'],
        mode='max',  # Maximize F1 score
        verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(save_dir, 'runs'))
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_auroc': [], 'val_auroc': [],
        'val_sensitivity': [], 'val_specificity': []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Early stopping patience: {config['patience']}")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=config['gradient_clip']
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auroc'].append(train_metrics['auroc'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        
        # TensorBoard logging
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('F1', {'train': train_metrics['f1'], 'val': val_metrics['f1']}, epoch)
        writer.add_scalars('AUROC', {'train': train_metrics['auroc'], 'val': val_metrics['auroc']}, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f} | F1: {train_metrics['f1']:.4f} | AUROC: {train_metrics['auroc']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | F1: {val_metrics['f1']:.4f} | AUROC: {val_metrics['auroc']:.4f}")
        print(f"          Sensitivity: {val_metrics['sensitivity']:.4f} | Specificity: {val_metrics['specificity']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }
            
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Best model saved! (F1: {best_f1:.4f})")
        
        print()
        
        # Early stopping check
        if early_stopping(val_metrics['f1']):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    writer.close()
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best F1 score: {best_f1:.4f} (Epoch {best_epoch+1})")
    print(f"Final F1 score: {val_metrics['f1']:.4f}")
    print(f"Models saved to: {save_dir}")
    print("="*70 + "\n")
    
    return history, best_f1


# ============================================================================
# MAIN - COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train seizure detection model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed .npz file')
    parser.add_argument('--class_weights_path', type=str, required=True,
                       help='Path to class weights .npy file')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\nLoading preprocessed data...")
    windows, labels, _ = load_preprocessed_data(args.data_path)
    class_weights = torch.from_numpy(np.load(args.class_weights_path))
    
    # Create train/val split
    split_data = create_train_val_split(windows, labels, val_size=args.val_size)
    
    # Create dataloaders
    loaders = create_dataloaders(
        split_data['X_train'], split_data['y_train'],
        split_data['X_val'], split_data['y_val'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampling=True,
        augment_train=True
    )
    
    # Create model
    model = create_seizure_detector(model_size=args.model_size)
    model = model.to(device)
    
    # Training configuration
    config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'gradient_clip': args.gradient_clip,
        'patience': args.patience,
        'model_size': args.model_size,
        'seed': args.seed
    }
    
    # Train
    history, best_f1 = train_model(
        model, loaders['train_loader'], loaders['val_loader'],
        class_weights, config, device, args.save_dir
    )
    
    print(f"\n✓ Training complete! Best F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()