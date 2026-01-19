"""
EEG Artifact Detection - Training Script
Ceribell Project

Trains artifact detector using same proven CNN-LSTM architecture
that achieved 98.6% ROC-AUC on seizure detection.

Binary Classification:
- Class 0: Clean EEG
- Class 1: Artifact (any type)

Usage:
    python train_artifact_detector.py --data_path ./preprocessed/artifacts_01_tcp_ar.npz
    python train_artifact_detector.py --epochs 50 --batch_size 64

Author: Ceribell Seizure Detector Project
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Import proven model architecture from seizure detector
sys.path.append('../data_viz')
from model import SeizureDetectorCNNLSTM, create_seizure_detector

# Import dataset utilities
sys.path.append('../training')
try:
    from modules.dataset import EEGSeizureDataset, create_train_val_split
except:
    print("Warning: Could not import dataset utilities. Using manual implementation.")
    from torch.utils.data import Dataset
    
    class EEGSeizureDataset(Dataset):
        def __init__(self, windows, labels, augment=False):
            self.windows = windows.astype(np.float32)
            self.labels = labels.astype(np.int64)
            self.augment = augment
        
        def __len__(self):
            return len(self.windows)
        
        def __getitem__(self, idx):
            window = torch.from_numpy(self.windows[idx]).float()
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return window, label


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset."""
    n_samples = len(labels)
    n_artifact = np.sum(labels)
    n_clean = n_samples - n_artifact
    
    # Inverse frequency weighting
    weight_clean = n_samples / (2 * n_clean) if n_clean > 0 else 1.0
    weight_artifact = n_samples / (2 * n_artifact) if n_artifact > 0 else 1.0
    
    weights = torch.tensor([weight_clean, weight_artifact], dtype=torch.float32)
    
    print(f"\n⚖️  Class Weights:")
    print(f"  • Clean weight: {weight_clean:.4f}")
    print(f"  • Artifact weight: {weight_artifact:.4f}")
    
    return weights


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, (windows, labels) in enumerate(train_loader):
        windows = windows.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(windows)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of artifact class
        preds = torch.argmax(logits, dim=1)
        
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    epoch_f1 = f1_score(all_labels, all_preds)
    
    return epoch_loss, epoch_auc, epoch_f1


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for windows, labels in val_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _ = model(windows)
            loss = criterion(logits, labels)
            
            # Track metrics
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(val_loader)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    epoch_f1 = f1_score(all_labels, all_preds)
    
    # Calculate additional metrics
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    return epoch_loss, epoch_auc, epoch_f1, pr_auc, all_probs, all_labels


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[0, 1].plot(history['train_auc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_auc'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title('Training & Validation ROC-AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.5, 1.0])
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_f1'], label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training & Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best metrics summary
    best_val_auc_epoch = np.argmax(history['val_auc'])
    best_val_f1_epoch = np.argmax(history['val_f1'])
    
    summary_text = f"Best Val ROC-AUC: {history['val_auc'][best_val_auc_epoch]:.4f} (Epoch {best_val_auc_epoch+1})\n"
    summary_text += f"Best Val F1: {history['val_f1'][best_val_f1_epoch]:.4f} (Epoch {best_val_f1_epoch+1})\n"
    summary_text += f"Final Val ROC-AUC: {history['val_auc'][-1]:.4f}\n"
    summary_text += f"Final Val F1: {history['val_f1'][-1]:.4f}"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Training Summary')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history saved to: {save_path}")
    plt.close()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train artifact detector')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed .npz file')
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size (default: medium)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation split size')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EEG ARTIFACT DETECTION - TRAINING")
    print("Ceribell Project")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  • Data: {args.data_path}")
    print(f"  • Model size: {args.model_size}")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • Validation split: {args.val_size*100:.0f}%")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    
    data = np.load(args.data_path, allow_pickle=True)
    windows = data['windows']
    labels = data['labels']
    
    print(f"Loaded {len(windows):,} windows")
    print(f"  • Artifact: {np.sum(labels):,} ({100*np.mean(labels):.2f}%)")
    print(f"  • Clean: {len(labels) - np.sum(labels):,} ({100*(1-np.mean(labels)):.2f}%)")
    
    # =========================================================================
    # CREATE TRAIN/VAL SPLIT
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("CREATING TRAIN/VAL SPLIT")
    print(f"{'='*70}")
    
    try:
        split_data = create_train_val_split(windows, labels, val_size=args.val_size, random_state=42)
    except:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            windows, labels, test_size=args.val_size, random_state=42, stratify=labels
        )
        split_data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}
    
    # Create datasets
    train_dataset = EEGSeizureDataset(split_data['X_train'], split_data['y_train'], augment=True)
    val_dataset = EEGSeizureDataset(split_data['X_val'], split_data['y_val'], augment=False)
    
    # Calculate class weights
    class_weights = calculate_class_weights(split_data['y_train'])
    class_weights = class_weights.to(device)
    
    # Create dataloaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("CREATING MODEL")
    print(f"{'='*70}")
    
    # Reuse proven architecture from seizure detector
    model = create_seizure_detector(model_size=args.model_size)
    model = model.to(device)
    
    # =========================================================================
    # TRAINING SETUP
    # =========================================================================
    print(f"\n{'='*70}")
    print("TRAINING SETUP")
    print(f"{'='*70}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    print(f"✓ Loss function: CrossEntropyLoss (weighted)")
    print(f"✓ Optimizer: Adam (lr={args.lr})")
    print(f"✓ Scheduler: ReduceLROnPlateau")
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    history = {
        'train_loss': [], 'train_auc': [], 'train_f1': [],
        'val_loss': [], 'val_auc': [], 'val_f1': [], 'val_pr_auc': []
    }
    best_val_auc = 0.0
    best_epoch = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Train
        train_loss, train_auc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_auc, val_f1, val_pr_auc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_auc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        history['val_pr_auc'].append(val_pr_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            
            best_model_path = os.path.join(args.save_dir, 'best_artifact_detector.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_f1': val_f1,
                'class_weights': class_weights,
                'args': args
            }, best_model_path)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | PR-AUC: {val_pr_auc:.4f}")
        if val_auc == best_val_auc:
            print(f"  ✓ New best model saved!")
        print()
    
    total_time = time.time() - start_time
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best Val ROC-AUC: {best_val_auc:.4f} (Epoch {best_epoch+1})")
    print(f"Final Val ROC-AUC: {history['val_auc'][-1]:.4f}")
    print(f"Final Val F1: {history['val_f1'][-1]:.4f}")
    print(f"Final Val PR-AUC: {history['val_pr_auc'][-1]:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_artifact_detector.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'args': args
    }, final_model_path)
    
    print(f"\n📁 Model files:")
    print(f"  • Best model: {best_model_path}")
    print(f"  • Final model: {final_model_path}")
    
    # Plot training history
    plot_path = os.path.join(args.save_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Evaluate: python evaluate_artifact_detector.py --model_path {best_model_path}")
    print(f"  2. Build pipeline: python unified_pipeline.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
