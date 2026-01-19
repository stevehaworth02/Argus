"""
Argus Forecasting Training Script
===================================
Train seizure forecasting model on TUSZ dataset.

Target: 70-80% ROC-AUC for 30-minute warning window
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecasting_model import MultiTaskSeizureModel, ForecastingOnlyModel


class ForecastingDataset(Dataset):
    """Dataset for seizure forecasting."""
    
    def __init__(self, data_file, normalize=True):
        """
        Args:
            data_file: Path to pickle file with preprocessed windows
            normalize: Whether to apply z-score normalization per channel
        """
        with open(data_file, 'rb') as f:
            self.windows = pickle.load(f)
        
        self.normalize = normalize
        
        # Convert labels to binary
        self.label_map = {'pre_ictal': 1, 'inter_ictal': 0}
        
        print(f"Loaded {len(self.windows)} windows from {data_file}")
        
        # Count labels
        labels = [self.label_map[w['label']] for w in self.windows]
        print(f"  Pre-ictal: {sum(labels)} ({sum(labels)/len(labels)*100:.2f}%)")
        print(f"  Inter-ictal: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.2f}%)")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # Get data and label
        data = window['data'].astype(np.float32)  # (channels, time)
        label = self.label_map[window['label']]
        
        # Normalize per channel
        if self.normalize:
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True) + 1e-8
            data = (data - mean) / std
        
        # Convert to tensor
        data = torch.from_numpy(data)
        label = torch.tensor(label, dtype=torch.float32)
        
        return data, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for data, labels in tqdm(dataloader, desc="Training", leave=False):
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data).squeeze()
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    
    # Compute additional metrics
    all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
    cm = confusion_matrix(all_labels, all_preds_binary)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'ppv': ppv,
        'npv': npv,
        'preds': all_preds,
        'labels': all_labels
    }
    
    return metrics


def plot_training_curves(train_losses, train_aucs, val_aucs, save_path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # AUC curve
    ax2.plot(train_aucs, label='Train AUC', linewidth=2)
    ax2.plot(val_aucs, label='Val AUC', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('ROC-AUC Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (70%)')
    ax2.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Goal (80%)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(labels, preds, save_path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Seizure Forecasting ROC Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training pipeline."""
    
    # ============================================
    # CONFIGURATION - CHANGE THESE PATHS
    # ============================================
    
    config = {
        'data_dir': '/mnt/c/Users/0218s/Desktop/Argus/data/tusz_forecasting',
        'output_dir': '/mnt/c/Users/0218s/Desktop/Argus/models/forecasting',
        
        # Model
        'n_channels': 32,
        'sequence_length': 2000,
        'hidden_size': 128,
        'num_lstm_layers': 2,
        'dropout': 0.4,
        'use_attention': True,
        
        # Training
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 10,  # Early stopping patience
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 50)
    print("ARGUS SEIZURE FORECASTING TRAINING")
    print("=" * 50)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # ============================================
    # LOAD DATA
    # ============================================
    
    print("\n" + "=" * 50)
    print("Loading Data")
    print("=" * 50)
    
    train_dataset = ForecastingDataset(
        os.path.join(config['data_dir'], 'train_forecasting_balanced.pkl'),
        normalize=True
    )
    
    val_dataset = ForecastingDataset(
        os.path.join(config['data_dir'], 'dev_forecasting_balanced.pkl'),
        normalize=True
    )
    
    test_dataset = ForecastingDataset(
        os.path.join(config['data_dir'], 'eval_forecasting_balanced.pkl'),
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ============================================
    # CREATE MODEL
    # ============================================
    
    print("\n" + "=" * 50)
    print("Creating Model")
    print("=" * 50)
    
    model = ForecastingOnlyModel(
        n_channels=config['n_channels'],
        sequence_length=config['sequence_length'],
        hidden_size=config['hidden_size'],
        num_lstm_layers=config['num_lstm_layers'],
        dropout=config['dropout'],
        use_attention=config['use_attention']
    )
    
    model = model.to(config['device'])
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1024 / 1024:.2f} MB")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # ============================================
    # TRAINING LOOP
    # ============================================
    
    print("\n" + "=" * 50)
    print("Training")
    print("=" * 50)
    
    best_val_auc = 0
    patience_counter = 0
    train_losses = []
    train_aucs = []
    val_aucs = []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, config['device'])
        val_auc = val_metrics['auc']
        
        # Update scheduler
        scheduler.step(val_auc)
        
        # Track history
        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'config': config
            }, os.path.join(config['output_dir'], 'best_forecasting_model.pth'))
            
            print(f"✓ Saved best model (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # ============================================
    # FINAL EVALUATION
    # ============================================
    
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config['output_dir'], 'best_forecasting_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, config['device'])
    
    print(f"\nTest Results:")
    print(f"  ROC-AUC: {test_metrics['auc']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  PPV: {test_metrics['ppv']:.4f}")
    print(f"  NPV: {test_metrics['npv']:.4f}")
    
    # ============================================
    # SAVE VISUALIZATIONS
    # ============================================
    
    print("\nGenerating visualizations...")
    
    # Training curves
    plot_training_curves(
        train_losses, train_aucs, val_aucs,
        os.path.join(config['output_dir'], 'training_curves.png')
    )
    
    # ROC curve
    plot_roc_curve(
        test_metrics['labels'], test_metrics['preds'],
        os.path.join(config['output_dir'], 'roc_curve.png')
    )
    
    print("\n✓ Training complete!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    
    if test_metrics['auc'] >= 0.70:
        print("\n🎯 Target achieved! (AUC ≥ 70%)")
    if test_metrics['auc'] >= 0.80:
        print("🚀 Stretch goal achieved! (AUC ≥ 80%)")


if __name__ == "__main__":
    main()