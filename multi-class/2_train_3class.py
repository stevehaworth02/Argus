"""
Step 2: Train 3-Class CNN-LSTM Model
Trains on combined TUAR + TUSZ train data with 3 classes

NO DATA LEAKAGE:
- Trains on train split only
- 80/20 train/val split from training data
- Dev set never used during training

Author: Ceribell Multi-Class System
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ============================================================================
# DATASET
# ============================================================================

class EEGDataset(Dataset):
    """PyTorch Dataset for 3-class EEG data"""
    
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ThreeClassCNNLSTM(nn.Module):
    """
    3-Class CNN-LSTM for EEG Classification
    
    Architecture:
    - CNN: Extract spatial features from channels
    - LSTM: Capture temporal patterns
    - FC: 3-class classification (background, artifact, seizure)
    """
    
    def __init__(self, num_channels=22, num_classes=3, hidden_size=128, num_layers=2):
        super(ThreeClassCNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # 3 classes
        )
    
    def forward(self, x):
        # x shape: (batch, channels, time_steps)
        
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Classification
        x = self.fc(x)
        
        return x


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for windows, labels in train_loader:
        windows = windows.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(windows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for windows, labels in val_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            
            outputs = model(windows)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def calculate_per_class_metrics(preds, labels, num_classes=3):
    """Calculate precision, recall, F1 per class"""
    metrics = {}
    
    for class_idx in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((preds == class_idx) & (labels == class_idx))
        fp = np.sum((preds == class_idx) & (labels != class_idx))
        fn = np.sum((preds != class_idx) & (labels == class_idx))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = ['Background', 'Artifact', 'Seizure'][class_idx]
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train 3-class EEG classifier')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to 3-class training data (.npz)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("="*70)
    print("STEP 2: TRAINING 3-CLASS MODEL")
    print("Background, Artifact, and Seizure Classification")
    print("="*70)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("LOADING TRAINING DATA")
    print("="*70)
    
    if not os.path.exists(args.data_path):
        print(f"\n[ERROR] Data file not found: {args.data_path}")
        print("\nPlease run data preparation first:")
        print("  python 1_prepare_3class_data.py")
        exit(1)
    
    print(f"Loading: {args.data_path}")
    data = np.load(args.data_path, allow_pickle=True)
    windows = data['windows']
    labels = data['labels']
    
    print(f"\n[OK] Data Loaded:")
    print(f"  * Total windows: {len(windows):,}")
    print(f"  * Shape: {windows.shape}")
    print(f"  * Classes: 0=Background, 1=Artifact, 2=Seizure")
    print(f"\n  Class distribution:")
    for i in range(3):
        class_name = ['Background', 'Artifact', 'Seizure'][i]
        count = np.sum(labels == i)
        print(f"    - Class {i} ({class_name}): {count:,} ({100*count/len(labels):.2f}%)")
    
    # ========================================================================
    # TRAIN/VAL SPLIT
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*70)
    
    print("Splitting 80% train / 20% validation (from train data)...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        windows, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    
    print(f"\n[OK] Split Complete:")
    print(f"  * Train: {len(X_train):,} samples")
    for i in range(3):
        count = np.sum(y_train == i)
        print(f"    - Class {i}: {count:,} ({100*count/len(y_train):.2f}%)")
    
    print(f"  * Val: {len(X_val):,} samples")
    for i in range(3):
        count = np.sum(y_val == i)
        print(f"    - Class {i}: {count:,} ({100*count/len(y_val):.2f}%)")
    
    print(f"\n  [CRITICAL] This is validation from TRAIN split")
    print(f"  [CRITICAL] DEV split reserved for final testing!")
    
    # ========================================================================
    # CALCULATE CLASS WEIGHTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("CALCULATING CLASS WEIGHTS")
    print("="*70)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print("Class weights (for handling imbalance):")
    for i in range(3):
        class_name = ['Background', 'Artifact', 'Seizure'][i]
        print(f"  * Class {i} ({class_name}): {class_weights[i]:.3f}")
    
    # ========================================================================
    # CREATE DATALOADERS
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING DATALOADERS")
    print("="*70)
    
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"[OK] DataLoaders created:")
    print(f"  * Train batches: {len(train_loader)}")
    print(f"  * Val batches: {len(val_loader)}")
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model = ThreeClassCNNLSTM(num_channels=22, num_classes=3).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("[OK] Model created:")
    print(f"  * Architecture: CNN-LSTM")
    print(f"  * Input: 22 channels × 2000 samples")
    print(f"  * Output: 3 classes (softmax)")
    print(f"  * Total parameters: {total_params:,}")
    print(f"  * Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # SETUP TRAINING
    # ========================================================================
    
    print("\n" + "="*70)
    print("SETUP TRAINING")
    print("="*70)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"[OK] Training setup:")
    print(f"  * Loss: Weighted CrossEntropyLoss")
    print(f"  * Optimizer: Adam (lr={args.lr})")
    print(f"  * Scheduler: ReduceLROnPlateau")
    print(f"  * Epochs: {args.epochs}")
    print(f"  * Batch size: {args.batch_size}")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("(This will take 45-60 minutes)\n")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Calculate per-class metrics
        per_class = calculate_per_class_metrics(val_preds, val_labels)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  Per-class F1:")
        for class_name, metrics in per_class.items():
            print(f"    {class_name}: {metrics['f1']:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            save_path = os.path.join(args.save_dir, 'best_3class_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'per_class_metrics': per_class
            }, save_path)
            
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        
        print()
    
    total_time = time.time() - start_time
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    print(f"\n[TRAINING SUMMARY]:")
    print(f"  * Total time: {total_time/60:.1f} minutes")
    print(f"  * Best epoch: {best_epoch}")
    print(f"  * Best val loss: {best_val_loss:.4f}")
    print(f"  * Best val accuracy: {best_val_acc:.2f}%")
    
    print(f"\n[MODEL SAVED]:")
    print(f"  * Path: {save_path}")
    print(f"  * Size: {os.path.getsize(save_path) / 1e6:.1f} MB")
    
    print(f"\n[NEXT STEP: EVALUATE ON DEV SET]")
    print(f"\nRun evaluation (ONCE!):")
    print(f"  python 3_evaluate_3class.py --model_path {save_path} --test_data ../preprocessed/dev.npz --save_dir ./results")
    
    print(f"\n[CRITICAL REMINDER]:")
    print(f"  * This was trained on TRAIN split only")
    print(f"  * Now test on DEV split (53 different patients)")
    print(f"  * Run evaluation ONCE - no iteration!")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
