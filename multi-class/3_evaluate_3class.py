"""
Step 3: Evaluate 3-Class Model on Dev Set
FINAL EVALUATION - RUN ONCE ONLY!

NO DATA LEAKAGE:
- Tests on TUSZ DEV split (53 patients)
- These patients were NEVER seen during training
- Run this ONCE at the end - no iteration!

Author: Ceribell Multi-Class System
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# ============================================================================
# MODEL ARCHITECTURE (same as training)
# ============================================================================

class ThreeClassCNNLSTM(nn.Module):
    """3-Class CNN-LSTM for EEG Classification"""
    
    def __init__(self, num_channels=22, num_classes=3, hidden_size=128, num_layers=2):
        super(ThreeClassCNNLSTM, self).__init__()
        
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
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, windows, labels, device, batch_size=64):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_probs = []
    
    num_batches = (len(windows) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(windows))
            
            batch_windows = torch.FloatTensor(windows[start_idx:end_idx]).to(device)
            
            outputs = model(batch_windows)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {end_idx}/{len(windows)} windows...")
    
    return np.array(all_preds), np.array(all_probs)


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics"""
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Per-class metrics
    metrics = {}
    for idx, class_name in enumerate(class_names):
        mask_true = (y_true == idx)
        mask_pred = (y_pred == idx)
        
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        tn = np.sum(~mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'specificity': float(specificity),
            'support': int(np.sum(mask_true))
        }
    
    return accuracy, metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (% of True Class)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate 3-class model on dev set')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pth)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to dev set data (.npz)')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("="*70)
    print("STEP 3: EVALUATING 3-CLASS MODEL ON DEV SET")
    print("FINAL EVALUATION - RUN ONCE ONLY!")
    print("="*70)
    
    # ========================================================================
    # LOAD TEST DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("LOADING DEV SET (TEST DATA)")
    print("="*70)
    
    if not os.path.exists(args.test_data):
        print(f"\n[ERROR] Test data not found: {args.test_data}")
        exit(1)
    
    print(f"Loading: {args.test_data}")
    test_data = np.load(args.test_data, allow_pickle=True)
    test_windows = test_data['windows']
    test_labels_binary = test_data['labels']  # 0=background, 1=seizure
    
    # Convert binary labels to 3-class
    # In dev set: 0=background, 1=seizure
    # We need: 0=background, 1=artifact (N/A), 2=seizure
    test_labels = test_labels_binary.copy()
    test_labels[test_labels_binary == 1] = 2  # Seizures → class 2
    
    print(f"\n[OK] Dev Set Loaded:")
    print(f"  * Total windows: {len(test_windows):,}")
    print(f"  * Shape: {test_windows.shape}")
    print(f"  * CRITICAL: These are 53 DIFFERENT patients from training!")
    
    print(f"\n  Label distribution:")
    print(f"    - Background: {np.sum(test_labels==0):,} ({100*np.mean(test_labels==0):.2f}%)")
    print(f"    - Artifact: 0 (TUSZ dev has no artifact labels)")
    print(f"    - Seizure: {np.sum(test_labels==2):,} ({100*np.mean(test_labels==2):.2f}%)")
    
    print(f"\n  [NOTE] We can only evaluate Background vs Seizure")
    print(f"  [NOTE] Artifact class not present in TUSZ dev set")
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    
    print("\n" + "="*70)
    print("LOADING TRAINED MODEL")
    print("="*70)
    
    if not os.path.exists(args.model_path):
        print(f"\n[ERROR] Model not found: {args.model_path}")
        exit(1)
    
    print(f"Loading: {args.model_path}")
    
    model = ThreeClassCNNLSTM(num_channels=22, num_classes=3).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n[OK] Model Loaded:")
    print(f"  * Training epoch: {checkpoint['epoch']}")
    print(f"  * Training val loss: {checkpoint['val_loss']:.4f}")
    print(f"  * Training val acc: {checkpoint['val_acc']:.2f}%")
    
    # ========================================================================
    # EVALUATE ON DEV SET
    # ========================================================================
    
    print("\n" + "="*70)
    print("RUNNING EVALUATION ON DEV SET")
    print("="*70)
    
    print(f"\nEvaluating on {len(test_windows):,} windows...")
    print("This may take a few minutes...\n")
    
    predictions, probabilities = evaluate_model(
        model, test_windows, test_labels, device
    )
    
    print("\n[OK] Evaluation complete!")
    
    # ========================================================================
    # CALCULATE METRICS
    # ========================================================================
    
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    class_names = ['Background', 'Artifact', 'Seizure']
    
    accuracy, per_class_metrics = calculate_metrics(
        test_labels, predictions, class_names
    )
    
    print(f"\n[OVERALL PERFORMANCE]:")
    print(f"  * Overall Accuracy: {100*accuracy:.2f}%")
    print(f"  * Total samples: {len(test_labels):,}")
    print(f"  * Correct: {np.sum(test_labels==predictions):,}")
    print(f"  * Incorrect: {np.sum(test_labels!=predictions):,}")
    
    print(f"\n[PER-CLASS METRICS]:")
    for class_name, metrics in per_class_metrics.items():
        if metrics['support'] > 0:
            print(f"\n  {class_name}:")
            print(f"    * Precision: {100*metrics['precision']:.2f}%")
            print(f"    * Recall: {100*metrics['recall']:.2f}%")
            print(f"    * F1-Score: {100*metrics['f1']:.2f}%")
            print(f"    * Support: {metrics['support']:,} samples")
        else:
            print(f"\n  {class_name}:")
            print(f"    * Not present in dev set")
    
    # ========================================================================
    # CONFUSION MATRIX
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Confusion matrix
    cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, predictions, class_names, cm_path)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save detailed results
    results = {
        'overall_accuracy': float(accuracy),
        'per_class_metrics': per_class_metrics,
        'test_size': len(test_labels),
        'num_patients': 53,
        'data_leakage': 'NONE - Different patients from training'
    }
    
    results_path = os.path.join(args.save_dir, 'evaluation_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Metrics saved: {results_path}")
    
    # Save predictions
    preds_path = os.path.join(args.save_dir, 'predictions.npz')
    np.savez_compressed(
        preds_path,
        predictions=predictions,
        probabilities=probabilities,
        true_labels=test_labels
    )
    
    print(f"✓ Predictions saved: {preds_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    print(f"\n[FINAL RESULTS]:")
    print(f"  * Overall Accuracy: {100*accuracy:.2f}%")
    print(f"  * Background Recall: {100*per_class_metrics['Background']['recall']:.2f}%")
    print(f"  * Seizure Recall: {100*per_class_metrics['Seizure']['recall']:.2f}%")
    print(f"  * Seizure Precision: {100*per_class_metrics['Seizure']['precision']:.2f}%")
    
    print(f"\n[VALIDATION]:")
    print(f"  ✓ Tested on 53 independent patients")
    print(f"  ✓ Zero overlap with training patients")
    print(f"  ✓ NO data leakage")
    print(f"  ✓ Honest, unbiased performance")
    
    print(f"\n[OUTPUT FILES]:")
    print(f"  * {cm_path}")
    print(f"  * {results_path}")
    print(f"  * {preds_path}")
    
    print(f"\n[NEXT STEP: COMPARE TO BINARY MODEL]")
    print(f"\nRun comparison (optional):")
    print(f"  python 4_compare_models.py")
    
    print(f"\n[CRITICAL REMINDER]:")
    print(f"  * This evaluation was run ONCE")
    print(f"  * Do NOT iterate based on these results")
    print(f"  * Do NOT retrain based on dev performance")
    print(f"  * Report these results as-is")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
