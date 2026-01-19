"""
EEG Artifact Detection - Evaluation Script
Ceribell Project

Comprehensive evaluation of trained artifact detector:
- ROC-AUC, PR-AUC, F1, Precision, Recall
- Confusion matrix
- Per-artifact-type performance (if metadata available)
- Threshold analysis

Usage:
    python evaluate_artifact_detector.py --model_path ./models/best_artifact_detector.pth --data_path ./preprocessed/artifacts_01_tcp_ar.npz

Author: Ceribell Seizure Detector Project
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import model
sys.path.append('../data_viz')
from model import SeizureDetectorCNNLSTM

# Import dataset
sys.path.append('../training')
try:
    from modules.dataset import EEGSeizureDataset
except:
    from torch.utils.data import Dataset
    
    class EEGSeizureDataset(Dataset):
        def __init__(self, windows, labels, augment=False):
            self.windows = windows.astype(np.float32)
            self.labels = labels.astype(np.int64)
        
        def __len__(self):
            return len(self.windows)
        
        def __getitem__(self, idx):
            window = torch.from_numpy(self.windows[idx]).float()
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return window, label


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions + labels."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for windows, labels in dataloader:
            windows = windows.to(device)
            
            logits, _ = model(windows)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of artifact
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Artifact Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ ROC curve saved to: {save_path}")
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(labels, probs, save_path):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Artifact Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ PR curve saved to: {save_path}")
    plt.close()
    
    return pr_auc


def plot_confusion_matrix(labels, preds, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Clean', 'Artifact'],
                yticklabels=['Clean', 'Artifact'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Artifact Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()
    
    return cm


def plot_threshold_analysis(labels, probs, save_path):
    """Plot performance metrics vs threshold."""
    thresholds = np.linspace(0, 1, 100)
    
    f1_scores = []
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        f1_scores.append(f1_score(labels, preds, zero_division=0))
        precisions.append(precision_score(labels, preds, zero_division=0))
        recalls.append(recall_score(labels, preds, zero_division=0))
    
    # Find optimal threshold (max F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Threshold analysis saved to: {save_path}")
    plt.close()
    
    return optimal_threshold, f1_scores[optimal_idx]


def print_evaluation_summary(labels, preds, probs, roc_auc, pr_auc, optimal_threshold):
    """Print comprehensive evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # Overall metrics
    print(f"\n📊 Overall Performance:")
    print(f"  • ROC-AUC: {roc_auc:.4f}")
    print(f"  • PR-AUC: {pr_auc:.4f}")
    print(f"  • F1 Score: {f1_score(labels, preds):.4f}")
    print(f"  • Precision: {precision_score(labels, preds):.4f}")
    print(f"  • Recall: {recall_score(labels, preds):.4f}")
    print(f"  • Optimal Threshold: {optimal_threshold:.4f}")
    
    # Confusion matrix stats
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n🎯 Detailed Metrics:")
    print(f"  • True Positives (Artifact detected): {tp:,}")
    print(f"  • True Negatives (Clean detected): {tn:,}")
    print(f"  • False Positives (Clean → Artifact): {fp:,}")
    print(f"  • False Negatives (Artifact → Clean): {fn:,}")
    print(f"  • Specificity: {specificity:.4f}")
    print(f"  • Negative Predictive Value: {npv:.4f}")
    
    # Clinical interpretation
    print(f"\n🏥 Clinical Interpretation:")
    sensitivity = recall_score(labels, preds)
    ppv = precision_score(labels, preds)
    
    print(f"  • Sensitivity (Artifact catch rate): {sensitivity:.2%}")
    print(f"  • Specificity (Clean pass rate): {specificity:.2%}")
    print(f"  • PPV (Artifact flag accuracy): {ppv:.2%}")
    print(f"  • NPV (Clean flag accuracy): {npv:.2%}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate artifact detector')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model .pth file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed .npz file')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='./evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EEG ARTIFACT DETECTION - EVALUATION")
    print("Ceribell Project")
    print("="*70)
    
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
    
    # Create dataset and dataloader (num_workers=0 for Windows compatibility)
    dataset = EEGSeizureDataset(windows, labels, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Create model (need to infer architecture from checkpoint)
    model = SeizureDetectorCNNLSTM(
        num_channels=22,
        num_samples=2000,
        num_classes=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Model loaded from: {args.model_path}")
    print(f"  • Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  • Val AUC: {checkpoint.get('val_auc', 'unknown')}")
    
    # =========================================================================
    # EVALUATE
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("EVALUATING MODEL")
    print(f"{'='*70}\n")
    
    preds, true_labels, probs = evaluate_model(model, dataloader, device)
    
    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("GENERATING EVALUATION PLOTS")
    print(f"{'='*70}\n")
    
    # ROC curve
    roc_path = os.path.join(args.save_dir, 'roc_curve.png')
    roc_auc = plot_roc_curve(true_labels, probs, roc_path)
    
    # PR curve
    pr_path = os.path.join(args.save_dir, 'pr_curve.png')
    pr_auc = plot_precision_recall_curve(true_labels, probs, pr_path)
    
    # Confusion matrix
    cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    cm = plot_confusion_matrix(true_labels, preds, cm_path)
    
    # Threshold analysis
    threshold_path = os.path.join(args.save_dir, 'threshold_analysis.png')
    optimal_threshold, max_f1 = plot_threshold_analysis(true_labels, probs, threshold_path)
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    
    print_evaluation_summary(true_labels, preds, probs, roc_auc, pr_auc, optimal_threshold)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    results = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_score': f1_score(true_labels, preds),
        'precision': precision_score(true_labels, preds),
        'recall': recall_score(true_labels, preds),
        'optimal_threshold': optimal_threshold,
        'max_f1': max_f1,
        'confusion_matrix': cm,
        'predictions': preds,
        'probabilities': probs,
        'true_labels': true_labels
    }
    
    results_path = os.path.join(args.save_dir, 'evaluation_results.npz')
    np.savez(results_path, **results)
    
    print(f"📁 Evaluation files:")
    print(f"  • ROC curve: {roc_path}")
    print(f"  • PR curve: {pr_path}")
    print(f"  • Confusion matrix: {cm_path}")
    print(f"  • Threshold analysis: {threshold_path}")
    print(f"  • Results data: {results_path}")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Build unified pipeline: python unified_pipeline.py")
    print(f"  2. Create visualizations for artifact examples")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
