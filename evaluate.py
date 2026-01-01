"""
EEG Seizure Detection - Evaluation & Inference
Ceribell Project

Features:
    - Load trained model and evaluate on test set
    - Generate comprehensive metrics report
    - Plot confusion matrix and ROC curve
    - Analyze per-window predictions
    - Export predictions for clinical review

Author: Ceribell Seizure Detector Project
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from typing import Dict, Tuple
import json

from modules.model import create_seizure_detector
from modules.dataset import load_preprocessed_data, EEGSeizureDataset
from torch.utils.data import DataLoader


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: np.ndarray) -> Dict:
    """
    Calculate detailed evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for seizure class
    
    Returns:
        Dictionary of metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Clinical metrics
    sensitivity = recall  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Predictive values
    ppv = precision  # Positive predictive value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    # ROC-AUC
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0.0
    
    # Precision-Recall AUC
    if len(np.unique(y_true)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }
    
    return metrics


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model: torch.nn.Module, data_loader: DataLoader,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device (cpu/cuda)
    
    Returns:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for seizure class
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Forward pass
            logits, _ = model(data)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Seizure class probability
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background', 'Seizure'],
                yticklabels=['Background', 'Seizure'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {save_path}")
    
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray,
                                save_path: str = None):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (PPV)', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved to: {save_path}")
    
    plt.close()


def plot_prediction_distribution(y_true: np.ndarray, y_prob: np.ndarray,
                                 save_path: str = None):
    """
    Plot distribution of prediction probabilities.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Background samples
    bg_probs = y_prob[y_true == 0]
    axes[0].hist(bg_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    axes[0].set_xlabel('Predicted Seizure Probability', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Background Windows', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Seizure samples
    sz_probs = y_prob[y_true == 1]
    axes[1].hist(sz_probs, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    axes[1].set_xlabel('Predicted Seizure Probability', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Seizure Windows', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction distribution saved to: {save_path}")
    
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_evaluation_report(metrics: Dict, save_path: str = None):
    """
    Generate formatted evaluation report.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save report (text file)
    """
    report = []
    report.append("=" * 70)
    report.append("SEIZURE DETECTION MODEL - EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Performance metrics
    report.append("PERFORMANCE METRICS")
    report.append("-" * 70)
    report.append(f"  Accuracy:        {metrics['accuracy']:.4f}")
    report.append(f"  F1 Score:        {metrics['f1']:.4f}")
    report.append(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    report.append(f"  PR-AUC:          {metrics['pr_auc']:.4f}")
    report.append("")
    
    # Clinical metrics
    report.append("CLINICAL METRICS")
    report.append("-" * 70)
    report.append(f"  Sensitivity:     {metrics['sensitivity']:.4f}  (True Positive Rate)")
    report.append(f"  Specificity:     {metrics['specificity']:.4f}  (True Negative Rate)")
    report.append(f"  Precision (PPV): {metrics['precision']:.4f}  (Positive Predictive Value)")
    report.append(f"  NPV:             {metrics['npv']:.4f}  (Negative Predictive Value)")
    report.append("")
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    report.append("CONFUSION MATRIX")
    report.append("-" * 70)
    report.append(f"  True Negatives:  {cm['tn']:>6d}")
    report.append(f"  False Positives: {cm['fp']:>6d}")
    report.append(f"  False Negatives: {cm['fn']:>6d}")
    report.append(f"  True Positives:  {cm['tp']:>6d}")
    report.append("")
    
    # Clinical interpretation
    report.append("CLINICAL INTERPRETATION")
    report.append("-" * 70)
    total = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
    report.append(f"  Total windows analyzed: {total}")
    report.append(f"  Correctly classified:   {cm['tp'] + cm['tn']} ({100*metrics['accuracy']:.1f}%)")
    report.append(f"  ")
    report.append(f"  Of {cm['tp'] + cm['fn']} actual seizure windows:")
    report.append(f"    - Correctly detected:  {cm['tp']} ({100*metrics['sensitivity']:.1f}%)")
    report.append(f"    - Missed:              {cm['fn']} ({100*cm['fn']/(cm['tp']+cm['fn']):.1f}%)")
    report.append(f"  ")
    report.append(f"  Of {cm['tn'] + cm['fp']} actual background windows:")
    report.append(f"    - Correctly rejected:  {cm['tn']} ({100*metrics['specificity']:.1f}%)")
    report.append(f"    - False alarms:        {cm['fp']} ({100*cm['fp']/(cm['tn']+cm['fp']):.1f}%)")
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Report saved to: {save_path}")
    
    return report_text


# ============================================================================
# MAIN - COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate seizure detection model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data (.npz)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Create model
    config = checkpoint.get('config', {})
    model_size = config.get('model_size', 'medium')
    model = create_seizure_detector(model_size=model_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint.get('epoch', '?')} epochs)")
    
    # Load test data
    print(f"\nLoading test data: {args.data_path}")
    windows, labels, _ = load_preprocessed_data(args.data_path)
    
    # Create dataset and dataloader
    test_dataset = EEGSeizureDataset(windows, labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_detailed_metrics(y_true, y_pred, y_prob)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    generate_evaluation_report(metrics, save_path=report_path)
    
    # Save metrics as JSON
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    if len(np.unique(y_true)) > 1:
        plot_roc_curve(y_true, y_prob, save_path=os.path.join(args.output_dir, 'roc_curve.png'))
        plot_precision_recall_curve(y_true, y_prob, save_path=os.path.join(args.output_dir, 'pr_curve.png'))
    
    plot_prediction_distribution(y_true, y_prob, save_path=os.path.join(args.output_dir, 'prediction_distribution.png'))
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.npz')
    np.savez(predictions_path,
             y_true=y_true,
             y_pred=y_pred,
             y_prob=y_prob)
    print(f"✓ Predictions saved to: {predictions_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()