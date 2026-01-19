"""
Argus Forecasting Model Evaluation
====================================
Evaluate trained forecasting model on test set.
Generate comprehensive metrics and visualizations.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from forecasting_model import ForecastingOnlyModel
from train_forecasting import ForecastingDataset


def evaluate_model(model, dataloader, device):
    """Comprehensive model evaluation."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            
            # Forward pass
            outputs = model(data).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    return all_labels, all_probs, all_preds


def compute_metrics(labels, probs, preds):
    """Compute all evaluation metrics."""
    # ROC-AUC
    auc = roc_auc_score(labels, probs)
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # F1 score
    precision = ppv
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average Precision
    ap = average_precision_score(labels, probs)
    
    metrics = {
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'ap': ap,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return metrics


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Inter-ictal', 'Pre-ictal'],
                yticklabels=['Inter-ictal', 'Pre-ictal'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Seizure Forecasting')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve with optimal threshold."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    # Find optimal threshold (Youden's index)
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]
    optimal_tpr = tpr[youden_idx]
    optimal_fpr = fpr[youden_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.scatter(optimal_fpr, optimal_tpr, color='red', s=100, zorder=5,
                label=f'Optimal (θ={optimal_threshold:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Seizure Forecasting')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return optimal_threshold


def plot_precision_recall_curve(labels, probs, save_path):
    """Plot precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR (AP = {ap:.4f})')
    plt.axhline(y=np.mean(labels), color='k', linestyle='--', 
                alpha=0.3, label='Baseline')
    
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve - Seizure Forecasting')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_distribution(labels, probs, save_path):
    """Plot distribution of predictions by class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by class
    inter_ictal_probs = probs[labels == 0]
    pre_ictal_probs = probs[labels == 1]
    
    # Plot histograms
    ax.hist(inter_ictal_probs, bins=50, alpha=0.6, 
            label=f'Inter-ictal (n={len(inter_ictal_probs)})', color='blue')
    ax.hist(pre_ictal_probs, bins=50, alpha=0.6,
            label=f'Pre-ictal (n={len(pre_ictal_probs)})', color='red')
    
    ax.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, 
               label='Decision Threshold')
    
    ax.set_xlabel('Predicted Probability (Pre-ictal)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Model Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(metrics, save_path):
    """Generate text report of metrics."""
    report = []
    report.append("=" * 60)
    report.append("ARGUS SEIZURE FORECASTING - EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("Task: Predict seizures 30 minutes in advance")
    report.append("Target: ROC-AUC ≥ 70%")
    report.append("")
    report.append("-" * 60)
    report.append("PRIMARY METRICS")
    report.append("-" * 60)
    report.append(f"ROC-AUC:           {metrics['auc']:.4f}")
    report.append(f"Average Precision: {metrics['ap']:.4f}")
    report.append(f"Accuracy:          {metrics['accuracy']:.4f}")
    report.append("")
    report.append("-" * 60)
    report.append("CLINICAL METRICS")
    report.append("-" * 60)
    report.append(f"Sensitivity (Recall):  {metrics['sensitivity']:.4f}")
    report.append(f"  → Detects {metrics['sensitivity']*100:.1f}% of pre-ictal periods")
    report.append(f"  → Misses {(1-metrics['sensitivity'])*100:.1f}% (False Negatives)")
    report.append("")
    report.append(f"Specificity:           {metrics['specificity']:.4f}")
    report.append(f"  → Correctly identifies {metrics['specificity']*100:.1f}% of inter-ictal")
    report.append(f"  → False alarm rate: {(1-metrics['specificity'])*100:.1f}%")
    report.append("")
    report.append(f"PPV (Precision):       {metrics['ppv']:.4f}")
    report.append(f"  → {metrics['ppv']*100:.1f}% of warnings are correct")
    report.append(f"  → 1 correct warning per {1/metrics['ppv']:.1f} alarms")
    report.append("")
    report.append(f"NPV:                   {metrics['npv']:.4f}")
    report.append(f"  → {metrics['npv']*100:.1f}% confidence when predicting no seizure")
    report.append("")
    report.append(f"F1 Score:              {metrics['f1']:.4f}")
    report.append("")
    report.append("-" * 60)
    report.append("CONFUSION MATRIX")
    report.append("-" * 60)
    report.append(f"True Positives (TP):   {metrics['tp']:,}")
    report.append(f"True Negatives (TN):   {metrics['tn']:,}")
    report.append(f"False Positives (FP):  {metrics['fp']:,}")
    report.append(f"False Negatives (FN):  {metrics['fn']:,}")
    report.append("")
    report.append("-" * 60)
    report.append("PERFORMANCE ASSESSMENT")
    report.append("-" * 60)
    
    if metrics['auc'] >= 0.80:
        report.append("🚀 EXCELLENT: Exceeded stretch goal (AUC ≥ 80%)")
    elif metrics['auc'] >= 0.70:
        report.append("✓ GOOD: Target achieved (AUC ≥ 70%)")
    elif metrics['auc'] >= 0.60:
        report.append("⚠ FAIR: Below target but shows promise (60-70%)")
    else:
        report.append("✗ NEEDS IMPROVEMENT: Below acceptable threshold (<60%)")
    
    report.append("")
    report.append("Clinical Utility:")
    if metrics['ppv'] >= 0.30 and metrics['sensitivity'] >= 0.60:
        report.append("✓ Clinically useful: Good balance of warnings and detection")
    elif metrics['ppv'] < 0.20:
        report.append("⚠ High false alarm rate may reduce clinical utility")
    elif metrics['sensitivity'] < 0.50:
        report.append("⚠ Missing too many pre-ictal periods")
    
    report.append("")
    report.append("=" * 60)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    # Print to console
    print('\n'.join(report))


def main():
    """Main evaluation pipeline."""
    
    # CHANGE THESE PATHS TO MATCH YOUR SETUP
    config = {
        'model_path': '/mnt/c/Users/0218s/Desktop/Argus/models/forecasting/best_forecasting_model.pth',
        'test_data': '/mnt/c/Users/0218s/Desktop/Argus/data/tusz_forecasting/eval_forecasting_balanced.pkl',
        'output_dir': '/mnt/c/Users/0218s/Desktop/Argus/models/forecasting/evaluation',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("=" * 60)
    print("ARGUS FORECASTING MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(config['model_path'], map_location=config['device'])
    
    model = ForecastingOnlyModel(
        n_channels=32,
        sequence_length=2000,
        hidden_size=128,
        num_lstm_layers=2,
        dropout=0.4,
        use_attention=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config['device'])
    model.eval()
    
    print("✓ Model loaded")
    
    # Load test data
    print("\nLoading test data...")
    test_dataset = ForecastingDataset(config['test_data'], normalize=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    print("✓ Data loaded")
    
    # Evaluate
    print("\nEvaluating model...")
    labels, probs, preds = evaluate_model(model, test_loader, config['device'])
    print("✓ Evaluation complete")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(labels, probs, preds)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(
        cm,
        os.path.join(config['output_dir'], 'confusion_matrix.png')
    )
    
    optimal_threshold = plot_roc_curve(
        labels, probs,
        os.path.join(config['output_dir'], 'roc_curve.png')
    )
    
    plot_precision_recall_curve(
        labels, probs,
        os.path.join(config['output_dir'], 'precision_recall_curve.png')
    )
    
    plot_prediction_distribution(
        labels, probs,
        os.path.join(config['output_dir'], 'prediction_distribution.png')
    )
    
    print("✓ Visualizations saved")
    
    # Generate report
    print("\nGenerating report...")
    generate_report(
        metrics,
        os.path.join(config['output_dir'], 'evaluation_report.txt')
    )
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"\n✓ Evaluation complete!")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()