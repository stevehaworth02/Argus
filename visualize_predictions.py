"""
EEG Seizure Detection - Prediction Visualizations
Ceribell Project

Creates compelling visualizations of:
- EEG epochs with seizure/non-seizure predictions
- Model attention heatmaps
- Confidence scores over time
- Comparison of true vs predicted labels

Author: Ceribell Seizure Detector Project
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Tuple, List

from modules.model import create_seizure_detector
from modules.dataset import load_preprocessed_data, EEGSeizureDataset
from torch.utils.data import DataLoader

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def load_model_and_data(checkpoint_path: str, data_path: str, device: torch.device):
    """Load trained model and data."""
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_size = config.get('model_size', 'medium')
    
    model = create_seizure_detector(model_size=model_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data
    windows, labels, metadata = load_preprocessed_data(data_path)
    
    return model, windows, labels, metadata


def get_predictions_with_attention(model: torch.nn.Module, windows: np.ndarray,
                                   device: torch.device, batch_size: int = 32):
    """Get predictions and attention weights for all windows."""
    
    model.eval()
    all_probs = []
    all_attentions = []
    
    dataset = EEGSeizureDataset(windows, np.zeros(len(windows)), augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_windows, _ in loader:
            batch_windows = batch_windows.to(device)
            
            logits, attention = model(batch_windows)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Seizure probability
            
            all_probs.extend(probs.cpu().numpy())
            if attention is not None:
                all_attentions.extend(attention.cpu().numpy())
    
    return np.array(all_probs), np.array(all_attentions) if all_attentions else None


def plot_eeg_with_prediction(window: np.ndarray, true_label: int, pred_prob: float,
                             attention: np.ndarray = None, save_path: str = None):
    """
    Plot a single EEG window with prediction overlay.
    
    Args:
        window: EEG data (channels, samples)
        true_label: Ground truth label (0=background, 1=seizure)
        pred_prob: Predicted seizure probability
        attention: Attention weights (optional)
        save_path: Path to save figure
    """
    pred_label = int(pred_prob > 0.5)
    
    # Create figure
    if attention is not None:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 1, height_ratios=[3, 1, 0.3], hspace=0.3)
        ax_eeg = fig.add_subplot(gs[0])
        ax_attn = fig.add_subplot(gs[1], sharex=ax_eeg)
        ax_info = fig.add_subplot(gs[2])
        ax_info.axis('off')
    else:
        fig, ax_eeg = plt.subplots(figsize=(16, 8))
    
    # Plot EEG channels
    num_channels = min(22, window.shape[0])
    time = np.arange(window.shape[1]) / 200  # 200 Hz sampling rate
    
    for i in range(num_channels):
        # Offset channels for visibility
        offset = i * 8
        color = 'red' if true_label == 1 else 'blue'
        alpha = 0.8 if i < 10 else 0.5  # Emphasize first 10 channels
        
        ax_eeg.plot(time, window[i] + offset, color=color, alpha=alpha, linewidth=0.5)
    
    # Add title with prediction info
    correct = "✓" if (pred_label == true_label) else "✗"
    title = f"{correct} True: {'SEIZURE' if true_label else 'BACKGROUND'} | "
    title += f"Predicted: {'SEIZURE' if pred_label else 'BACKGROUND'} ({pred_prob:.2%} confidence)"
    
    ax_eeg.set_title(title, fontsize=16, fontweight='bold',
                    color='green' if pred_label == true_label else 'red')
    ax_eeg.set_ylabel('Channels (offset)', fontsize=12)
    ax_eeg.set_xlim([0, 10])
    ax_eeg.grid(True, alpha=0.3)
    
    # Add prediction confidence bar
    ax_eeg.axhline(y=-10, xmin=0, xmax=pred_prob, color='red', linewidth=10, alpha=0.7)
    ax_eeg.text(5, -12, f'Seizure Confidence: {pred_prob:.1%}', ha='center', fontsize=10)
    
    # Plot attention if available
    if attention is not None:
        # Resample attention to match time axis
        time_attention = np.linspace(0, 10, len(attention))
        
        ax_attn.fill_between(time_attention, 0, attention, color='orange', alpha=0.6)
        ax_attn.plot(time_attention, attention, color='darkorange', linewidth=2)
        ax_attn.set_ylabel('Attention\nWeight', fontsize=12)
        ax_attn.set_ylim([0, attention.max() * 1.1])
        ax_attn.grid(True, alpha=0.3)
        ax_attn.set_title('Model Attention (where the model is focusing)', fontsize=12)
        
        # Highlight max attention region
        max_idx = np.argmax(attention)
        ax_attn.axvline(time_attention[max_idx], color='red', linestyle='--', alpha=0.7)
        ax_eeg.axvline(time_attention[max_idx], color='red', linestyle='--', alpha=0.3)
    
    ax_eeg.set_xlabel('Time (seconds)', fontsize=12) if attention is None else None
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_example_predictions(model: torch.nn.Module, windows: np.ndarray,
                             labels: np.ndarray, device: torch.device,
                             output_dir: str, num_examples: int = 6):
    """Plot multiple example predictions."""
    
    print(f"\nGenerating {num_examples} example predictions...")
    
    # Get predictions
    probs, attentions = get_predictions_with_attention(model, windows, device)
    
    # Find interesting examples
    predictions = (probs > 0.5).astype(int)
    
    # Categories
    true_pos = np.where((labels == 1) & (predictions == 1))[0]  # Correct seizure detection
    true_neg = np.where((labels == 0) & (predictions == 0))[0]  # Correct background
    false_pos = np.where((labels == 0) & (predictions == 1))[0]  # False alarm
    false_neg = np.where((labels == 1) & (predictions == 0))[0]  # Missed seizure
    
    examples = []
    
    # Select diverse examples
    if len(true_pos) > 0:
        examples.append(('true_positive', np.random.choice(true_pos)))
    if len(true_neg) > 0:
        examples.append(('true_negative', np.random.choice(true_neg)))
    if len(false_pos) > 0:
        examples.append(('false_positive', np.random.choice(false_pos)))
    if len(false_neg) > 0:
        examples.append(('false_negative', np.random.choice(false_neg)))
    
    # Add more confident predictions
    if len(true_pos) > 0:
        high_conf_seizure = true_pos[np.argsort(probs[true_pos])[-1]]
        examples.append(('high_conf_seizure', high_conf_seizure))
    if len(true_neg) > 0:
        high_conf_background = true_neg[np.argsort(probs[true_neg])[0]]
        examples.append(('high_conf_background', high_conf_background))
    
    # Plot each example
    for name, idx in examples[:num_examples]:
        save_path = os.path.join(output_dir, f'prediction_{name}.png')
        
        attention = attentions[idx] if attentions is not None else None
        
        plot_eeg_with_prediction(
            windows[idx],
            labels[idx],
            probs[idx],
            attention,
            save_path
        )
        
        print(f"  ✓ Saved: {save_path}")


def plot_temporal_predictions(model: torch.nn.Module, windows: np.ndarray,
                              labels: np.ndarray, device: torch.device,
                              output_dir: str, num_windows: int = 100):
    """
    Plot predictions over time to show temporal patterns.
    """
    
    print(f"\nGenerating temporal prediction visualization...")
    
    # Get predictions for subset
    subset_size = min(num_windows, len(windows))
    indices = np.random.choice(len(windows), subset_size, replace=False)
    indices.sort()
    
    subset_windows = windows[indices]
    subset_labels = labels[indices]
    
    probs, _ = get_predictions_with_attention(model, subset_windows, device)
    predictions = (probs > 0.5).astype(int)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    
    time_axis = np.arange(len(subset_labels)) * 10  # 10 seconds per window
    
    # Plot 1: True labels
    axes[0].fill_between(time_axis, 0, subset_labels, alpha=0.5, color='red', label='Seizure', step='post')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_ylim([-0.1, 1.1])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Background', 'Seizure'])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Ground Truth Labels', fontsize=14, fontweight='bold')
    
    # Plot 2: Predictions
    axes[1].fill_between(time_axis, 0, predictions, alpha=0.5, color='blue', label='Predicted Seizure', step='post')
    axes[1].set_ylabel('Prediction', fontsize=12)
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Background', 'Seizure'])
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Model Predictions', fontsize=14, fontweight='bold')
    
    # Plot 3: Confidence scores
    axes[2].plot(time_axis, probs, color='purple', linewidth=2, label='Seizure Probability')
    axes[2].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
    axes[2].fill_between(time_axis, 0, probs, where=(probs > 0.5), alpha=0.3, color='red')
    axes[2].fill_between(time_axis, 0, probs, where=(probs <= 0.5), alpha=0.3, color='blue')
    axes[2].set_ylabel('Probability', fontsize=12)
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    axes[2].set_title('Seizure Probability Over Time', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Temporal Predictions ({subset_size} windows, {subset_size*10/60:.1f} minutes)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'temporal_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def plot_confidence_analysis(model: torch.nn.Module, windows: np.ndarray,
                             labels: np.ndarray, device: torch.device,
                             output_dir: str):
    """Analyze prediction confidence for different classes."""
    
    print(f"\nGenerating confidence analysis...")
    
    probs, _ = get_predictions_with_attention(model, windows, device)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Confidence distributions
    axes[0, 0].hist(probs[labels == 0], bins=50, alpha=0.7, color='blue', 
                   label='Background', edgecolor='black')
    axes[0, 0].hist(probs[labels == 1], bins=50, alpha=0.7, color='red',
                   label='Seizure', edgecolor='black')
    axes[0, 0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 0].set_xlabel('Predicted Seizure Probability', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Confidence Distribution by True Label', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence vs correctness
    predictions = (probs > 0.5).astype(int)
    correct = (predictions == labels)
    
    correct_probs = probs[correct]
    incorrect_probs = probs[~correct]
    
    axes[0, 1].hist(correct_probs, bins=50, alpha=0.7, color='green',
                   label=f'Correct ({len(correct_probs)})', edgecolor='black')
    axes[0, 1].hist(incorrect_probs, bins=50, alpha=0.7, color='red',
                   label=f'Incorrect ({len(incorrect_probs)})', edgecolor='black')
    axes[0, 1].axvline(0.5, color='black', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Confidence', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confidence heatmap
    bins = np.linspace(0, 1, 21)
    seizure_probs = probs[labels == 1]
    background_probs = probs[labels == 0]
    
    seizure_hist, _ = np.histogram(seizure_probs, bins=bins)
    background_hist, _ = np.histogram(background_probs, bins=bins)
    
    heatmap_data = np.vstack([background_hist, seizure_hist])
    
    im = axes[1, 0].imshow(heatmap_data, aspect='auto', cmap='RdYlGn', origin='lower')
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['Background', 'Seizure'])
    axes[1, 0].set_xlabel('Predicted Probability Bin', fontsize=12)
    axes[1, 0].set_title('Confidence Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 0], label='Count')
    
    # Plot 4: Metrics by confidence threshold
    thresholds = np.linspace(0, 1, 100)
    sensitivities = []
    specificities = []
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        tp = np.sum((labels == 1) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sens)
        specificities.append(spec)
    
    axes[1, 1].plot(thresholds, sensitivities, label='Sensitivity', linewidth=2, color='red')
    axes[1, 1].plot(thresholds, specificities, label='Specificity', linewidth=2, color='blue')
    axes[1, 1].axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Default (0.5)')
    axes[1, 1].set_xlabel('Decision Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Metric Value', fontsize=12)
    axes[1, 1].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confidence_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Create compelling EEG prediction visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_examples', type=int, default=6,
                       help='Number of example predictions to plot')
    parser.add_argument('--num_temporal', type=int, default=100,
                       help='Number of windows for temporal plot')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model and data
    print(f"\nLoading model and data...")
    model, windows, labels, metadata = load_model_and_data(
        args.checkpoint, args.data_path, device
    )
    
    print(f"Loaded {len(windows)} windows")
    print(f"Seizure rate: {100*np.mean(labels):.2f}%")
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # 1. Example predictions
    plot_example_predictions(model, windows, labels, device, args.output_dir, args.num_examples)
    
    # 2. Temporal predictions
    plot_temporal_predictions(model, windows, labels, device, args.output_dir, args.num_temporal)
    
    # 3. Confidence analysis
    plot_confidence_analysis(model, windows, labels, device, args.output_dir)
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"All visualizations saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()