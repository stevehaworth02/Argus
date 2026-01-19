"""
Cross-Validation: Artifact Detector on Seizure Data
Ceribell Project

This script tests the artifact detector on TUSZ seizure data to validate that:
1. Real seizures are NOT flagged as artifacts (high specificity)
2. Non-seizure background is correctly identified as clean
3. The artifact detector complements seizure detection (doesn't interfere)

Critical Question: Does artifact detection hurt seizure detection?

Author: Ceribell Seizure Detector Project
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
import os
import sys

# Import model architecture from training directory
sys.path.append('..')
sys.path.append('../training')
sys.path.append('../training/modules')

# Try to import from training/modules
try:
    from modules.model import SeizureDetectorCNNLSTM
    MODEL_IMPORTED = True
except:
    try:
        from model import SeizureDetectorCNNLSTM
        MODEL_IMPORTED = True
    except:
        MODEL_IMPORTED = False
        print("Warning: Could not import SeizureDetectorCNNLSTM model")
        print("Will define a basic placeholder model")

# If import failed, define basic model structure
if not MODEL_IMPORTED:
    class SeizureDetectorCNNLSTM(nn.Module):
        """Basic CNN-LSTM model matching the architecture"""
        def __init__(self, num_channels=22, num_samples=2000, num_classes=2, 
                     cnn_filters=[32, 64, 128], lstm_hidden=128, lstm_layers=2, dropout=0.5, attention=True):
            super().__init__()
            
            # CNN layers
            self.conv_layers = nn.ModuleList()
            in_channels = num_channels
            for out_channels in cnn_filters:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout)
                ))
                in_channels = out_channels
            
            # LSTM
            self.lstm = nn.LSTM(
                input_size=cnn_filters[-1],
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=True
            )
            
            # Attention
            self.attention = nn.Sequential(
                nn.Linear(lstm_hidden * 2, lstm_hidden),
                nn.Tanh(),
                nn.Linear(lstm_hidden, 1)
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(lstm_hidden * 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            # CNN
            for conv in self.conv_layers:
                x = conv(x)
            
            # LSTM
            x = x.permute(0, 2, 1)
            lstm_out, _ = self.lstm(x)
            
            # Attention
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            attended = torch.sum(lstm_out * attn_weights, dim=1)
            
            # Classify
            out = self.classifier(attended)
            return out


# ============================================================================
# PYTORCH DATASET FOR TUSZ
# ============================================================================

class TUSZDataset(Dataset):
    """Simple dataset for TUSZ seizure data"""
    
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_artifact_on_seizures(model, dataloader, device):
    """
    Evaluate artifact detector on seizure data.
    
    Returns:
        preds: Predicted classes (0=clean, 1=artifact)
        true_labels: True labels (0=background, 1=seizure)
        probs: Prediction probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating artifact detector on seizure data...")
    
    with torch.no_grad():
        for windows, labels in dataloader:
            windows = windows.to(device)
            
            # Get artifact predictions
            # Model returns (outputs, attention_weights) tuple
            model_output = model(windows)
            if isinstance(model_output, tuple):
                outputs, _ = model_output  # Unpack tuple
            else:
                outputs = model_output  # Already just outputs
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of artifact
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def analyze_cross_performance(artifact_preds, seizure_labels, artifact_probs):
    """
    Analyze how artifact detector performs on seizure vs non-seizure data.
    
    Args:
        artifact_preds: Artifact detector predictions (0=clean, 1=artifact)
        seizure_labels: True seizure labels (0=background, 1=seizure)
        artifact_probs: Artifact prediction probabilities
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*70)
    
    # Separate by seizure vs background
    seizure_mask = seizure_labels == 1
    background_mask = seizure_labels == 0
    
    n_seizures = np.sum(seizure_mask)
    n_background = np.sum(background_mask)
    
    print(f"\nData Distribution:")
    print(f"  • Seizure windows: {n_seizures:,}")
    print(f"  • Background windows: {n_background:,}")
    
    # Artifact detection on SEIZURE windows
    seizure_artifact_rate = np.mean(artifact_preds[seizure_mask])
    seizure_clean_rate = 1 - seizure_artifact_rate
    
    print(f"\n🔴 SEIZURE Windows:")
    print(f"  • Flagged as CLEAN: {seizure_clean_rate*100:.2f}% ✅ (Want HIGH)")
    print(f"  • Flagged as ARTIFACT: {seizure_artifact_rate*100:.2f}% ⚠️ (Want LOW)")
    print(f"  → Seizures should be clean, not artifacts!")
    
    # Artifact detection on BACKGROUND windows
    bg_artifact_rate = np.mean(artifact_preds[background_mask])
    bg_clean_rate = 1 - bg_artifact_rate
    
    print(f"\n🟢 BACKGROUND Windows:")
    print(f"  • Flagged as CLEAN: {bg_clean_rate*100:.2f}%")
    print(f"  • Flagged as ARTIFACT: {bg_artifact_rate*100:.2f}%")
    
    # Critical metrics
    print(f"\n⚡ CRITICAL METRICS:")
    print(f"  • Seizure Preservation Rate: {seizure_clean_rate*100:.2f}%")
    print(f"    (How many seizures pass through artifact filter)")
    print(f"  • Background Artifact Rate: {bg_artifact_rate*100:.2f}%")
    print(f"    (How much background is filtered)")
    
    # Calculate if artifact detector helps or hurts
    if seizure_clean_rate > 0.85:
        print(f"\n✅ EXCELLENT: Artifact detector preserves {seizure_clean_rate*100:.1f}% of seizures!")
    elif seizure_clean_rate > 0.70:
        print(f"\n⚠️ WARNING: Artifact detector blocks {(1-seizure_clean_rate)*100:.1f}% of seizures")
    else:
        print(f"\n❌ PROBLEM: Artifact detector blocks {(1-seizure_clean_rate)*100:.1f}% of seizures!")
    
    return {
        'seizure_clean_rate': seizure_clean_rate,
        'seizure_artifact_rate': seizure_artifact_rate,
        'bg_clean_rate': bg_clean_rate,
        'bg_artifact_rate': bg_artifact_rate,
        'n_seizures': n_seizures,
        'n_background': n_background
    }


def plot_cross_validation_results(artifact_preds, seizure_labels, artifact_probs, save_dir='./cross_validation'):
    """Create visualizations for cross-validation analysis"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    seizure_mask = seizure_labels == 1
    background_mask = seizure_labels == 0
    
    # 1. Confusion-style matrix (not really confusion, but comparison)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top left: Bar chart comparing rates
    ax = axes[0, 0]
    categories = ['Seizure Windows', 'Background Windows']
    clean_rates = [
        np.mean(artifact_preds[seizure_mask] == 0) * 100,
        np.mean(artifact_preds[background_mask] == 0) * 100
    ]
    artifact_rates = [
        np.mean(artifact_preds[seizure_mask] == 1) * 100,
        np.mean(artifact_preds[background_mask] == 1) * 100
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clean_rates, width, label='Flagged as CLEAN', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, artifact_rates, width, label='Flagged as ARTIFACT', color='red', alpha=0.7)
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Artifact Detector on Seizure Data', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Top right: Probability distributions
    ax = axes[0, 1]
    ax.hist(artifact_probs[seizure_mask], bins=50, alpha=0.6, label='Seizure', color='red', density=True)
    ax.hist(artifact_probs[background_mask], bins=50, alpha=0.6, label='Background', color='blue', density=True)
    ax.set_xlabel('Artifact Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Artifact Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    
    # Bottom left: 2x2 comparison matrix
    ax = axes[1, 0]
    
    # Create comparison matrix
    seizure_clean = np.sum((seizure_labels == 1) & (artifact_preds == 0))
    seizure_artifact = np.sum((seizure_labels == 1) & (artifact_preds == 1))
    bg_clean = np.sum((seizure_labels == 0) & (artifact_preds == 0))
    bg_artifact = np.sum((seizure_labels == 0) & (artifact_preds == 1))
    
    matrix = np.array([
        [seizure_clean, seizure_artifact],
        [bg_clean, bg_artifact]
    ])
    
    sns.heatmap(matrix, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['Clean', 'Artifact'],
                yticklabels=['Seizure', 'Background'],
                cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Artifact Detector Prediction', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Seizure Label', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Matrix', fontsize=14, fontweight='bold')
    
    # Bottom right: Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    seizure_clean_rate = seizure_clean / (seizure_clean + seizure_artifact) * 100
    bg_artifact_rate = bg_artifact / (bg_clean + bg_artifact) * 100
    
    summary_text = f"""
    CROSS-VALIDATION SUMMARY
    ═══════════════════════════════════
    
    🔴 SEIZURE WINDOWS:
      Total: {seizure_clean + seizure_artifact:,}
      → Flagged as CLEAN: {seizure_clean:,} ({seizure_clean_rate:.1f}%)
      → Flagged as ARTIFACT: {seizure_artifact:,} ({100-seizure_clean_rate:.1f}%)
    
    🟢 BACKGROUND WINDOWS:
      Total: {bg_clean + bg_artifact:,}
      → Flagged as CLEAN: {bg_clean:,} ({100-bg_artifact_rate:.1f}%)
      → Flagged as ARTIFACT: {bg_artifact:,} ({bg_artifact_rate:.1f}%)
    
    ⚡ KEY INSIGHT:
      Seizure Preservation: {seizure_clean_rate:.1f}%
      {"✅ Excellent!" if seizure_clean_rate > 85 else "⚠️ Needs review" if seizure_clean_rate > 70 else "❌ Problem!"}
    
    📊 INTERPRETATION:
      The artifact detector should flag
      seizures as CLEAN (neurological activity)
      and artifacts as ARTIFACT (non-neural noise).
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cross_validation_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Cross-validation analysis saved to: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test artifact detector on seizure data')
    parser.add_argument('--artifact_model', type=str, required=True,
                       help='Path to artifact detector model')
    parser.add_argument('--seizure_data', type=str, required=True,
                       help='Path to TUSZ seizure dataset (.npz)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='./cross_validation')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("="*70)
    print("ARTIFACT DETECTOR ON SEIZURE DATA - CROSS VALIDATION")
    print("Ceribell Project")
    print("="*70)
    
    # =========================================================================
    # LOAD SEIZURE DATA (TUSZ)
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("LOADING SEIZURE DATA")
    print(f"{'='*70}")
    
    data = np.load(args.seizure_data, allow_pickle=True)
    windows = data['windows']
    labels = data['labels']  # 0=background, 1=seizure
    
    print(f"Loaded {len(windows):,} windows from TUSZ")
    print(f"  • Seizure: {np.sum(labels):,} ({100*np.mean(labels):.2f}%)")
    print(f"  • Background: {len(labels) - np.sum(labels):,} ({100*(1-np.mean(labels)):.2f}%)")
    print(f"  • Shape: {windows.shape}")
    
    # Create dataset
    dataset = TUSZDataset(windows, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # =========================================================================
    # LOAD ARTIFACT DETECTOR
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("LOADING ARTIFACT DETECTOR")
    print(f"{'='*70}")
    
    checkpoint = torch.load(args.artifact_model, map_location=device, weights_only=False)
    
    # Try to get model size from checkpoint
    model_size = checkpoint.get('model_size', 'medium')
    
    # Create model - try different initialization approaches
    print(f"Attempting to load model (size: {model_size})...")
    
    try:
        # Try with full parameters (cnn_filters not cnn_channels!)
        model = SeizureDetectorCNNLSTM(
            num_channels=22,
            num_samples=2000,
            num_classes=2,
            cnn_filters=[32, 64, 128],
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.5,
            attention=True
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully with explicit parameters")
    except Exception as e1:
        print(f"  Attempt 1 failed: {e1}")
        try:
            # Try with model_size parameter (if using create_seizure_detector helper)
            model = SeizureDetectorCNNLSTM(model_size=model_size).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded successfully with model_size='{model_size}'")
        except Exception as e2:
            print(f"  Attempt 2 failed: {e2}")
            try:
                # Try with no parameters (uses defaults)
                model = SeizureDetectorCNNLSTM().to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Model loaded successfully with default parameters")
            except Exception as e3:
                print(f"  Attempt 3 failed: {e3}")
                print("\n❌ Could not load model!")
                print(f"Please check model architecture matches checkpoint")
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
                return
    
    model.eval()
    
    print(f"  • Trained epoch: {checkpoint['epoch']}")
    print(f"  • Training Val AUC: {checkpoint.get('val_auc', 'N/A')}")
    
    # =========================================================================
    # EVALUATE
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("RUNNING CROSS-VALIDATION")
    print(f"{'='*70}")
    
    artifact_preds, seizure_labels, artifact_probs = evaluate_artifact_on_seizures(
        model, dataloader, device
    )
    
    # =========================================================================
    # ANALYZE RESULTS
    # =========================================================================
    
    results = analyze_cross_performance(artifact_preds, seizure_labels, artifact_probs)
    
    # =========================================================================
    # VISUALIZE
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_cross_validation_results(artifact_preds, seizure_labels, artifact_probs, args.save_dir)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, 'cross_validation_results.npz')
    
    np.savez(
        results_path,
        artifact_preds=artifact_preds,
        seizure_labels=seizure_labels,
        artifact_probs=artifact_probs,
        **results
    )
    
    print(f"✓ Results saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n📁 Output files:")
    print(f"  • Visualization: {args.save_dir}/cross_validation_analysis.png")
    print(f"  • Results data: {results_path}")
    
    print(f"\n🎯 Final Verdict:")
    if results['seizure_clean_rate'] > 0.85:
        print(f"  ✅ PASS: Artifact detector preserves {results['seizure_clean_rate']*100:.1f}% of seizures")
        print(f"  → Safe to use in production pipeline!")
    elif results['seizure_clean_rate'] > 0.70:
        print(f"  ⚠️ CAUTION: Artifact detector blocks {(1-results['seizure_clean_rate'])*100:.1f}% of seizures")
        print(f"  → May need threshold adjustment")
    else:
        print(f"  ❌ FAIL: Artifact detector blocks {(1-results['seizure_clean_rate'])*100:.1f}% of seizures")
        print(f"  → Not safe for production without retraining")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
