"""
Hierarchical 3-Class Pipeline - FIXED VERSION
Combines Binary Artifact Filter + Argus Seizure Detector

Uses actual user architectures from model.py

Architecture:
    Input → Binary Artifact Filter (Is it artifact?)
         ├─ YES → "ARTIFACT"
         └─ NO → Argus Seizure Detector (Is it seizure?)
              ├─ YES → "SEIZURE"
              └─ NO → "BACKGROUND"

Author: Ceribell Hierarchical System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from typing import Tuple, Optional

# ============================================================================
# IMPORT USER'S MODEL ARCHITECTURE
# ============================================================================

# Add parent directory to path to import user's modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import the actual model architecture
try:
    from modules.model import SeizureDetectorCNNLSTM, TemporalAttention, TemporalConvBlock
    print("✓ Successfully imported user's model architecture")
except ImportError as e:
    print(f"✗ Could not import from modules.model: {e}")
    print("✓ Defining architecture inline...")
    
    # Define architecture inline if import fails
    class TemporalAttention(nn.Module):
        """Temporal attention mechanism"""
        def __init__(self, hidden_dim: int):
            super(TemporalAttention, self).__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            attention_scores = self.attention(lstm_output)
            attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
            context = context.squeeze(1)
            return context, attention_weights

    class TemporalConvBlock(nn.Module):
        """Temporal convolution block with residual connection"""
        def __init__(self, in_channels: int, out_channels: int, 
                     kernel_size: int, dropout: float = 0.3):
            super(TemporalConvBlock, self).__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                   padding=kernel_size//2)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   padding=kernel_size//2)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)
            self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.residual(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = out + residual
            out = F.relu(out)
            return out

    class SeizureDetectorCNNLSTM(nn.Module):
        """Robust CNN-LSTM architecture for seizure detection"""
        def __init__(self, 
                     num_channels: int = 22,
                     num_samples: int = 2000,
                     num_classes: int = 2,
                     cnn_filters: list = [32, 64, 128],
                     lstm_hidden: int = 128,
                     lstm_layers: int = 2,
                     dropout: float = 0.4,
                     attention: bool = True):
            super(SeizureDetectorCNNLSTM, self).__init__()
            
            self.num_channels = num_channels
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.use_attention = attention
            
            # Temporal convolution blocks
            self.temporal_conv1 = TemporalConvBlock(num_channels, cnn_filters[0], 
                                                    kernel_size=7, dropout=dropout)
            self.temporal_conv2 = TemporalConvBlock(cnn_filters[0], cnn_filters[1],
                                                    kernel_size=11, dropout=dropout)
            self.temporal_conv3 = TemporalConvBlock(cnn_filters[1], cnn_filters[2],
                                                    kernel_size=15, dropout=dropout)
            
            self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
            
            self.feature_size = cnn_filters[2]
            self.seq_len = num_samples // (4 ** 3)
            
            # LSTM
            self.lstm = nn.LSTM(
                input_size=self.feature_size,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=True
            )
            
            lstm_output_dim = lstm_hidden * 2
            
            # Attention
            if self.use_attention:
                self.attention = TemporalAttention(lstm_output_dim)
                classifier_input_dim = lstm_output_dim
            else:
                classifier_input_dim = lstm_output_dim
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(classifier_input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # Temporal convolution
            out = self.temporal_conv1(x)
            out = self.pool(out)
            out = self.temporal_conv2(out)
            out = self.pool(out)
            out = self.temporal_conv3(out)
            out = self.pool(out)
            
            # LSTM
            out = out.permute(0, 2, 1)
            lstm_out, _ = self.lstm(out)
            
            # Attention or pooling
            if self.use_attention:
                context, attention_weights = self.attention(lstm_out)
            else:
                context = torch.mean(lstm_out, dim=1)
                attention_weights = None
            
            # Classification
            logits = self.classifier(context)
            
            return logits, attention_weights


# ============================================================================
# HIERARCHICAL PIPELINE
# ============================================================================

class HierarchicalPipeline:
    """
    Two-stage hierarchical classifier:
    Stage 1: Binary Artifact Filter
    Stage 2: Argus Seizure Detector
    """
    
    def __init__(self, artifact_model, seizure_model, device):
        self.artifact_model = artifact_model
        self.seizure_model = seizure_model
        self.device = device
        
        self.artifact_model.eval()
        self.seizure_model.eval()
    
    def predict(self, windows, batch_size=64):
        """
        Hierarchical prediction on batch of windows
        
        Returns:
            predictions: 0=Background, 1=Artifact, 2=Seizure
            probabilities: Confidence scores
            seizure_probs: Probability of seizure class (for ROC/PR curves)  # ← ADDED
            stage_info: Which stage made final decision
        """
        predictions = []
        probabilities = []
        seizure_probs = []  # ← ADDED FOR ROC/PR CURVES
        stage_info = []
        
        num_batches = (len(windows) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(windows))
                batch = torch.FloatTensor(windows[start_idx:end_idx]).to(self.device)
                
                # Stage 1: Artifact Detection
                artifact_outputs, _ = self.artifact_model(batch)
                artifact_probs = torch.softmax(artifact_outputs, dim=1)
                artifact_predictions = torch.argmax(artifact_probs, dim=1)
                
                # Process each window in batch
                for j in range(len(batch)):
                    # If artifact (class 1), output ARTIFACT
                    if artifact_predictions[j] == 1:
                        predictions.append(1)  # Artifact
                        probabilities.append(artifact_probs[j, 1].item())
                        seizure_probs.append(0.0)  # ← ADDED: Artifact = not seizure, so 0.0
                        stage_info.append('artifact_filter')
                    
                    # If clean (class 0), go to Stage 2
                    else:
                        # Stage 2: Seizure Detection
                        seizure_output, _ = self.seizure_model(batch[j:j+1])
                        seizure_probs_tensor = torch.softmax(seizure_output, dim=1)
                        seizure_pred = torch.argmax(seizure_probs_tensor, dim=1)
                        
                        # ← ADDED: Always save probability of seizure class
                        seizure_prob_value = seizure_probs_tensor[0, 1].item()
                        seizure_probs.append(seizure_prob_value)
                        
                        # If seizure (class 1), output SEIZURE
                        if seizure_pred[0] == 1:
                            predictions.append(2)  # Seizure
                            probabilities.append(seizure_prob_value)
                            stage_info.append('seizure_detector')
                        
                        # If background (class 0), output BACKGROUND
                        else:
                            predictions.append(0)  # Background
                            probabilities.append(seizure_probs_tensor[0, 0].item())
                            stage_info.append('seizure_detector')
        
        return np.array(predictions), np.array(probabilities), np.array(seizure_probs), stage_info  # ← MODIFIED


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_hierarchical(pipeline, windows, true_labels, class_names):
    """Evaluate hierarchical pipeline"""
    
    print("\nRunning hierarchical pipeline...")
    print("Stage 1: Artifact Filter")
    print("Stage 2: Seizure Detector")
    
    predictions, probabilities, seizure_probs, stage_info = pipeline.predict(windows)  # ← MODIFIED
    
    # Calculate metrics
    accuracy = np.mean(predictions == true_labels)
    
    # Per-class metrics
    metrics = {}
    for idx, class_name in enumerate(class_names):
        mask_true = (true_labels == idx)
        mask_pred = (predictions == idx)
        
        if np.sum(mask_true) == 0:
            continue
        
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(mask_true))
        }
    
    # Stage usage statistics
    stage_counts = {
        'artifact_filter': sum(1 for s in stage_info if s == 'artifact_filter'),
        'seizure_detector': sum(1 for s in stage_info if s == 'seizure_detector')
    }
    
    return accuracy, metrics, predictions, seizure_probs, stage_counts, stage_info  # ← MODIFIED


def plot_results(true_labels, predictions, class_names, save_dir):
    """Plot confusion matrix and results"""
    
    cm = confusion_matrix(true_labels, predictions)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Hierarchical Pipeline - Confusion Matrix (Counts)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Hierarchical Pipeline - Confusion Matrix (% of True Class)',
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'hierarchical_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hierarchical 3-class pipeline')
    parser.add_argument('--artifact_model', type=str, required=True,
                       help='Path to binary artifact model')
    parser.add_argument('--seizure_model', type=str, required=True,
                       help='Path to Argus seizure model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data (dev.npz)')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("="*70)
    print("HIERARCHICAL 3-CLASS PIPELINE")
    print("Binary Artifact Filter + Argus Seizure Detector")
    print("="*70)
    
    # Load test data
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    
    data = np.load(args.test_data, allow_pickle=True)
    test_windows = data['windows']
    test_labels_binary = data['labels']
    
    # Convert to 3-class: 0=background, 1=artifact (N/A), 2=seizure
    test_labels = test_labels_binary.copy()
    test_labels[test_labels_binary == 1] = 2
    
    print(f"\nTest set: {len(test_windows):,} windows")
    print(f"  Background: {np.sum(test_labels==0):,}")
    print(f"  Artifact: {np.sum(test_labels==1):,} (N/A in TUSZ)")
    print(f"  Seizure: {np.sum(test_labels==2):,}")
    
    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    print("\n[1/2] Loading Binary Artifact Filter...")
    artifact_checkpoint = torch.load(args.artifact_model, map_location=device, weights_only=False)
    
    # Determine artifact model configuration from checkpoint
    artifact_state = artifact_checkpoint['model_state_dict']
    # The artifact detector likely uses default 'medium' config
    artifact_model = SeizureDetectorCNNLSTM(
        num_channels=22,
        num_samples=2000,
        num_classes=2,
        cnn_filters=[32, 64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.4,
        attention=True
    ).to(device)
    
    artifact_model.load_state_dict(artifact_state, strict=True)
    artifact_model.eval()
    
    print(f"  ✓ Loaded (trained epoch {artifact_checkpoint['epoch']})")
    auc_val = artifact_checkpoint.get('val_auc', artifact_checkpoint.get('val_metrics', {}).get('auroc', 'N/A'))
    print(f"  ✓ Training AUC: {auc_val}")
    
    print("\n[2/2] Loading Argus Seizure Detector...")
    seizure_checkpoint = torch.load(args.seizure_model, map_location=device, weights_only=False)
    
    # Create seizure model with 'medium' configuration
    seizure_model = SeizureDetectorCNNLSTM(
        num_channels=22,
        num_samples=2000,
        num_classes=2,
        cnn_filters=[32, 64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.4,
        attention=True
    ).to(device)
    
    seizure_model.load_state_dict(seizure_checkpoint['model_state_dict'], strict=True)
    seizure_model.eval()
    
    print(f"  ✓ Loaded (trained epoch {seizure_checkpoint['epoch']})")
    auc_val = seizure_checkpoint.get('val_auc', seizure_checkpoint.get('val_metrics', {}).get('auroc', 'N/A'))
    print(f"  ✓ Training AUC: {auc_val}")
    
    # Create pipeline
    print("\n" + "="*70)
    print("CREATING HIERARCHICAL PIPELINE")
    print("="*70)
    
    pipeline = HierarchicalPipeline(artifact_model, seizure_model, device)
    
    print("\nPipeline architecture:")
    print("  Input → Artifact Filter (Is it artifact?)")
    print("       ├─ YES → Output: ARTIFACT")
    print("       └─ NO → Seizure Detector (Is it seizure?)")
    print("            ├─ YES → Output: SEIZURE")
    print("            └─ NO → Output: BACKGROUND")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING ON DEV SET")
    print("="*70)
    
    class_names = ['Background', 'Artifact', 'Seizure']
    
    accuracy, metrics, predictions, seizure_probs, stage_counts, stage_info = evaluate_hierarchical(  # ← MODIFIED
        pipeline, test_windows, test_labels, class_names
    )
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n[OVERALL PERFORMANCE]:")
    print(f"  * Overall Accuracy: {100*accuracy:.2f}%")
    print(f"  * Total samples: {len(test_labels):,}")
    print(f"  * Correct: {np.sum(predictions==test_labels):,}")
    print(f"  * Incorrect: {np.sum(predictions!=test_labels):,}")
    
    print(f"\n[STAGE USAGE]:")
    print(f"  * Classified by Artifact Filter: {stage_counts['artifact_filter']:,} ({100*stage_counts['artifact_filter']/len(predictions):.1f}%)")
    print(f"  * Classified by Seizure Detector: {stage_counts['seizure_detector']:,} ({100*stage_counts['seizure_detector']/len(predictions):.1f}%)")
    
    print(f"\n[PER-CLASS METRICS]:")
    for class_name, class_metrics in metrics.items():
        if class_metrics['support'] > 0:
            print(f"\n  {class_name}:")
            print(f"    * Precision: {100*class_metrics['precision']:.2f}%")
            print(f"    * Recall: {100*class_metrics['recall']:.2f}%")
            print(f"    * F1-Score: {100*class_metrics['f1']:.2f}%")
            print(f"    * Support: {class_metrics['support']:,} samples")
        else:
            print(f"\n  {class_name}: Not present in test set")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Plot
    plot_results(test_labels, predictions, class_names, args.save_dir)
    
    # Save metrics
    results = {
        'overall_accuracy': float(accuracy),
        'per_class_metrics': metrics,
        'stage_usage': stage_counts,
        'test_size': len(test_labels),
        'architecture': 'hierarchical_pipeline'
    }
    
    results_path = os.path.join(args.save_dir, 'hierarchical_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    # ========== ADDED: SAVE PREDICTIONS FOR ROC/PR CURVES ==========
    print("\n" + "="*70)
    print("SAVING PREDICTIONS FOR ROC/PR CURVES")
    print("="*70)
    
    # Convert 3-class labels to binary (0=background/artifact, 1=seizure)
    y_true_binary = (test_labels == 2).astype(int)
    y_pred_binary = (predictions == 2).astype(int)
    
    predictions_path = os.path.join(args.save_dir, 'hierarchical_predictions.npz')
    np.savez(
        predictions_path,
        y_true=y_true_binary,
        y_pred=y_pred_binary,
        y_pred_proba=seizure_probs,
        predictions_3class=predictions,
        true_labels_3class=test_labels,
        stage_decisions=stage_info
    )
    
    print(f"\n✓ Predictions saved to: {predictions_path}")
    print(f"  - y_true: Binary true labels (0=background/artifact, 1=seizure)")
    print(f"  - y_pred: Binary predictions (0=background/artifact, 1=seizure)")
    print(f"  - y_pred_proba: Seizure probabilities (for ROC/PR curves)")
    print(f"  - predictions_3class: 3-class predictions (0/1/2)")
    print(f"  - stage_decisions: Which stage made each decision")
    
    # Verify saved data
    verify = np.load(predictions_path)
    print(f"\n✓ Verification:")
    print(f"  Keys in file: {list(verify.keys())}")
    print(f"  y_true shape: {verify['y_true'].shape}")
    print(f"  y_pred shape: {verify['y_pred'].shape}")
    print(f"  y_pred_proba shape: {verify['y_pred_proba'].shape}")
    print(f"  Probability range: [{verify['y_pred_proba'].min():.4f}, {verify['y_pred_proba'].max():.4f}]")
    
    print("\n  You can now run: python generate_roc_pr_curves.py")
    print("="*70)
    # ================================================================
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n[HIERARCHICAL PIPELINE PERFORMANCE]:")
    print(f"  * Uses existing models (no retraining!)")
    print(f"  * Overall Accuracy: {100*accuracy:.2f}%")
    print(f"  * Seizure Recall: {100*metrics.get('Seizure', {}).get('recall', 0):.2f}%")
    print(f"  * Background Recall: {100*metrics.get('Background', {}).get('recall', 0):.2f}%")
    
    print(f"\n[ADVANTAGES]:")
    print(f"  ✓ Modular (can update each model independently)")
    print(f"  ✓ Leverages Argus's existing knowledge")
    print(f"  ✓ Uses artifact filter's preservation")
    print(f"  ✓ No retraining required")
    print(f"  ✓ Interpretable (two-stage decision process)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()