"""
Hierarchical 3-Class Pipeline
Combines Binary Artifact Filter + Argus Seizure Detector

Architecture:
    Input → Binary Artifact Filter (Is it artifact?)
         ├─ YES → "ARTIFACT"
         └─ NO → Argus Seizure Detector (Is it seizure?)
              ├─ YES → "SEIZURE"
              └─ NO → "BACKGROUND"

Advantages:
- Uses existing high-performing models (no retraining!)
- Binary filter: 99.88% seizure preservation
- Argus: 98.06% seizure detection AUC
- Expected: 95%+ overall accuracy

Author: Ceribell Hierarchical System
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
# MODEL ARCHITECTURES
# ============================================================================

class BinaryArtifactCNNLSTM(nn.Module):
    """Binary Artifact Filter (Artifact vs Clean)"""
    
    def __init__(self, num_channels=22, hidden_size=128, num_layers=2):
        super(BinaryArtifactCNNLSTM, self).__init__()
        
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
            nn.Linear(64, 2)  # Binary: clean vs artifact
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


class ArgusSeizureCNNLSTM(nn.Module):
    """Argus Seizure Detector (Seizure vs Background)"""
    
    def __init__(self, num_channels=22, hidden_size=128, num_layers=2):
        super(ArgusSeizureCNNLSTM, self).__init__()
        
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
            nn.Linear(64, 2)  # Binary: background vs seizure
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
            stage_info: Which stage made final decision
        """
        predictions = []
        probabilities = []
        stage_info = []
        
        num_batches = (len(windows) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(windows))
                batch = torch.FloatTensor(windows[start_idx:end_idx]).to(self.device)
                
                # Stage 1: Artifact Detection
                artifact_outputs = self.artifact_model(batch)
                artifact_probs = torch.softmax(artifact_outputs, dim=1)
                artifact_predictions = torch.argmax(artifact_probs, dim=1)
                
                # Process each window in batch
                for j in range(len(batch)):
                    # If artifact (class 1), output ARTIFACT
                    if artifact_predictions[j] == 1:
                        predictions.append(1)  # Artifact
                        probabilities.append(artifact_probs[j, 1].item())
                        stage_info.append('artifact_filter')
                    
                    # If clean (class 0), go to Stage 2
                    else:
                        # Stage 2: Seizure Detection
                        seizure_output = self.seizure_model(batch[j:j+1])
                        seizure_probs = torch.softmax(seizure_output, dim=1)
                        seizure_pred = torch.argmax(seizure_probs, dim=1)
                        
                        # If seizure (class 1), output SEIZURE
                        if seizure_pred[0] == 1:
                            predictions.append(2)  # Seizure
                            probabilities.append(seizure_probs[0, 1].item())
                            stage_info.append('seizure_detector')
                        
                        # If background (class 0), output BACKGROUND
                        else:
                            predictions.append(0)  # Background
                            probabilities.append(seizure_probs[0, 0].item())
                            stage_info.append('seizure_detector')
        
        return np.array(predictions), np.array(probabilities), stage_info


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_hierarchical(pipeline, windows, true_labels, class_names):
    """Evaluate hierarchical pipeline"""
    
    print("\nRunning hierarchical pipeline...")
    print("Stage 1: Artifact Filter")
    print("Stage 2: Seizure Detector")
    
    predictions, probabilities, stage_info = pipeline.predict(windows)
    
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
    
    return accuracy, metrics, predictions, stage_counts


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
    artifact_model = BinaryArtifactCNNLSTM().to(device)
    artifact_checkpoint = torch.load(args.artifact_model, map_location=device, weights_only=False)
    artifact_model.load_state_dict(artifact_checkpoint['model_state_dict'])
    print(f"  ✓ Loaded (trained epoch {artifact_checkpoint['epoch']})")
    print(f"  ✓ Training AUC: {artifact_checkpoint.get('val_auc', 'N/A')}")
    
    print("\n[2/2] Loading Argus Seizure Detector...")
    seizure_model = ArgusSeizureCNNLSTM().to(device)
    seizure_checkpoint = torch.load(args.seizure_model, map_location=device, weights_only=False)
    seizure_model.load_state_dict(seizure_checkpoint['model_state_dict'])
    print(f"  ✓ Loaded (trained epoch {seizure_checkpoint['epoch']})")
    print(f"  ✓ Training AUC: {seizure_checkpoint.get('val_auc', 'N/A')}")
    
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
    
    accuracy, metrics, predictions, stage_counts = evaluate_hierarchical(
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
    print(f"  ✓ Leverages Argus's 98% seizure knowledge")
    print(f"  ✓ Uses artifact filter's 99.88% preservation")
    print(f"  ✓ No retraining required")
    print(f"  ✓ Interpretable (two-stage decision process)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
