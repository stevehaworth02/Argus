"""
EEG Analysis - Unified Two-Stage Pipeline
Ceribell Project

Two-stage classification system:
    Stage 1: Artifact Detection (Clean vs Artifact)
    Stage 2: Seizure Detection (Clean vs Seizure)

Pipeline Logic:
    Raw EEG → Artifact Detector → [REJECT if artifact] → Seizure Detector → [ALERT if seizure]

This architecture prevents false seizure alarms caused by artifacts,
demonstrating production-ready clinical deployment thinking.

Usage:
    python unified_pipeline.py --artifact_model ./models/best_artifact_detector.pth --seizure_model ../data_viz/best_model.pth
    python unified_pipeline.py --test_data ./test_data.npz

Author: Ceribell Seizure Detector Project
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import time

# Import model architecture
sys.path.append('../data_viz')
from model import SeizureDetectorCNNLSTM


# ============================================================================
# UNIFIED PIPELINE CLASS
# ============================================================================

class UnifiedEEGPipeline:
    """
    Two-stage EEG analysis pipeline.
    
    Stage 1: Artifact Detection
        - Input: Raw EEG window
        - Output: Clean (pass to Stage 2) or Artifact (reject)
    
    Stage 2: Seizure Detection
        - Input: Clean EEG window (from Stage 1)
        - Output: Background or Seizure (alert)
    
    Clinical Decision Logic:
        IF artifact → REJECT (no seizure check)
        IF clean → Check for seizure
            IF seizure → ALERT
            IF background → PASS
    """
    
    def __init__(self, 
                 artifact_model_path: str,
                 seizure_model_path: str,
                 artifact_threshold: float = 0.5,
                 seizure_threshold: float = 0.5,
                 device: str = 'cuda'):
        """
        Initialize unified pipeline.
        
        Args:
            artifact_model_path: Path to trained artifact detector
            seizure_model_path: Path to trained seizure detector
            artifact_threshold: Threshold for artifact detection (0-1)
            seizure_threshold: Threshold for seizure detection (0-1)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.artifact_threshold = artifact_threshold
        self.seizure_threshold = seizure_threshold
        
        print(f"\n{'='*70}")
        print("INITIALIZING UNIFIED EEG PIPELINE")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Artifact threshold: {artifact_threshold}")
        print(f"Seizure threshold: {seizure_threshold}")
        
        # Load artifact detector
        print(f"\n[1/2] Loading artifact detector...")
        self.artifact_model = self._load_model(artifact_model_path)
        print(f"✓ Artifact detector loaded")
        
        # Load seizure detector
        print(f"\n[2/2] Loading seizure detector...")
        self.seizure_model = self._load_model(seizure_model_path)
        print(f"✓ Seizure detector loaded")
        
        print(f"\n{'='*70}")
        print("PIPELINE READY")
        print(f"{'='*70}\n")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = SeizureDetectorCNNLSTM(
            num_channels=22,
            num_samples=2000,
            num_classes=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict_artifact(self, window: torch.Tensor) -> Tuple[int, float]:
        """
        Stage 1: Artifact detection.
        
        Args:
            window: EEG window tensor (channels, samples)
        
        Returns:
            prediction: 0=clean, 1=artifact
            probability: Confidence score (0-1)
        """
        with torch.no_grad():
            if window.dim() == 2:
                window = window.unsqueeze(0)  # Add batch dimension
            
            window = window.to(self.device)
            logits, _ = self.artifact_model(window)
            probs = torch.softmax(logits, dim=1)
            
            artifact_prob = probs[0, 1].item()  # Probability of artifact class
            prediction = 1 if artifact_prob >= self.artifact_threshold else 0
            
            return prediction, artifact_prob
    
    def predict_seizure(self, window: torch.Tensor) -> Tuple[int, float]:
        """
        Stage 2: Seizure detection.
        
        Args:
            window: EEG window tensor (channels, samples)
        
        Returns:
            prediction: 0=background, 1=seizure
            probability: Confidence score (0-1)
        """
        with torch.no_grad():
            if window.dim() == 2:
                window = window.unsqueeze(0)  # Add batch dimension
            
            window = window.to(self.device)
            logits, _ = self.seizure_model(window)
            probs = torch.softmax(logits, dim=1)
            
            seizure_prob = probs[0, 1].item()  # Probability of seizure class
            prediction = 1 if seizure_prob >= self.seizure_threshold else 0
            
            return prediction, seizure_prob
    
    def process_window(self, window: torch.Tensor) -> Dict:
        """
        Process single EEG window through two-stage pipeline.
        
        Args:
            window: EEG window tensor (channels, samples)
        
        Returns:
            Dictionary with pipeline results
        """
        # Stage 1: Check for artifacts
        artifact_pred, artifact_prob = self.predict_artifact(window)
        
        result = {
            'artifact_detected': bool(artifact_pred),
            'artifact_probability': artifact_prob,
            'seizure_detected': False,
            'seizure_probability': 0.0,
            'clinical_decision': 'REJECT - Artifact' if artifact_pred else None
        }
        
        # Stage 2: Only check seizure if clean
        if not artifact_pred:
            seizure_pred, seizure_prob = self.predict_seizure(window)
            result['seizure_detected'] = bool(seizure_pred)
            result['seizure_probability'] = seizure_prob
            
            if seizure_pred:
                result['clinical_decision'] = 'ALERT - Seizure Detected'
            else:
                result['clinical_decision'] = 'PASS - Clean Background'
        
        return result
    
    def process_batch(self, windows: torch.Tensor, verbose: bool = False) -> list:
        """
        Process batch of windows through pipeline.
        
        Args:
            windows: Batch of EEG windows (batch, channels, samples)
            verbose: Print progress
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, window in enumerate(windows):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(windows)} windows...")
            
            result = self.process_window(window)
            results.append(result)
        
        return results


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_pipeline(pipeline: UnifiedEEGPipeline,
                     windows: np.ndarray,
                     artifact_labels: np.ndarray = None,
                     seizure_labels: np.ndarray = None) -> Dict:
    """
    Evaluate unified pipeline on test data.
    
    Args:
        pipeline: UnifiedEEGPipeline instance
        windows: EEG windows array (num_windows, channels, samples)
        artifact_labels: Ground truth artifact labels (optional)
        seizure_labels: Ground truth seizure labels (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print("EVALUATING UNIFIED PIPELINE")
    print(f"{'='*70}\n")
    
    print(f"Processing {len(windows):,} windows...")
    start_time = time.time()
    
    # Convert to tensor
    windows_tensor = torch.from_numpy(windows).float()
    
    # Process through pipeline
    results = pipeline.process_batch(windows_tensor, verbose=True)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Processed in {elapsed:.2f} seconds ({len(windows)/elapsed:.1f} windows/sec)")
    
    # Extract predictions
    artifact_preds = [r['artifact_detected'] for r in results]
    seizure_preds = [r['seizure_detected'] for r in results]
    
    # Calculate statistics
    n_artifacts = sum(artifact_preds)
    n_seizures = sum(seizure_preds)
    n_clean = len(windows) - n_artifacts - n_seizures
    
    print(f"\n📊 Pipeline Results:")
    print(f"  • Total windows: {len(windows):,}")
    print(f"  • Artifacts detected (rejected): {n_artifacts:,} ({100*n_artifacts/len(windows):.2f}%)")
    print(f"  • Seizures detected (alerted): {n_seizures:,} ({100*n_seizures/len(windows):.2f}%)")
    print(f"  • Clean background (passed): {n_clean:,} ({100*n_clean/len(windows):.2f}%)")
    
    eval_results = {
        'n_windows': len(windows),
        'n_artifacts': n_artifacts,
        'n_seizures': n_seizures,
        'n_clean': n_clean,
        'processing_time': elapsed,
        'windows_per_second': len(windows) / elapsed,
        'results': results
    }
    
    # If ground truth available, calculate accuracy
    if artifact_labels is not None:
        from sklearn.metrics import accuracy_score, f1_score
        artifact_acc = accuracy_score(artifact_labels, artifact_preds)
        artifact_f1 = f1_score(artifact_labels, artifact_preds)
        print(f"\n🎯 Artifact Detection Performance:")
        print(f"  • Accuracy: {artifact_acc:.4f}")
        print(f"  • F1 Score: {artifact_f1:.4f}")
        
        eval_results['artifact_accuracy'] = artifact_acc
        eval_results['artifact_f1'] = artifact_f1
    
    if seizure_labels is not None:
        from sklearn.metrics import accuracy_score, f1_score
        # Only evaluate seizure detection on windows that passed artifact filter
        clean_mask = [not r['artifact_detected'] for r in results]
        clean_indices = [i for i, is_clean in enumerate(clean_mask) if is_clean]
        
        if len(clean_indices) > 0:
            clean_seizure_preds = [seizure_preds[i] for i in clean_indices]
            clean_seizure_labels = [seizure_labels[i] for i in clean_indices]
            
            seizure_acc = accuracy_score(clean_seizure_labels, clean_seizure_preds)
            seizure_f1 = f1_score(clean_seizure_labels, clean_seizure_preds)
            
            print(f"\n🎯 Seizure Detection Performance (on clean windows):")
            print(f"  • Accuracy: {seizure_acc:.4f}")
            print(f"  • F1 Score: {seizure_f1:.4f}")
            
            eval_results['seizure_accuracy'] = seizure_acc
            eval_results['seizure_f1'] = seizure_f1
    
    print(f"{'='*70}\n")
    
    return eval_results


def visualize_pipeline_results(results: list, save_path: str):
    """Visualize pipeline decision distribution."""
    decisions = [r['clinical_decision'] for r in results]
    
    decision_counts = {
        'REJECT - Artifact': sum(1 for d in decisions if 'Artifact' in d),
        'ALERT - Seizure': sum(1 for d in decisions if 'Seizure' in d),
        'PASS - Clean': sum(1 for d in decisions if 'Clean' in d)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # Red, Yellow, Green
    bars = ax.bar(decision_counts.keys(), decision_counts.values(), color=colors, alpha=0.8)
    
    ax.set_ylabel('Number of Windows', fontsize=12)
    ax.set_title('Unified Pipeline - Clinical Decisions', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Pipeline visualization saved to: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run unified two-stage EEG pipeline')
    parser.add_argument('--artifact_model', type=str, required=True,
                       help='Path to trained artifact detector')
    parser.add_argument('--seizure_model', type=str, required=True,
                       help='Path to trained seizure detector')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data .npz file')
    parser.add_argument('--artifact_threshold', type=float, default=0.5,
                       help='Artifact detection threshold')
    parser.add_argument('--seizure_threshold', type=float, default=0.5,
                       help='Seizure detection threshold')
    parser.add_argument('--save_dir', type=str, default='./pipeline_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("UNIFIED TWO-STAGE EEG ANALYSIS PIPELINE")
    print("Ceribell Project")
    print("="*70)
    
    # =========================================================================
    # INITIALIZE PIPELINE
    # =========================================================================
    
    pipeline = UnifiedEEGPipeline(
        artifact_model_path=args.artifact_model,
        seizure_model_path=args.seizure_model,
        artifact_threshold=args.artifact_threshold,
        seizure_threshold=args.seizure_threshold,
        device=args.device
    )
    
    # =========================================================================
    # LOAD TEST DATA
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("LOADING TEST DATA")
    print(f"{'='*70}\n")
    
    data = np.load(args.test_data, allow_pickle=True)
    windows = data['windows']
    
    # Check if ground truth labels exist
    artifact_labels = data.get('artifact_labels', None)
    seizure_labels = data.get('seizure_labels', None)
    
    print(f"Loaded {len(windows):,} test windows")
    print(f"Window shape: {windows.shape}")
    
    # =========================================================================
    # RUN PIPELINE
    # =========================================================================
    
    eval_results = evaluate_pipeline(
        pipeline, windows,
        artifact_labels=artifact_labels,
        seizure_labels=seizure_labels
    )
    
    # =========================================================================
    # VISUALIZE RESULTS
    # =========================================================================
    
    viz_path = os.path.join(args.save_dir, 'pipeline_decisions.png')
    visualize_pipeline_results(eval_results['results'], viz_path)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    results_path = os.path.join(args.save_dir, 'pipeline_results.npz')
    np.savez(results_path, **eval_results)
    
    print(f"\n📁 Results saved:")
    print(f"  • Pipeline results: {results_path}")
    print(f"  • Visualization: {viz_path}")
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
