"""
EEG Artifact Detection - Run Complete Preprocessing
Ceribell Project

This script runs the complete preprocessing pipeline for TUAR:
1. Preprocesses raw EDF files with artifact annotations
2. Creates train/val split
3. Saves preprocessed data for training

Usage:
    python run_preprocessing.py --max_files 50
    python run_preprocessing.py --save_dir ./preprocessed

Author: Ceribell Seizure Detector Project
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import dataset utilities
sys.path.append('..')
from artifact_preprocessing import preprocess_dataset, Config

# Import dataset utilities from seizure detector (reuse code!)
sys.path.append('../training')
try:
    from modules.dataset import create_train_val_split
except:
    print("Warning: Could not import dataset module. Train/val split will be manual.")


def visualize_samples(windows, labels, save_path='./sample_visualization.png'):
    """
    Visualize sample windows (both artifact and clean).
    
    Args:
        windows: Preprocessed windows
        labels: Binary labels (0=clean, 1=artifact)
        save_path: Where to save visualization
    """
    print("\nGenerating sample visualizations...")
    
    # Find artifact and clean samples
    artifact_idx = np.where(labels == 1)[0]
    clean_idx = np.where(labels == 0)[0]
    
    if len(artifact_idx) == 0:
        print("Warning: No artifact samples found!")
        return
    
    if len(clean_idx) == 0:
        print("Warning: No clean samples found!")
        return
    
    # Select random samples
    artifact_sample = windows[np.random.choice(artifact_idx)]
    clean_sample = windows[np.random.choice(clean_idx)]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot artifact sample
    for i in range(min(22, artifact_sample.shape[0])):
        axes[0].plot(artifact_sample[i] + i*5, linewidth=0.5, color='red', alpha=0.7)
    axes[0].set_title('Artifact Window (10 seconds)', fontsize=14, fontweight='bold', color='red')
    axes[0].set_xlabel('Samples (200 Hz)')
    axes[0].set_ylabel('Channels (offset)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot clean sample
    for i in range(min(22, clean_sample.shape[0])):
        axes[1].plot(clean_sample[i] + i*5, linewidth=0.5, color='green', alpha=0.7)
    axes[1].set_title('Clean Window (10 seconds)', fontsize=14, fontweight='bold', color='green')
    axes[1].set_xlabel('Samples (200 Hz)')
    axes[1].set_ylabel('Channels (offset)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    plt.close()


def print_summary(windows, labels, split_data=None):
    """Print comprehensive data summary."""
    print("\n" + "="*70)
    print("DATA SUMMARY - ARTIFACT DETECTION")
    print("="*70)
    
    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"  • Total windows: {len(windows):,}")
    print(f"  • Window shape: {windows.shape}")
    print(f"  • Artifact windows: {np.sum(labels):,} ({100*np.mean(labels):.2f}%)")
    print(f"  • Clean windows: {len(labels) - np.sum(labels):,} ({100*(1-np.mean(labels)):.2f}%)")
    print(f"  • Duration per window: {Config.WINDOW_SIZE} seconds")
    print(f"  • Total duration: {len(windows) * Config.WINDOW_SIZE / 3600:.2f} hours")
    
    # Train/val split statistics
    if split_data:
        print(f"\n📈 Train/Validation Split:")
        print(f"  • Train samples: {len(split_data['X_train']):,}")
        print(f"    - Artifact: {np.sum(split_data['y_train']):,} ({100*np.mean(split_data['y_train']):.2f}%)")
        print(f"    - Clean: {len(split_data['y_train']) - np.sum(split_data['y_train']):,}")
        print(f"  • Val samples: {len(split_data['X_val']):,}")
        print(f"    - Artifact: {np.sum(split_data['y_val']):,} ({100*np.mean(split_data['y_val']):.2f}%)")
        print(f"    - Clean: {len(split_data['y_val']) - np.sum(split_data['y_val']):,}")
    
    # Data quality metrics
    print(f"\n🔍 Data Quality:")
    print(f"  • Mean signal amplitude: {np.mean(np.abs(windows)):.4f}")
    print(f"  • Signal std deviation: {np.std(windows):.4f}")
    print(f"  • Min value: {np.min(windows):.4f}")
    print(f"  • Max value: {np.max(windows):.4f}")
    
    # Class imbalance ratio
    imbalance_ratio = (len(labels) - np.sum(labels)) / max(np.sum(labels), 1)
    print(f"\n⚖️  Class Imbalance:")
    print(f"  • Clean:Artifact ratio = {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 3:
        print(f"  • Use weighted sampling & class weights for training!")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess TUAR data for artifact detection')
    parser.add_argument('--montage', type=str, default='01_tcp_ar',
                       choices=['01_tcp_ar', '02_tcp_le', '03_tcp_ar_a'],
                       help='Montage to process (default: 01_tcp_ar)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (None = all)')
    parser.add_argument('--save_dir', type=str, default='./preprocessed',
                       help='Directory to save preprocessed data')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size (fraction)')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip preprocessing (load existing .npz file)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate sample visualizations')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EEG ARTIFACT DETECTION - PREPROCESSING PIPELINE")
    print("Ceribell Project")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  • Montage: {args.montage}")
    print(f"  • Max files: {args.max_files if args.max_files else 'All'}")
    print(f"  • Save directory: {args.save_dir}")
    print(f"  • Validation size: {args.val_size*100:.0f}%")
    
    # Define save path
    save_path = os.path.join(args.save_dir, f'artifacts_{args.montage}.npz')
    
    # =========================================================================
    # STEP 1: PREPROCESS RAW DATA
    # =========================================================================
    
    if args.skip_preprocessing and os.path.exists(save_path):
        print(f"\n⏭️  Skipping preprocessing - loading existing file: {save_path}")
        data = np.load(save_path, allow_pickle=True)
        windows = data['windows']
        labels = data['labels']
        metadata = data['metadata']
    else:
        print(f"\n{'='*70}")
        print(f"STEP 1: PREPROCESSING RAW DATA")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Run preprocessing
        result = preprocess_dataset(
            montage=args.montage,
            max_files=args.max_files,
            save_path=save_path
        )
        
        if result is None:
            print("\n❌ Preprocessing failed - no valid data produced")
            return
        
        windows = result['windows']
        labels = result['labels']
        metadata = result['metadata']
        
        elapsed = time.time() - start_time
        print(f"\n✓ Preprocessing completed in {elapsed/60:.1f} minutes")
        print(f"✓ Data saved to: {save_path}")
    
    # =========================================================================
    # STEP 2: CREATE TRAIN/VAL SPLIT
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"STEP 2: CREATING TRAIN/VALIDATION SPLIT")
    print(f"{'='*70}")
    
    try:
        split_data = create_train_val_split(
            windows, labels,
            val_size=args.val_size,
            random_state=42
        )
    except:
        # Manual split if import failed
        print("Using manual train/val split...")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            windows, labels, 
            test_size=args.val_size,
            random_state=42,
            stratify=labels
        )
        split_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }
    
    # Save split indices for reproducibility
    split_indices_path = os.path.join(args.save_dir, f'artifacts_{args.montage}_split_indices.npz')
    np.savez(
        split_indices_path,
        train_indices=np.arange(len(split_data['X_train'])),
        val_indices=np.arange(len(split_data['X_val']))
    )
    print(f"✓ Split indices saved to: {split_indices_path}")
    
    # =========================================================================
    # STEP 3: VISUALIZATION (OPTIONAL)
    # =========================================================================
    
    if args.visualize:
        print(f"\n{'='*70}")
        print(f"STEP 3: GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        viz_path = os.path.join(args.save_dir, f'artifacts_{args.montage}_samples.png')
        visualize_samples(windows, labels, save_path=viz_path)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print_summary(windows, labels, split_data)
    
    # =========================================================================
    # READY FOR TRAINING
    # =========================================================================
    
    print("="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\n📁 Output Files:")
    print(f"  • Preprocessed data: {save_path}")
    print(f"  • Split indices: {split_indices_path}")
    if args.visualize:
        print(f"  • Visualizations: {viz_path}")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Train artifact detector: python train_artifact_detector.py")
    print(f"  2. Evaluate performance: python evaluate_artifact_detector.py")
    print(f"  3. Build unified pipeline: python unified_pipeline.py")
    
    print(f"\n💡 Training Command:")
    print(f"  python train_artifact_detector.py --data_path {save_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
