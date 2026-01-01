"""
EEG Seizure Detection - Run Complete Preprocessing
Ceribell Project

This script runs the complete preprocessing pipeline:
1. Preprocesses raw EDF files
2. Creates train/val split
3. Creates PyTorch DataLoaders
4. Saves everything for training

Usage:
    python run_preprocessing.py --split dev --max_files 100
    python run_preprocessing.py --split train --save_dir ./preprocessed

Author: Ceribell Seizure Detector Project
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from modules.preprocessing import preprocess_dataset, Config
from modules.dataset import load_preprocessed_data, create_train_val_split, create_dataloaders


def visualize_samples(windows, labels, save_path='./sample_visualization.png'):
    """
    Visualize sample windows (both seizure and background).
    
    Args:
        windows: Preprocessed windows
        labels: Binary labels
        save_path: Where to save visualization
    """
    print("\nGenerating sample visualizations...")
    
    # Find seizure and background samples
    seizure_idx = np.where(labels == 1)[0]
    background_idx = np.where(labels == 0)[0]
    
    if len(seizure_idx) == 0:
        print("Warning: No seizure samples found!")
        return
    
    # Select random samples
    sz_sample = windows[np.random.choice(seizure_idx)]
    bg_sample = windows[np.random.choice(background_idx)]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot seizure sample
    for i in range(min(22, sz_sample.shape[0])):
        axes[0].plot(sz_sample[i] + i*5, linewidth=0.5, color='red', alpha=0.7)
    axes[0].set_title('Seizure Window (10 seconds)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Samples (200 Hz)')
    axes[0].set_ylabel('Channels (offset)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot background sample
    for i in range(min(22, bg_sample.shape[0])):
        axes[1].plot(bg_sample[i] + i*5, linewidth=0.5, color='blue', alpha=0.7)
    axes[1].set_title('Background Window (10 seconds)', fontsize=14, fontweight='bold')
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
    print("DATA SUMMARY")
    print("="*70)
    
    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"  • Total windows: {len(windows):,}")
    print(f"  • Window shape: {windows.shape}")
    print(f"  • Seizure windows: {np.sum(labels):,} ({100*np.mean(labels):.2f}%)")
    print(f"  • Background windows: {len(labels) - np.sum(labels):,} ({100*(1-np.mean(labels)):.2f}%)")
    print(f"  • Duration per window: {Config.WINDOW_SIZE} seconds")
    print(f"  • Total duration: {len(windows) * Config.WINDOW_SIZE / 3600:.2f} hours")
    
    # Train/val split statistics
    if split_data:
        print(f"\n📈 Train/Validation Split:")
        print(f"  • Train samples: {len(split_data['X_train']):,}")
        print(f"    - Seizure: {np.sum(split_data['y_train']):,} ({100*np.mean(split_data['y_train']):.2f}%)")
        print(f"    - Background: {len(split_data['y_train']) - np.sum(split_data['y_train']):,}")
        print(f"  • Val samples: {len(split_data['X_val']):,}")
        print(f"    - Seizure: {np.sum(split_data['y_val']):,} ({100*np.mean(split_data['y_val']):.2f}%)")
        print(f"    - Background: {len(split_data['y_val']) - np.sum(split_data['y_val']):,}")
    
    # Data quality metrics
    print(f"\n🔍 Data Quality:")
    print(f"  • Mean signal amplitude: {np.mean(np.abs(windows)):.4f}")
    print(f"  • Signal std deviation: {np.std(windows):.4f}")
    print(f"  • Min value: {np.min(windows):.4f}")
    print(f"  • Max value: {np.max(windows):.4f}")
    
    # Class imbalance ratio
    imbalance_ratio = (len(labels) - np.sum(labels)) / max(np.sum(labels), 1)
    print(f"\n⚖️  Class Imbalance:")
    print(f"  • Background:Seizure ratio = {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 10:
        print(f"  • ⚠️  High imbalance - use weighted sampling & class weights!")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG data for seizure detection')
    parser.add_argument('--split', type=str, default='dev', 
                       choices=['dev', 'train', 'eval'],
                       help='Data split to process (default: dev)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (None = all)')
    parser.add_argument('--save_dir', type=str, default='./preprocessed',
                       help='Directory to save preprocessed data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for DataLoaders')
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
    print("EEG SEIZURE DETECTION - PREPROCESSING PIPELINE")
    print("Ceribell Project")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  • Data split: {args.split}")
    print(f"  • Max files: {args.max_files if args.max_files else 'All'}")
    print(f"  • Save directory: {args.save_dir}")
    print(f"  • Validation size: {args.val_size*100:.0f}%")
    print(f"  • Batch size: {args.batch_size}")
    
    # Define save path
    save_path = os.path.join(args.save_dir, f'{args.split}.npz')
    
    # =========================================================================
    # STEP 1: PREPROCESS RAW DATA
    # =========================================================================
    
    if args.skip_preprocessing and os.path.exists(save_path):
        print(f"\n⏭️  Skipping preprocessing - loading existing file: {save_path}")
        windows, labels, metadata = load_preprocessed_data(save_path)
    else:
        print(f"\n{'='*70}")
        print(f"STEP 1: PREPROCESSING RAW DATA")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Run preprocessing
        result = preprocess_dataset(
            data_split=args.split,
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
    
    split_data = create_train_val_split(
        windows, labels,
        val_size=args.val_size,
        random_state=42
    )
    
    # Save split indices for reproducibility
    split_indices_path = os.path.join(args.save_dir, f'{args.split}_split_indices.npz')
    np.savez(
        split_indices_path,
        train_indices=np.arange(len(split_data['X_train'])),
        val_indices=np.arange(len(split_data['X_val']))
    )
    print(f"✓ Split indices saved to: {split_indices_path}")
    
    # =========================================================================
    # STEP 3: CREATE DATALOADERS
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"STEP 3: CREATING PYTORCH DATALOADERS")
    print(f"{'='*70}")
    
    loaders = create_dataloaders(
        X_train=split_data['X_train'],
        y_train=split_data['y_train'],
        X_val=split_data['X_val'],
        y_val=split_data['y_val'],
        batch_size=args.batch_size,
        num_workers=4,
        use_weighted_sampling=True,
        augment_train=True
    )
    
    # Save class weights
    class_weights_path = os.path.join(args.save_dir, f'{args.split}_class_weights.npy')
    np.save(class_weights_path, loaders['class_weights'].numpy())
    print(f"✓ Class weights saved to: {class_weights_path}")
    
    # =========================================================================
    # STEP 4: TEST DATALOADERS
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"STEP 4: TESTING DATALOADERS")
    print(f"{'='*70}")
    
    print("\nLoading sample batches...")
    
    # Test training loader
    train_loader = loaders['train_loader']
    train_seizure_counts = []
    
    for i, (batch_windows, batch_labels) in enumerate(train_loader):
        train_seizure_counts.append(batch_labels.sum().item())
        if i >= 9:  # Check first 10 batches
            break
    
    print(f"\nTrain Loader (first 10 batches):")
    print(f"  • Avg seizures per batch: {np.mean(train_seizure_counts):.1f}/{args.batch_size}")
    print(f"  • Expected ~50% due to weighted sampling")
    
    # Test validation loader
    val_loader = loaders['val_loader']
    val_seizure_counts = []
    
    for i, (batch_windows, batch_labels) in enumerate(val_loader):
        val_seizure_counts.append(batch_labels.sum().item())
        if i >= 9:
            break
    
    print(f"\nVal Loader (first 10 batches):")
    print(f"  • Avg seizures per batch: {np.mean(val_seizure_counts):.1f}/{args.batch_size}")
    print(f"  • Natural distribution (no weighted sampling)")
    
    # =========================================================================
    # STEP 5: VISUALIZATION (OPTIONAL)
    # =========================================================================
    
    if args.visualize:
        print(f"\n{'='*70}")
        print(f"STEP 5: GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        viz_path = os.path.join(args.save_dir, f'{args.split}_samples.png')
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
    print(f"  • Class weights: {class_weights_path}")
    if args.visualize:
        print(f"  • Visualizations: {viz_path}")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Design your model architecture (CNN-LSTM recommended)")
    print(f"  2. Implement training loop with weighted loss")
    print(f"  3. Define evaluation metrics (F1, AUROC, sensitivity)")
    print(f"  4. Train on this preprocessed data")
    
    print(f"\n💡 Quick Training Example:")
    print(f"```python")
    print(f"from modules.dataset import load_preprocessed_data, create_dataloaders")
    print(f"")
    print(f"# Load your preprocessed data")
    print(f"windows, labels, _ = load_preprocessed_data('{save_path}')")
    print(f"")
    print(f"# Create dataloaders (already done above)")
    print(f"# loaders = create_dataloaders(...)")
    print(f"")
    print(f"# Load class weights for loss function")
    print(f"class_weights = np.load('{class_weights_path}')")
    print(f"criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights))")
    print(f"")
    print(f"# Start training!")
    print(f"# for epoch in range(num_epochs):")
    print(f"#     for batch_windows, batch_labels in train_loader:")
    print(f"#         # ... training loop ...")
    print(f"```")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()