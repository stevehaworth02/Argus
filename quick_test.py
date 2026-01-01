"""
Quick Test Script - Train and Evaluate on Small Dataset
Ceribell Project

This script quickly trains and evaluates a model on your preprocessed data
to verify everything works before scaling up.

Usage:
    python quick_test.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from modules.model import create_seizure_detector
from modules.dataset import load_preprocessed_data, create_train_val_split, create_dataloaders
from train import train_model
from evaluate import evaluate_model, calculate_detailed_metrics, generate_evaluation_report
from torch.utils.data import DataLoader


def quick_test():
    """Run quick training and evaluation test."""
    
    print("\n" + "="*70)
    print("QUICK TEST - SEIZURE DETECTOR")
    print("Ceribell Project")
    print("="*70 + "\n")
    
    # Check if data exists
    data_path = './preprocessed/dev.npz'
    weights_path = './preprocessed/dev_class_weights.npy'
    
    if not os.path.exists(data_path):
        print("❌ Error: Preprocessed data not found!")
        print("Please run: python run_preprocessing.py --split dev --max_files 10")
        return
    
    if not os.path.exists(weights_path):
        print("❌ Error: Class weights not found!")
        print("Please run: python run_preprocessing.py --split dev --max_files 10")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load data
    print("Loading data...")
    windows, labels, _ = load_preprocessed_data(data_path)
    class_weights = torch.from_numpy(np.load(weights_path))
    
    # Create train/val split
    print("Creating train/val split...")
    split_data = create_train_val_split(windows, labels, val_size=0.2)
    
    # Create dataloaders
    print("Creating dataloaders...")
    loaders = create_dataloaders(
        split_data['X_train'], split_data['y_train'],
        split_data['X_val'], split_data['y_val'],
        batch_size=32,
        num_workers=2,  # Reduced for quick test
        use_weighted_sampling=True,
        augment_train=True
    )
    
    # Create model (small for quick testing)
    print("\nCreating SMALL model for quick testing...")
    model = create_seizure_detector('small')
    model = model.to(device)
    
    # Training configuration
    config = {
        'num_epochs': 20,  # Short training for quick test
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'gradient_clip': 1.0,
        'patience': 10,
        'model_size': 'small',
        'seed': 42
    }
    
    # Create save directory
    save_dir = './checkpoints/quick_test'
    os.makedirs(save_dir, exist_ok=True)
    
    # Train
    print("\n" + "="*70)
    print("TRAINING (20 epochs for quick test)")
    print("="*70 + "\n")
    
    history, best_f1 = train_model(
        model, 
        loaders['train_loader'], 
        loaders['val_loader'],
        class_weights,
        config,
        device,
        save_dir
    )
    
    # Load best model for evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70 + "\n")
    
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate on validation set
    y_true, y_pred, y_prob = evaluate_model(model, loaders['val_loader'], device)
    
    # Calculate metrics
    metrics = calculate_detailed_metrics(y_true, y_pred, y_prob)
    
    # Generate report
    generate_evaluation_report(metrics)
    
    # Summary
    print("\n" + "="*70)
    print("QUICK TEST COMPLETE!")
    print("="*70)
    print(f"\n✓ Model trained successfully")
    print(f"✓ Best validation F1: {best_f1:.4f}")
    print(f"✓ Final metrics calculated")
    print(f"\n📊 Key Results:")
    print(f"   • Sensitivity: {metrics['sensitivity']:.3f} (catch {100*metrics['sensitivity']:.1f}% of seizures)")
    print(f"   • Specificity: {metrics['specificity']:.3f} (reject {100*metrics['specificity']:.1f}% of background)")
    print(f"   • F1 Score:    {metrics['f1']:.3f}")
    print(f"   • ROC-AUC:     {metrics['roc_auc']:.3f}")
    
    print(f"\n📁 Results saved to: {save_dir}")
    print(f"   • Best model:     {os.path.join(save_dir, 'best_model.pth')}")
    print(f"   • Training history: {os.path.join(save_dir, 'history.json')}")
    print(f"   • TensorBoard logs: {os.path.join(save_dir, 'runs')}")
    
    print("\n🚀 Next Steps:")
    print("   1. Review training curves: tensorboard --logdir ./checkpoints/quick_test/runs")
    print("   2. If results look good, scale to full dataset:")
    print("      python run_preprocessing.py --split dev")
    print("      python train.py --data_path ./preprocessed/dev.npz --model_size medium")
    print("   3. For production model, train on full train split")
    
    print("\n" + "="*70 + "\n")
    
    # Return metrics for further use
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'best_f1': best_f1
    }


if __name__ == "__main__":
    results = quick_test()