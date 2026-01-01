"""
EEG Seizure Detection - PyTorch Dataset
Ceribell Project

This module provides:
- PyTorch Dataset for EEG windows
- Data augmentation for EEG signals
- Class imbalance handling (weighted sampling, oversampling)
- Train/validation splitting

Author: Ceribell Seizure Detector Project
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class EEGAugmentation:
    """Data augmentation techniques for EEG signals"""
    
    @staticmethod
    def add_gaussian_noise(data: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        Add Gaussian noise to EEG signal.
        
        Args:
            data: EEG window (channels, samples)
            noise_level: Standard deviation of noise relative to signal
        
        Returns:
            Augmented EEG data
        """
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    @staticmethod
    def scale_amplitude(data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Randomly scale signal amplitude.
        
        Args:
            data: EEG window (channels, samples)
            scale_range: (min_scale, max_scale)
        
        Returns:
            Augmented EEG data
        """
        scale = np.random.uniform(*scale_range)
        return data * scale
    
    @staticmethod
    def time_shift(data: np.ndarray, max_shift: int = 50) -> np.ndarray:
        """
        Randomly shift signal in time.
        
        Args:
            data: EEG window (channels, samples)
            max_shift: Maximum shift in samples
        
        Returns:
            Augmented EEG data
        """
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if shift > 0:
            shifted = np.concatenate([data[:, shift:], data[:, :shift]], axis=1)
        elif shift < 0:
            shifted = np.concatenate([data[:, shift:], data[:, :shift]], axis=1)
        else:
            shifted = data
        
        return shifted
    
    @staticmethod
    def channel_dropout(data: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
        """
        Randomly zero out entire channels (simulates electrode failure).
        
        Args:
            data: EEG window (channels, samples)
            dropout_prob: Probability of dropping each channel
        
        Returns:
            Augmented EEG data
        """
        augmented = data.copy()
        mask = np.random.random(data.shape[0]) > dropout_prob
        augmented[~mask] = 0
        return augmented
    
    @staticmethod
    def apply_augmentation(data: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """
        Apply random combination of augmentations.
        
        Args:
            data: EEG window (channels, samples)
            prob: Probability of applying each augmentation
        
        Returns:
            Augmented EEG data
        """
        augmented = data.copy()
        
        if np.random.random() < prob:
            augmented = EEGAugmentation.add_gaussian_noise(augmented)
        
        if np.random.random() < prob:
            augmented = EEGAugmentation.scale_amplitude(augmented)
        
        if np.random.random() < prob:
            augmented = EEGAugmentation.time_shift(augmented)
        
        if np.random.random() < prob * 0.5:  # Less frequent
            augmented = EEGAugmentation.channel_dropout(augmented)
        
        return augmented


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class EEGSeizureDataset(Dataset):
    """
    PyTorch Dataset for EEG seizure detection.
    
    Handles:
    - Loading preprocessed windows and labels
    - Data augmentation during training
    - Normalization
    """
    
    def __init__(self, windows: np.ndarray, labels: np.ndarray,
                 augment: bool = False, augment_prob: float = 0.5):
        """
        Initialize dataset.
        
        Args:
            windows: EEG windows (num_windows, channels, samples)
            labels: Binary labels (num_windows,)
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying augmentation
        """
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.augment_prob = augment_prob
        
        print(f"Dataset initialized:")
        print(f"  - Total samples: {len(self.windows):,}")
        print(f"  - Seizure samples: {np.sum(self.labels):,} ({100*np.mean(self.labels):.2f}%)")
        print(f"  - Background samples: {len(self.labels) - np.sum(self.labels):,}")
        print(f"  - Input shape: {self.windows.shape}")
        print(f"  - Augmentation: {self.augment}")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            window: EEG window tensor (channels, samples)
            label: Binary label tensor (scalar)
        """
        window = self.windows[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation if enabled (only for seizure samples in training)
        if self.augment and label == 1:
            window = EEGAugmentation.apply_augmentation(window, self.augment_prob)
        
        # Convert to PyTorch tensors with explicit float32 dtype
        window = torch.from_numpy(window).float()  # Ensure float32
        label = torch.tensor(label, dtype=torch.long)  # Ensure long for classification
        
        return window, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for weighted loss (handles imbalance).
        
        Returns:
            Class weights tensor [weight_background, weight_seizure]
        """
        n_samples = len(self.labels)
        n_seizure = np.sum(self.labels)
        n_background = n_samples - n_seizure
        
        # Inverse frequency weighting
        weight_background = n_samples / (2 * n_background) if n_background > 0 else 1.0
        weight_seizure = n_samples / (2 * n_seizure) if n_seizure > 0 else 1.0
        
        weights = torch.tensor([weight_background, weight_seizure], dtype=torch.float32)
        
        print(f"\nClass weights calculated:")
        print(f"  - Background weight: {weight_background:.4f}")
        print(f"  - Seizure weight: {weight_seizure:.4f}")
        
        return weights
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate per-sample weights for WeightedRandomSampler.
        
        Returns:
            Sample weights tensor (num_samples,)
        """
        class_weights = self.get_class_weights().numpy()
        sample_weights = np.array([class_weights[label] for label in self.labels])
        return torch.from_numpy(sample_weights)


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_preprocessed_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load preprocessed data from .npz file.
    
    Args:
        npz_path: Path to .npz file
    
    Returns:
        windows: EEG windows array
        labels: Labels array
        metadata: List of metadata dicts
    """
    print(f"Loading preprocessed data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    windows = data['windows']
    labels = data['labels']
    metadata = data['metadata']
    
    print(f"Loaded {len(windows):,} windows")
    print(f"  - Shape: {windows.shape}")
    print(f"  - Seizure rate: {100*np.mean(labels):.2f}%")
    
    return windows, labels, metadata


def create_train_val_split(windows: np.ndarray, labels: np.ndarray,
                           val_size: float = 0.2, 
                           random_state: int = 42) -> Dict:
    """
    Split data into training and validation sets.
    
    Args:
        windows: EEG windows
        labels: Binary labels
        val_size: Fraction of data for validation
        random_state: Random seed
    
    Returns:
        Dictionary with train/val splits
    """
    X_train, X_val, y_train, y_val = train_test_split(
        windows, labels, 
        test_size=val_size,
        random_state=random_state,
        stratify=labels  # Maintain class distribution
    )
    
    print(f"\nTrain/Val split created:")
    print(f"  - Train: {len(X_train):,} samples ({100*np.mean(y_train):.2f}% seizure)")
    print(f"  - Val: {len(X_val):,} samples ({100*np.mean(y_val):.2f}% seizure)")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }


def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       use_weighted_sampling: bool = True,
                       augment_train: bool = True) -> Dict:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        X_train: Training windows
        y_train: Training labels
        X_val: Validation windows
        y_val: Validation labels
        batch_size: Batch size
        num_workers: Number of workers for data loading
        use_weighted_sampling: Whether to use weighted random sampling
        augment_train: Whether to augment training data
    
    Returns:
        Dictionary with train_loader, val_loader, and class_weights
    """
    print(f"\n{'='*60}")
    print("Creating DataLoaders")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = EEGSeizureDataset(X_train, y_train, augment=augment_train)
    val_dataset = EEGSeizureDataset(X_val, y_val, augment=False)
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    
    # Create samplers
    if use_weighted_sampling:
        print("\nUsing WeightedRandomSampler for balanced batches")
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle_train = False
    else:
        train_sampler = None
        shuffle_train = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'class_weights': class_weights,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EEG SEIZURE DETECTION - PYTORCH DATASET")
    print("Ceribell Project")
    print("="*60 + "\n")
    
    # Example: Create dummy data to demonstrate
    print("Creating example dataset...")
    
    # Simulate preprocessed data
    num_samples = 1000
    num_channels = 22
    window_samples = 2000  # 10 seconds at 200 Hz
    
    windows = np.random.randn(num_samples, num_channels, window_samples).astype(np.float32)
    labels = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])  # 15% seizure
    
    print(f"Example data shape: {windows.shape}")
    print(f"Example labels shape: {labels.shape}")
    print(f"Seizure rate: {100*np.mean(labels):.2f}%\n")
    
    # Create train/val split
    split_data = create_train_val_split(windows, labels, val_size=0.2)
    
    # Create dataloaders
    loaders = create_dataloaders(
        split_data['X_train'], split_data['y_train'],
        split_data['X_val'], split_data['y_val'],
        batch_size=32,
        use_weighted_sampling=True,
        augment_train=True
    )
    
    # Test loading a batch
    print("Testing batch loading...")
    train_loader = loaders['train_loader']
    
    for batch_windows, batch_labels in train_loader:
        print(f"\nBatch loaded successfully!")
        print(f"  - Batch windows shape: {batch_windows.shape}")
        print(f"  - Batch labels shape: {batch_labels.shape}")
        print(f"  - Seizure samples in batch: {batch_labels.sum().item()}/{len(batch_labels)}")
        break
    
    print("\n" + "="*60)
    print("Dataset ready for model training!")
    print("="*60 + "\n")