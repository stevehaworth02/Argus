"""
EEG Seizure Detection - Modules Package
Ceribell Project

This package contains all preprocessing and data loading utilities.
"""

from .preprocessing import (
    preprocess_eeg_file,
    preprocess_dataset,
    get_file_paths,
    Config
)

from .dataset import (
    EEGSeizureDataset,
    EEGAugmentation,
    load_preprocessed_data,
    create_train_val_split,
    create_dataloaders
)

__all__ = [
    # Preprocessing
    'preprocess_eeg_file',
    'preprocess_dataset',
    'get_file_paths',
    'Config',
    
    # Dataset
    'EEGSeizureDataset',
    'EEGAugmentation',
    'load_preprocessed_data',
    'create_train_val_split',
    'create_dataloaders',
]

__version__ = '1.0.0'
__author__ = 'Ceribell Seizure Detector Project'