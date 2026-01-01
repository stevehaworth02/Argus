"""
EEG Seizure Detection - Data Preprocessing Pipeline
Ceribell Project

This module handles:
- EDF file loading with MNE
- Multi-channel EEG preprocessing
- Sliding window generation with labels
- Class imbalance handling
- PyTorch Dataset integration

Author: Ceribell Seizure Detector Project
Dataset: Temple University Hospital EEG Seizure Corpus (TUSZ) v2.0.3
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy import signal
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for preprocessing pipeline"""
    
    # Data paths
    DATA_ROOT = "/mnt/c/Users/0218s/Desktop/Ceribell-SZ-DTCTR/data"
    TUSZ_PATH = os.path.join(DATA_ROOT, "tusz")
    TUAR_PATH = os.path.join(DATA_ROOT, "tuar")
    
    # Preprocessing parameters
    SAMPLING_RATE = 250  # Hz (original TUSZ sampling rate)
    TARGET_SAMPLING_RATE = 200  # Hz (downsample for efficiency)
    WINDOW_SIZE = 10  # seconds
    WINDOW_OVERLAP = 5  # seconds (50% overlap)
    
    # Frequency filtering
    LOWCUT = 0.5  # Hz (remove DC drift)
    HIGHCUT = 50  # Hz (remove line noise and high-freq artifacts)
    NOTCH_FREQ = 60  # Hz (US power line frequency)
    
    # Channel selection (TCP montage - standard 22 bipolar channels)
    STANDARD_CHANNELS = [
        'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
        'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FZ-CZ', 'CZ-PZ',
        'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'
    ]
    
    # Seizure types (map to binary or multi-class)
    SEIZURE_TYPES = ['gnsz', 'fnsz', 'cpsz', 'tcsz', 'absz', 'tnsz']
    
    # Class imbalance handling
    BALANCE_STRATEGY = 'weighted'  # Options: 'weighted', 'oversample', 'undersample'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_edf_file(edf_path: str, target_channels: List[str] = None) -> Tuple[np.ndarray, List[str], float]:
    """
    Load EEG data from EDF file using MNE.
    
    Args:
        edf_path: Path to .edf file
        target_channels: List of channels to load (None = load all)
    
    Returns:
        data: EEG data array (channels, samples)
        channels: List of channel names
        sfreq: Sampling frequency
    """
    try:
        # Load EDF file
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Get channel names and sampling frequency
        channels = raw.ch_names
        sfreq = raw.info['sfreq']
        
        # Filter to target channels if specified
        if target_channels:
            # Find intersection of available and target channels
            available_channels = [ch for ch in target_channels if ch in channels]
            if available_channels:
                raw.pick_channels(available_channels)
                channels = available_channels
        
        # Get data as numpy array
        data = raw.get_data()
        
        return data, channels, sfreq
        
    except Exception as e:
        print(f"Error loading {edf_path}: {e}")
        return None, None, None


def parse_annotations(csv_path: str) -> pd.DataFrame:
    """
    Parse seizure annotations from CSV file.
    
    Args:
        csv_path: Path to .csv annotation file
    
    Returns:
        DataFrame with columns: channel, start_time, stop_time, label, confidence
    """
    try:
        if not os.path.exists(csv_path):
            # No annotations = all background
            return pd.DataFrame(columns=['channel', 'start_time', 'stop_time', 'label', 'confidence'])
        
        # Read CSV, skipping comment lines (lines starting with #)
        # Also handle potential header variations
        df = pd.read_csv(csv_path, comment='#', skip_blank_lines=True)
        
        # Check if we got valid data
        if df.empty or len(df.columns) < 5:
            return pd.DataFrame(columns=['channel', 'start_time', 'stop_time', 'label', 'confidence'])
        
        # Ensure we have the right columns
        expected_cols = ['channel', 'start_time', 'stop_time', 'label', 'confidence']
        if not all(col in df.columns for col in expected_cols):
            # Try to infer column names if header is wrong
            if len(df.columns) == 5:
                df.columns = expected_cols
            else:
                return pd.DataFrame(columns=expected_cols)
        
        # Filter to seizure events only (exclude 'bckg')
        df = df[df['label'] != 'bckg'].copy()
        
        return df
        
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
        return pd.DataFrame(columns=['channel', 'start_time', 'stop_time', 'label', 'confidence'])


def apply_bandpass_filter(data: np.ndarray, sfreq: float, 
                          lowcut: float = Config.LOWCUT, 
                          highcut: float = Config.HIGHCUT) -> np.ndarray:
    """
    Apply bandpass Butterworth filter to EEG data.
    
    Args:
        data: EEG data (channels, samples)
        sfreq: Sampling frequency
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
    
    Returns:
        Filtered EEG data
    """
    nyquist = sfreq / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth filter (4th order)
    sos = signal.butter(4, [low, high], btype='band', output='sos')
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.sosfiltfilt(sos, data[i])
    
    return filtered_data


def apply_notch_filter(data: np.ndarray, sfreq: float, 
                       freq: float = Config.NOTCH_FREQ, Q: float = 30) -> np.ndarray:
    """
    Apply notch filter to remove power line noise.
    
    Args:
        data: EEG data (channels, samples)
        sfreq: Sampling frequency
        freq: Frequency to remove (Hz)
        Q: Quality factor
    
    Returns:
        Filtered EEG data
    """
    b, a = signal.iirnotch(freq, Q, sfreq)
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    return filtered_data


def normalize_channels(data: np.ndarray, method: str = 'robust') -> np.ndarray:
    """
    Normalize EEG channels to standard scale.
    
    Args:
        data: EEG data (channels, samples)
        method: 'robust' (median/IQR) or 'standard' (mean/std)
    
    Returns:
        Normalized EEG data
    """
    normalized_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        if method == 'robust':
            # Robust scaling (less sensitive to outliers)
            median = np.median(data[i])
            q75, q25 = np.percentile(data[i], [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                normalized_data[i] = (data[i] - median) / iqr
            else:
                normalized_data[i] = data[i] - median
        else:
            # Standard scaling
            mean = np.mean(data[i])
            std = np.std(data[i])
            if std > 0:
                normalized_data[i] = (data[i] - mean) / std
            else:
                normalized_data[i] = data[i] - mean
    
    return normalized_data


def resample_data(data: np.ndarray, original_sfreq: float, 
                  target_sfreq: float) -> np.ndarray:
    """
    Resample EEG data to target sampling frequency.
    
    Args:
        data: EEG data (channels, samples)
        original_sfreq: Original sampling frequency
        target_sfreq: Target sampling frequency
    
    Returns:
        Resampled EEG data
    """
    if original_sfreq == target_sfreq:
        return data
    
    # Calculate resampling ratio
    num_samples = int(data.shape[1] * target_sfreq / original_sfreq)
    
    # Resample each channel
    resampled_data = np.zeros((data.shape[0], num_samples))
    for i in range(data.shape[0]):
        resampled_data[i] = signal.resample(data[i], num_samples)
    
    return resampled_data


# ============================================================================
# WINDOW GENERATION WITH LABELS
# ============================================================================

def create_windows_with_labels(data: np.ndarray, annotations: pd.DataFrame,
                                sfreq: float, window_size: float, 
                                overlap: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows from EEG data with seizure labels.
    
    Args:
        data: EEG data (channels, samples)
        annotations: DataFrame with seizure annotations
        sfreq: Sampling frequency
        window_size: Window size in seconds
        overlap: Overlap in seconds
    
    Returns:
        windows: Array of windows (num_windows, channels, window_samples)
        labels: Binary labels (num_windows,) - 1 if seizure present, 0 otherwise
        times: Start time of each window in seconds (num_windows,)
    """
    window_samples = int(window_size * sfreq)
    step_samples = int((window_size - overlap) * sfreq)
    
    # Check if recording is long enough for at least one window
    if data.shape[1] < window_samples:
        # Recording too short - return empty arrays with proper shape
        return np.array([]).reshape(0, data.shape[0], window_samples), np.array([]), np.array([])
    
    num_windows = (data.shape[1] - window_samples) // step_samples + 1
    
    # Safety check
    if num_windows <= 0:
        return np.array([]).reshape(0, data.shape[0], window_samples), np.array([]), np.array([])
    
    windows = np.zeros((num_windows, data.shape[0], window_samples))
    labels = np.zeros(num_windows, dtype=np.int32)
    times = np.zeros(num_windows)
    
    for i in range(num_windows):
        start_sample = i * step_samples
        end_sample = start_sample + window_samples
        
        # Extract window
        windows[i] = data[:, start_sample:end_sample]
        
        # Calculate window time range
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq
        times[i] = start_time
        
        # Check if any seizure overlaps with this window
        if len(annotations) > 0:
            for _, ann in annotations.iterrows():
                # Check temporal overlap
                if not (end_time <= ann['start_time'] or start_time >= ann['stop_time']):
                    labels[i] = 1
                    break
    
    return windows, labels, times

# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================

def preprocess_eeg_file(edf_path: str, csv_path: str, 
                        config: Config = Config()) -> Dict:
    """
    Complete preprocessing pipeline for a single EEG file.
    
    Args:
        edf_path: Path to EDF file
        csv_path: Path to annotation CSV
        config: Configuration object
    
    Returns:
        Dictionary containing:
            - windows: Preprocessed windows (num_windows, channels, samples)
            - labels: Binary labels (num_windows,)
            - times: Window start times (num_windows,)
            - channels: List of channel names
            - metadata: Recording metadata
    """
    # Load EDF file
    data, channels, sfreq = load_edf_file(edf_path, config.STANDARD_CHANNELS)
    
    if data is None:
        return None
    
    # Handle channel count mismatch - standardize to fixed number of channels
    target_channel_count = len(config.STANDARD_CHANNELS)
    current_channel_count = data.shape[0]
    
    if current_channel_count < target_channel_count:
        # Pad with zeros if we have fewer channels
        padding = np.zeros((target_channel_count - current_channel_count, data.shape[1]))
        data = np.vstack([data, padding])
        channels = channels + ['PADDING'] * (target_channel_count - current_channel_count)
    elif current_channel_count > target_channel_count:
        # Truncate if we have more channels (keep first target_channel_count)
        data = data[:target_channel_count, :]
        channels = channels[:target_channel_count]
    
    # Load annotations
    annotations = parse_annotations(csv_path)
    
    # Apply bandpass filter
    data = apply_bandpass_filter(data, sfreq, config.LOWCUT, config.HIGHCUT)
    
    # Apply notch filter (remove power line noise)
    data = apply_notch_filter(data, sfreq, config.NOTCH_FREQ)
    
    # Resample to target frequency
    if sfreq != config.TARGET_SAMPLING_RATE:
        data = resample_data(data, sfreq, config.TARGET_SAMPLING_RATE)
        sfreq = config.TARGET_SAMPLING_RATE
    
    # Normalize channels
    data = normalize_channels(data, method='robust')
    
    # Create windows with labels
    windows, labels, times = create_windows_with_labels(
        data, annotations, sfreq, 
        config.WINDOW_SIZE, config.WINDOW_OVERLAP
    )
    
    # Metadata
    metadata = {
        'file': os.path.basename(edf_path),
        'num_windows': len(windows),
        'num_seizure_windows': np.sum(labels),
        'duration_seconds': data.shape[1] / sfreq,
        'sampling_rate': sfreq,
        'num_channels': len(channels)
    }
    
    return {
        'windows': windows,
        'labels': labels,
        'times': times,
        'channels': channels,
        'metadata': metadata
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def get_file_paths(data_split: str = 'dev', corpus: str = 'tusz') -> List[Tuple[str, str]]:
    """
    Get all EDF and CSV file paths for a data split.
    
    Args:
        data_split: 'dev', 'train', or 'eval'
        corpus: 'tusz' or 'tuar'
    
    Returns:
        List of (edf_path, csv_path) tuples
    """
    if corpus == 'tusz':
        base_path = os.path.join(Config.TUSZ_PATH, 'edf', data_split)
    else:
        base_path = os.path.join(Config.TUAR_PATH, 'edf', data_split)
    
    file_pairs = []
    
    # Walk through directory structure
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.edf'):
                edf_path = os.path.join(root, file)
                csv_path = edf_path.replace('.edf', '.csv')
                
                # Check if CSV exists (some files may not have annotations)
                file_pairs.append((edf_path, csv_path))
    
    return file_pairs


def preprocess_dataset(data_split: str = 'dev', 
                       max_files: Optional[int] = None,
                       save_path: Optional[str] = None,
                       batch_size: int = 100) -> Dict:
    """
    Preprocess entire dataset split with memory-efficient batching.
    
    Args:
        data_split: 'dev', 'train', or 'eval'
        max_files: Maximum number of files to process (None = all)
        save_path: Path to save preprocessed data (None = don't save)
        batch_size: Number of files to process before saving (memory management)
    
    Returns:
        Dictionary containing all preprocessed windows and labels
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing {data_split.upper()} split")
    print(f"{'='*60}\n")
    
    file_pairs = get_file_paths(data_split)
    
    if max_files:
        file_pairs = file_pairs[:max_files]
    
    print(f"Found {len(file_pairs)} files to process")
    print(f"Processing in batches of {batch_size} files to manage memory\n")
    
    # Process in batches to avoid OOM
    num_batches = (len(file_pairs) + batch_size - 1) // batch_size
    batch_files = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(file_pairs))
        batch_pairs = file_pairs[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (files {start_idx+1}-{end_idx})...")
        
        batch_windows = []
        batch_labels = []
        batch_metadata = []
        skipped = 0
        
        for i, (edf_path, csv_path) in enumerate(batch_pairs):
            if (i + 1) % 10 == 0:
                print(f"  File {start_idx + i + 1}/{len(file_pairs)}...")
            
            result = preprocess_eeg_file(edf_path, csv_path)
            
            if result:
                batch_windows.append(result['windows'])
                batch_labels.append(result['labels'])
                batch_metadata.append(result['metadata'])
            else:
                skipped += 1
        
        if len(batch_windows) > 0:
            # Save batch
            batch_data = {
                'windows': np.concatenate(batch_windows, axis=0),
                'labels': np.concatenate(batch_labels, axis=0),
                'metadata': batch_metadata
            }
            
            # Save to temporary file
            if save_path:
                batch_path = save_path.replace('.npz', f'_batch{batch_idx}.npz')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez_compressed(batch_path, **batch_data)
                print(f"  ✓ Batch saved: {batch_path}")
            
            batch_files.append(batch_path if save_path else batch_data)
        
        print(f"  Batch complete. Skipped: {skipped}\n")
    
    # Combine all batches
    print("Combining all batches...")
    
    if save_path and len(batch_files) > 0:
        all_windows = []
        all_labels = []
        all_metadata = []
        
        for batch_file in batch_files:
            if isinstance(batch_file, str):
                data = np.load(batch_file, allow_pickle=True)
                all_windows.append(data['windows'])
                all_labels.append(data['labels'])
                all_metadata.extend(data['metadata'])
        
        # Concatenate and save final file
        final_windows = np.concatenate(all_windows, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        
        np.savez_compressed(
            save_path,
            windows=final_windows,
            labels=final_labels,
            metadata=all_metadata
        )
        
        # Clean up batch files
        for batch_file in batch_files:
            if isinstance(batch_file, str) and os.path.exists(batch_file):
                os.remove(batch_file)
        
        print(f"\n{'='*60}")
        print(f"Preprocessing Complete!")
        print(f"{'='*60}")
        print(f"Total windows: {len(final_windows):,}")
        print(f"Seizure windows: {np.sum(final_labels):,} ({100*np.mean(final_labels):.2f}%)")
        print(f"Background windows: {len(final_labels) - np.sum(final_labels):,}")
        print(f"Window shape: {final_windows.shape}")
        print(f"{'='*60}\n")
        print(f"Saved preprocessed data to: {save_path}\n")
        
        return {
            'windows': final_windows,
            'labels': final_labels,
            'metadata': all_metadata
        }
    
    return None

# ============================================================================
# MAIN - EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EEG SEIZURE DETECTION - PREPROCESSING PIPELINE")
    print("Ceribell Project")
    print("="*60 + "\n")
    
    # Test on a single file first
    print("Testing on single file...")
    file_pairs = get_file_paths('dev')
    
    if file_pairs:
        edf_path, csv_path = file_pairs[0]
        print(f"Processing: {os.path.basename(edf_path)}\n")
        
        result = preprocess_eeg_file(edf_path, csv_path)
        
        if result:
            print(f"✓ Successfully preprocessed!")
            print(f"  - Windows created: {result['metadata']['num_windows']}")
            print(f"  - Seizure windows: {result['metadata']['num_seizure_windows']}")
            print(f"  - Window shape: {result['windows'].shape}")
            print(f"  - Channels: {len(result['channels'])}")
    
    print("\n" + "="*60)
    print("Ready to preprocess full dataset!")
    print("="*60)
    print("\nUsage:")
    print("  # Preprocess dev split (first 100 files)")
    print("  data = preprocess_dataset('dev', max_files=100, save_path='./preprocessed/dev.npz')")
    print("\n  # Load preprocessed data")
    print("  data = np.load('./preprocessed/dev.npz')")
    print("  windows = data['windows']")
    print("  labels = data['labels']")