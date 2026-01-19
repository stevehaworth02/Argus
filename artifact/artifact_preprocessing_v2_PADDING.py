"""
EEG Artifact Detection - Data Preprocessing Pipeline
Ceribell Project

This module handles:
- EDF file loading from TUAR corpus
- Multi-channel EEG preprocessing
- Sliding window generation with binary artifact labels
- Artifact annotation parsing from CSV files
- PyTorch Dataset integration

Author: Ceribell Seizure Detector Project
Dataset: Temple University Hospital EEG Artifact Corpus (TUAR) v3.0.1
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
    """Central configuration for artifact preprocessing pipeline"""
    
    # Data paths
    DATA_ROOT = "C:/Users/0218s/Desktop/Ceribell-SZ-DTCTR/data"
    TUAR_PATH = os.path.join(DATA_ROOT, "tuar")
    
    # Preprocessing parameters (MATCH seizure detector exactly)
    SAMPLING_RATE = 250  # Hz (original TUAR sampling rate)
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
    
    # Artifact types in TUAR
    ARTIFACT_TYPES = ['eyem', 'chew', 'shiv', 'musc', 'elec']
    
    # Montage to use (01_tcp_ar matches seizure detector)
    MONTAGE = '01_tcp_ar'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_edf_file(edf_path: str, target_channels: List[str] = None) -> Tuple[np.ndarray, List[str], float]:
    """
    Load EEG data from EDF file using MNE.
    Pads missing channels with zeros to ensure consistent shape.
    
    Args:
        edf_path: Path to .edf file
        target_channels: List of channels to load (None = load all)
    
    Returns:
        data: EEG data array (channels, samples) - always has len(target_channels) rows
        channels: List of channel names (ordered as target_channels)
        sfreq: Sampling frequency
    """
    try:
        # Load EDF file
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Get channel names and sampling frequency
        available_channels = raw.ch_names
        sfreq = raw.info['sfreq']
        
        # If target channels specified, ensure consistent output
        if target_channels:
            # Find which target channels are available
            present_channels = [ch for ch in target_channels if ch in available_channels]
            
            if len(present_channels) == 0:
                # No target channels found - skip file
                return None, None, None
            
            # Pick available channels
            raw.pick_channels(present_channels)
            data_available = raw.get_data()  # Shape: (n_present, n_samples)
            
            # Create full data array with zeros for missing channels
            n_samples = data_available.shape[1]
            data = np.zeros((len(target_channels), n_samples), dtype=np.float32)
            
            # Fill in the available channels at their correct positions
            for i, target_ch in enumerate(target_channels):
                if target_ch in present_channels:
                    # Find index of this channel in present_channels
                    present_idx = present_channels.index(target_ch)
                    data[i, :] = data_available[present_idx, :]
                # else: leave as zeros (missing channel)
            
            channels = target_channels  # Return full target list
        else:
            # No target channels - return all available
            data = raw.get_data()
            channels = available_channels
        
        return data, channels, sfreq
        
    except Exception as e:
        # Return None for files that can't be loaded
        return None, None, None


def parse_artifact_annotations(csv_path: str) -> pd.DataFrame:
    """
    Parse artifact annotations from TUAR CSV file.
    
    CSV format:
    channel,start_time,stop_time,label,confidence
    FP1-F7,22.9737,30.0688,eyem,1.000000
    
    Args:
        csv_path: Path to .csv annotation file
    
    Returns:
        DataFrame with columns: channel, start_time, stop_time, label, confidence
    """
    try:
        if not os.path.exists(csv_path):
            # No annotations = all background (clean)
            return pd.DataFrame(columns=['channel', 'start_time', 'stop_time', 'label', 'confidence'])
        
        # Read CSV, skipping comment lines (lines starting with #)
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
        
        # Convert time columns to float
        df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
        df['stop_time'] = pd.to_numeric(df['stop_time'], errors='coerce')
        
        # Filter out any rows with NaN times
        df = df.dropna(subset=['start_time', 'stop_time'])
        
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


def resample_data(data: np.ndarray, original_sfreq: float, target_sfreq: float) -> np.ndarray:
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
    
    num_samples = int(data.shape[1] * target_sfreq / original_sfreq)
    resampled_data = np.zeros((data.shape[0], num_samples))
    
    for i in range(data.shape[0]):
        resampled_data[i] = signal.resample(data[i], num_samples)
    
    return resampled_data


def create_windows_with_artifact_labels(data: np.ndarray, 
                                        annotations: pd.DataFrame,
                                        sfreq: float,
                                        window_size: float = Config.WINDOW_SIZE,
                                        stride: float = Config.WINDOW_OVERLAP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows with binary artifact labels.
    
    BINARY LABELING STRATEGY:
    - Label = 1 (ARTIFACT) if ANY artifact annotation overlaps with window
    - Label = 0 (CLEAN) if no artifact annotations in window
    
    Args:
        data: EEG data (channels, samples)
        annotations: DataFrame with artifact annotations
        sfreq: Sampling frequency
        window_size: Window duration in seconds
        stride: Stride between windows in seconds
    
    Returns:
        windows: Array of windows (num_windows, channels, window_samples)
        labels: Binary labels (num_windows,) - 0=clean, 1=artifact
        times: Start time of each window (num_windows,)
    """
    window_samples = int(window_size * sfreq)
    stride_samples = int(stride * sfreq)
    
    num_windows = (data.shape[1] - window_samples) // stride_samples + 1
    
    windows = []
    labels = []
    times = []
    
    for i in range(num_windows):
        start_sample = i * stride_samples
        end_sample = start_sample + window_samples
        
        # Extract window
        window = data[:, start_sample:end_sample]
        
        # Calculate window time boundaries
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq
        
        # Check if ANY artifact annotation overlaps with this window
        is_artifact = False
        
        if not annotations.empty:
            # Check for overlap: annotation overlaps if it starts before window ends
            # AND ends after window starts
            overlaps = (
                (annotations['start_time'] < end_time) &
                (annotations['stop_time'] > start_time)
            )
            
            if overlaps.any():
                is_artifact = True
        
        windows.append(window)
        labels.append(1 if is_artifact else 0)  # 1 = artifact, 0 = clean
        times.append(start_time)
    
    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(times, dtype=np.float32)
    )


# ============================================================================
# FILE PREPROCESSING
# ============================================================================

def preprocess_artifact_file(edf_path: str, csv_path: str, config=Config) -> Optional[Dict]:
    """
    Preprocess a single TUAR EDF file with artifact annotations.
    Missing channels are padded with zeros to ensure consistent 22-channel output.
    
    Args:
        edf_path: Path to .edf file
        csv_path: Path to .csv annotation file
        config: Configuration object
    
    Returns:
        Dictionary with windows, labels, times, channels, and metadata
    """
    # Load EDF file (missing channels are padded with zeros)
    data, channels, sfreq = load_edf_file(edf_path, config.STANDARD_CHANNELS)
    
    if data is None:
        return None
    
    # Data now guaranteed to have shape (22, n_samples) due to padding
    assert data.shape[0] == len(config.STANDARD_CHANNELS), \
        f"Expected {len(config.STANDARD_CHANNELS)} channels, got {data.shape[0]}"
    
    # Parse artifact annotations
    annotations = parse_artifact_annotations(csv_path)
    
    # Apply preprocessing (SAME as seizure detector for consistency)
    # 1. Bandpass filter
    data = apply_bandpass_filter(data, sfreq, config.LOWCUT, config.HIGHCUT)
    
    # 2. Notch filter (remove 60 Hz line noise)
    data = apply_notch_filter(data, sfreq, config.NOTCH_FREQ)
    
    # 3. Resample to target frequency
    if sfreq != config.TARGET_SAMPLING_RATE:
        data = resample_data(data, sfreq, config.TARGET_SAMPLING_RATE)
        sfreq = config.TARGET_SAMPLING_RATE
    
    # 4. Normalize channels
    data = normalize_channels(data, method='robust')
    
    # 5. Create windows with binary artifact labels
    windows, labels, times = create_windows_with_artifact_labels(
        data, annotations, sfreq, 
        config.WINDOW_SIZE, config.WINDOW_OVERLAP
    )
    
    # Metadata
    metadata = {
        'file': os.path.basename(edf_path),
        'num_windows': len(windows),
        'num_artifact_windows': np.sum(labels),
        'num_clean_windows': len(labels) - np.sum(labels),
        'duration_seconds': data.shape[1] / sfreq,
        'sampling_rate': sfreq,
        'num_channels': len(channels),
        'num_annotations': len(annotations)
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

def get_file_paths(montage: str = '01_tcp_ar') -> List[Tuple[str, str]]:
    """
    Get all EDF and CSV file paths for TUAR corpus.
    
    Args:
        montage: Montage directory ('01_tcp_ar', '02_tcp_le', '03_tcp_ar_a')
    
    Returns:
        List of (edf_path, csv_path) tuples
    """
    base_path = os.path.join(Config.TUAR_PATH, 'edf', montage)
    
    file_pairs = []
    
    # Walk through directory structure
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.edf'):
                edf_path = os.path.join(root, file)
                # CSV has same name as EDF
                csv_path = edf_path.replace('.edf', '.csv')
                
                # Only add if both files exist
                if os.path.exists(csv_path):
                    file_pairs.append((edf_path, csv_path))
    
    return file_pairs


def preprocess_dataset(montage: str = '01_tcp_ar',
                       max_files: Optional[int] = None,
                       save_path: Optional[str] = None,
                       batch_size: int = 50) -> Dict:
    """
    Preprocess entire TUAR dataset with memory-efficient batching.
    
    Args:
        montage: Montage to process ('01_tcp_ar', '02_tcp_le', '03_tcp_ar_a')
        max_files: Maximum number of files to process (None = all)
        save_path: Path to save preprocessed data (None = don't save)
        batch_size: Number of files to process before saving (memory management)
    
    Returns:
        Dictionary containing all preprocessed windows and labels
    """
    print(f"\n{'='*70}")
    print(f"Preprocessing TUAR Artifact Corpus - {montage}")
    print(f"{'='*70}\n")
    
    file_pairs = get_file_paths(montage)
    
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
            
            result = preprocess_artifact_file(edf_path, csv_path)
            
            if result:
                batch_windows.append(result['windows'])
                batch_labels.append(result['labels'])
                batch_metadata.append(result['metadata'])
            else:
                skipped += 1
        
        if len(batch_windows) > 0:
            # Validate all windows have same shape before concatenating
            window_shapes = [w.shape for w in batch_windows]
            if len(set(window_shapes)) > 1:
                print(f"  Warning: Inconsistent window shapes in batch: {set(window_shapes)}")
                # Filter to only keep windows with the most common shape
                from collections import Counter
                most_common_shape = Counter(window_shapes).most_common(1)[0][0]
                print(f"  Keeping only windows with shape: {most_common_shape}")
                
                filtered_windows = []
                filtered_labels = []
                filtered_metadata = []
                
                for w, l, m in zip(batch_windows, batch_labels, batch_metadata):
                    if w.shape == most_common_shape:
                        filtered_windows.append(w)
                        filtered_labels.append(l)
                        filtered_metadata.append(m)
                
                batch_windows = filtered_windows
                batch_labels = filtered_labels
                batch_metadata = filtered_metadata
                
                if len(batch_windows) == 0:
                    print(f"  No valid windows after filtering!")
                    continue
            
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
        
        print(f"  Batch complete. Processed: {len(batch_windows)}, Skipped: {skipped}\n")
    
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
        
        print(f"\n{'='*70}")
        print(f"Preprocessing Complete!")
        print(f"{'='*70}")
        print(f"Total windows: {len(final_windows):,}")
        print(f"Artifact windows: {np.sum(final_labels):,} ({100*np.mean(final_labels):.2f}%)")
        print(f"Clean windows: {len(final_labels) - np.sum(final_labels):,} ({100*(1-np.mean(final_labels)):.2f}%)")
        print(f"Window shape: {final_windows.shape}")
        print(f"{'='*70}\n")
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
    print("\n" + "="*70)
    print("EEG ARTIFACT DETECTION - PREPROCESSING PIPELINE")
    print("Ceribell Project")
    print("="*70 + "\n")
    
    # Test on a single file first
    print("Testing on single file...")
    file_pairs = get_file_paths('01_tcp_ar')
    
    if file_pairs:
        edf_path, csv_path = file_pairs[0]
        print(f"Processing: {os.path.basename(edf_path)}\n")
        
        result = preprocess_artifact_file(edf_path, csv_path)
        
        if result:
            print(f"✓ Successfully preprocessed!")
            print(f"  - Windows created: {result['metadata']['num_windows']}")
            print(f"  - Artifact windows: {result['metadata']['num_artifact_windows']}")
            print(f"  - Clean windows: {result['metadata']['num_clean_windows']}")
            print(f"  - Window shape: {result['windows'].shape}")
            print(f"  - Channels: {len(result['channels'])}")
    
    print("\n" + "="*70)
    print("Ready to preprocess full dataset!")
    print("="*70)
    print("\nUsage:")
    print("  # Preprocess TUAR (first 100 files)")
    print("  data = preprocess_dataset('01_tcp_ar', max_files=100, save_path='./preprocessed/artifacts.npz')")
    print("\n  # Load preprocessed data")
    print("  data = np.load('./preprocessed/artifacts.npz')")
    print("  windows = data['windows']")
    print("  labels = data['labels']")
