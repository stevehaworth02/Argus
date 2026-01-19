"""
Argus Seizure Type Classification - Preprocessing
==================================================
Extract and label seizure types from TUSZ data using 22-channel TCP montage.

Uses the SAME channel configuration as the successful binary detector!

Classes:
- Background: No seizure activity
- Focal Seizure: FNSZ, CPSZ, SPSZ (localized to one brain region)
- Generalized Seizure: GNSZ, ABSZ, TCSZ, TNSZ, MYSZ (affects whole brain)

Author: Argus Seizure Detection System
"""

import numpy as np
import mne
from pathlib import Path
from tqdm import tqdm
import pickle
from collections import Counter
from scipy import signal as scipy_signal
import warnings
warnings.filterwarnings('ignore')


class SeizureTypePreprocessor:
    """Preprocess TUSZ data for 3-class clinical seizure classification."""
    
    # Clinical seizure type mapping (3 classes)
    SEIZURE_TYPES = {
        # Background
        'bckg': 0,
        
        # Focal seizures
        'fnsz': 1,  # Focal non-specific
        'cpsz': 1,  # Complex partial
        'spsz': 1,  # Simple partial
        
        # Generalized seizures
        'gnsz': 2,  # Generalized non-specific
        'absz': 2,  # Absence
        'tcsz': 2,  # Tonic-clonic
        'tnsz': 2,  # Tonic
        'mysz': 2,  # Myoclonic
    }
    
    TYPE_NAMES = ['Background', 'Focal Seizure', 'Generalized Seizure']
    
    # Standard 22-channel TCP bipolar montage (same as binary model!)
    BIPOLAR_CHANNELS = [
        'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
        'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FZ-CZ', 'CZ-PZ',
        'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'
    ]
    
    def __init__(self, tusz_path, output_path, window_duration=10, 
                 target_sample_rate=200):
        """
        Initialize preprocessor.
        
        Args:
            tusz_path: Path to TUSZ edf directory
            output_path: Where to save preprocessed data
            window_duration: Window size in seconds
            target_sample_rate: Target sampling rate
        """
        self.tusz_path = Path(tusz_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.window_duration = window_duration
        self.target_sample_rate = target_sample_rate
        
        print(f"Seizure Type Preprocessor initialized (22-channel TCP montage):")
        print(f"  TUSZ path: {self.tusz_path}")
        print(f"  Output path: {self.output_path}")
        print(f"  Window duration: {self.window_duration}s")
        print(f"  Target sample rate: {self.target_sample_rate} Hz")
        print(f"  Channels: 22 (TCP bipolar montage)")
        print(f"  Number of classes: {len(self.SEIZURE_TYPES)}")
    
    
    def load_edf_with_target_channels(self, edf_file):
        """
        Load EDF file with target bipolar channels (same as binary model).
        Channels already exist as bipolar in TUSZ EDF files!
        
        Args:
            edf_file: Path to .edf file
            
        Returns:
            data: (22, samples) array
            sample_rate: Original sampling rate
        """
        try:
            # Load with MNE
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            
            channels = raw.ch_names
            sfreq = raw.info['sfreq']
            
            # Find intersection of available and target bipolar channels
            available_channels = [ch for ch in self.BIPOLAR_CHANNELS if ch in channels]
            
            if len(available_channels) < 15:  # Need at least 15 of 22 channels
                return None, None
            
            # Pick available channels
            raw.pick_channels(available_channels)
            data = raw.get_data()
            
            # Pad to 22 channels if needed
            if data.shape[0] < 22:
                padding = np.zeros((22 - data.shape[0], data.shape[1]))
                data = np.vstack([data, padding])
            elif data.shape[0] > 22:
                data = data[:22, :]
            
            return data, sfreq
            
        except Exception as e:
            return None, None
    
    def load_seizure_annotations(self, edf_file):
        """
        Load seizure type annotations from CSV file.
        
        Returns:
            List of (start_time, end_time, seizure_type) tuples
        """
        csv_file = str(edf_file).replace('.edf', '.csv')
        
        if not Path(csv_file).exists():
            return []
        
        annotations = []
        
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('#') or line.startswith('channel,'):
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 4:
                        start_time = float(parts[1])
                        stop_time = float(parts[2])
                        label = parts[3].lower()
                        
                        # Store all annotations (including bckg)
                        annotations.append((start_time, stop_time, label))
        except:
            pass
        
        return annotations
    
    def get_window_type(self, window_start, window_end, annotations):
        """
        Determine seizure type for a window based on annotations.
        
        Priority: If window overlaps with ANY seizure, label as that seizure type.
        If multiple seizures overlap, use the one with most overlap.
        If no seizure, label as BCKG.
        
        Args:
            window_start: Window start time
            window_end: Window end time
            annotations: List of (start, end, type) annotations
        
        Returns:
            seizure_type: String label ('bckg', 'fnsz', etc.)
        """
        max_overlap = 0
        dominant_type = 'bckg'
        
        for ann_start, ann_end, ann_type in annotations:
            # Skip background annotations
            if ann_type == 'bckg':
                continue
            
            # Calculate overlap
            overlap_start = max(window_start, ann_start)
            overlap_end = min(window_end, ann_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                dominant_type = ann_type
        
        # If no seizure overlap, it's background
        return dominant_type
    
    def process_file(self, edf_file):
        """Process single EDF file by loading bipolar channels directly."""
        try:
            # Load annotations
            annotations = self.load_seizure_annotations(edf_file)
            
            if not annotations:
                return []
            
            # Load EDF file with target channels (they already exist as bipolar!)
            data, sample_rate = self.load_edf_with_target_channels(edf_file)
            
            # Skip if can't load enough channels
            if data is None:
                return []
            
            # Get duration
            duration = data.shape[1] / sample_rate
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                num_samples_new = int(data.shape[1] * self.target_sample_rate / sample_rate)
                data = scipy_signal.resample(data, num_samples_new, axis=1)
                sample_rate = self.target_sample_rate
            
            # Extract windows
            samples_per_window = int(self.window_duration * sample_rate)
            num_windows = int(duration / self.window_duration)
            
            windows = []
            
            for i in range(num_windows):
                start_sample = i * samples_per_window
                end_sample = start_sample + samples_per_window
                
                if end_sample > data.shape[1]:
                    break
                
                window_data = data[:, start_sample:end_sample]
                
                # Get time range
                window_start_time = i * self.window_duration
                window_end_time = window_start_time + self.window_duration
                
                # Get seizure type for this window
                seizure_type = self.get_window_type(
                    window_start_time, 
                    window_end_time, 
                    annotations
                )
                
                # Skip unknown types
                if seizure_type not in self.SEIZURE_TYPES:
                    continue
                
                windows.append({
                    'data': window_data,
                    'type': seizure_type,
                    'label': self.SEIZURE_TYPES[seizure_type],
                    'file': str(edf_file.name),
                    'window_idx': i,
                    'time': window_start_time
                })
            
            return windows
            
        except Exception as e:
            return []
    
    def process_dataset(self, split='train', batch_size=50):
        """Process entire dataset split with memory-efficient batching."""
        print(f"\nProcessing {split} split for seizure type classification...")
        
        # Find all EDF files
        split_path = self.tusz_path / split
        edf_files = list(split_path.rglob('*.edf'))
        
        print(f"Found {len(edf_files)} EDF files")
        print(f"Processing in batches of {batch_size} files...")
        
        # Create temp directory
        temp_dir = self.output_path / f"temp_{split}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Count per type
        type_counts = {t: 0 for t in self.TYPE_NAMES}
        
        all_windows = []
        batch_num = 0
        
        for i in tqdm(range(0, len(edf_files), batch_size), desc=f"Processing {split}"):
            batch_files = edf_files[i:i+batch_size]
            batch_windows = []
            
            for edf_file in batch_files:
                windows = self.process_file(edf_file)
                batch_windows.extend(windows)
                
                # Update counts
                for w in windows:
                    type_counts[self.TYPE_NAMES[w['label']]] += 1
            
            # Save batch
            if batch_windows:
                batch_file = temp_dir / f"batch_{batch_num}.pkl"
                with open(batch_file, 'wb') as f:
                    pickle.dump(batch_windows, f)
                batch_num += 1
            
            # Show running counts every 50 files
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"\nRunning counts after {i + batch_size} files:")
                for type_name, count in type_counts.items():
                    print(f"  {type_name}: {count}")
        
        # Combine all batches
        print(f"\nCombining {batch_num} batches...")
        
        for batch_file in temp_dir.glob("batch_*.pkl"):
            with open(batch_file, 'rb') as f:
                windows = pickle.load(f)
                all_windows.extend(windows)
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"\nFinal counts for {split}:")
        for type_name in self.TYPE_NAMES:
            count = sum(1 for w in all_windows if self.TYPE_NAMES[w['label']] == type_name)
            print(f"  {type_name}: {count}")
        
        # Save complete dataset
        output_file = self.output_path / f"{split}_seizure_types.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(all_windows, f)
        
        print(f"\nSaved {len(all_windows)} windows to {output_file}")
        
        return all_windows
    
    def create_balanced_dataset(self, split='train'):
        """
        Create class-balanced dataset.
        
        Since BCKG dominates, we'll undersample it.
        """
        # Load full dataset
        input_file = self.output_path / f"{split}_seizure_types.pkl"
        
        with open(input_file, 'rb') as f:
            all_windows = pickle.load(f)
        
        # Separate by type
        by_type = {t: [] for t in range(len(self.SEIZURE_TYPES))}
        
        for w in all_windows:
            by_type[w['label']].append(w)
        
        print(f"\nOriginal class distribution:")
        for i, type_name in enumerate(self.TYPE_NAMES):
            print(f"  {type_name}: {len(by_type[i])}")
        
        # Find max non-background count
        non_bckg_max = max(len(by_type[i]) for i in range(1, len(self.SEIZURE_TYPES)))
        
        # Balance: use all seizures, undersample background to 2x seizure count
        balanced_windows = []
        
        for i in range(len(self.SEIZURE_TYPES)):
            if i == 0:  # Background
                # Undersample to 2x the largest seizure class
                target = min(len(by_type[0]), non_bckg_max * 2)
                selected = np.random.choice(len(by_type[0]), target, replace=False)
                balanced_windows.extend([by_type[0][idx] for idx in selected])
            else:
                # Keep all seizure samples
                balanced_windows.extend(by_type[i])
        
        print(f"\nBalanced class distribution:")
        balanced_counts = Counter(w['label'] for w in balanced_windows)
        for i, type_name in enumerate(self.TYPE_NAMES):
            print(f"  {type_name}: {balanced_counts[i]}")
        
        # Save balanced dataset
        output_file = self.output_path / f"{split}_seizure_types_balanced.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(balanced_windows, f)
        
        print(f"\nSaved {len(balanced_windows)} balanced windows to {output_file}")
        
        return balanced_windows


def main():
    """Main preprocessing pipeline."""
    
    # CHANGE THESE PATHS
    tusz_data_path = "/mnt/c/Users/0218s/Desktop/Ceribell-SZ-DTCTR/data/tusz/edf"
    output_data_path = "/mnt/c/Users/0218s/Desktop/Argus/data/seizure_types"
    
    preprocessor = SeizureTypePreprocessor(
        tusz_path=tusz_data_path,
        output_path=output_data_path,
        window_duration=10,
        target_sample_rate=200
        # Now uses 22-channel TCP bipolar montage automatically!
    )
    
    # Process all splits
    for split in ['train', 'dev', 'eval']:
        preprocessor.process_dataset(split)
        preprocessor.create_balanced_dataset(split)
    
    print("\n✓ Seizure type preprocessing complete!")


if __name__ == "__main__":
    main()