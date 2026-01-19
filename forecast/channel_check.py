"""
Comprehensive Diagnostic: Check ALL TUSZ files for forecasting viability
========================================================================
Checks different pre-ictal window sizes: 5, 10, 15, 30 minutes
"""

import pyedflib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def load_seizure_times(edf_file):
    """Load seizure times from CSV file."""
    csv_file = str(edf_file).replace('.edf', '.csv')
    
    if not Path(csv_file).exists():
        return []
    
    seizures = []
    seizure_events = {}
    
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
                    label = parts[3]
                    
                    if label != 'bckg':
                        seizure_events[(start_time, stop_time)] = True
        
        seizures = list(seizure_events.keys())
    except:
        pass
    
    return seizures


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE FORECASTING VIABILITY CHECK - ALL FILES")
    print("="*70)
    
    data_path = Path("/mnt/c/Users/0218s/Desktop/Ceribell-SZ-DTCTR/data/tusz/edf/train")
    edf_files = list(data_path.rglob('*.edf'))
    
    print(f"\nChecking ALL {len(edf_files)} EDF files...")
    print("This may take 10-15 minutes...\n")
    
    stats = {
        'total_files': 0,
        'has_32_channels': 0,
        'has_seizures': 0,
    }
    
    # Check multiple window sizes
    windows = [5, 10, 15, 20, 30]
    viable_by_window = {w: 0 for w in windows}
    
    durations = []
    seizure_timings = []
    channel_counts = []
    
    for edf_file in tqdm(edf_files, desc="Processing"):
        try:
            with pyedflib.EdfReader(str(edf_file)) as f:
                n_channels = f.signals_in_file
                duration = f.file_duration
            
            stats['total_files'] += 1
            durations.append(duration)
            channel_counts.append(n_channels)
            
            # Only check 32-channel files
            if n_channels != 32:
                continue
            
            stats['has_32_channels'] += 1
            
            # Check for seizures
            seizures = load_seizure_times(edf_file)
            
            if not seizures:
                continue
            
            stats['has_seizures'] += 1
            
            # Check viability for each window size
            for seiz_start, seiz_end in seizures:
                seizure_timings.append(seiz_start)
                
                for window_min in windows:
                    if seiz_start >= window_min * 60:
                        viable_by_window[window_min] += 1
                
                break  # Only count first seizure
                
        except Exception as e:
            continue
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS - ALL FILES")
    print("="*70)
    
    print(f"\nTotal files processed:     {stats['total_files']}")
    print(f"Files with 32 channels:    {stats['has_32_channels']} ({stats['has_32_channels']/stats['total_files']*100:.1f}%)")
    print(f"Files with seizures:       {stats['has_seizures']}")
    
    print(f"\nVIABLE FILES by pre-ictal window size:")
    print("-" * 70)
    for window in windows:
        count = viable_by_window[window]
        if count > 0:
            print(f"  {window:2d}-minute window: {count:4d} files  ✓")
        else:
            print(f"  {window:2d}-minute window: {count:4d} files  ✗")
    
    print(f"\nRecording durations (ALL files):")
    print(f"  Mean:   {np.mean(durations)/60:.1f} minutes")
    print(f"  Median: {np.median(durations)/60:.1f} minutes")
    print(f"  Min:    {np.min(durations)/60:.1f} minutes")
    print(f"  Max:    {np.max(durations)/60:.1f} minutes")
    print(f"  25th percentile: {np.percentile(durations, 25)/60:.1f} minutes")
    print(f"  75th percentile: {np.percentile(durations, 75)/60:.1f} minutes")
    print(f"  90th percentile: {np.percentile(durations, 90)/60:.1f} minutes")
    
    if seizure_timings:
        print(f"\nSeizure timing (when seizures occur in recordings):")
        print(f"  Mean:   {np.mean(seizure_timings)/60:.1f} minutes")
        print(f"  Median: {np.median(seizure_timings)/60:.1f} minutes")
        print(f"  25th percentile: {np.percentile(seizure_timings, 25)/60:.1f} minutes")
        print(f"  75th percentile: {np.percentile(seizure_timings, 75)/60:.1f} minutes")
        print(f"  90th percentile: {np.percentile(seizure_timings, 90)/60:.1f} minutes")
    
    print(f"\nChannel count distribution:")
    counts = Counter(channel_counts)
    for channels, freq in sorted(counts.items())[:10]:
        print(f"  {channels:3d} channels: {freq:4d} files ({freq/len(channel_counts)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Find best window
    best_window = None
    best_count = 0
    for window in windows:
        if viable_by_window[window] > best_count:
            best_window = window
            best_count = viable_by_window[window]
    
    if best_window:
        print(f"\n✓ BEST OPTION: Use {best_window}-minute pre-ictal window")
        print(f"  This gives you {best_count} viable files with 32 channels")
        print(f"  Estimated ~{best_count * 100} pre-ictal windows (assuming ~100 windows per file)")
    else:
        print("\n✗ NO VIABLE OPTIONS FOUND")
        print("  Consider:")
        print("    - Using different channel count")
        print("    - Reducing window size below 5 minutes")
        print("    - Using a different dataset")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()