"""
Debug script to check what channels are in TUAR files
"""
import os
import mne
mne.set_log_level('ERROR')

# TUAR path
TUAR_PATH = "C:/Users/0218s/Desktop/Ceribell-SZ-DTCTR/data/tuar/edf/01_tcp_ar"

# Standard channels we're looking for
STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FZ-CZ', 'CZ-PZ',
    'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'
]

# Find first EDF file
print("Searching for TUAR files...")
for root, dirs, files in os.walk(TUAR_PATH):
    for file in files:
        if file.endswith('.edf'):
            edf_path = os.path.join(root, file)
            print(f"\nFound: {file}")
            print(f"Path: {edf_path}")
            
            # Load and check channels
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                channels = raw.ch_names
                
                print(f"\nFile has {len(channels)} channels:")
                for i, ch in enumerate(channels):
                    print(f"  {i+1}. '{ch}'")
                
                print(f"\n Looking for {len(STANDARD_CHANNELS)} standard channels:")
                for i, target in enumerate(STANDARD_CHANNELS[:5], 1):  # Show first 5
                    print(f"  {i}. '{target}'")
                print(f"  ... and {len(STANDARD_CHANNELS)-5} more")
                
                # Try matching
                matches = []
                for target in STANDARD_CHANNELS:
                    for avail in channels:
                        if avail.upper().strip() == target.upper().strip():
                            matches.append((target, avail))
                            break
                
                print(f"\nDirect matches found: {len(matches)}")
                if matches:
                    for target, avail in matches[:5]:
                        print(f"  '{target}' -> '{avail}'")
                
                # Try without dashes
                matches_flexible = []
                for target in STANDARD_CHANNELS:
                    target_clean = target.replace('-', '').replace(' ', '').upper()
                    for avail in channels:
                        avail_clean = avail.replace('-', '').replace(' ', '').upper()
                        if avail_clean == target_clean:
                            matches_flexible.append((target, avail))
                            break
                
                print(f"\nFlexible matches (no dashes): {len(matches_flexible)}")
                if matches_flexible:
                    for target, avail in matches_flexible[:5]:
                        print(f"  '{target}' -> '{avail}'")
                
            except Exception as e:
                print(f"Error loading file: {e}")
            
            # Only check first file
            break
    break

print("\nDone!")
