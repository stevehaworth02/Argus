"""
Test bipolar construction from TUAR referential data
"""
import os
import sys
import numpy as np
sys.path.append('.')
from artifact_preprocessing import load_edf_file, Config

# Test file
test_file = "C:/Users/0218s/Desktop/Ceribell-SZ-DTCTR/data/tuar/edf/01_tcp_ar/aaaaaaju_s005_t000.edf"

print("="*70)
print("TESTING BIPOLAR CONSTRUCTION")
print("="*70)

print(f"\nTest file: {os.path.basename(test_file)}")
print(f"\nTarget channels (first 5): {Config.STANDARD_CHANNELS[:5]}")
print(f"Total target channels: {len(Config.STANDARD_CHANNELS)}")

print("\n" + "="*70)
print("LOADING WITH DEBUG OUTPUT")
print("="*70)

# Load with debug enabled
data, channels, sfreq = load_edf_file(test_file, Config.STANDARD_CHANNELS, debug=True)

if data is not None:
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nLoaded data shape: {data.shape}")
    print(f"  Expected: ({len(Config.STANDARD_CHANNELS)}, N_SAMPLES)")
    print(f"  Actual: ({data.shape[0]}, {data.shape[1]})")
    print(f"\nChannels returned: {len(channels)}")
    print(f"Sampling rate: {sfreq} Hz")
    
    # Check which channels have data vs zeros
    non_zero_channels = []
    zero_channels = []
    
    for i, ch in enumerate(channels):
        if np.any(data[i] != 0):
            non_zero_channels.append(ch)
        else:
            zero_channels.append(ch)
    
    print(f"\nChannels with data: {len(non_zero_channels)}")
    print(f"  Examples: {non_zero_channels[:5]}")
    
    print(f"\nChannels zero-padded: {len(zero_channels)}")
    if zero_channels:
        print(f"  Examples: {zero_channels[:5]}")
    
    print("\n" + "="*70)
    print("✓ BIPOLAR CONSTRUCTION WORKS!")
    print("="*70)
    
else:
    print("\n" + "="*70)
    print("❌ FAILED TO LOAD")
    print("="*70)
    print("Could not construct bipolar channels from referential data")

print("\nDone!")
