"""
Helper script to run artifact detector cross-validation on seizure data
This finds your preprocessed seizure data and runs the test
"""

import os
import glob

print("="*70)
print("FINDING SEIZURE DATA")
print("="*70)

# Possible locations for seizure data
possible_paths = [
    '../training/data/preprocessed_windows.npz',
    '../training/preprocessed/dev.npz',
    '../training/preprocessed/train.npz',
    '../training/data/dev.npz',
    '../data/preprocessed_windows.npz',
]

# Find existing data files
found_files = []
for path in possible_paths:
    if os.path.exists(path):
        found_files.append(path)
        print(f"✓ Found: {path}")

# Also search in training directory
training_dir = '../training'
if os.path.exists(training_dir):
    npz_files = glob.glob(os.path.join(training_dir, '**/*.npz'), recursive=True)
    for npz_file in npz_files:
        if npz_file not in found_files:
            found_files.append(npz_file)
            print(f"✓ Found: {npz_file}")

if not found_files:
    print("\n❌ No seizure data found!")
    print("\nPlease preprocess your TUSZ data first:")
    print("  cd ../training")
    print("  python run_preprocessing.py --split dev --max_files 50")
    print("\nOr specify the correct path to your .npz file")
else:
    print(f"\n{'='*70}")
    print("FOUND SEIZURE DATA FILES")
    print("="*70)
    for i, f in enumerate(found_files, 1):
        print(f"{i}. {f}")
    
    # Use the first one found
    seizure_data = found_files[0]
    artifact_model = './models/best_artifact_detector.pth'
    
    print(f"\n{'='*70}")
    print("RUNNING CROSS-VALIDATION")
    print("="*70)
    print(f"\nUsing:")
    print(f"  • Artifact model: {artifact_model}")
    print(f"  • Seizure data: {seizure_data}")
    
    # Build command
    cmd = f'python test_artifact_on_seizures.py --artifact_model {artifact_model} --seizure_data {seizure_data}'
    
    print(f"\nCommand:")
    print(f"  {cmd}")
    print("\n" + "="*70)
    
    # Run it
    os.system(cmd)
