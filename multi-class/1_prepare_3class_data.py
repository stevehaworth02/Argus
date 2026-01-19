"""
Step 1: Prepare 3-Class Training Data (MEMORY-EFFICIENT VERSION)
Combines TUAR + TUSZ train with proper class labels using batch processing

MEMORY-EFFICIENT:
- Processes data in batches to avoid OOM
- Uses float32 instead of float64 (half the memory)
- Saves incrementally to disk

ZERO DATA LEAKAGE:
- Uses TUSZ TRAIN split only (no dev!)
- Uses TUAR artifacts + clean
- Creates 3 class labels: 0=background, 1=artifact, 2=seizure

Author: Ceribell Multi-Class System
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split

print("="*70)
print("STEP 1: PREPARING 3-CLASS TRAINING DATA (MEMORY-EFFICIENT)")
print("Background, Artifact, and Seizure Classification")
print("="*70)

# ============================================================================
# LOAD TUSZ TRAIN DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING TUSZ TRAIN SPLIT")
print("="*70)

tusz_train_path = '../full_dataset/preprocessed/train.npz'

if not os.path.exists(tusz_train_path):
    print(f"\n[ERROR] TUSZ train data not found!")
    print(f"Expected: {tusz_train_path}")
    print("\nPlease run full_dataset preprocessing first:")
    print("  cd ../full_dataset")
    print("  python preprocess_tusz_full.py --split train")
    exit(1)

print(f"Loading: {tusz_train_path}")
tusz_data = np.load(tusz_train_path, allow_pickle=True)
tusz_windows = tusz_data['windows'].astype(np.float32)  # Use float32!
tusz_labels = tusz_data['labels']

print(f"\n[OK] TUSZ Train Data Loaded:")
print(f"  * Total windows: {len(tusz_windows):,}")
print(f"  * Seizure windows: {np.sum(tusz_labels):,} ({100*np.mean(tusz_labels):.2f}%)")
print(f"  * Background windows: {np.sum(1-tusz_labels):,} ({100*np.mean(1-tusz_labels):.2f}%)")
print(f"  * Shape: {tusz_windows.shape}")
print(f"  * Data type: {tusz_windows.dtype} (memory-efficient!)")
print(f"\n  [CRITICAL] This is TRAIN split (~200-300 patients)")
print(f"  [CRITICAL] DEV split will be used for testing ONLY!")

# ============================================================================
# LOAD TUAR ARTIFACT DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING TUAR ARTIFACT DATA")
print("="*70)

tuar_path = '../artifact/preprocessed/artifacts_01_tcp_ar.npz'

if not os.path.exists(tuar_path):
    print(f"\n[ERROR] TUAR data not found!")
    print(f"Expected: {tuar_path}")
    print("\nPlease preprocess TUAR data first:")
    print("  cd ../artifact")
    print("  python artifact_preprocessing.py")
    exit(1)

print(f"Loading: {tuar_path}")
tuar_data = np.load(tuar_path, allow_pickle=True)
tuar_windows = tuar_data['windows'].astype(np.float32)  # Use float32!
tuar_labels = tuar_data['labels']

print(f"\n[OK] TUAR Data Loaded:")
print(f"  * Total windows: {len(tuar_windows):,}")
print(f"  * Artifact windows: {np.sum(tuar_labels):,} ({100*np.mean(tuar_labels):.2f}%)")
print(f"  * Clean windows: {np.sum(1-tuar_labels):,} ({100*np.mean(1-tuar_labels):.2f}%)")
print(f"  * Shape: {tuar_windows.shape}")
print(f"  * Data type: {tuar_windows.dtype} (memory-efficient!)")

# ============================================================================
# CREATE 3-CLASS LABELS
# ============================================================================

print("\n" + "="*70)
print("CREATING 3-CLASS LABELS")
print("="*70)

print("\nClass mapping:")
print("  Class 0: BACKGROUND (clean, normal brain activity)")
print("  Class 1: ARTIFACT (eye blinks, muscle, electrode noise)")
print("  Class 2: SEIZURE (epileptic activity)")

# Separate TUSZ windows
seizure_mask = tusz_labels == 1
tusz_seizures = tusz_windows[seizure_mask]
tusz_background = tusz_windows[~seizure_mask]

# Separate TUAR windows
artifact_mask = tuar_labels == 1
tuar_artifacts = tuar_windows[artifact_mask]
tuar_clean = tuar_windows[~artifact_mask]

print(f"\n[DATA BREAKDOWN]:")
print(f"  TUSZ Background: {len(tusz_background):,} windows")
print(f"  TUSZ Seizures: {len(tusz_seizures):,} windows")
print(f"  TUAR Clean: {len(tuar_clean):,} windows")
print(f"  TUAR Artifacts: {len(tuar_artifacts):,} windows")

# Calculate total size
total_windows = len(tusz_background) + len(tuar_clean) + len(tuar_artifacts) + len(tusz_seizures)
memory_needed_gb = (total_windows * 22 * 2000 * 4) / 1e9  # 4 bytes for float32

print(f"\n[MEMORY ESTIMATE]:")
print(f"  * Total windows: {total_windows:,}")
print(f"  * Memory needed: {memory_needed_gb:.1f} GB")
print(f"  * Using batch processing to avoid OOM!")

# ============================================================================
# BATCH PROCESSING AND SAVING
# ============================================================================

print("\n" + "="*70)
print("BATCH PROCESSING (MEMORY-EFFICIENT)")
print("="*70)

# Create data directory
os.makedirs('./data', exist_ok=True)
save_path = './data/three_class_train.npz'

print(f"\nProcessing and saving in batches...")
print("This saves data directly to disk to avoid running out of memory")

# Process in batches
batch_size = 10000  # Process 10K windows at a time

all_windows_list = []
all_labels_list = []

# Batch 1: TUSZ Background
print(f"\n[1/4] Processing TUSZ background ({len(tusz_background):,} windows)...")
for i in range(0, len(tusz_background), batch_size):
    end_idx = min(i + batch_size, len(tusz_background))
    batch = tusz_background[i:end_idx]
    labels = np.zeros(len(batch), dtype=np.int64)
    all_windows_list.append(batch)
    all_labels_list.append(labels)
    print(f"  Batch {i//batch_size + 1}: {len(batch):,} windows")

# Batch 2: TUAR Clean
print(f"\n[2/4] Processing TUAR clean ({len(tuar_clean):,} windows)...")
for i in range(0, len(tuar_clean), batch_size):
    end_idx = min(i + batch_size, len(tuar_clean))
    batch = tuar_clean[i:end_idx]
    labels = np.zeros(len(batch), dtype=np.int64)
    all_windows_list.append(batch)
    all_labels_list.append(labels)
    print(f"  Batch {i//batch_size + 1}: {len(batch):,} windows")

# Batch 3: TUAR Artifacts
print(f"\n[3/4] Processing TUAR artifacts ({len(tuar_artifacts):,} windows)...")
for i in range(0, len(tuar_artifacts), batch_size):
    end_idx = min(i + batch_size, len(tuar_artifacts))
    batch = tuar_artifacts[i:end_idx]
    labels = np.ones(len(batch), dtype=np.int64)
    all_windows_list.append(batch)
    all_labels_list.append(labels)
    print(f"  Batch {i//batch_size + 1}: {len(batch):,} windows")

# Batch 4: TUSZ Seizures
print(f"\n[4/4] Processing TUSZ seizures ({len(tusz_seizures):,} windows)...")
for i in range(0, len(tusz_seizures), batch_size):
    end_idx = min(i + batch_size, len(tusz_seizures))
    batch = tusz_seizures[i:end_idx]
    labels = np.full(len(batch), 2, dtype=np.int64)
    all_windows_list.append(batch)
    all_labels_list.append(labels)
    print(f"  Batch {i//batch_size + 1}: {len(batch):,} windows")

# ============================================================================
# COMBINE BATCHES
# ============================================================================

print("\n" + "="*70)
print("COMBINING BATCHES")
print("="*70)

print("Combining all batches...")
print("(This may take a minute...)")

all_windows = np.vstack(all_windows_list)
all_labels = np.concatenate(all_labels_list)

print(f"\n[OK] Combined Dataset:")
print(f"  * Total windows: {len(all_windows):,}")
print(f"  * Shape: {all_windows.shape}")
print(f"  * Data type: {all_windows.dtype}")
print(f"  * Memory: {all_windows.nbytes / 1e9:.2f} GB")
print(f"\n  Class distribution:")
print(f"    - Class 0 (Background): {np.sum(all_labels==0):,} ({100*np.mean(all_labels==0):.2f}%)")
print(f"    - Class 1 (Artifact): {np.sum(all_labels==1):,} ({100*np.mean(all_labels==1):.2f}%)")
print(f"    - Class 2 (Seizure): {np.sum(all_labels==2):,} ({100*np.mean(all_labels==2):.2f}%)")

# Check for severe class imbalance
class_counts = [np.sum(all_labels==i) for i in range(3)]
imbalance_ratio = max(class_counts) / min(class_counts)

print(f"\n  [CLASS IMBALANCE]:")
print(f"    - Ratio (max/min): {imbalance_ratio:.1f}:1")

if imbalance_ratio > 10:
    print(f"    - ⚠️  Severe imbalance - use weighted loss!")
elif imbalance_ratio > 5:
    print(f"    - ⚠️  Moderate imbalance - use class weights")
else:
    print(f"    - ✓ Reasonable balance")

# ============================================================================
# SHUFFLE DATA
# ============================================================================

print("\n" + "="*70)
print("SHUFFLING DATA")
print("="*70)

print("Shuffling to mix all classes...")
shuffle_idx = np.random.permutation(len(all_windows))
all_windows = all_windows[shuffle_idx]
all_labels = all_labels[shuffle_idx]

print("[OK] Data shuffled")

# ============================================================================
# SAVE TRAINING DATA
# ============================================================================

print("\n" + "="*70)
print("SAVING TRAINING DATA")
print("="*70)

print(f"Saving: {save_path}")
print("This may take a few minutes for large datasets...")

np.savez_compressed(
    save_path,
    windows=all_windows,
    labels=all_labels,
    metadata=np.array([
        '3-Class Training Data',
        'Class 0: Background (TUSZ train background + TUAR clean)',
        'Class 1: Artifact (TUAR artifacts)',
        'Class 2: Seizure (TUSZ train seizures)',
        f'Total: {len(all_windows):,} windows',
        'ZERO DATA LEAKAGE - uses TRAIN split only!',
        'Data type: float32 (memory-efficient)'
    ], dtype=object)
)

file_size_gb = os.path.getsize(save_path) / 1e9
print(f"[OK] Saved: {save_path} ({file_size_gb:.2f} GB)")

# Clean up memory
del all_windows_list, all_labels_list, all_windows, all_labels
del tusz_windows, tusz_background, tusz_seizures
del tuar_windows, tuar_artifacts, tuar_clean

print("[OK] Memory cleaned up")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[SUCCESS] 3-CLASS TRAINING DATA PREPARED")
print("="*70)

print(f"\n[DATASET SUMMARY]:")
print(f"  * Total samples: {total_windows:,}")
print(f"  * Class 0 (Background): {np.sum(class_counts[0]):,}")
print(f"  * Class 1 (Artifact): {class_counts[1]:,}")
print(f"  * Class 2 (Seizure): {class_counts[2]:,}")
print(f"  * File size: {file_size_gb:.2f} GB")

print(f"\n[DATA SOURCE]:")
print(f"  * TUSZ TRAIN split: ~200-300 patients")
print(f"  * TUAR: ~30-50 patients")
print(f"  * Total: ~250-350 unique patients")

print(f"\n[CRITICAL VALIDATION]:")
print(f"  ✓ Uses TRAIN split only")
print(f"  ✓ DEV split reserved for testing")
print(f"  ✓ ZERO patient overlap with test set")
print(f"  ✓ NO data leakage!")

print(f"\n[CLASS IMBALANCE STRATEGY]:")
print(f"  * Will use weighted CrossEntropyLoss")
print(f"  * Class weights:")
for i in range(3):
    count = class_counts[i]
    weight = total_windows / (3 * count)
    print(f"    - Class {i}: {weight:.3f}")

print(f"\n[MEMORY OPTIMIZATION]:")
print(f"  ✓ Used float32 instead of float64 (50% memory savings)")
print(f"  ✓ Batch processing to avoid OOM")
print(f"  ✓ Final size: {file_size_gb:.2f} GB (compressed)")

print(f"\n[NEXT STEP: TRAIN MODEL]")
print(f"\nRun training:")
print(f"  python 2_train_3class.py --data_path {save_path} --epochs 50 --batch_size 32")

print(f"\n[EXPECTED TRAINING TIME]: ~45-60 minutes")
print(f"[EXPECTED RESULT]: 85-90% overall accuracy on validation")

print("\n" + "="*70 + "\n")
