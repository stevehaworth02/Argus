# Save as: check_predictions_file.py
import numpy as np
from pathlib import Path

pred_file = Path(r"C:\Users\0218s\Desktop\Ceribell-SZ-DTCTR\multi_class\results\predictions.npz")

print("="*60)
print("CHECKING PREDICTIONS FILE")
print("="*60)

print(f"\n📁 File: {pred_file.name}")
print(f"   Size: {pred_file.stat().st_size / 1024:.2f} KB")
print(f"   Exists: {pred_file.exists()}")

if pred_file.exists():
    data = np.load(pred_file, allow_pickle=True)
    
    print(f"\n📊 Keys in file: {list(data.keys())}")
    
    print(f"\n📐 Data shapes:")
    for key in data.keys():
        try:
            shape = data[key].shape
            dtype = data[key].dtype
            print(f"   {key:20s}: shape {shape}, dtype {dtype}")
            
            # Show first few values
            if len(data[key].shape) == 1 and len(data[key]) > 0:
                print(f"      First 5 values: {data[key][:5]}")
                if key in ['y_pred', 'predictions', 'labels', 'y_true']:
                    unique = np.unique(data[key])
                    print(f"      Unique values: {unique}")
                    counts = [(val, np.sum(data[key] == val)) for val in unique]
                    print(f"      Value counts: {counts}")
        except Exception as e:
            print(f"   {key:20s}: (scalar or object) - {e}")
    
    print("\n" + "="*60)
    print("WHAT WE NEED FOR ROC/PR CURVES:")
    print("="*60)
    
    # Check what we have vs what we need
    needs = {
        'True labels': ['y_true', 'labels', 'true_labels'],
        'Predictions': ['y_pred', 'predictions', 'pred_labels'],
        'Probabilities': ['y_pred_proba', 'probabilities', 'probs']
    }
    
    print("\n✅ = Found  |  ❌ = Missing\n")
    for need, possible_keys in needs.items():
        found = False
        for key in possible_keys:
            if key in data.keys():
                print(f"✅ {need:20s}: '{key}'")
                found = True
                break
        if not found:
            print(f"❌ {need:20s}: Not found (checked: {possible_keys})")
    
    print("\n" + "="*60)