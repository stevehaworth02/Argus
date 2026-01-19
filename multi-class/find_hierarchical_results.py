import numpy as np
from pathlib import Path
import json

results_dir = Path(r"C:\Users\0218s\Desktop\Ceribell-SZ-DTCTR\multi_class\results")

print("="*60)
print("CHECKING ALL FILES IN RESULTS DIRECTORY")
print("="*60)

print(f"\n📁 Directory: {results_dir}\n")

for file in sorted(results_dir.iterdir()):
    print(f"📄 {file.name}")
    print(f"   Size: {file.stat().st_size / 1024:.2f} KB")
    
    # Check .npz files
    if file.suffix == '.npz':
        data = np.load(file, allow_pickle=True)
        print(f"   Keys: {list(data.keys())}")
        for key in data.keys():
            try:
                print(f"      {key}: shape {data[key].shape}")
            except:
                print(f"      {key}: (scalar)")
    
    # Check .json files
    elif file.suffix == '.json':
        with open(file, 'r') as f:
            content = json.load(f)
        print(f"   Keys: {list(content.keys())}")
        if 'accuracy' in content:
            print(f"      Accuracy: {content['accuracy']:.4f}")
        if 'seizure_recall' in content:
            print(f"      Seizure Recall: {content['seizure_recall']:.4f}")
    
    print()

print("="*60)
print("CHECKING FOR HIERARCHICAL-SPECIFIC FILES")
print("="*60)

# Look for hierarchical results
hierarchical_json = results_dir / 'hierarchical_results.json'
if hierarchical_json.exists():
    print(f"\n✅ Found: hierarchical_results.json")
    with open(hierarchical_json, 'r') as f:
        results = json.load(f)
    print(f"\n📊 Hierarchical Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key:30s}: {value:.4f}")
        else:
            print(f"   {key:30s}: {value}")
else:
    print(f"\n❌ No hierarchical_results.json found")

print("\n" + "="*60)