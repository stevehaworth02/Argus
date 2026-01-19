# Save as: find_my_files.py
from pathlib import Path

print("="*60)
print("SEARCHING FOR YOUR SEIZURE DETECTION FILES")
print("="*60)

# Search from the Ceribell project root
root = Path(r"C:\Users\0218s\Desktop\Ceribell-SZ-DTCTR")

print("\n📁 DIRECTORY STRUCTURE:")
print(root)

# List main directories
for item in sorted(root.iterdir()):
    if item.is_dir():
        print(f"  ├─ {item.name}/")

print("\n" + "="*60)
print("SEARCHING FOR KEY FILES...")
print("="*60)

# Find hierarchical pipeline files
print("\n🔍 Hierarchical Pipeline Scripts:")
for f in root.rglob("*hierarchical*.py"):
    print(f"  ✓ {f.relative_to(root)}")

# Find prediction files
print("\n🔍 Prediction Files (.npz):")
for f in root.rglob("*prediction*.npz"):
    print(f"  ✓ {f.relative_to(root)}")
    print(f"    Size: {f.stat().st_size / 1024:.2f} KB")

# Find results directories
print("\n🔍 Results Directories:")
for d in root.rglob("results"):
    if d.is_dir():
        print(f"  ✓ {d.relative_to(root)}/")
        # Show contents
        for f in sorted(d.iterdir())[:10]:  # First 10 files
            print(f"      - {f.name}")

# Find combine directory
print("\n🔍 Checking for 'combine' directory:")
combine_dir = root / "combine"
if combine_dir.exists():
    print(f"  ✅ Found: {combine_dir}")
    print(f"  Contents:")
    for item in sorted(combine_dir.iterdir())[:15]:
        print(f"      - {item.name}")
else:
    print(f"  ❌ No 'combine' directory found")

# Find multi_class directory
print("\n🔍 Checking for 'multi_class' directory:")
multi_class_dir = root / "multi_class"
if multi_class_dir.exists():
    print(f"  ✅ Found: {multi_class_dir}")
    print(f"  Contents:")
    for item in sorted(multi_class_dir.iterdir())[:15]:
        print(f"      - {item.name}")
else:
    print(f"  ❌ No 'multi_class' directory found")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nWhich directory contains your main work?")
print("Look for the directory with 'hierarchical_pipeline' scripts!")