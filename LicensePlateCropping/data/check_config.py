# save as check_config.py
import os
from pathlib import Path

print("🔍 Configuration Check")
print("=" * 50)

# Check virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path and 'yolovenv' in venv_path:
    print("✓ Using yolovenv:", venv_path)
else:
    print("⚠️  Not in yolovenv. Run: source ../yolovenv/bin/activate")

# Check current directory
cwd = Path.cwd()
print(f"✓ Current directory: {cwd}")

# Check if in right project folder
if cwd.name == 'licenseplatecropping':
    print("✓ In correct project folder")
else:
    print("⚠️  Not in licenseplatecropping folder")

# Check directories
dirs = {
    'Raw Images': 'data/raw_images',
    'Cropped Plates': 'data/cropped_plates', 
    'Model Weights': 'models/weights',
    'Annotations': 'data/annotations'
}

print("\n📁 Directory Structure:")
for name, path in dirs.items():
    p = Path(path)
    if p.exists():
        files = len(list(p.glob('*')))
        print(f"  ✓ {name}: {path} ({files} files)")
    else:
        p.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {name}: {path} (created)")

print("\n💾 Save Locations:")
print(f"  Models will save to: {Path('models/weights').absolute()}")
print(f"  Crops will save to: {Path('data/cropped_plates').absolute()}")
print("=" * 50)