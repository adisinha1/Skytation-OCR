from pathlib import Path

# Check directories exist
dirs = ['data/raw_images', 'data/cropped_plates', 'models/weights']
for d in dirs:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    print(f"✓ {d} exists: {p.exists()}")

# Test imports
try:
    import cv2
    print("✓ OpenCV installed")
except:
    print("✗ OpenCV not installed")

try:
    from ultralytics import YOLO
    print("✓ YOLO installed")
except:
    print("✗ YOLO not installed")

print(f"\n📁 Project root: {Path.cwd()}")
print(f"📁 Models will save to: {Path('models/weights').absolute()}")
print(f"📁 Cropped plates will save to: {Path('data/cropped_plates').absolute()}")
