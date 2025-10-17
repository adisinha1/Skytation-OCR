# License Plate OCR Project

Real-time optical character recognition (OCR) for license plates using an iPhone camera, EasyOCR, and React Native/Expo.

## Project Structure

```
~/Documents/GitHub/Skytation-OCR/
├── LicensePlateOCR/              (Mobile app - Expo/React Native/TypeScript)
│   ├── app/(tabs)/ocr.tsx        (Main OCR screen - MODIFIED)
│   ├── app.json                  (App config - MODIFIED)
│   └── package.json
│
└── LicensePlateOCR-Backend/      (Backend server - Node.js + Python)
    ├── server.js                 (Express server)
    ├── process_frame.py          (EasyOCR + OpenCV processing)
    └── package.json
```

## Quick Start

### 1. Terminal 1 - Backend Server

```bash
cd ~/Documents/GitHub/Skytation-OCR/LicensePlateOCR-Backend

# Activate virtual environment (if using one)
source ~/ocr-env/bin/activate

# Start backend
npm start
```

Should show:

```
✅ Server running on http://0.0.0.0:5001
```

### 2. Terminal 2 - Mobile App

```bash
cd ~/Documents/GitHub/Skytation-OCR/LicensePlateOCR

# Start Expo
npm start

# Press 'i' for iOS simulator or scan QR code with Expo Go on iPhone
```

## Important Configuration

### Backend URL (MUST UPDATE)

In `LicensePlateOCR/app/(tabs)/ocr.tsx`, find this line:

```typescript
const BACKEND_URL = "http://0.0.0.0:5001";
```

**Change `10.0.0.38` to your computer's IP address:**

```bash
# Find your IP:
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Then update in `ocr.tsx` and reload app with `r`.

## What's Working

✅ **Mobile App**

- Real-time camera feed
- High-quality photo capture (quality: 1.0)
- 500ms auto-focus delay
- On-screen guidance for users
- Formatted results display

✅ **Backend Processing**

- EasyOCR text detection (0.95+ confidence on good images)
- Blur detection (rejects images < 50 blur score)
- Brightness detection (auto-adjusts dark/bright images)
- Bounding box visualization (debug images)

✅ **Classification**

- Detects US state names → converts to 2-letter abbreviation
- Extracts 5-7 digit license numbers
- Finds expiration dates (MM/YY or MM-XX format)
- Identifies state slogans (FREEDOM → Pennsylvania, etc.)
- Returns structured JSON with all classifications

## Debug Images

All debug images saved to: `/tmp/ocr_debug/`

Files generated per capture:

- `1_raw_*.jpg` - Original photo
- `2_preprocessed_*.jpg` - After upscaling & denoising
- `3_detections_*.jpg` - With green bounding boxes

View them:

```bash
open /tmp/ocr_debug/
```

## Testing with Static Image (Temporary)

To test with a specific file instead of camera:

**1. Edit `LicensePlateOCR-Backend/server.js`:**

Change line:

```javascript
const testImagePath = "/path/to/your/image.jpg";
```

**2. Restart backend**: `npm start`

**3. Each app button tap** will process that image

**To switch back to camera:** Remove the temp code in `server.js` and restart.

## Current Limitations & Notes

⚠️ **Phone Camera Quality**

- Real photos may have lower confidence than file tests
- Best results: good lighting, straight angle, 1-2 feet away, steady hand

⚠️ **Temporary Settings**

- `server.js` may have debug image path - revert for production
- Remove test-specific code before final use

⚠️ **Virtual Environment**

- Using `~/ocr-env` Python environment
- Must activate before running backend: `source ~/ocr-env/bin/activate`

## Dependencies Installed

**Mobile (already in package.json):**

- expo
- expo-camera
- expo-permissions
- react-native

**Backend:**

- Node: express, cors
- Python: opencv-python, easyocr, pytesseract, pillow, numpy

**System:**

- Tesseract OCR (installed via brew)
- Python 3.12 (at `/opt/homebrew/bin/python3`)

## Model Files

EasyOCR downloads ~200MB of models on first run:

```bash
python3 << 'EOF'
import easyocr
reader = easyocr.Reader(['en'])
EOF
```

Models cached after first download.

## Troubleshooting

**Port 5001 in use:**

```bash
kill -9 $(lsof -t -i :5001)
# Or edit server.js to use different port
```

**Camera won't load in app:**

- Check permissions: Settings → Privacy & Security → Camera
- Reload app with `r` in Expo
- Run: `npm start -- --clear`

**Backend can't find process_frame.py:**

- Make sure you're in `LicensePlateOCR-Backend/` folder
- Check file exists: `ls process_frame.py`

**Python module not found (cv2, easyocr, etc):**

```bash
# Activate environment first
source ~/ocr-env/bin/activate

# Reinstall
pip install opencv-python easyocr pytesseract pillow numpy
```

**Low confidence results:**

- Test with a real printed license plate (not phone screen)
- Ensure good lighting
- Camera should be 1-2 feet away
- Point straight at plate, not at angle
- Check debug image: `open /tmp/ocr_debug/`

## Next Steps / Improvements

- [ ] Train YOLOv8 specifically for license plate detection
- [ ] Add region-of-interest (ROI) cropping to focus on plate area
- [ ] Implement continuous capture mode (not button-based)
- [ ] Add database to store results
- [ ] Cross-reference license numbers against vehicle registration DB
- [ ] Add vehicle make/model detection
- [ ] Improve state slogan dictionary (currently ~10 states)
- [ ] Add support for Canadian provinces
- [ ] Performance optimization for faster processing

## Key Files to Know

| File                                       | Purpose                        | Last Modified |
| ------------------------------------------ | ------------------------------ | ------------- |
| `LicensePlateOCR/app/(tabs)/ocr.tsx`       | Main UI, camera, button logic  | Today         |
| `LicensePlateOCR-Backend/process_frame.py` | OCR processing, classification | Today         |
| `LicensePlateOCR-Backend/server.js`        | Node server, receives frames   | Today         |
| `LicensePlateOCR/app.json`                 | App config, camera permissions | Today         |

## Contact / References

- EasyOCR Docs: https://github.com/JaidedAI/EasyOCR
- OpenCV Docs: https://docs.opencv.org/
- Expo Docs: https://docs.expo.dev/
- React Native: https://reactnative.dev/

---

**Last Updated:** October 16, 2025
**Status:** Working - Manual photo capture, classification functional
**Next Session:** Start backend first, then mobile app, update IP address
